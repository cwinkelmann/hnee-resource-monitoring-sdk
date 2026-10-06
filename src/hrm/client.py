from __future__ import annotations

import contextlib
import getpass
import http.client
import json
import os
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from typing import Callable, Iterable, List, Optional, Union

from . import __version__, planning
from .errors import (Conflict, HRMError, NoCapacity, ServerUnavailable, UnknownUser,
                     error_for_status)
from .models import Booking, BookingList, BookingResult, Usage

DEFAULT_URL = "http://10.188.1.1:8765"
MAX_ATTEMPTS = 3  # book(): POST attempts in total, across candidate GPUs
MIB_PER_GIB = 1024


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_datetime(value: Union[datetime, str], name: str) -> datetime:
    """A datetime or ISO string; naive values stay naive (Berlin, see planning)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    raise ValueError("bad %s %r, expected a datetime or ISO string" % (name, value))


@contextlib.contextmanager
def _shape(path: str):
    """Turn parsing failures on a well-formed but wrongly shaped body into HRMError."""
    try:
        yield
    except (KeyError, TypeError, ValueError, AttributeError):
        raise HRMError("unexpected response shape from %s" % path) from None


class Client:
    """A client for the booking server.  Raises subclasses of HRMError."""

    def __init__(self, url: Optional[str] = None, user: Optional[str] = None,
                 timeout: float = 10.0,
                 clock: Optional[Callable[[], datetime]] = None) -> None:
        self.url = (url or os.environ.get("HRM_URL") or DEFAULT_URL).rstrip("/")
        self.user = user
        self.timeout = timeout
        # zero-arg callable returning an aware UTC datetime; injectable for tests
        self._clock = clock or _utc_now
        self._users_cache: Optional[List[str]] = None

    # -- transport -------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        headers = {"User-Agent": "hrm-sdk/" + __version__, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        # urllib sends no Origin header; the server rejects cross-origin writes.
        if not self.url.lower().startswith(("http://", "https://")):
            raise HRMError("invalid server URL %r: must start with http:// or https://" % self.url)
        try:
            req = urllib.request.Request(self.url + path, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                parsed = json.loads(e.read())
            except (ValueError, OSError, http.client.HTTPException):
                parsed = None  # unreadable error body: still report the status
            raise error_for_status(e.code, parsed) from None
        except ValueError as e:  # malformed URL: raised before any request is made
            raise HRMError("invalid server URL %r: %s" % (self.url, e)) from None
        except urllib.error.URLError as e:
            raise ServerUnavailable(
                "cannot reach %s: %s" % (self.url, e.reason), None) from None
        except (OSError, http.client.HTTPException) as e:  # timeouts, resets mid-read
            raise ServerUnavailable("cannot reach %s: %s" % (self.url, e), None) from None
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise HRMError("server sent invalid JSON for %s %s" % (method, path)) from None
        if not isinstance(parsed, dict):
            raise HRMError("server sent unexpected JSON for %s %s" % (method, path))
        return parsed

    # -- read API --------------------------------------------------------
    def users(self) -> List[str]:
        data = self._request("GET", "/api/users")
        with _shape("/api/users"):
            users = data["users"]
            if not isinstance(users, list):
                raise TypeError("users is not a list")
            return list(users)

    def usage(self) -> Usage:
        data = self._request("GET", "/api/now")
        with _shape("/api/now"):
            return Usage.from_json(data)

    def bookings(self, days_ahead: int = 14, days_back: int = 7,
                 include_cancelled: bool = False) -> BookingList:
        if not 1 <= days_ahead <= 14:
            raise ValueError("days_ahead must be 1..14, got %r" % (days_ahead,))
        if not 1 <= days_back <= 30:
            raise ValueError("days_back must be 1..30, got %r" % (days_back,))
        data = self._request("GET", "/api/claims?days=%d&back=%d" % (days_ahead, days_back))
        with _shape("/api/claims"):
            found = [Booking.from_json(c) for c in data["claims"]]
        return BookingList(b for b in found if include_cancelled or not b.cancelled)

    # -- user resolution -------------------------------------------------
    def resolve_user(self, explicit: Optional[str] = None) -> str:
        """explicit -> connect(user=) -> HRM_USER -> local login if a known account."""
        user = explicit or self.user or os.environ.get("HRM_USER")
        if user:
            return user
        login = getpass.getuser()
        if self._users_cache is None:
            self._users_cache = self.users()
        if login in self._users_cache:
            return login
        raise UnknownUser(
            "set hrm.connect(user=...) or HRM_USER; %r is not a carrot account" % login)

    # -- write API -------------------------------------------------------
    def _post_claim(self, path: str, body: dict) -> Booking:
        data = self._request("POST", path, body)
        with _shape(path):
            return Booking.from_json(data["claim"])

    def book(self, days: Optional[Iterable[Union[date, str]]] = None, *,
             memory: Optional[float] = None, gpu: Optional[int] = None,
             start: Optional[Union[datetime, str]] = None,
             end: Optional[Union[datetime, str]] = None,
             note: Optional[str] = None, user: Optional[str] = None) -> BookingResult:
        """Book ``memory`` GiB (None: the whole card) on one GPU for ``days``
        (Europe/Berlin dates) or for ``start``..``end``.  One booking per run
        of consecutive days; the lowest-index GPU that fits is used."""
        has_range = start is not None or end is not None
        if (days is None) == (not has_range):
            raise ValueError("pass either days or start/end, not both")
        if memory is not None and not memory > 0:
            raise ValueError("memory must be > 0 GiB, got %r" % (memory,))
        now = self._clock()
        if days is not None:
            day_list = planning.parse_days(days)
            if not day_list:
                raise ValueError("days must name at least one day")
            windows = planning.windows_for_days(day_list, now)
        else:
            if start is None or end is None:
                raise ValueError("pass both start and end")
            windows = [planning.window_from(
                _to_datetime(start, "start"), _to_datetime(end, "end"), now)]
        who = self.resolve_user(user)

        cards = [(g.index, g.total_mib) for g in self.usage().gpus]
        if gpu is not None:
            cards = [c for c in cards if c[0] == gpu]
            if not cards:
                raise ValueError("GPU %r does not exist" % (gpu,))
        claims = self.bookings(days_ahead=14, days_back=1)
        need_mib = None if memory is None else int(round(memory * MIB_PER_GIB))
        candidates = planning.choose_gpus(cards, claims, windows, need_mib)
        if not candidates:
            raise NoCapacity(self._no_capacity(cards, claims, windows, memory, gpu))

        conflict: Optional[Conflict] = None
        for index in candidates[:MAX_ATTEMPTS]:
            try:
                made = self._book_windows(index, windows, who, memory, note)
            except Conflict as e:  # a race: someone booked since we planned
                conflict = e
                continue
            return BookingResult(index, made)
        assert conflict is not None
        raise conflict

    def _book_windows(self, gpu: int, windows: List[planning.Window], user: str,
                      memory: Optional[float], note: Optional[str]) -> List[Booking]:
        """POST every window on ``gpu``; on any failure cancel what was made, re-raise."""
        made: List[Booking] = []
        try:
            for start, end in windows:
                body = {"user": user, "gpu": gpu,
                        "start": start.isoformat(), "end": end.isoformat()}
                if memory is not None:
                    body["vram_gib"] = float(memory)
                if note is not None:
                    body["note"] = note
                made.append(self._post_claim("/api/claims", body))
        except Exception:
            for booking in made:  # best effort: never mask the original error
                try:
                    self.cancel(booking)
                except Exception:
                    pass
            raise
        return made

    @staticmethod
    def _no_capacity(cards, claims, windows, memory, gpu) -> str:
        index, free_mib = planning.best_effort(cards, claims, windows)
        free_gib = free_mib / MIB_PER_GIB
        if gpu is not None:
            want = "the whole card" if memory is None else "%g GiB" % memory
            return "GPU %d has only %.1f GiB free for the whole window, need %s" % (
                index, free_gib, want)
        if memory is None:
            head = "no GPU is entirely free for the whole window"
        else:
            head = "no GPU has %g GiB free for the whole window" % memory
        return "%s; best: GPU %d with %.1f GiB" % (head, index, free_gib)

    def cancel(self, booking_or_id: Union[Booking, int]) -> Booking:
        """Cancel a booking (idempotent); returns it with ``cancelled_at`` set."""
        if isinstance(booking_or_id, Booking):
            booking_id = booking_or_id.id
        else:
            booking_id = int(booking_or_id)
        return self._post_claim("/api/claims/%d/cancel" % booking_id, {})

    def hold(self, gpu: int, user: Optional[str] = None) -> Booking:
        """Quick hold: the whole card from now until the next 09:00 Berlin."""
        return self._post_claim("/api/claims/quick",
                                {"user": self.resolve_user(user), "gpu": gpu})

    def release(self, gpu: int) -> None:
        """End the GPU's active quick holds."""
        self._request("POST", "/api/claims/quick", {"user": None, "gpu": gpu})
