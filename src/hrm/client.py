from __future__ import annotations

import contextlib
import getpass
import http.client
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from . import __version__
from .errors import HRMError, ServerUnavailable, UnknownUser, error_for_status
from .models import Booking, BookingList, Usage

DEFAULT_URL = "http://10.188.1.1:8765"


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
                 timeout: float = 10.0) -> None:
        self.url = (url or os.environ.get("HRM_URL") or DEFAULT_URL).rstrip("/")
        self.user = user
        self.timeout = timeout
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
