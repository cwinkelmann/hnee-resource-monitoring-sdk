"""A fake GPU-booking server for tests (stdlib only, no dependency on hrm).

It serves the anonymised fixtures and implements the write side of the real
server's contract against an in-memory claims store, so client and booking
tests can run end to end.

Usage (see the ``fake`` fixture in conftest.py)::

    fake.url                                   # "http://127.0.0.1:<port>"
    fake.requests                              # every request, in order
    fake.script("POST", "/api/claims", 409, {"error": "conflict"}, times=1)
    fake.now = datetime(...)                   # the server's clock

Scripted responses win over the real handlers.  They are queued per route:
each one is consumed in order (``times=None`` means "forever"); once the
queue is empty the route falls back to the normal in-memory behaviour.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

FIXTURES = Path(__file__).parent / "fixtures"
BERLIN = ZoneInfo("Europe/Berlin")
MIB_PER_GIB = 1024


def _load(name: str) -> Any:
    return json.loads((FIXTURES / (name + ".json")).read_text())


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


class Recorded:
    """One request as the server saw it."""

    def __init__(self, method, path, query, headers, body, raw):
        self.method = method
        self.path = path  # without the query string
        self.query = query  # {name: first value}
        self.headers = headers  # lower-cased names
        self.body = body  # parsed JSON, or None
        self.raw = raw  # raw bytes

    def __repr__(self):
        return "<Recorded %s %s %r>" % (self.method, self.path, self.body)


class FakeServer:
    def __init__(self) -> None:
        self._now_fixture = _load("now")
        self.users: List[str] = _load("users")["users"]
        self.claims: List[Dict[str, Any]] = [dict(c) for c in _load("claims")["claims"]]
        self._next_id = max(c["id"] for c in self.claims) + 1
        # The server's clock.  Frozen at the fixtures' moment by default so
        # fixture claims stay "current"; tests may assign a datetime.
        self.now: datetime = _parse(_load("claims")["now"])
        self.requests: List[Recorded] = []
        self._scripts: Dict[tuple, list] = {}
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._httpd.daemon_threads = True
        self.url = "http://127.0.0.1:%d" % self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "FakeServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    # -- scripting -------------------------------------------------------
    def script(self, method: str, path: str, status: int, body: Any = None,
               times: Optional[int] = None, delay: float = 0.0) -> None:
        """Answer ``method path`` with ``status``/``body`` (``times`` times,
        or forever when None).  ``body`` may be a str to send invalid JSON.
        ``delay`` sleeps first (for timeout tests)."""
        with self._lock:
            self._scripts.setdefault((method, path), []).append(
                {"status": status, "body": body, "left": times, "delay": delay})

    def requests_to(self, method: str, path: str) -> List[Recorded]:
        return [r for r in self.requests if r.method == method and r.path == path]

    def _next_script(self, method: str, path: str):
        with self._lock:
            queue = self._scripts.get((method, path), [])
            if not queue:
                return None
            entry = queue[0]
            if entry["left"] is not None:
                entry["left"] -= 1
                if entry["left"] <= 0:
                    queue.pop(0)
            return entry

    # -- in-memory domain ------------------------------------------------
    def _total_mib(self, gpu: int) -> Optional[int]:
        for g in self._now_fixture["gpus"]:
            if g["gpu"] == gpu:
                return g["total_mib"]
        return None

    def _active_at(self, claim: Dict[str, Any], t: datetime) -> bool:
        return (claim["cancelled_at"] is None
                and _parse(claim["start"]) <= t < _parse(claim["end"]))

    def _check_capacity(self, gpu: int, need_mib: int, start: datetime, end: datetime):
        """Return None if it fits, else the 409 body.

        Rule: at every boundary instant in [start, end) (start itself and every
        existing booking start inside the window), the active uncancelled
        bookings on the GPU plus the new one must be <= the card total."""
        total = self._total_mib(gpu)
        others = [c for c in self.claims if c["gpu"] == gpu and c["cancelled_at"] is None]
        instants = {start}
        for c in others:
            s = _parse(c["start"])
            if start < s < end:
                instants.add(s)
        worst = max(sum(c["vram_mib"] for c in others if self._active_at(c, t))
                    for t in instants)
        if worst + need_mib > total:
            free_gib = max(total - worst, 0) / MIB_PER_GIB
            return {"error": "conflict",
                    "detail": "GPU %d has only %.1f GiB unbooked in that window" % (gpu, free_gib)}
        return None

    def _new_claim(self, user, gpu, vram_mib, start, end, note, kind) -> Dict[str, Any]:
        claim = {
            "id": self._next_id, "user": user, "gpu": gpu,
            "vram_mib": vram_mib, "vram_gib": round(vram_mib / MIB_PER_GIB, 1),
            "start": _iso(start), "end": _iso(end), "note": note,
            "created_at": _iso(self.now), "created_ip": "192.0.2.10",
            "cancelled_at": None, "cancelled_ip": None, "kind": kind,
        }
        self._next_id += 1
        self.claims.append(claim)
        return claim

    def _usage_now(self) -> Dict[str, Any]:
        """The now fixture, with bookings recomputed from the store."""
        data = json.loads(json.dumps(self._now_fixture))
        for g in data["gpus"]:
            active = [c for c in self.claims if c["gpu"] == g["gpu"]
                      and c["cancelled_at"] is None and _parse(c["end"]) > self.now]
            g["bookings"] = active
            g["booked_mib"] = min(sum(c["vram_mib"] for c in active), g["total_mib"])
            g["free_mib"] = max(g["total_mib"] - g["used_mib"] - g["booked_mib"], 0)
        return data

    # Each handler returns (status, body).
    def create_claim(self, body: Any):
        if not isinstance(body, dict):
            return 400, {"error": "invalid", "detail": "body must be a JSON object"}
        user, gpu = body.get("user"), body.get("gpu")
        if user not in self.users:
            return 400, {"error": "invalid", "detail": "unknown user"}
        if not isinstance(gpu, int) or isinstance(gpu, bool) or self._total_mib(gpu) is None:
            return 400, {"error": "invalid", "detail": "unknown gpu"}
        try:
            start, end = _parse(body["start"]), _parse(body["end"])
        except (KeyError, TypeError, ValueError):
            return 400, {"error": "invalid", "detail": "start and end must be ISO times"}
        if start.tzinfo is None or end.tzinfo is None:
            return 400, {"error": "invalid", "detail": "times need an offset"}
        if end <= start:
            return 400, {"error": "invalid", "detail": "end must be after start"}
        if end - start > timedelta(days=14):
            return 400, {"error": "invalid", "detail": "a booking lasts at most 14 days"}
        if start < self.now - timedelta(minutes=5):
            return 400, {"error": "invalid", "detail": "start is in the past"}
        total = self._total_mib(gpu)
        if body.get("vram_gib") is None:
            need = total  # absent means the whole card
        else:
            gib = body["vram_gib"]
            if not isinstance(gib, (int, float)) or isinstance(gib, bool):
                return 400, {"error": "invalid", "detail": "vram_gib must be a number"}
            need = int(round(gib * MIB_PER_GIB))
            if need < MIB_PER_GIB or need > total:
                return 400, {"error": "invalid", "detail": "vram_gib out of range"}
        conflict = self._check_capacity(gpu, need, start, end)
        if conflict:
            return 409, conflict
        claim = self._new_claim(user, gpu, need, start, end, body.get("note"), "calendar")
        return 201, {"claim": claim}

    def cancel_claim(self, claim_id: int):
        for c in self.claims:
            if c["id"] == claim_id:
                if c["cancelled_at"] is None:  # idempotent: second cancel changes nothing
                    c["cancelled_at"] = _iso(self.now)
                    c["cancelled_ip"] = "192.0.2.10"
                return 200, {"claim": c}
        return 404, {"error": "not found"}

    def quick_claim(self, body: Any):
        """{user, gpu}: user -> whole card until the next 09:00 Berlin;
        user null -> release the GPU's active quick bookings."""
        if not isinstance(body, dict) or self._total_mib(body.get("gpu")) is None:
            return 400, {"error": "invalid", "detail": "unknown gpu"}
        gpu, user = body["gpu"], body.get("user")
        if user is None:
            released = [c for c in self.claims if c["gpu"] == gpu and c["kind"] == "quick"
                        and self._active_at(c, self.now)]
            for c in released:
                c["cancelled_at"], c["cancelled_ip"] = _iso(self.now), "192.0.2.10"
            return 200, {"claim": None}
        if user not in self.users:
            return 400, {"error": "invalid", "detail": "unknown user"}
        local = self.now.astimezone(BERLIN)
        end = local.replace(hour=9, minute=0, second=0, microsecond=0)
        if end <= local:
            end += timedelta(days=1)
        total = self._total_mib(gpu)
        conflict = self._check_capacity(gpu, total, self.now, end)
        if conflict:
            return 409, conflict
        return 201, {"claim": self._new_claim(user, gpu, total, self.now, end,
                                              "quick booking", "quick")}

    # -- HTTP plumbing ---------------------------------------------------
    def _route(self, method: str, path: str, query: Dict[str, str], body: Any):
        if method == "GET":
            if path == "/api/users":
                return 200, {"users": self.users}
            if path == "/api/now":
                return 200, self._usage_now()
            if path == "/api/claims":
                try:
                    days, back = int(query.get("days", 14)), int(query.get("back", 7))
                except ValueError:
                    return 400, {"error": "invalid", "detail": "days/back must be integers"}
                if not (1 <= days <= 14 and 1 <= back <= 30):
                    return 400, {"error": "invalid", "detail": "days 1..14, back 1..30"}
                lo, hi = self.now - timedelta(days=back), self.now + timedelta(days=days)
                shown = [c for c in self.claims
                         if _parse(c["end"]) > lo and _parse(c["start"]) < hi]
                return 200, {"now": _iso(self.now), "claims": shown}
        elif method == "POST":
            if path == "/api/claims":
                return self.create_claim(body)
            if path == "/api/claims/quick":
                return self.quick_claim(body)
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "claims"] and parts[3] == "cancel" \
                    and parts[2].isdigit():
                return self.cancel_claim(int(parts[2]))
        return 404, {"error": "not found"}

    def _make_handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # keep test output quiet
                pass

            def _handle(self, method: str) -> None:
                parts = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                headers = {k.lower(): v for k, v in self.headers.items()}
                body = None
                if raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        body = None
                with fake._lock:
                    fake.requests.append(Recorded(method, parts.path, query, headers, body, raw))
                script = fake._next_script(method, parts.path)
                if script is not None:
                    if script["delay"]:
                        time.sleep(script["delay"])
                    status, payload = script["status"], script["body"]
                elif method == "POST" and "application/json" not in headers.get("content-type", ""):
                    status, payload = 415, {"error": "unsupported media type"}
                elif method == "POST" and raw and body is None:
                    status, payload = 400, {"error": "invalid", "detail": "invalid JSON"}
                else:
                    with fake._lock:  # the store is shared between handler threads
                        status, payload = fake._route(method, parts.path, query, body)
                data = payload if isinstance(payload, str) else json.dumps(payload)
                data = data.encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up (timeout test)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        return Handler
