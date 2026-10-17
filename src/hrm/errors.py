from __future__ import annotations

from typing import Optional


class HRMError(Exception):
    """Base class for all SDK errors."""

    def __init__(self, detail: str = "", status: Optional[int] = None,
                 extra: Optional[dict] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        # Whatever the server attached beyond the fixed shape: on a 409 that is
        # `earliest` and `free_now`, or `max_pending_per_user` and `pending`. Without
        # it a caller is told "conflict" and has nothing to act on.
        self.extra = extra or {}

    def __str__(self) -> str:
        if not self.detail and self.status is not None:
            return "HTTP %d" % self.status  # e.g. a bodyless 502 from a proxy
        return self.detail


class InvalidRequest(HRMError):
    """HTTP 400, 413 or 415."""


class Forbidden(HRMError):
    """HTTP 403."""


class NotFound(HRMError):
    """HTTP 404."""


class Conflict(HRMError):
    """HTTP 409."""


class RateLimited(HRMError):
    """HTTP 429."""


class ServerUnavailable(HRMError):
    """HTTP 503 and other 5xx, network errors, timeouts."""


class NoCapacity(HRMError):
    """Raised by the client when no GPU can hold the booking."""


class UnknownUser(HRMError):
    """Raised by the client when the user cannot be resolved."""


class Preempted(HRMError):
    """An important booking took this slot's card. Checkpoint and stop.

    ``event`` is the server's payload: how much was lost, by whom, and whether the
    row was truncated or cancelled.
    """

    def __init__(self, detail: str = "", event: Optional[dict] = None) -> None:
        super().__init__(detail)
        self.event = event or {}


def error_for_status(status: int, body: Optional[dict]) -> HRMError:
    """Map an HTTP error status and decoded JSON body to an exception."""
    body = body if isinstance(body, dict) else {}
    detail = body.get("detail") or body.get("error") or "HTTP %d" % status
    extra = {k: v for k, v in body.items() if k not in ("detail", "error")}
    if status in (400, 413, 415):
        cls = InvalidRequest
    elif status == 403:
        cls = Forbidden
    elif status == 404:
        cls = NotFound
    elif status == 409:
        cls = Conflict
    elif status == 429:
        cls = RateLimited
    elif status >= 500:
        cls = ServerUnavailable
    else:
        cls = HRMError
    return cls(str(detail), status, extra)
