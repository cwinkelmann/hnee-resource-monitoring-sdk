from __future__ import annotations

from typing import Optional


class HRMError(Exception):
    """Base class for all SDK errors."""

    def __init__(self, detail: str = "", status: Optional[int] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail

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


def error_for_status(status: int, body: Optional[dict]) -> HRMError:
    """Map an HTTP error status and decoded JSON body to an exception."""
    body = body if isinstance(body, dict) else {}
    detail = body.get("detail") or body.get("error") or "HTTP %d" % status
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
    return cls(str(detail), status)
