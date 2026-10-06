"""hrm: book GPUs on the shared box from Python.

    import hrm
    hrm.connect(user="alice")          # or set HRM_USER / HRM_URL
    hrm.usage().to_frame()
    print(hrm.book(days=["2026-10-06", "2026-10-07"], memory=48))
"""
from __future__ import annotations

__version__ = "0.1.0"  # before the imports below: client.py reads it

from datetime import date, datetime  # noqa: E402
from typing import Iterable, List, Optional, Union  # noqa: E402

from .client import Client  # noqa: E402
from .errors import (Conflict, Forbidden, HRMError, InvalidRequest, NoCapacity,  # noqa: E402
                     NotFound, RateLimited, ServerUnavailable, UnknownUser)
from .models import (Alert, Booking, BookingList, BookingResult, Gpu, Process,  # noqa: E402
                     Usage)

__all__ = [
    "__version__",
    "connect", "users", "usage", "bookings", "book", "cancel", "hold", "release",
    "Client",
    "Alert", "Booking", "BookingList", "BookingResult", "Gpu", "Process", "Usage",
    "HRMError", "InvalidRequest", "Forbidden", "NotFound", "Conflict", "RateLimited",
    "ServerUnavailable", "NoCapacity", "UnknownUser",
]

_default: Optional[Client] = None


def connect(user: Optional[str] = None, url: Optional[str] = None) -> Client:
    """(Re)create the default client used by the module-level functions."""
    global _default
    _default = Client(url=url, user=user)
    return _default


def _client() -> Client:
    global _default
    if _default is None:
        _default = Client()
    return _default


def users() -> List[str]:
    return _client().users()


def usage() -> Usage:
    return _client().usage()


def bookings(days_ahead: int = 14, days_back: int = 7,
             include_cancelled: bool = False) -> BookingList:
    return _client().bookings(days_ahead=days_ahead, days_back=days_back,
                              include_cancelled=include_cancelled)


def book(days: Optional[Iterable[Union[date, str]]] = None, *,
         memory: Optional[float] = None, gpu: Optional[int] = None,
         start: Optional[Union[datetime, str]] = None,
         end: Optional[Union[datetime, str]] = None,
         note: Optional[str] = None, user: Optional[str] = None) -> BookingResult:
    return _client().book(days, memory=memory, gpu=gpu, start=start, end=end,
                          note=note, user=user)


def cancel(booking_or_id: Union[Booking, int]) -> Booking:
    return _client().cancel(booking_or_id)


def hold(gpu: int, user: Optional[str] = None) -> Booking:
    return _client().hold(gpu, user=user)


def release(gpu: int) -> None:
    return _client().release(gpu)
