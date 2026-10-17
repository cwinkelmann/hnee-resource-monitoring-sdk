"""A reserved slot: what a job holds, waits for, is told about, and gives back.

The booking server assigns the slot; this module is the job's side of that
contract. A reservation never fails for lack of capacity -- it comes back with a
start, which may be in the future and may move earlier -- so the job's shape is
always: reserve, wait for the start, run, release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import time

from .errors import Conflict
from .models import Booking, parse_time

MIB_PER_GIB = 1024
MAX_HOURS = 14 * 24
MAX_GPUS = 8


def _check_shape(memory: Optional[float], hours: float, count: int) -> None:
    """The client-side rules, checked before anything is sent: a bad shape is a
    programming error in the caller, not a server round trip."""
    if memory is not None and (isinstance(memory, bool)
                               or not isinstance(memory, (int, float))
                               or not memory > 0):
        raise ValueError("memory must be a number of GiB above 0, got %r" % (memory,))
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) \
            or not 0 < hours <= MAX_HOURS:
        raise ValueError("hours must be a number in (0, %d], got %r" % (MAX_HOURS, hours))
    if isinstance(count, bool) or not isinstance(count, int) \
            or not 1 <= count <= MAX_GPUS:
        raise ValueError("count must be a whole number of GPUs, 1..%d, got %r"
                         % (MAX_GPUS, count))


@dataclass(frozen=True)
class Capacity:
    """What one card has free now, and when a given shape would next fit.

    Ask the server for this rather than summing bookings yourself: clients that
    reimplemented the arithmetic got it wrong, which is why the route exists.
    """
    free_mib: int
    free_gib: float
    earliest: Optional[datetime]      # None: nothing fits inside the server's horizon
    gpu: Optional[int]
    gpus: List[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Capacity":
        earliest = d.get("earliest")
        free_mib = int(d["free_mib"])
        return cls(
            free_mib=free_mib,
            free_gib=float(d.get("free_gib", free_mib / MIB_PER_GIB)),
            earliest=None if earliest is None else parse_time(earliest),
            gpu=d.get("gpu"),
            gpus=list(d.get("gpus") or []),
        )


class Slot:
    """One reserved window, over one card or several.

    A slot is not a lock: it is the calendar row the server assigned, and the
    honour system does the rest. `start` may be in the future -- wait for it with
    wait_until_start() -- and may move earlier while you wait.
    """

    def __init__(self, client, bookings: List[Booking], assigned_later: bool,
                 revoked: Optional[List[Dict[str, Any]]] = None) -> None:
        if not bookings:
            raise ValueError("a slot needs at least one booking")
        self.client = client
        self.bookings = sorted(bookings, key=lambda b: (b.start, b.id))
        self.assigned_later = bool(assigned_later)
        self.revoked = list(revoked or [])

    def __repr__(self) -> str:
        return "Slot(ids=%r, gpus=%r, %s..%s)" % (
            [b.id for b in self.bookings], self.gpus,
            self.start.isoformat(), self.end.isoformat())

    @property
    def booking(self) -> Booking:
        return self.bookings[0]

    @property
    def id(self) -> int:
        return self.bookings[0].id

    @property
    def gpu(self) -> int:
        return self.bookings[0].gpu

    @property
    def gpus(self) -> List[int]:
        return sorted(b.gpu for b in self.bookings)

    @property
    def start(self) -> datetime:
        return min(b.start for b in self.bookings)

    @property
    def end(self) -> datetime:
        return max(b.end for b in self.bookings)

    @property
    def memory_gib(self) -> float:
        return round(self.bookings[0].vram_mib / MIB_PER_GIB, 2)

    @property
    def cancelled(self) -> bool:
        """True once every row of this slot is cancelled."""
        return all(b.cancelled_at is not None for b in self.bookings)

    def refresh(self) -> "Slot":
        """Re-read this slot's rows. The server may have moved them earlier, and a
        slot the server no longer lists keeps the rows we already have."""
        ids = {b.id for b in self.bookings}
        found = [b for b in self.client.bookings(include_cancelled=True)
                 if b.id in ids]
        if found:
            self.bookings = sorted(found, key=lambda b: (b.start, b.id))
        return self

    def wait_until_start(self, poll: float = 10.0, timeout: Optional[float] = None,
                         sleep: Optional[Callable[[float], None]] = None) -> "Slot":
        """Block until the slot's window has begun, then return self.

        Re-reads the rows on every tick, so a slot that moves earlier shortens the
        wait. Raises Conflict if the slot is cancelled under us -- otherwise a job
        would wait for a window that no longer exists -- and TimeoutError if
        `timeout` seconds pass first.
        """
        rest = sleep or time.sleep
        waited = 0.0
        while True:
            if self.cancelled:
                raise Conflict("slot %d was cancelled while waiting for its start"
                               % self.id)
            if self.start <= self.client.now():
                return self
            if timeout is not None and waited >= timeout:
                raise TimeoutError("slot %d still starts at %s after %.0fs"
                                   % (self.id, self.start.isoformat(), waited))
            rest(poll)
            waited += poll
            self.refresh()
