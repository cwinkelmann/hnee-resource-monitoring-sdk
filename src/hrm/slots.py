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
