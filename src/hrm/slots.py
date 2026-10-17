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

import logging
import threading
import time

from .errors import Conflict, NotFound, Preempted
from .models import Booking, parse_time

log = logging.getLogger("hrm.slots")

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


class Watcher:
    """Parks one long-poll per row of a slot, in daemon threads.

    The threads only ever *set* things; the job asks at a safe point with
    Slot.check(). Raising into another thread is not something Python does
    reliably, and the middle of a CUDA kernel is not where anyone wants to
    checkpoint.
    """

    def __init__(self, slot: "Slot",
                 on_taken: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.slot = slot
        self.on_taken = on_taken
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._live = 0
        self._lock = threading.Lock()

    @property
    def watching(self) -> bool:
        """True while at least one long-poll is parked. A server with no /events
        route leaves this False, and the job runs on unwatched."""
        with self._lock:
            return self._live > 0

    def __enter__(self) -> "Watcher":
        for booking in self.slot.bookings:
            thread = threading.Thread(target=self._loop, args=(booking.id,),
                                      name="hrm-watch-%d" % booking.id, daemon=True)
            with self._lock:
                self._live += 1
            self._threads.append(thread)
            thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        """Stop listening and return at once.

        It does not wait for a parked long-poll: the server holds one for up to a
        minute, the threads are daemons, and this runs on the exit path that
        notice_seconds=0 promises to be quick. A poll that answers after this is
        discarded -- see _deliver -- so nothing can set `taken` on a slot that has
        already been released.
        """
        self._stop.set()
        with self._lock:
            self._live = 0              # say so now, not in five seconds
        self._threads = []

    def _deliver(self, event: Dict[str, Any]) -> bool:
        """Hand an event to the slot, unless we have been stopped. Returns whether
        it was delivered."""
        if self._stop.is_set():
            return False
        self.slot._record_taken(event)
        if self.on_taken is not None:
            try:
                self.on_taken(event)
            except Exception:
                log.exception("on_taken callback raised")
        return True

    def _missed(self, booking_id: int, end: datetime) -> Optional[Dict[str, Any]]:
        """Did the card go while nobody was listening?

        The server drops an event with no watcher parked, so a take delivered in the
        gap between two long-polls is lost for good. The row still shows it: a
        cancelled booking, or an end that moved earlier, means the card went.
        """
        try:
            rows = {b.id: b for b in
                    self.slot.client.bookings(include_cancelled=True)}
        except Exception as exc:
            log.warning("could not re-read booking %d (%s)", booking_id,
                        exc.__class__.__name__)
            return None
        row = rows.get(booking_id)
        if row is None:
            return None
        if row.cancelled_at is not None:
            return {"event": "cancelled", "effective_mib": 0, "lost_mib": row.vram_mib,
                    "by": [], "truncated": False, "cancelled": True,
                    "deadline": row.cancelled_at.isoformat(), "noticed_late": True}
        if row.end < end:
            return {"event": "taken", "effective_mib": 0, "lost_mib": row.vram_mib,
                    "by": sorted({u for seg in row.taken for u in seg.get("by", ())}),
                    "truncated": True, "cancelled": False,
                    "deadline": row.end.isoformat(), "noticed_late": True}
        return None

    def _loop(self, booking_id: int) -> None:
        end = next((b.end for b in self.slot.bookings if b.id == booking_id),
                   self.slot.end)
        try:
            while not self._stop.is_set():
                try:
                    event = self.slot.client.watch_claim(booking_id)
                except NotFound:
                    log.warning("this server has no /events route for booking %d (or "
                                "the booking is gone); running unwatched", booking_id)
                    return
                except Conflict:
                    return              # cancelled or over: there is nothing to watch
                except Exception as exc:        # a blip, a restart, a timeout
                    log.warning("watch on booking %d failed (%s); retrying",
                                booking_id, exc.__class__.__name__)
                    if self._stop.wait(5.0):
                        return
                    continue
                if event is None:
                    # 204: the hold expired. Before reconnecting, check whether the
                    # card went while no poll was parked.
                    missed = self._missed(booking_id, end)
                    if missed is None:
                        continue
                    event = missed
                self._deliver(event)
                return
        finally:
            with self._lock:
                self._live = max(self._live - 1, 0)


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
        self.taken: Optional[Dict[str, Any]] = None   # set by a Watcher, read by check()
        self.watcher: Optional[Watcher] = None
        self.wait_poll = 10.0        # seconds between looks while `with slot:` waits

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

    def release(self, end: Optional[Union[datetime, str]] = None) -> List[Dict[str, Any]]:
        """Give every row of this slot back, and return the slots that moved earlier.

        Safe on every path: this is what a job's `finally:` calls, so a slot that is
        already cancelled, already over, or gone from the server is not an error
        worth taking a job down for. Anything else -- a 500, an unreachable server --
        still raises.
        """
        moved: List[Dict[str, Any]] = []
        for booking in self.bookings:
            try:
                moved.extend(self.client.release_claim(booking.id, end)["advanced"])
            except (NotFound, Conflict):
                continue      # already cancelled, already over, or never existed
        return moved

    # --- being told the card was taken ---------------------------------------------
    def _record_taken(self, event: Dict[str, Any]) -> None:
        self.taken = event

    @property
    def watcher_parked(self) -> bool:
        """Is anyone actually listening for this slot right now? The server reads the
        same fact from its end, and only honours a declared notice while it holds."""
        return self.watcher is not None and self.watcher.watching

    def watch(self, on_taken: Optional[Callable[[Dict[str, Any]], None]] = None
              ) -> Watcher:
        """Park a long-poll per row, so the server can say in milliseconds that an
        important booking took the card.

            with slot.watch():
                for epoch in range(100):
                    slot.check()        # raises Preempted once it is taken
                    train_one_epoch()
        """
        self.watcher = Watcher(self, on_taken)
        return self.watcher

    def check(self) -> None:
        """Raise Preempted if the card has been taken. Call it where stopping is
        safe -- between epochs, after a checkpoint."""
        if self.taken is not None:
            by = ", ".join(self.taken.get("by") or []) or "someone"
            raise Preempted("GPU %d was taken by %s" % (self.gpu, by), self.taken)

    # --- the whole job pattern -----------------------------------------------------
    def __enter__(self) -> "Slot":
        """Wait for the window, then park the watcher.

            with hrm.reserve(memory=19.5, hours=1, notice_seconds=0) as slot:
                for epoch in range(epochs):
                    slot.check()          # raises Preempted if the card was taken
                    train_one_epoch()
                    save_checkpoint()

        Leaving the block stops the watcher and releases the booking, on every path.
        """
        try:
            self.wait_until_start(poll=self.wait_poll)
        except BaseException:
            # The wait is inside __enter__, so __exit__ never runs if it raises --
            # a Conflict or a Ctrl-C while queued would otherwise leave the
            # reservation holding a window nobody will use.
            self.release()
            raise
        self.watch().__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        try:
            if self.watcher is not None:
                self.watcher.stop()
        finally:
            self.release()
        return False
