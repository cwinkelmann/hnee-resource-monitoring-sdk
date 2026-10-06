from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional, Sequence, Tuple, Union
from zoneinfo import ZoneInfo

from .models import Booking

TZ = ZoneInfo("Europe/Berlin")
MAX_DAYS = 14

Window = Tuple[datetime, datetime]


def parse_days(days: Sequence[Union[date, str]]) -> List[date]:
    """Accept dates or 'YYYY-MM-DD' strings; dedup and sort."""
    out = set()
    for x in days:
        if isinstance(x, datetime):
            out.add(x.date())
        elif isinstance(x, date):
            out.add(x)
        elif isinstance(x, str):
            try:
                out.add(datetime.strptime(x, "%Y-%m-%d").date())
            except ValueError:
                raise ValueError("bad day %r, expected YYYY-MM-DD" % (x,)) from None
        else:
            raise ValueError("bad day %r, expected date or YYYY-MM-DD" % (x,))
    return sorted(out)


def day_runs(days: List[date]) -> List[List[date]]:
    """Split sorted days into runs of consecutive dates."""
    runs: List[List[date]] = []
    for day in days:
        if runs and day - runs[-1][-1] == timedelta(days=1):
            runs[-1].append(day)
        else:
            runs.append([day])
    return runs


def _local_midnight(day: date) -> datetime:
    return datetime.combine(day, time(0), tzinfo=TZ)


def windows_for_days(days: List[date], now: datetime) -> List[Window]:
    """One UTC window per run of consecutive days (Europe/Berlin days)."""
    today = now.astimezone(TZ).date()
    for day in days:
        if day < today:
            raise ValueError("%s is in the past" % day)
    windows: List[Window] = []
    for run in day_runs(days):
        if len(run) > MAX_DAYS:
            raise ValueError("a booking may last at most %d days" % MAX_DAYS)
        start = max(now, _local_midnight(run[0]))
        end = _local_midnight(run[-1] + timedelta(days=1))
        assert end > start, "empty window"
        windows.append((start.astimezone(timezone.utc), end.astimezone(timezone.utc)))
    return windows


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt


def window_from(start: datetime, end: datetime, now: datetime) -> Window:
    """Explicit window; naive datetimes are Europe/Berlin."""
    start, end = _aware(start), _aware(end)
    if end <= start:
        raise ValueError("end must be after start")
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def free_mib_at_worst(
    bookings: Sequence[Booking], gpu: int, total_mib: int, start: datetime, end: datetime
) -> int:
    """Free VRAM at the busiest instant of [start, end), as the server computes it."""
    mine = [b for b in bookings if b.gpu == gpu and not b.cancelled]
    instants = [start] + [b.start for b in mine if start < b.start < end]
    worst = max(sum(b.vram_mib for b in mine if b.active_at(t)) for t in instants)
    return total_mib - worst


def _min_free(gpu: Tuple[int, int], bookings, windows) -> int:
    index, total = gpu
    return min(free_mib_at_worst(bookings, index, total, s, e) for s, e in windows)


def choose_gpus(
    gpus: Sequence[Tuple[int, int]],
    bookings: Sequence[Booking],
    windows: Sequence[Window],
    need_mib: Optional[int],
) -> List[int]:
    """GPUs that fit every window, ascending; need_mib=None means the whole card."""
    out = []
    for g in sorted(gpus):
        free = _min_free(g, bookings, windows)
        if (free == g[1]) if need_mib is None else (free >= need_mib):
            out.append(g[0])
    return out


def best_effort(
    gpus: Sequence[Tuple[int, int]], bookings: Sequence[Booking], windows: Sequence[Window]
) -> Tuple[int, int]:
    """(index, min free MiB) of the GPU with the most room, for error messages."""
    return max(
        ((g[0], _min_free(g, bookings, windows)) for g in sorted(gpus)),
        key=lambda x: x[1],
    )
