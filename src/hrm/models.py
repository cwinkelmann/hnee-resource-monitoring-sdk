from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
_PANDAS_MSG = 'pip install "hnee-resource-monitoring[pandas]"'


def parse_time(s: str) -> datetime:
    """Parse a server ISO timestamp into an aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt


def _opt_time(s: Optional[str]) -> Optional[datetime]:
    return parse_time(s) if s else None


def _pandas():
    try:
        import pandas
    except ImportError:
        raise ImportError(_PANDAS_MSG) from None
    return pandas


@dataclass(frozen=True)
class Process:
    pid: int
    user: Optional[str]
    name: str
    used_mib: int

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Process":
        return cls(pid=d["pid"], user=d.get("user"), name=d["name"], used_mib=d["used_mib"])


@dataclass(frozen=True)
class Booking:
    id: int
    user: str
    gpu: int
    vram_mib: int
    start: datetime
    end: datetime
    note: Optional[str]
    created_at: datetime
    created_ip: Optional[str]
    cancelled_at: Optional[datetime]
    cancelled_ip: Optional[str]
    kind: str

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Booking":
        return cls(
            id=d["id"],
            user=d["user"],
            gpu=d["gpu"],
            vram_mib=d["vram_mib"],
            start=parse_time(d["start"]),
            end=parse_time(d["end"]),
            note=d.get("note"),
            created_at=parse_time(d["created_at"]),
            created_ip=d.get("created_ip"),
            cancelled_at=_opt_time(d.get("cancelled_at")),
            cancelled_ip=d.get("cancelled_ip"),
            kind=d.get("kind", "calendar"),
        )

    @property
    def vram_gib(self) -> float:
        return round(self.vram_mib / 1024, 1)

    @property
    def cancelled(self) -> bool:
        return self.cancelled_at is not None

    def active_at(self, t: datetime) -> bool:
        return not self.cancelled and self.start <= t < self.end

    def __str__(self) -> str:
        fmt = "%a %d %b %H:%M"
        return "#%d GPU %d · %.1f GiB · %s · %s → %s" % (
            self.id,
            self.gpu,
            self.vram_gib,
            self.user,
            self.start.astimezone(BERLIN).strftime(fmt),
            self.end.astimezone(BERLIN).strftime(fmt),
        )


@dataclass(frozen=True)
class Alert:
    kind: str
    key: str
    gpu: Optional[int]
    user: Optional[str]
    text: str
    sent: Any

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Alert":
        return cls(
            kind=d["kind"],
            key=d.get("key"),
            gpu=d.get("gpu"),
            user=d.get("user"),
            text=d.get("text", ""),
            sent=d.get("sent"),
        )


@dataclass(frozen=True)
class Gpu:
    index: int
    total_mib: int
    used_mib: int
    util_pct: float
    power_w: float
    booked_mib: int
    free_mib: int
    procs: Tuple[Process, ...]
    bookings: Tuple[Booking, ...]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Gpu":
        return cls(
            index=d["gpu"],
            total_mib=d["total_mib"],
            used_mib=d["used_mib"],
            util_pct=d["util_pct"],
            power_w=d["power_w"],
            booked_mib=d["booked_mib"],
            free_mib=d["free_mib"],
            procs=tuple(Process.from_json(p) for p in d.get("procs", [])),
            bookings=tuple(Booking.from_json(b) for b in d.get("bookings", [])),
        )


@dataclass(frozen=True)
class Usage:
    taken_at: datetime
    age_s: float
    stale: bool
    slack: Any
    gpus: Tuple[Gpu, ...]
    alerts: Tuple[Alert, ...]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Usage":
        return cls(
            taken_at=parse_time(d["ts"]),
            age_s=d["age_s"],
            stale=d["stale"],
            slack=d.get("slack"),
            gpus=tuple(Gpu.from_json(g) for g in d.get("gpus", [])),
            alerts=tuple(Alert.from_json(a) for a in d.get("alerts", [])),
        )

    def to_frame(self):
        pd = _pandas()
        rows = []
        for g in self.gpus:
            holders = sorted({p.user or "unattributed" for p in g.procs})
            booked = ", ".join(
                "%s %g GiB until %s"
                % (b.user, b.vram_gib, b.end.astimezone(BERLIN).strftime("%a %H:%M"))
                for b in g.bookings
                if not b.cancelled
            )
            rows.append(
                {
                    "gpu": g.index,
                    "used_gib": round(g.used_mib / 1024, 1),
                    "total_gib": round(g.total_mib / 1024, 1),
                    "free_gib": round(g.free_mib / 1024, 1),
                    "util_pct": g.util_pct,
                    "power_w": g.power_w,
                    "users": ",".join(holders),
                    "booked": booked,
                }
            )
        cols = ["gpu", "used_gib", "total_gib", "free_gib", "util_pct", "power_w", "users", "booked"]
        return pd.DataFrame(rows, columns=cols)


class BookingList(List[Booking]):
    """A list of bookings with a pandas export."""

    def to_frame(self):
        pd = _pandas()
        rows = [
            {
                "id": b.id,
                "user": b.user,
                "gpu": b.gpu,
                "vram_gib": b.vram_gib,
                "start": b.start.astimezone(BERLIN),
                "end": b.end.astimezone(BERLIN),
                "note": b.note,
                "kind": b.kind,
                "cancelled": b.cancelled,
            }
            for b in self
        ]
        cols = ["id", "user", "gpu", "vram_gib", "start", "end", "note", "kind", "cancelled"]
        return pd.DataFrame(rows, columns=cols)


@dataclass(frozen=True)
class BookingResult:
    """What book() created: one booking per run of consecutive days, all on one GPU."""

    gpu: int
    bookings: List[Booking]

    @property
    def ids(self) -> List[int]:
        return [b.id for b in self.bookings]

    def __str__(self) -> str:
        fmt = "%a %d %b %H:%M"
        return "\n".join(
            "booked GPU %d, %g GiB, %s → %s (Europe/Berlin), id %d" % (
                b.gpu,
                b.vram_gib,
                b.start.astimezone(BERLIN).strftime(fmt),
                b.end.astimezone(BERLIN).strftime(fmt),
                b.id,
            )
            for b in self.bookings
        )
