from __future__ import annotations

import sys
from datetime import timedelta

import pytest

from hrm.errors import (
    Conflict, Forbidden, HRMError, InvalidRequest, NotFound, RateLimited,
    ServerUnavailable, error_for_status,
)
from hrm.models import Booking, BookingList, Usage, parse_time


@pytest.fixture
def usage(load_fixture):
    return Usage.from_json(load_fixture("now"))


@pytest.fixture
def bookings(load_fixture):
    return BookingList(Booking.from_json(c) for c in load_fixture("claims")["claims"])


def test_parse_time_z_and_offset():
    a = parse_time("2026-10-06T12:00:00Z")
    assert a == parse_time("2026-10-06T12:00:00+00:00")
    assert a.tzinfo is not None


def test_usage_parses(usage):
    assert len(usage.gpus) == 8
    assert usage.taken_at.utcoffset() is not None
    assert usage.stale is False
    g0 = usage.gpus[0]
    assert g0.index == 0 and g0.total_mib == 81559
    assert isinstance(g0.bookings, tuple) and isinstance(g0.bookings[0], Booking)
    assert isinstance(g0.procs, tuple)
    assert any(g.procs for g in usage.gpus)
    assert usage.alerts == ()


def test_bookings_parse(bookings):
    assert len(bookings) > 3
    for b in bookings:
        assert b.start.utcoffset() is not None and b.created_at.utcoffset() is not None
    assert bookings[0].cancelled and bookings[0].cancelled_at.utcoffset() is not None
    assert not [b for b in bookings if b.id == 3][0].cancelled


def test_vram_gib(bookings):
    assert bookings[0].vram_gib == 1.0
    assert [b for b in bookings if b.id == 6][0].vram_gib == 79.6


def test_active_at_boundaries(bookings):
    b = [b for b in bookings if b.id == 3][0]
    assert b.active_at(b.start)
    assert not b.active_at(b.start - timedelta(microseconds=1))
    assert b.active_at(b.end - timedelta(microseconds=1))
    assert not b.active_at(b.end)


def test_cancelled_never_active(bookings):
    b = bookings[0]
    assert b.cancelled and not b.active_at(b.start + timedelta(minutes=1))


def test_booking_str(bookings):
    b = [b for b in bookings if b.id == 3][0]
    assert str(b) == "#3 GPU 6 · 40.0 GiB · carol · Tue 06 Oct 14:57 → Fri 09 Oct 18:00"


@pytest.mark.parametrize("status,cls", [
    (400, InvalidRequest), (413, InvalidRequest), (415, InvalidRequest),
    (403, Forbidden), (404, NotFound), (409, Conflict), (429, RateLimited),
    (503, ServerUnavailable), (500, ServerUnavailable), (502, ServerUnavailable),
])
def test_error_for_status(status, cls):
    e = error_for_status(status, {"error": "bad", "detail": "why"})
    assert type(e) is cls and e.status == status and e.detail == "why"
    assert isinstance(e, HRMError)


def test_error_detail_fallback_and_unknown():
    assert error_for_status(409, {"error": "conflict"}).detail == "conflict"
    assert error_for_status(404, None).detail == ""
    e = error_for_status(418, {"error": "teapot"})
    assert type(e) is HRMError and e.status == 418


def test_to_frame_pandas(usage, bookings):
    pd = pytest.importorskip("pandas")
    df = usage.to_frame()
    assert list(df.columns) == ["gpu", "used_gib", "total_gib", "free_gib",
                                "util_pct", "power_w", "users", "booked"]
    assert len(df) == 8
    assert "alice" in df.loc[0, "booked"] and "GiB until" in df.loc[0, "booked"]
    bf = bookings.to_frame()
    assert list(bf.columns) == ["id", "user", "gpu", "vram_gib", "start", "end",
                                "note", "kind", "cancelled"]
    assert str(bf["start"].dt.tz) == "Europe/Berlin"


def test_to_frame_without_pandas(monkeypatch, usage, bookings):
    monkeypatch.setitem(sys.modules, "pandas", None)
    msg = 'pip install "hnee-resource-monitoring[pandas]"'
    with pytest.raises(ImportError, match=r"hnee-resource-monitoring\[pandas\]") as ei:
        usage.to_frame()
    assert str(ei.value) == msg
    with pytest.raises(ImportError):
        bookings.to_frame()
