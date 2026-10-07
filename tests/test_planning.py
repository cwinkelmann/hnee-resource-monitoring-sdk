from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from hrm.models import Booking
from hrm.planning import (
    TZ,
    best_effort,
    choose_gpus,
    day_runs,
    free_mib_at_worst,
    parse_days,
    window_from,
    windows_for_days,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 6, 9, 30, tzinfo=TZ)


def d(day):
    return date(2026, 10, day)


def t(hour, day=6, month=10):
    return datetime(2026, month, day, hour, tzinfo=UTC)


def bk(gpu, mib, start, end, cancelled=False, id=1):
    return Booking(
        id=id, user="alice", gpu=gpu, vram_mib=mib, start=start, end=end, note=None,
        created_at=start, created_ip=None,
        cancelled_at=start if cancelled else None, cancelled_ip=None, kind="calendar",
    )


def test_parse_days_accepts_strings_and_dates_dedups_sorts():
    assert parse_days(["2026-10-08", d(7), "2026-10-07"]) == [d(7), d(8)]


def test_parse_days_bad_format():
    with pytest.raises(ValueError):
        parse_days(["06.10.2026"])


def test_runs():
    assert day_runs([d(6), d(7)]) == [[d(6), d(7)]]
    assert day_runs([d(6), d(8)]) == [[d(6)], [d(8)]]
    assert day_runs(parse_days([d(8), d(6), d(6)])) == [[d(6)], [d(8)]]


def test_today_starts_at_now():
    (s, e), = windows_for_days([d(6)], NOW)
    assert s == NOW
    assert e == datetime(2026, 10, 7, 0, 0, tzinfo=TZ)
    assert s.utcoffset() == timedelta(0) and e.utcoffset() == timedelta(0)


def test_future_day_starts_at_local_midnight():
    (s, e), = windows_for_days([d(8)], NOW)
    assert s == datetime(2026, 10, 8, 0, 0, tzinfo=TZ)
    assert e == datetime(2026, 10, 9, 0, 0, tzinfo=TZ)


def test_two_runs_two_windows():
    assert len(windows_for_days([d(6), d(8)], NOW)) == 2


def test_past_day_raises():
    with pytest.raises(ValueError, match="2026-10-05 is in the past"):
        windows_for_days([d(5)], NOW)


def test_14_days_ok_15_raise():
    ok = [date(2026, 10, 7) + timedelta(days=i) for i in range(14)]
    assert len(windows_for_days(ok, NOW)) == 1
    with pytest.raises(ValueError, match="at most 14 days"):
        windows_for_days(ok + [date(2026, 10, 21)], NOW)


def test_dst_autumn_end():
    (s, e), = windows_for_days([date(2026, 10, 24), date(2026, 10, 25)], NOW)
    assert e == datetime(2026, 10, 25, 23, 0, tzinfo=UTC)
    assert e.isoformat() == "2026-10-25T23:00:00+00:00"
    assert s == datetime(2026, 10, 23, 22, 0, tzinfo=UTC)


def test_dst_spring():
    (s, e), = windows_for_days([date(2026, 3, 29)], datetime(2026, 3, 1, tzinfo=TZ))
    assert s == datetime(2026, 3, 28, 23, 0, tzinfo=UTC)  # CET
    assert e == datetime(2026, 3, 29, 22, 0, tzinfo=UTC)  # CEST


def test_window_from_naive_is_berlin():
    s, e = window_from(datetime(2026, 10, 7, 12), datetime(2026, 10, 7, 14), NOW)
    assert s == datetime(2026, 10, 7, 10, tzinfo=UTC)
    assert e == datetime(2026, 10, 7, 12, tzinfo=UTC)


def test_window_from_end_must_follow_start():
    with pytest.raises(ValueError):
        window_from(datetime(2026, 10, 7, 12), datetime(2026, 10, 7, 12), NOW)


def test_free_none():
    assert free_mib_at_worst([], 0, 1000, t(10), t(20)) == 1000


def test_free_partial_overlap():
    b = [bk(0, 400, t(8), t(12))]
    assert free_mib_at_worst(b, 0, 1000, t(10), t(20)) == 600


def test_free_back_to_back_ignored():
    b = [bk(0, 400, t(8), t(10))]
    assert free_mib_at_worst(b, 0, 1000, t(10), t(20)) == 1000


def test_free_starting_at_window_end_ignored():
    b = [bk(0, 400, t(20), t(22))]
    assert free_mib_at_worst(b, 0, 1000, t(10), t(20)) == 1000


def test_free_cancelled_and_other_gpu_ignored():
    b = [bk(0, 400, t(8), t(12), cancelled=True), bk(1, 500, t(8), t(12))]
    assert free_mib_at_worst(b, 0, 1000, t(10), t(20)) == 1000


def test_free_three_booking_worst_instant():
    b = [
        bk(0, 100, t(8), t(13), id=1),
        bk(0, 200, t(12), t(16), id=2),
        bk(0, 300, t(12), t(18), id=3),
        bk(0, 50, t(17), t(19), id=4),
    ]
    # worst at 12:00: 100 + 200 + 300
    assert free_mib_at_worst(b, 0, 1000, t(10), t(20)) == 400


def test_choose_gpus_memory():
    gpus = [(0, 1000), (1, 1000), (2, 500)]
    b = [bk(0, 800, t(10), t(12)), bk(1, 300, t(10), t(12))]
    w = [(t(9), t(15))]
    assert choose_gpus(gpus, b, w, 500) == [1, 2]


def test_choose_gpus_every_window():
    gpus = [(0, 1000), (1, 1000)]
    b = [bk(0, 800, t(10), t(12))]
    w = [(t(0), t(5)), (t(9), t(15))]
    assert choose_gpus(gpus, b, w, 500) == [1]


def test_choose_gpus_whole_card():
    gpus = [(0, 1000), (1, 1000)]
    b = [bk(0, 1, t(10), t(12))]
    assert choose_gpus(gpus, b, [(t(9), t(15))], None) == [1]


def test_best_effort():
    gpus = [(0, 1000), (1, 1000)]
    b = [bk(0, 800, t(10), t(12)), bk(1, 300, t(10), t(12))]
    assert best_effort(gpus, b, [(t(9), t(15))]) == (1, 700)


def test_14_calendar_days_over_dst_exceeds_336h():
    run = [date(2026, 10, 19) + timedelta(days=i) for i in range(14)]
    with pytest.raises(ValueError, match="at most 14 days"):
        windows_for_days(run, NOW)


def test_window_from_start_in_past():
    with pytest.raises(ValueError, match="start lies in the past"):
        window_from(datetime(2026, 10, 6, 9, 0), datetime(2026, 10, 6, 12), NOW)


def test_window_from_five_minute_grace():
    s, _ = window_from(datetime(2026, 10, 6, 9, 26), datetime(2026, 10, 6, 12), NOW)
    assert s == datetime(2026, 10, 6, 7, 26, tzinfo=UTC)


def test_window_from_over_14_days():
    with pytest.raises(ValueError, match="at most 14 days"):
        window_from(datetime(2026, 10, 7), datetime(2026, 10, 21, 1), NOW)


def test_berlin_midnight_boundary():
    now = datetime(2026, 10, 6, 23, 30, tzinfo=UTC)  # 01:30 on 7 Oct in Berlin
    with pytest.raises(ValueError, match="2026-10-06 is in the past"):
        windows_for_days([d(6)], now)
    (s, _), = windows_for_days([d(7)], now)
    assert s == now


def test_empty_inputs_raise():
    w = [(t(9), t(15))]
    with pytest.raises(ValueError, match="windows"):
        choose_gpus([(0, 1000)], [], [], 100)
    with pytest.raises(ValueError, match="gpus"):
        choose_gpus([], [], w, 100)
    with pytest.raises(ValueError, match="windows"):
        best_effort([(0, 1000)], [], [])
    with pytest.raises(ValueError, match="gpus"):
        best_effort([], [], w)
