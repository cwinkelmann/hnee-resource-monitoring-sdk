"""The watch channel: a parked long-poll tells a job its card was taken.

These drive the fake server's scripted responses, which are consulted before its
routing table -- so the threading and the client's handling are tested here, and
whether the real server actually wakes a watcher is tested in test_live.py.
"""
from __future__ import annotations

import threading
import time

import pytest

import hrm
from hrm.errors import Preempted

TAKEN = {"event": "taken", "effective_mib": 0, "lost_mib": 19968,
         "by": ["dorian.zwanzig"], "truncated": True, "cancelled": False,
         "deadline": "2026-10-17T10:05:00+00:00"}
CANCELLED = {"event": "cancelled", "effective_mib": 0, "lost_mib": 19968,
             "by": [], "truncated": False, "cancelled": True, "deadline": None}
SLOT_BODY = {
    "claim": {"id": 11, "user": "cwinkelmann", "gpu": 4, "vram_mib": 19968,
              "start": "2026-10-17T10:00:00+00:00", "end": "2026-10-17T11:00:00+00:00",
              "note": None, "created_at": "2026-10-17T09:00:00+00:00",
              "created_ip": "192.0.2.10", "cancelled_at": None, "cancelled_ip": None,
              "kind": "calendar", "priority": "lendable", "auto_advance": True,
              "group_id": None, "notice_seconds": 0},
    "takes": [], "adjusted_start": None, "assigned_later": False, "revoked": [],
}
SLOT_BODY["claims"] = [SLOT_BODY["claim"]]
EVENTS = "/api/claims/11/events"


@pytest.fixture
def client(fake):
    return hrm.Client(url=fake.url, user="cwinkelmann")


@pytest.fixture
def slot(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    return client.reserve(memory=19.5, hours=1, notice_seconds=0)


def _wait_for(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_watched_slot_learns_it_was_taken(slot, fake):
    fake.script("GET", EVENTS, 200, TAKEN)
    with slot.watch():
        assert _wait_for(lambda: slot.taken is not None)
    assert slot.taken["by"] == ["dorian.zwanzig"]
    assert slot.taken["truncated"] is True


def test_check_raises_preempted_once_the_card_is_taken(slot, fake):
    fake.script("GET", EVENTS, 200, TAKEN)
    with slot.watch():
        assert _wait_for(lambda: slot.taken is not None)
        with pytest.raises(Preempted) as e:
            slot.check()
    assert e.value.event["event"] == "taken"
    assert "dorian.zwanzig" in str(e.value)
    assert "GPU 4" in str(e.value)


def test_check_is_quiet_while_the_slot_stands(slot, fake):
    fake.script("GET", EVENTS, 204, None, times=None)
    with slot.watch():
        slot.check()
        slot.check()
    assert slot.taken is None


def test_a_cancellation_is_a_notice_too(slot, fake):
    fake.script("GET", EVENTS, 200, CANCELLED)
    seen = []
    with slot.watch(on_taken=seen.append):
        assert _wait_for(lambda: bool(seen))
    assert seen[0]["event"] == "cancelled"
    with pytest.raises(Preempted):
        slot.check()


def test_a_server_without_the_events_route_does_not_take_the_job_down(slot, fake):
    fake.script("GET", EVENTS, 404, {"error": "not found"}, times=None)
    with slot.watch() as watcher:
        assert _wait_for(lambda: not watcher.watching)
        slot.check()                 # unwatched, but the job carries on
    assert slot.taken is None


def test_a_booking_the_server_calls_over_stops_the_watch(slot, fake):
    fake.script("GET", EVENTS, 409, {"error": "conflict", "detail": "booking is over"},
                times=None)
    with slot.watch() as watcher:
        assert _wait_for(lambda: not watcher.watching)
    assert slot.taken is None


def test_the_hold_expiring_just_reconnects(slot, fake):
    fake.script("GET", EVENTS, 204, None, times=None)
    with slot.watch():
        assert _wait_for(lambda: len(fake.requests_to("GET", EVENTS)) >= 2)
    assert slot.taken is None


def test_the_watcher_thread_does_not_outlive_its_block(slot, fake):
    fake.script("GET", EVENTS, 204, None, times=None)
    before = threading.active_count()
    with slot.watch():
        assert _wait_for(lambda: threading.active_count() > before)
    assert _wait_for(lambda: threading.active_count() == before)


def test_watcher_parked_says_whether_anyone_is_listening(slot, fake):
    fake.script("GET", EVENTS, 204, None, times=None)
    assert slot.watcher_parked is False
    with slot.watch():
        assert _wait_for(lambda: slot.watcher_parked)
    assert slot.watcher_parked is False


def test_a_group_slot_watches_every_row(client, fake):
    rows = []
    for i in range(2):
        row = dict(SLOT_BODY["claim"])
        row.update(id=20 + i, gpu=i, group_id="g20")
        rows.append(row)
    fake.script("POST", "/api/claims", 201,
                {"claim": rows[0], "claims": rows, "takes": [], "adjusted_start": None,
                 "assigned_later": False, "revoked": []})
    group = client.reserve(memory=1, hours=1, count=2)
    for row in rows:
        fake.script("GET", "/api/claims/%d/events" % row["id"], 204, None, times=None)
    with group.watch():
        for row in rows:
            assert _wait_for(lambda r=row: fake.requests_to(
                "GET", "/api/claims/%d/events" % r["id"]))


# --- fixes from the whole-branch review ---------------------------------------------
def test_stopping_does_not_wait_for_a_parked_poll(slot, fake):
    """stop() used to join for 5 s per row. The exit path is exactly what
    notice_seconds=0 promises to be fast, and the thread is a daemon anyway."""
    fake.script("GET", EVENTS, 204, None, delay=3.0)
    watcher = slot.watch()
    watcher.__enter__()
    assert _wait_for(lambda: watcher.watching)
    began = time.monotonic()
    watcher.stop()
    assert time.monotonic() - began < 1.0
    assert watcher.watching is False      # it says so at once, not in five seconds


def test_an_event_that_lands_after_stop_is_discarded(slot, fake):
    """A late answer used to set slot.taken after the slot was released, so check()
    raised Preempted for a booking that was already given back."""
    fake.script("GET", EVENTS, 200, TAKEN, delay=1.0)
    with slot.watch():
        time.sleep(0.1)
    time.sleep(1.5)                       # the parked answer arrives now
    assert slot.taken is None
    slot.check()                          # and check() stays quiet


def test_a_revocation_missed_while_reconnecting_is_noticed_on_the_next_read(slot, fake):
    """Watchers.notify drops an event with no listener, so a take delivered in the
    gap between two long-polls is lost for good. The row still shows it: a truncated
    end means the card went."""
    fake.script("GET", EVENTS, 204, None, times=1)       # the hold expires
    truncated = dict(SLOT_BODY["claim"])
    truncated["end"] = "2026-10-17T10:20:00+00:00"       # someone cut it short
    fake.script("GET", "/api/claims", 200,
                {"now": "2026-10-17T10:20:00+00:00", "claims": [truncated]}, times=None)
    fake.script("GET", EVENTS, 204, None, times=None)
    with slot.watch():
        assert _wait_for(lambda: slot.taken is not None, seconds=10)
    assert slot.taken["event"] == "taken"
    assert slot.taken["truncated"] is True
    with pytest.raises(Preempted):
        slot.check()


def test_a_cancellation_missed_while_reconnecting_is_noticed_too(slot, fake):
    fake.script("GET", EVENTS, 204, None, times=1)
    cancelled = dict(SLOT_BODY["claim"])
    cancelled["cancelled_at"] = "2026-10-17T10:20:00+00:00"
    fake.script("GET", "/api/claims", 200,
                {"now": "2026-10-17T10:20:00+00:00", "claims": [cancelled]}, times=None)
    fake.script("GET", EVENTS, 204, None, times=None)
    with slot.watch():
        assert _wait_for(lambda: slot.taken is not None, seconds=10)
    assert slot.taken["event"] == "cancelled"
