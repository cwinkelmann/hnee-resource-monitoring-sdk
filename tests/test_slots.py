"""Slot and Capacity: the client-side half. The contract lives in test_live.py."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

import hrm
from hrm.slots import Capacity

TAKEN_EVENT = {"event": "taken", "effective_mib": 0, "lost_mib": 19968,
               "by": ["duser"], "truncated": True, "cancelled": False,
               "deadline": None}
CAP_BODY = {"free_mib": 16376, "free_gib": 16.0,
            "earliest": "2026-10-17T07:48:03+00:00", "gpu": 0, "gpus": [0]}


@pytest.fixture
def client(fake):
    return hrm.Client(url=fake.url, user="cwinkelmann")


def test_capacity_parses_the_servers_answer():
    cap = Capacity.from_json(CAP_BODY)
    assert cap.free_mib == 16376
    assert cap.free_gib == 16.0
    assert isinstance(cap.earliest, datetime) and cap.earliest.tzinfo is not None
    assert cap.gpu == 0 and cap.gpus == [0]


def test_capacity_with_nothing_fitting_has_no_earliest():
    cap = Capacity.from_json({"free_mib": 0, "free_gib": 0.0, "earliest": None,
                              "gpu": None, "gpus": []})
    assert cap.earliest is None and cap.gpu is None and cap.gpus == []


def test_capacity_derives_free_gib_when_the_server_omits_it():
    cap = Capacity.from_json({"free_mib": 2048, "earliest": None, "gpu": None})
    assert cap.free_gib == pytest.approx(2.0)


def test_capacity_asks_the_server_and_sends_the_shape(client, fake):
    fake.script("GET", "/api/capacity", 200, CAP_BODY)
    cap = client.capacity(memory=19.5, hours=2, gpu=4, count=1)
    assert cap.free_mib == 16376
    asked = fake.requests_to("GET", "/api/capacity")[-1]
    assert asked.query["vram_gib"] == "19.5"
    assert asked.query["duration_minutes"] == "120"
    assert asked.query["gpu"] == "4"
    assert asked.query["count"] == "1"


def test_capacity_omits_a_card_it_was_not_given(client, fake):
    fake.script("GET", "/api/capacity", 200, CAP_BODY)
    client.capacity(memory=1, hours=1)
    assert "gpu" not in fake.requests_to("GET", "/api/capacity")[-1].query


@pytest.mark.parametrize("kwargs", [
    {"memory": 0}, {"memory": -1}, {"memory": "lots"}, {"memory": True},
    {"hours": 0}, {"hours": -1}, {"hours": 15 * 24}, {"hours": "two"},
    {"count": 0}, {"count": 9}, {"count": 1.5}, {"count": True},
])
def test_a_bad_shape_is_a_valueerror_before_anything_is_sent(client, fake, kwargs):
    args = {"memory": 1, "hours": 1}
    args.update(kwargs)
    before = len(fake.requests)
    with pytest.raises(ValueError):
        client.capacity(**args)
    assert len(fake.requests) == before


def test_capacity_is_on_the_module_too(fake):
    fake.script("GET", "/api/capacity", 200, CAP_BODY)
    hrm.connect(user="cwinkelmann", url=fake.url)
    assert hrm.capacity(memory=1, hours=1).free_gib == 16.0


SLOT_BODY = {
    "claim": {"id": 11, "user": "cwinkelmann", "gpu": 4, "vram_mib": 19968,
              "start": "2026-10-17T10:00:00+00:00", "end": "2026-10-17T11:00:00+00:00",
              "note": "job", "created_at": "2026-10-17T09:00:00+00:00",
              "created_ip": "192.0.2.10", "cancelled_at": None, "cancelled_ip": None,
              "kind": "calendar", "priority": "lendable", "auto_advance": True,
              "group_id": None, "notice_seconds": 0},
    "takes": [], "adjusted_start": None, "assigned_later": True, "revoked": [],
}
SLOT_BODY["claims"] = [SLOT_BODY["claim"]]


def _group_body(count):
    rows = []
    for i in range(count):
        row = dict(SLOT_BODY["claim"])
        row.update(id=20 + i, gpu=i, group_id="g20")
        rows.append(row)
    return {"claim": rows[0], "claims": rows, "takes": [], "adjusted_start": None,
            "assigned_later": False, "revoked": []}


def test_a_reservation_parses_into_a_slot(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    slot = client.reserve(memory=19.5, hours=1, note="job")
    assert slot.id == 11
    assert slot.gpu == 4 and slot.gpus == [4]
    assert slot.assigned_later is True
    assert slot.memory_gib == pytest.approx(19.5, abs=0.01)
    assert slot.end - slot.start == timedelta(hours=1)
    assert "Slot(" in repr(slot)


def test_reserve_sends_a_duration_and_no_window(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    client.reserve(memory=19.5, hours=2)
    body = fake.requests_to("POST", "/api/claims")[-1].body
    assert body["duration_minutes"] == 120
    assert body["auto_advance"] is True        # the default: take an earlier slot
    assert body["count"] == 1
    assert body["priority"] == "lendable"
    assert body["user"] == "cwinkelmann"
    assert "start" not in body and "end" not in body


def test_notice_important_and_not_later_than_reach_the_server(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    client.reserve(memory=1, hours=1, notice_seconds=0, important=True,
                   auto_advance=False, gpu=6,
                   not_later_than="2026-10-17T12:00:00+00:00")
    body = fake.requests_to("POST", "/api/claims")[-1].body
    assert body["notice_seconds"] == 0
    assert body["priority"] == "important"
    assert body["auto_advance"] is False
    assert body["gpu"] == 6
    assert body["not_later_than"].startswith("2026-10-17T12:00:00")


def test_a_group_reservation_is_one_slot_over_several_cards(client, fake):
    fake.script("POST", "/api/claims", 201, _group_body(4))
    slot = client.reserve(memory=1, hours=1, count=4)
    assert len(slot.bookings) == 4
    assert slot.gpus == [0, 1, 2, 3]
    assert len({b.group_id for b in slot.bookings}) == 1
    assert slot.id == 20


def test_an_older_server_without_a_claims_list_still_parses(client, fake):
    body = {k: v for k, v in SLOT_BODY.items() if k != "claims"}
    fake.script("POST", "/api/claims", 201, body)
    assert client.reserve(memory=1, hours=1).id == 11


@pytest.mark.parametrize("kwargs", [
    {"memory": 0}, {"hours": 15 * 24}, {"count": 9}, {"notice_seconds": -1},
    {"notice_seconds": 1.5}, {"notice_seconds": True},
])
def test_reserve_rejects_a_bad_shape_before_sending(client, fake, kwargs):
    args = {"memory": 1, "hours": 1}
    args.update(kwargs)
    before = len(fake.requests)
    with pytest.raises(ValueError):
        client.reserve(**args)
    assert len(fake.requests) == before


def test_reserve_is_on_the_module_too(fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    hrm.connect(user="cwinkelmann", url=fake.url)
    assert hrm.reserve(memory=1, hours=1).id == 11


# --- waiting -----------------------------------------------------------------------
# wait_until_start() is pure polling over client.bookings() and client.now(), so these
# drive a stub client: no HTTP, no clock to race, and the tick count is an assertion.
class StubClient:
    def __init__(self, rows, now):
        self._rows = list(rows)
        self._now = now
        self.reads = 0

    def bookings(self, include_cancelled=False):
        self.reads += 1
        return list(self._rows)

    def now(self):
        return self._now

    def release_claim(self, booking_or_id, end=None):   # replaced per test
        raise AssertionError("not wired")

    def watch_claim(self, booking_or_id, timeout=90.0):  # replaced per test
        raise AssertionError("not wired")

    def set(self, rows=None, now=None):
        if rows is not None:
            self._rows = list(rows)
        if now is not None:
            self._now = now


def _row(**over):
    d = dict(SLOT_BODY["claim"])
    d.update(over)
    return hrm.Booking.from_json(d)


T_START = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)


def test_waiting_for_a_slot_that_already_started_returns_at_once():
    row = _row()
    stub = StubClient([row], T_START)
    slot = hrm.Slot(stub, [row], assigned_later=False)
    slept = []
    assert slot.wait_until_start(sleep=slept.append) is slot
    assert slept == []


def test_waiting_polls_until_the_start_arrives():
    row = _row()
    stub = StubClient([row], T_START - timedelta(minutes=20))
    slot = hrm.Slot(stub, [row], assigned_later=True)
    ticks = []

    def sleep(seconds):
        ticks.append(seconds)
        stub.set(now=stub.now() + timedelta(minutes=10))
    slot.wait_until_start(poll=5, sleep=sleep)
    assert ticks == [5, 5]
    assert stub.reads == 2                     # it re-reads on every tick


def test_waiting_notices_the_slot_moved_earlier():
    row = _row(start="2026-10-17T18:00:00+00:00", end="2026-10-17T19:00:00+00:00")
    stub = StubClient([row], T_START)
    slot = hrm.Slot(stub, [row], assigned_later=True)
    far = slot.start

    def sleep(_seconds):
        stub.set(rows=[_row()])                # someone released early
    slot.wait_until_start(poll=5, sleep=sleep)
    assert slot.start < far
    assert slot.start == T_START


def test_a_cancelled_slot_stops_the_wait_with_a_conflict():
    row = _row(start="2026-10-17T18:00:00+00:00", end="2026-10-17T19:00:00+00:00")
    stub = StubClient([row], T_START)
    slot = hrm.Slot(stub, [row], assigned_later=True)

    def sleep(_seconds):
        stub.set(rows=[_row(start="2026-10-17T18:00:00+00:00",
                            end="2026-10-17T19:00:00+00:00",
                            cancelled_at="2026-10-17T10:05:00+00:00")])
    with pytest.raises(hrm.Conflict):
        slot.wait_until_start(poll=1, sleep=sleep)


def test_waiting_gives_up_after_the_timeout():
    row = _row(start="2026-10-20T10:00:00+00:00", end="2026-10-20T11:00:00+00:00")
    stub = StubClient([row], T_START)
    slot = hrm.Slot(stub, [row], assigned_later=True)
    waited = [0.0]

    def sleep(seconds):
        waited[0] += seconds
    with pytest.raises(TimeoutError):
        slot.wait_until_start(poll=5, timeout=12, sleep=sleep)
    assert waited[0] <= 15


def test_refresh_keeps_the_rows_when_the_server_forgets_them():
    row = _row()
    stub = StubClient([], T_START)
    slot = hrm.Slot(stub, [row], assigned_later=False)
    assert slot.refresh().id == row.id


# --- releasing ---------------------------------------------------------------------
def _released(**over):
    row = dict(SLOT_BODY["claim"])
    row.update(over)
    return {"claim": row, "advanced": []}


def _reserved(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    return client.reserve(memory=19.5, hours=1)


def test_release_truncates_and_sends_no_end_by_default(client, fake):
    slot = _reserved(client, fake)
    fake.script("POST", "/api/claims/11/release", 200,
                _released(end="2026-10-17T10:20:00+00:00"))
    assert slot.release() == []
    assert fake.requests_to("POST", "/api/claims/11/release")[-1].body == {}


def test_release_passes_an_explicit_end_through(client, fake):
    slot = _reserved(client, fake)
    fake.script("POST", "/api/claims/11/release", 200, _released())
    slot.release(end="2026-10-17T10:30:00+00:00")
    body = fake.requests_to("POST", "/api/claims/11/release")[-1].body
    assert body["end"].startswith("2026-10-17T10:30:00")


def test_release_reports_the_slots_it_advanced(client, fake):
    slot = _reserved(client, fake)
    fake.script("POST", "/api/claims/11/release", 200,
                {"claim": SLOT_BODY["claim"],
                 "advanced": [{"id": 99, "user": "alice", "gpu": 0,
                               "old_start": "2026-10-17T17:00:00+00:00",
                               "new_start": "2026-10-17T11:00:00+00:00"}]})
    assert [m["id"] for m in slot.release()] == [99]


def test_release_releases_every_row_of_a_group(client, fake):
    fake.script("POST", "/api/claims", 201, _group_body(3))
    slot = client.reserve(memory=1, hours=1, count=3)
    for row in slot.bookings:
        fake.script("POST", "/api/claims/%d/release" % row.id, 200, _released(id=row.id))
    slot.release()
    for row in slot.bookings:
        assert fake.requests_to("POST", "/api/claims/%d/release" % row.id)


@pytest.mark.parametrize("status,body", [
    (404, {"error": "not found"}),
    (409, {"error": "conflict", "detail": "already cancelled"}),
    (409, {"error": "conflict", "detail": "booking is over"}),
])
def test_release_is_safe_on_a_slot_that_is_already_gone(client, fake, status, body):
    """It runs from a finally: block on every path, including ones where the slot has
    already been cancelled, already ended, or never existed."""
    slot = _reserved(client, fake)
    fake.script("POST", "/api/claims/11/release", status, body)
    assert slot.release() == []


def test_release_lets_a_real_error_through(client, fake):
    slot = _reserved(client, fake)
    fake.script("POST", "/api/claims/11/release", 500, {"error": "internal"})
    with pytest.raises(hrm.ServerUnavailable):
        slot.release()


def test_release_twice_is_two_harmless_calls(client, fake):
    slot = _reserved(client, fake)
    fake.script("POST", "/api/claims/11/release", 200, _released())
    fake.script("POST", "/api/claims/11/release", 409,
                {"error": "conflict", "detail": "already cancelled"})
    slot.release()
    slot.release()


# --- the whole pattern -------------------------------------------------------------
def test_the_context_manager_waits_parks_and_releases(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    fake.script("GET", "/api/claims/11/events", 204, None, times=None)
    fake.script("POST", "/api/claims/11/release", 200, _released())
    slot = client.reserve(memory=19.5, hours=1)
    slot.client = StubClient([_row()], T_START)      # the window has begun
    slot.client.release_claim = client.release_claim  # but the writes are real
    with slot as entered:
        assert entered is slot
        assert slot.watcher is not None
    assert slot.watcher.watching is False
    assert fake.requests_to("POST", "/api/claims/11/release")


def test_the_context_manager_releases_even_when_the_body_raises(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    fake.script("GET", "/api/claims/11/events", 204, None, times=None)
    fake.script("POST", "/api/claims/11/release", 200, _released())
    slot = client.reserve(memory=19.5, hours=1)
    slot.client = StubClient([_row()], T_START)
    slot.client.release_claim = client.release_claim
    with pytest.raises(ZeroDivisionError):
        with slot:
            1 / 0
    assert fake.requests_to("POST", "/api/claims/11/release")
    assert slot.watcher.watching is False


def test_preempted_from_check_still_releases_the_slot(client, fake):
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    fake.script("GET", "/api/claims/11/events", 200, TAKEN_EVENT)
    fake.script("POST", "/api/claims/11/release", 200, _released())
    slot = client.reserve(memory=19.5, hours=1, notice_seconds=0)
    slot.client = StubClient([_row()], T_START)
    slot.client.release_claim = client.release_claim
    slot.client.watch_claim = client.watch_claim
    with pytest.raises(hrm.Preempted):
        with slot:
            for _ in range(200):
                slot.check()
                time.sleep(0.02)
    assert fake.requests_to("POST", "/api/claims/11/release")


def test_entering_releases_the_slot_if_the_wait_fails(client, fake):
    """__enter__ waits for the start, so a Conflict or Ctrl-C while queued used to
    skip __exit__ entirely and leave the reservation holding a future window."""
    fake.script("POST", "/api/claims/11/release", 200, _released())
    # A slot queued for this evening, which someone cancels while we wait for it.
    queued = _row(start="2026-10-17T18:00:00+00:00", end="2026-10-17T19:00:00+00:00")
    cancelled = _row(start="2026-10-17T18:00:00+00:00",
                     end="2026-10-17T19:00:00+00:00",
                     cancelled_at="2026-10-17T10:05:00+00:00")
    stub = StubClient([cancelled], T_START)
    stub.release_claim = client.release_claim
    slot = hrm.Slot(stub, [queued], assigned_later=True)
    slot.wait_poll = 0.05
    with pytest.raises(hrm.Conflict):
        with slot:
            raise AssertionError("the body must never run")
    assert fake.requests_to("POST", "/api/claims/11/release")


def test_a_booking_is_still_hashable(client, fake):
    """Booking is a frozen dataclass and set(bookings) worked before `taken` was
    added; a list field would have made it unhashable."""
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    slot = client.reserve(memory=1, hours=1)
    assert len({slot.booking, slot.booking}) == 1
