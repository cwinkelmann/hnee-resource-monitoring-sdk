"""book / cancel / hold / release against the fake server.

The client's clock is tied to the fake server's clock (``clock=lambda:
fake.now``), so both sides agree on "now" and every date is derived from it.
At the fixtures' moment (Tue 2026-10-06 16:37 Berlin) GPUs 4 and 5 are
entirely free, GPU 1 has 49.6 GiB free until Wed 14:18 UTC, GPUs 6 and 7 have
40.6 GiB free, and GPUs 0, 2 and 3 are held until Wed 07:00 UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from hrm.client import Client
from hrm.errors import (Conflict, InvalidRequest, NoCapacity, ServerUnavailable,
                        UnknownUser)
from hrm.models import Booking, BookingResult

BERLIN = ZoneInfo("Europe/Berlin")
UTC = timezone.utc


@pytest.fixture
def client(fake):
    return Client(fake.url, user="alice", clock=lambda: fake.now)


def today(fake):
    return fake.now.astimezone(BERLIN).date()


def midnight_utc(day):
    return datetime(day.year, day.month, day.day, tzinfo=BERLIN).astimezone(UTC)


def posts(fake):
    return [r.body for r in fake.requests_to("POST", "/api/claims")]


def live_claim(fake, claim_id):
    return next(c for c in fake.claims if c["id"] == claim_id)


def move_clock(fake, t):
    fake.now = t  # the client clock follows (lambda: fake.now)


# -- the flagship call ---------------------------------------------------------

def test_two_consecutive_days_book_one_run_on_lowest_fitting_gpu(fake, client):
    d1 = today(fake)
    d2 = d1 + timedelta(days=1)
    res = client.book(days=[d1.isoformat(), d2.isoformat()], memory=48)

    assert isinstance(res, BookingResult)
    assert len(res.bookings) == 1
    # GPU 0 is held; GPU 1 has 49.6 GiB free for the whole window
    assert res.gpu == 1
    [body] = posts(fake)
    assert body == {
        "user": "alice", "gpu": 1, "vram_gib": 48.0,
        "start": fake.now.astimezone(UTC).isoformat(),
        "end": midnight_utc(d2 + timedelta(days=1)).isoformat(),
    }
    b = res.bookings[0]
    assert (b.gpu, b.vram_mib, b.user) == (1, 48 * 1024, "alice")
    assert b.start == fake.now
    assert b.end == datetime(2026, 10, 7, 22, 0, tzinfo=UTC)
    assert res.ids == [b.id]


def test_result_str(fake, client):
    d1 = today(fake)
    res = client.book(days=[d1, d1 + timedelta(days=1)], memory=48)
    assert str(res) == ("booked GPU 1, 48 GiB, Tue 06 Oct 16:37 → Thu 08 Oct 00:00 "
                        "(Europe/Berlin), id %d" % res.ids[0])


def test_result_str_one_line_per_booking(fake, client):
    d1 = today(fake)
    res = client.book(days=[d1, d1 + timedelta(days=2)], memory=1.5)
    lines = str(res).splitlines()
    assert len(lines) == 2
    assert lines[1] == ("booked GPU 1, 1.5 GiB, Thu 08 Oct 00:00 → Fri 09 Oct 00:00 "
                        "(Europe/Berlin), id %d" % res.ids[1])


def test_non_consecutive_days_give_two_bookings_on_same_gpu(fake, client):
    d1 = today(fake)
    d3 = d1 + timedelta(days=2)
    res = client.book(days=[d3, d1], memory=48)
    assert [b.gpu for b in res.bookings] == [1, 1]
    assert [(b["start"], b["end"]) for b in posts(fake)] == [
        (fake.now.astimezone(UTC).isoformat(), midnight_utc(d1 + timedelta(days=1)).isoformat()),
        (midnight_utc(d3).isoformat(), midnight_utc(d3 + timedelta(days=1)).isoformat()),
    ]
    assert len(set(res.ids)) == 2


def test_planning_reads_usage_and_current_claims(fake, client):
    client.book(days=[today(fake)], memory=1)
    gets = [(r.path, r.query) for r in fake.requests if r.method == "GET"]
    assert ("/api/now", {}) in gets
    assert ("/api/claims", {"days": "14", "back": "1"}) in gets


# -- retries -------------------------------------------------------------------

def test_conflict_on_gpu0_then_success_on_gpu1(fake, client):
    # Thursday morning: the quick holds are over, GPUs 0..5 all fit 48 GiB
    move_clock(fake, datetime(2026, 10, 8, 8, 0, tzinfo=UTC))
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race"}, times=1)
    res = client.book(days=[today(fake)], memory=48)
    assert [b["gpu"] for b in posts(fake)] == [0, 1]
    assert res.gpu == 1 and res.bookings[0].gpu == 1


def test_three_conflicts_raise_conflict_and_stop(fake, client):
    move_clock(fake, datetime(2026, 10, 8, 8, 0, tzinfo=UTC))  # 6 candidates
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race 1"}, times=1)
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race 2"}, times=1)
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race 3"}, times=1)
    with pytest.raises(Conflict) as ei:
        client.book(days=[today(fake)], memory=48)
    assert ei.value.detail == "race 3" and ei.value.status == 409
    assert [b["gpu"] for b in posts(fake)] == [0, 1, 2]  # 3 attempts in total


def test_conflicts_exhaust_fewer_candidates(fake, client):
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "taken"})
    with pytest.raises(Conflict, match="taken"):
        client.book(days=[today(fake)], memory=48)
    assert [b["gpu"] for b in posts(fake)] == [1, 4, 5]


def test_conflict_on_second_window_rolls_back_then_retries_next_gpu(fake, client):
    d1 = today(fake)
    fake.script("POST", "/api/claims", None, times=1)  # first window: real booking
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race"}, times=1)
    res = client.book(days=[d1, d1 + timedelta(days=2)], memory=48)
    first_id = fake.claims[-3]["id"]  # created on GPU 1, then rolled back
    assert live_claim(fake, first_id)["cancelled_at"] is not None
    assert len(fake.requests_to("POST", "/api/claims/%d/cancel" % first_id)) == 1
    assert [b["gpu"] for b in posts(fake)] == [1, 1, 4, 4]
    assert res.gpu == 4 and [b.gpu for b in res.bookings] == [4, 4]


# -- rollback on other errors --------------------------------------------------

def test_second_window_400_cancels_first_then_raises(fake, client):
    d1 = today(fake)
    fake.script("POST", "/api/claims", None, times=1)
    fake.script("POST", "/api/claims", 400, {"error": "invalid", "detail": "nope"}, times=1)
    with pytest.raises(InvalidRequest, match="nope"):
        client.book(days=[d1, d1 + timedelta(days=2)], memory=48)
    assert len(posts(fake)) == 2  # no retry on a non-409
    created = fake.claims[-1]
    assert created["gpu"] == 1 and created["cancelled_at"] is not None
    assert len(fake.requests_to("POST", "/api/claims/%d/cancel" % created["id"])) == 1


def test_failing_rollback_does_not_mask_original_error(fake, client):
    d1 = today(fake)
    fake.script("POST", "/api/claims", None, times=1)
    fake.script("POST", "/api/claims", 400, {"error": "invalid", "detail": "nope"}, times=1)
    created_id = fake._next_id
    fake.script("POST", "/api/claims/%d/cancel" % created_id, 503, {"error": "busy"})
    with pytest.warns(RuntimeWarning, match=r"could not cancel booking %d during rollback"
                      r" — cancel it manually: hrm\.cancel\(%d\)" % (created_id, created_id)):
        with pytest.raises(InvalidRequest, match="nope"):
            client.book(days=[d1, d1 + timedelta(days=2)], memory=48)
    assert len(fake.requests_to("POST", "/api/claims/%d/cancel" % created_id)) == 1


def test_orphan_stops_retries_on_other_gpus(fake, client):
    d1 = today(fake)
    fake.script("POST", "/api/claims", None, times=1)
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race"}, times=1)
    orphan = fake._next_id
    fake.script("POST", "/api/claims/%d/cancel" % orphan, 503, {"error": "busy"})
    with pytest.warns(RuntimeWarning, match=r"hrm\.cancel\(%d\)" % orphan):
        with pytest.raises(Conflict, match="race"):
            client.book(days=[d1, d1 + timedelta(days=2)], memory=48)
    assert [b["gpu"] for b in posts(fake)] == [1, 1]  # no second live booking elsewhere
    assert live_claim(fake, orphan)["cancelled_at"] is None  # really orphaned


# -- a POST whose response was lost after the server committed it --------------

def made_since(fake, first_id):
    return [c for c in fake.claims if c["id"] >= first_id]


def test_lost_post_response_is_reconciled_and_cancelled(fake):
    client = Client(fake.url, user="alice", clock=lambda: fake.now, timeout=0.3)
    d1 = today(fake)
    first_id = fake._next_id
    fake.script("POST", "/api/claims", None, times=1)               # run 1: fine
    fake.script("POST", "/api/claims", None, times=1, delay=1.0)    # run 2: committed, late
    with pytest.raises(ServerUnavailable):
        client.book(days=[d1, d1 + timedelta(days=2)], memory=48)
    created = made_since(fake, first_id)
    assert len(created) == 2  # the server did commit both runs
    assert all(c["cancelled_at"] is not None for c in created)
    assert len(posts(fake)) == 2  # no retry on another GPU
    lookups = [r.query for r in fake.requests_to("GET", "/api/claims")]
    assert lookups == [{"days": "14", "back": "1"}] * 2  # planning, then reconcile


def test_ctrl_c_after_commit_is_reconciled(fake, client, monkeypatch):
    d1 = today(fake)
    first_id = fake._next_id
    real_post = client._post_claim

    def post(path, body):
        result = real_post(path, body)
        if path == "/api/claims":
            raise KeyboardInterrupt  # the cell was interrupted as the reply arrived
        return result

    monkeypatch.setattr(client, "_post_claim", post)
    with pytest.raises(KeyboardInterrupt):
        client.book(days=[d1], memory=48)
    [created] = made_since(fake, first_id)
    assert created["cancelled_at"] is not None


def test_reconcile_leaves_unrelated_and_preexisting_claims_alone(fake, client, monkeypatch):
    d1 = today(fake)
    earlier = client.book(days=[d1 + timedelta(days=2)], memory=1, note="x").bookings[0]
    first_id = fake._next_id
    real_post = client._post_claim

    def post(path, body):
        result = real_post(path, body)
        if path == "/api/claims":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(client, "_post_claim", post)
    with pytest.raises(KeyboardInterrupt):
        client.book(days=[d1 + timedelta(days=2)], memory=1, note="x")
    [created] = made_since(fake, first_id)
    assert created["cancelled_at"] is not None
    assert live_claim(fake, earlier.id)["cancelled_at"] is None  # same shape, not ours
    cancels = [r.path for r in fake.requests if r.path.endswith("/cancel")]
    assert cancels == ["/api/claims/%d/cancel" % created["id"]]


def test_reconcile_lookup_failure_warns(fake):
    client = Client(fake.url, user="alice", clock=lambda: fake.now, timeout=0.3)
    d1 = today(fake)
    first_id = fake._next_id
    fake.script("GET", "/api/claims", None, times=1)  # planning read works
    fake.script("GET", "/api/claims", 503, {"error": "busy"})  # the reconcile read fails
    fake.script("POST", "/api/claims", None, times=1, delay=1.0)
    with pytest.warns(RuntimeWarning, match=r"a booking for GPU 1 2026-10-06T16:37\+02:00"
                      r"→2026-10-07T00:00\+02:00 may have been created; check "
                      r"hrm\.bookings\(\) and cancel it"):
        with pytest.raises(ServerUnavailable):
            client.book(days=[d1], memory=48)
    [created] = made_since(fake, first_id)
    assert created["cancelled_at"] is None  # really left behind, but named in the warning


def test_definitive_server_error_does_not_reconcile(fake, client):
    fake.script("POST", "/api/claims", 400, {"error": "invalid", "detail": "nope"}, times=1)
    with pytest.raises(InvalidRequest):
        client.book(days=[today(fake)], memory=48)
    gets = [r for r in fake.requests if r.method == "GET" and r.path == "/api/claims"]
    assert len(gets) == 1  # only the planning read


# -- days given as a single string -------------------------------------------------

@pytest.mark.parametrize("days", ["2026-10-06", b"2026-10-06"])
def test_days_as_single_string_is_type_error(fake, client, days):
    with pytest.raises(TypeError, match=r"days must be a list of dates, "
                                        r"e\.g\. days=\['2026-10-06'\]"):
        client.book(days=days, memory=1)
    assert posts(fake) == []


def test_ctrl_c_during_second_window_rolls_back_first(fake, client, monkeypatch):
    d1 = today(fake)
    real_post = client._post_claim
    calls = []

    def post(path, body):
        if path == "/api/claims":
            calls.append(body)
            if len(calls) == 2:
                raise KeyboardInterrupt
        return real_post(path, body)

    monkeypatch.setattr(client, "_post_claim", post)
    with pytest.raises(KeyboardInterrupt):
        client.book(days=[d1, d1 + timedelta(days=2)], memory=48)
    first = fake.claims[-1]
    assert first["gpu"] == 1 and first["cancelled_at"] is not None
    assert len(calls) == 2  # no retry after Ctrl-C


def test_first_window_server_error_raises_without_rollback(fake, client):
    fake.script("POST", "/api/claims", 503, {"error": "busy"}, times=1)
    with pytest.raises(ServerUnavailable):
        client.book(days=[today(fake)], memory=48)
    assert len(posts(fake)) == 1
    assert not [r for r in fake.requests if r.path.endswith("/cancel")]


# -- capacity ------------------------------------------------------------------

def test_pinned_gpu_that_does_not_fit_raises_without_post(fake, client):
    with pytest.raises(NoCapacity, match="GPU 0") as ei:
        client.book(days=[today(fake)], memory=48, gpu=0)
    assert "0.0 GiB" in str(ei.value)
    assert posts(fake) == []


def test_pinned_gpu_that_fits_is_used(fake, client):
    res = client.book(days=[today(fake)], memory=48, gpu=5)
    assert res.gpu == 5 and [b["gpu"] for b in posts(fake)] == [5]


def test_pinned_gpu_conflict_raises_conflict(fake, client):
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "race"}, times=1)
    with pytest.raises(Conflict, match="race"):
        client.book(days=[today(fake)], memory=48, gpu=5)
    assert len(posts(fake)) == 1


def test_unknown_pinned_gpu_raises_value_error(fake, client):
    with pytest.raises(ValueError, match="GPU 9"):
        client.book(days=[today(fake)], memory=1, gpu=9)
    assert posts(fake) == []


def test_no_candidate_names_best_gpu(fake, client):
    client.hold(4)
    client.hold(5)
    with pytest.raises(NoCapacity) as ei:
        client.book(days=[today(fake)], memory=50)
    assert str(ei.value) == (
        "no GPU has 50 GiB free for the whole window; best: GPU 1 with 49.6 GiB")
    assert posts(fake) == []


def test_no_whole_card_free(fake, client):
    client.hold(4)
    client.hold(5)
    with pytest.raises(NoCapacity, match="no GPU is entirely free for the whole window; "
                                         "best: GPU 1 with 49.6 GiB"):
        client.book(days=[today(fake)])


def test_memory_none_omits_vram_gib_and_takes_whole_card(fake, client):
    res = client.book(days=[today(fake)], note="whole card")
    [body] = posts(fake)
    assert "vram_gib" not in body
    assert body["gpu"] == 4 and body["note"] == "whole card"
    assert res.bookings[0].vram_mib == 81559 and res.bookings[0].note == "whole card"


def test_note_omitted_when_none(fake, client):
    client.book(days=[today(fake)], memory=2)
    assert "note" not in posts(fake)[0]


# -- start / end -----------------------------------------------------------------

def test_naive_start_end_are_berlin(fake, client):
    res = client.book(start=datetime(2026, 10, 7, 9, 0), end=datetime(2026, 10, 7, 17, 30),
                      memory=10)
    [body] = posts(fake)
    assert body["start"] == "2026-10-07T07:00:00+00:00"
    assert body["end"] == "2026-10-07T15:30:00+00:00"
    # GPU 0's quick hold ends at 07:00 UTC exactly (end is exclusive)
    assert body["gpu"] == 0 and body["vram_gib"] == 10.0
    assert res.bookings[0].start == datetime(2026, 10, 7, 7, 0, tzinfo=UTC)


def test_iso_string_start_end(fake, client):
    client.book(start="2026-10-07T09:00", end="2026-10-07T12:00:00+02:00", memory=10)
    [body] = posts(fake)
    assert (body["start"], body["end"]) == ("2026-10-07T07:00:00+00:00",
                                            "2026-10-07T10:00:00+00:00")


@pytest.mark.parametrize("kw, msg", [
    ({}, "either days or start/end"),
    ({"days": ["2026-10-07"], "start": "2026-10-07T09:00", "end": "2026-10-07T10:00"},
     "either days or start/end"),
    ({"start": "2026-10-07T09:00"}, "start and end"),
    ({"end": "2026-10-07T09:00"}, "start and end"),
    ({"days": ["2026-10-07"], "memory": 0}, "memory"),
    ({"days": ["2026-10-07"], "memory": -1}, "memory"),
    ({"days": ["2026-10-05"]}, "2026-10-05 is in the past"),
    ({"start": "yesterday", "end": "2026-10-07T09:00"}, "yesterday"),
])
def test_bad_arguments_raise_value_error_without_post(fake, client, kw, msg):
    with pytest.raises(ValueError, match=msg):
        client.book(**kw)
    assert posts(fake) == []


def test_explicit_user_and_unknown_user(fake, client, monkeypatch):
    client.book(days=[today(fake)], memory=1, user="bob")
    assert posts(fake)[0]["user"] == "bob"
    monkeypatch.delenv("HRM_USER", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "mallory")
    anon = Client(fake.url, clock=lambda: fake.now)
    with pytest.raises(UnknownUser):
        anon.book(days=[today(fake)], memory=1)
    assert len(posts(fake)) == 1


# -- cancel / hold / release -----------------------------------------------------

def test_cancel_by_object_and_by_id(fake, client):
    res = client.book(days=[today(fake)], memory=1)
    b = client.cancel(res.bookings[0])
    assert isinstance(b, Booking) and b.id == res.ids[0] and b.cancelled
    cancel_req = fake.requests_to("POST", "/api/claims/%d/cancel" % b.id)[0]
    assert cancel_req.body == {}
    again = client.cancel(3)
    assert again.id == 3 and again.cancelled and again.user == "carol"


def test_hold_and_release_bodies(fake, client):
    held = client.hold(4)
    assert isinstance(held, Booking)
    assert (held.gpu, held.kind, held.user, held.vram_mib) == (4, "quick", "alice", 81559)
    assert client.hold(5, user="bob").user == "bob"
    assert client.release(4) is None
    bodies = [r.body for r in fake.requests_to("POST", "/api/claims/quick")]
    assert bodies == [{"user": "alice", "gpu": 4}, {"user": "bob", "gpu": 5},
                      {"user": None, "gpu": 4}]
    assert live_claim(fake, held.id)["cancelled_at"] is not None


def test_default_clock_is_real_utc_now(fake):
    """The one test that goes through the real default clock."""
    fake.now = datetime.now(UTC)
    before = datetime.now(UTC)
    c = Client(fake.url, user="alice")
    res = c.book(days=[fake.now.astimezone(BERLIN).date()], memory=1)
    after = datetime.now(UTC)
    start = res.bookings[0].start
    assert before <= start <= after
    assert start.utcoffset() == timedelta(0)
