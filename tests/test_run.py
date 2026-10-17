"""Slot.run: stop the job the moment the card is taken, not at the next checkpoint.

The epoch-boundary idiom (slot.check()) leaves the card held until the current
unit of work ends. This is the other half: the job runs in a child process and the
parent ends it at once, which is the shape playground/revocation_mvp.py prototyped.
"""
from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

import hrm

TAKEN = {"event": "taken", "effective_mib": 0, "lost_mib": 19968,
         "by": ["dorian.zwanzig"], "truncated": True, "cancelled": False,
         "deadline": None}
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
def slot(fake):
    client = hrm.Client(url=fake.url, user="cwinkelmann")
    fake.script("POST", "/api/claims", 201, SLOT_BODY)
    return client.reserve(memory=19.5, hours=1, notice_seconds=0)


def test_a_job_that_finishes_on_its_own_is_completed(slot, fake):
    fake.script("GET", EVENTS, 204, None, times=None)
    import mp_targets
    assert slot.run(mp_targets.finishes_quickly, args=(0.2,)) == "completed"
    assert slot.taken is None


def test_a_taken_card_ends_the_job_at_once(slot, fake, tmp_path):
    """Not at the next checkpoint: the parent sets the stop event as soon as the
    watcher is told, and the child is gone within the grace."""
    fake.script("GET", EVENTS, 200, TAKEN)
    import mp_targets
    marker = tmp_path / "marker"
    began = time.monotonic()
    outcome = slot.run(mp_targets.runs_until_stopped, args=(str(marker), 30.0))
    elapsed = time.monotonic() - began
    assert outcome == "taken"
    assert elapsed < 10            # not the 30 s the job would have run for
    assert marker.read_text() == "stopped gracefully"


def test_a_job_that_will_not_listen_is_ended_anyway(slot, fake):
    """The graceful stop is tried first; a child that ignores it is terminated. Only
    ever our own child -- nothing else is signalled, ever."""
    fake.script("GET", EVENTS, 200, TAKEN)
    import mp_targets
    began = time.monotonic()
    outcome = slot.run(mp_targets.ignores_the_stop, args=(30.0,), grace=1.0)
    elapsed = time.monotonic() - began
    assert outcome == "taken"
    assert elapsed < 10


def test_run_reports_what_the_watcher_was_told(slot, fake, tmp_path):
    fake.script("GET", EVENTS, 200, TAKEN)
    import mp_targets
    slot.run(mp_targets.runs_until_stopped, args=(str(tmp_path / "m"), 30.0))
    assert slot.taken["by"] == ["dorian.zwanzig"]
    with pytest.raises(hrm.Preempted):
        slot.check()


def test_run_leaves_no_child_behind(slot, fake, tmp_path):
    fake.script("GET", EVENTS, 200, TAKEN)
    import mp_targets
    before = len(multiprocessing.active_children())
    slot.run(mp_targets.runs_until_stopped, args=(str(tmp_path / "m"), 30.0))
    assert len(multiprocessing.active_children()) == before
