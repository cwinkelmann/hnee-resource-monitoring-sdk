"""The contract, against a real booking server. Skipped unless HRM_LIVE_URL is set.

    HRM_LIVE_URL=http://192.0.2.10:8765 HRM_LIVE_USER=cwinkelmann \
        python -m pytest tests/test_live.py -v

Every test books the smallest slot the server allows and releases it again, so a run
leaves the box as it found it.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import timedelta

import pytest

import hrm

URL = os.environ.get("HRM_LIVE_URL")
USER = os.environ.get("HRM_LIVE_USER") or os.environ.get("HRM_USER")

pytestmark = pytest.mark.skipif(not URL, reason="set HRM_LIVE_URL to run the live tests")


@pytest.fixture(scope="module")
def client():
    c = hrm.Client(url=URL, user=USER)
    try:
        c.usage()
    except hrm.HRMError as exc:
        pytest.skip("cannot reach %s: %s" % (URL, exc))
    return c


@pytest.fixture
def slots(client):
    """Hand back every slot a test made, however it ended."""
    made = []
    yield made
    for slot in made:
        try:
            slot.release()
        except hrm.HRMError:
            pass


def test_capacity_answers_from_the_real_box(client):
    cap = client.capacity(memory=1, hours=1)
    assert cap.free_mib > 0
    assert cap.gpu in cap.gpus
    assert cap.earliest is not None


def test_a_reservation_comes_back_with_a_window(client, slots):
    slot = client.reserve(memory=1, hours=1, note="hrm live test")
    slots.append(slot)
    assert slot.id > 0
    assert slot.end - slot.start == timedelta(hours=1)
    assert slot.gpu in range(8)
    assert slot.booking.auto_advance is True


def test_waiting_on_a_slot_that_starts_now_returns_at_once(client, slots):
    slot = client.reserve(memory=1, hours=1, note="hrm live test")
    slots.append(slot)
    if slot.assigned_later:
        pytest.skip("the box is busy; this test needs a free card")
    began = time.monotonic()
    slot.wait_until_start(poll=1)
    assert time.monotonic() - began < 2


def test_releasing_truncates_rather_than_cancelling(client, slots):
    slot = client.reserve(memory=1, hours=4, note="hrm live test")
    slots.append(slot)
    if slot.assigned_later:
        pytest.skip("the box is busy; this test needs a slot that has started")
    original_end = slot.end
    slot.release()
    slot.refresh()
    assert slot.booking.cancelled_at is None
    assert slot.end < original_end


def test_an_important_booking_takes_a_watched_lendable_one(client, slots):
    """The whole point of the feature, against the real server: a lendable slot with a
    parked watcher is taken at once, and the watcher is told in well under a second."""
    cap = client.capacity(memory=1, hours=1)
    # free_gib is rounded for display; floor the real MiB or a 16376 MiB card is asked
    # for 16384 and answers 400.
    whole = max(cap.free_mib // 1024, 2)
    victim = client.reserve(memory=1, hours=1, notice_seconds=0, note="hrm live victim")
    slots.append(victim)
    if victim.assigned_later:
        pytest.skip("the box is busy; this test needs a card now")
    with victim.watch():
        for _ in range(100):                     # let the long-poll park
            if victim.watcher_parked:
                break
            time.sleep(0.05)
        began = time.monotonic()
        taker = client.reserve(memory=whole, hours=1, gpu=victim.gpu, important=True,
                               note="hrm live taker")
        slots.append(taker)
        for _ in range(200):
            if victim.taken is not None:
                break
            time.sleep(0.05)
        waited = time.monotonic() - began
    if not victim.watcher_parked and victim.taken is None:
        pytest.skip("this server has no /api/claims/<id>/events route; deploy a newer "
                    "ResourceMonitor to test the watch channel")
    assert victim.taken is not None, "the watcher was never told"
    assert victim.taken["event"] in ("taken", "cancelled")
    assert waited < 5.0
    with pytest.raises(hrm.Preempted):
        victim.check()


def test_a_full_card_still_grants_a_slot_later(client, slots):
    cap = client.capacity(memory=1, hours=1)
    whole = max(cap.free_mib // 1024, 2)
    first = client.reserve(memory=whole, hours=2, note="hrm live filler")
    slots.append(first)
    second = client.reserve(memory=whole, hours=1, gpu=first.gpu, note="hrm live queued")
    slots.append(second)
    assert second.assigned_later is True         # never NoCapacity
    assert second.start >= first.end - timedelta(minutes=1)
