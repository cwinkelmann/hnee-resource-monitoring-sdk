"""Slot and Capacity: the client-side half. The contract lives in test_live.py."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import hrm
from hrm.slots import Capacity

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
