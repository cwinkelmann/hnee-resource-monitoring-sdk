"""Slot and Capacity: the client-side half. The contract lives in test_live.py."""
from __future__ import annotations

from datetime import datetime

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
