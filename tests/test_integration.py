"""Read-only checks against a live server.  Opt-in: HRM_INTEGRATION=1.

    HRM_INTEGRATION=1 pytest tests/test_integration.py      # HRM_URL to override the server
"""
from __future__ import annotations

import os

import pytest

import hrm

pytestmark = pytest.mark.skipif(
    os.environ.get("HRM_INTEGRATION") != "1",
    reason="live server test; set HRM_INTEGRATION=1 to run",
)


@pytest.fixture
def client():
    return hrm.Client()


def test_users_is_non_empty(client):
    users = client.users()
    assert users and all(isinstance(u, str) for u in users)


def test_usage_has_eight_gpus(client):
    usage = client.usage()
    assert len(usage.gpus) == 8
    assert all(g.total_mib > 0 for g in usage.gpus)


def test_bookings_parse(client):
    bookings = client.bookings(include_cancelled=True)
    assert isinstance(bookings, hrm.BookingList)
    assert all(isinstance(b, hrm.Booking) for b in bookings)
