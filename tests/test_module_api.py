"""The module-level API delegates to the client made by connect()."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import hrm

BERLIN = ZoneInfo("Europe/Berlin")


@pytest.fixture(autouse=True)
def fresh_default(monkeypatch):
    monkeypatch.setattr(hrm, "_default", None)
    monkeypatch.delenv("HRM_URL", raising=False)
    monkeypatch.delenv("HRM_USER", raising=False)


def test_connect_sets_user_and_url(fake):
    c = hrm.connect(user="bob", url=fake.url)
    assert isinstance(c, hrm.Client)
    assert (c.user, c.url) == ("bob", fake.url)
    assert hrm.users()[:3] == ["alice", "bob", "carol"]
    assert len(hrm.usage().gpus) == 8
    assert isinstance(hrm.bookings(), hrm.BookingList)
    held = hrm.hold(4)
    assert held.user == "bob"
    assert hrm.cancel(held).cancelled
    assert hrm.release(4) is None
    bodies = [r.body for r in fake.requests_to("POST", "/api/claims/quick")]
    assert bodies == [{"user": "bob", "gpu": 4}, {"user": None, "gpu": 4}]


def test_connect_replaces_default_client(fake):
    hrm.connect(user="alice", url="http://192.0.2.10:1")
    hrm.connect(user="carol", url=fake.url)
    assert hrm.hold(5).user == "carol"


def test_module_book_uses_connected_client(fake):
    fake.now = datetime.now(timezone.utc)  # module-level book uses the real clock
    hrm.connect(user="carol", url=fake.url)
    res = hrm.book(days=[fake.now.astimezone(BERLIN).date()], memory=1, note="nb")
    assert isinstance(res, hrm.BookingResult)
    body = fake.requests_to("POST", "/api/claims")[0].body
    assert body["user"] == "carol" and body["note"] == "nb"


def test_default_client_created_lazily_from_env(fake, monkeypatch):
    monkeypatch.setenv("HRM_URL", fake.url)
    monkeypatch.setenv("HRM_USER", "alice")
    assert hrm.hold(4).user == "alice"


def test_all_exports():
    names = {"connect", "users", "usage", "bookings", "book", "cancel", "hold", "release",
             "Client", "BookingResult", "Booking", "BookingList", "Usage", "Gpu", "Process",
             "Alert", "HRMError", "InvalidRequest", "Forbidden", "NotFound", "Conflict",
             "RateLimited", "ServerUnavailable", "NoCapacity", "UnknownUser", "__version__"}
    assert names <= set(hrm.__all__)
    for n in hrm.__all__:
        assert hasattr(hrm, n), n
    assert issubclass(hrm.NoCapacity, hrm.HRMError)
