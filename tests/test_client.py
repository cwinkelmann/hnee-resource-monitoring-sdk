from __future__ import annotations

import socket

import pytest

import hrm
from hrm import client as client_mod
from hrm.client import DEFAULT_URL, Client
from hrm.errors import (Conflict, Forbidden, HRMError, InvalidRequest, NotFound,
                        RateLimited, ServerUnavailable, UnknownUser)
from hrm.models import BookingList, Usage


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("HRM_URL", raising=False)
    monkeypatch.delenv("HRM_USER", raising=False)


# -- read endpoints ----------------------------------------------------------
def test_users(fake):
    users = Client(fake.url).users()
    assert users[:3] == ["alice", "bob", "carol"] and len(users) == 25


def test_usage(fake):
    usage = Client(fake.url).usage()
    assert isinstance(usage, Usage)
    assert len(usage.gpus) == 8
    assert usage.gpus[0].bookings[0].user == "alice"


def test_bookings_defaults_hide_cancelled(fake):
    bl = Client(fake.url).bookings()
    assert isinstance(bl, BookingList)
    assert bl and all(not b.cancelled for b in bl)
    req = fake.requests_to("GET", "/api/claims")[0]
    assert req.query == {"days": "14", "back": "7"}


def test_bookings_include_cancelled(fake):
    c = Client(fake.url)
    assert len(c.bookings(include_cancelled=True)) > len(c.bookings())


def test_bookings_passes_range(fake):
    Client(fake.url).bookings(days_ahead=3, days_back=30)
    assert fake.requests[-1].query == {"days": "3", "back": "30"}


@pytest.mark.parametrize("kw", [{"days_ahead": 0}, {"days_ahead": 15},
                                {"days_back": 0}, {"days_back": 31}])
def test_bad_ranges_raise_without_http(fake, kw):
    with pytest.raises(ValueError):
        Client(fake.url).bookings(**kw)
    assert fake.requests == []


# -- url resolution ----------------------------------------------------------
def test_default_url(monkeypatch):
    assert Client().url == DEFAULT_URL == "http://10.188.1.1:8765"


def test_env_url_and_arg_precedence(monkeypatch):
    monkeypatch.setenv("HRM_URL", "http://env.example:1/")
    assert Client().url == "http://env.example:1"
    assert Client("http://arg.example:2/").url == "http://arg.example:2"


# -- headers -----------------------------------------------------------------
def test_user_agent_and_no_origin(fake):
    Client(fake.url).users()
    h = fake.requests[0].headers
    assert h["user-agent"] == "hrm-sdk/" + hrm.__version__
    assert "origin" not in h


def test_post_is_json(fake):
    with pytest.raises(NotFound):  # unknown id; we only care how the request looked
        Client(fake.url)._request("POST", "/api/claims/999/cancel", {})
    req = fake.requests[0]
    assert req.headers["content-type"] == "application/json"
    assert "origin" not in req.headers and req.body == {}


# -- errors ------------------------------------------------------------------
@pytest.mark.parametrize("status,exc", [
    (400, InvalidRequest), (413, InvalidRequest), (415, InvalidRequest),
    (403, Forbidden), (404, NotFound), (409, Conflict),
    (429, RateLimited), (503, ServerUnavailable)])
def test_error_status_mapping(fake, status, exc):
    fake.script("GET", "/api/users", status, {"error": "boom", "detail": "details here"})
    with pytest.raises(exc) as ei:
        Client(fake.url).users()
    assert ei.value.status == status and ei.value.detail == "details here"


def test_detail_falls_back_to_error(fake):
    fake.script("GET", "/api/users", 503, {"error": "no history yet"})
    with pytest.raises(ServerUnavailable) as ei:
        Client(fake.url).users()
    assert ei.value.detail == "no history yet"


def test_error_body_not_json(fake):
    fake.script("GET", "/api/users", 404, "<html>nope</html>")
    with pytest.raises(NotFound) as ei:
        Client(fake.url).users()
    assert ei.value.status == 404


def test_invalid_json_success_is_hrmerror(fake):
    fake.script("GET", "/api/users", 200, "not json")
    with pytest.raises(HRMError) as ei:
        Client(fake.url).users()
    assert not isinstance(ei.value, ServerUnavailable)


def test_unreachable_server():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here now
    with pytest.raises(ServerUnavailable) as ei:
        Client("http://127.0.0.1:%d" % port).users()
    assert "cannot reach" in ei.value.detail and ei.value.status is None


def test_timeout_is_server_unavailable(fake):
    fake.script("GET", "/api/users", 200, {"users": []}, delay=1.0)
    with pytest.raises(ServerUnavailable):
        Client(fake.url, timeout=0.2).users()


# -- user resolution ---------------------------------------------------------
def test_explicit_user_wins(fake, monkeypatch):
    monkeypatch.setenv("HRM_USER", "bob")
    assert Client(fake.url, user="carol").resolve_user("alice") == "alice"
    assert fake.requests == []  # no lookup needed


def test_connected_user_beats_env(fake, monkeypatch):
    monkeypatch.setenv("HRM_USER", "bob")
    assert Client(fake.url, user="carol").resolve_user() == "carol"


def test_env_user(fake, monkeypatch):
    monkeypatch.setenv("HRM_USER", "bob")
    assert Client(fake.url).resolve_user() == "bob"


def test_login_user_if_known(fake, monkeypatch):
    monkeypatch.setattr(client_mod.getpass, "getuser", lambda: "user07")
    c = Client(fake.url)
    assert c.resolve_user() == "user07"
    c.resolve_user()
    assert len(fake.requests_to("GET", "/api/users")) == 1  # cached


def test_unknown_login_user(fake, monkeypatch):
    monkeypatch.setattr(client_mod.getpass, "getuser", lambda: "stranger")
    with pytest.raises(UnknownUser) as ei:
        Client(fake.url).resolve_user()
    assert "'stranger'" in ei.value.detail and "HRM_USER" in ei.value.detail


# -- review fixes -------------------------------------------------------------
def test_error_body_read_failure_still_maps_status(fake, monkeypatch):
    import io
    import urllib.error

    class Broken(urllib.error.HTTPError):
        def read(self, *a):
            raise ConnectionResetError("reset while reading")

    def boom(req, timeout=None):
        raise Broken(req.full_url, 409, "Conflict", {}, io.BytesIO())

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", boom)
    with pytest.raises(Conflict) as ei:
        Client(fake.url).users()
    assert ei.value.status == 409


@pytest.mark.parametrize("path,body,call", [
    ("/api/users", {"nope": 1}, lambda c: c.users()),
    ("/api/users", {"users": 5}, lambda c: c.users()),
    ("/api/claims", {"nope": 1}, lambda c: c.bookings()),
    ("/api/claims", {"claims": [{"id": 1}]}, lambda c: c.bookings()),
    ("/api/now", {"gpus": "x"}, lambda c: c.usage()),
])
def test_unexpected_shape_is_hrmerror(fake, path, body, call):
    fake.script("GET", path, 200, body)
    with pytest.raises(HRMError) as ei:
        call(Client(fake.url))
    assert "unexpected response shape" in ei.value.detail and path in ei.value.detail


@pytest.mark.parametrize("url", ["http://[", "no-scheme-host:1"])
def test_malformed_url_is_hrmerror(url):
    with pytest.raises(HRMError) as ei:
        Client(url).users()
    assert "invalid server URL" in ei.value.detail


@pytest.mark.parametrize("status", [500, 502])
def test_5xx_is_server_unavailable(fake, status):
    fake.script("GET", "/api/users", status, {"error": "bad"})
    with pytest.raises(ServerUnavailable):
        Client(fake.url).users()


@pytest.mark.parametrize("kw", [{"days_ahead": 1, "days_back": 1},
                                {"days_ahead": 14, "days_back": 30}])
def test_range_boundaries_accepted(fake, kw):
    Client(fake.url).bookings(**kw)
    assert fake.requests[-1].query == {"days": str(kw["days_ahead"]), "back": str(kw["days_back"])}
