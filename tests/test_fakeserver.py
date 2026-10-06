"""The fake server is shared test infrastructure, so pin its behaviour."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest


def call(fake, method, path, body=None):
    req = urllib.request.Request(
        fake.url + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(fake, **kw):
    body = {"user": "alice", "gpu": 4, "start": "2026-10-06T16:00:00+00:00",
            "end": "2026-10-07T16:00:00+00:00"}
    body.update(kw)
    return call(fake, "POST", "/api/claims", body)


def test_whole_card_when_vram_absent_then_conflict(fake):
    # GPU 4 has no active fixture bookings
    s, b = post(fake)
    assert s == 201 and b["claim"]["vram_mib"] == 81559
    s, b = post(fake, vram_gib=1)
    assert s == 409 and b["error"] == "conflict" and "GPU 4 has only 0.0 GiB" in b["detail"]


def test_capacity_checked_at_inner_boundaries(fake):
    assert post(fake, vram_gib=60, start="2026-10-06T20:00:00+00:00",
                end="2026-10-07T20:00:00+00:00")[0] == 201
    # new booking starts earlier but overlaps the 60 GiB one that begins inside it
    s, _ = post(fake, vram_gib=30)
    assert s == 409
    assert post(fake, vram_gib=15)[0] == 201


def test_validation(fake):
    assert post(fake, user="nobody")[0] == 400
    assert post(fake, vram_gib=0.5)[0] == 400
    assert post(fake, end="2026-10-06T16:00:00+00:00")[0] == 400
    assert post(fake, start="2026-10-06T10:00:00+00:00")[0] == 400  # past


def test_cancel_idempotent_and_404(fake):
    cid = post(fake)[1]["claim"]["id"]
    s1, b1 = call(fake, "POST", "/api/claims/%d/cancel" % cid, {})
    s2, b2 = call(fake, "POST", "/api/claims/%d/cancel" % cid, {})
    assert s1 == s2 == 200 and b1 == b2 and b1["claim"]["cancelled_at"]
    assert call(fake, "POST", "/api/claims/9999/cancel", {})[0] == 404


def test_quick_hold_and_release(fake):
    s, b = call(fake, "POST", "/api/claims/quick", {"user": "bob", "gpu": 5})
    assert s == 201 and b["claim"]["kind"] == "quick"
    assert call(fake, "POST", "/api/claims/quick", {"user": "bob", "gpu": 5})[0] == 409
    assert call(fake, "POST", "/api/claims/quick", {"user": None, "gpu": 5}) == (200, {"claim": None})
    assert call(fake, "POST", "/api/claims/quick", {"user": "bob", "gpu": 5})[0] == 201


def test_reads_reflect_store(fake):
    cid = post(fake)[1]["claim"]["id"]
    ids = [c["id"] for c in call(fake, "GET", "/api/claims")[1]["claims"]]
    assert cid in ids
    gpu4 = call(fake, "GET", "/api/now")[1]["gpus"][4]
    assert [c["id"] for c in gpu4["bookings"]] == [cid] and gpu4["free_mib"] == 0


def test_scripted_queue_then_fallthrough(fake):
    fake.script("POST", "/api/claims", 409, {"error": "conflict", "detail": "x"}, times=1)
    assert post(fake)[0] == 409
    assert post(fake)[0] == 201
    assert len(fake.requests_to("POST", "/api/claims")) == 2
    assert fake.requests[0].body["gpu"] == 4


def test_requires_json_content_type(fake):
    req = urllib.request.Request(fake.url + "/api/claims", method="POST", data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 415


def test_scripted_passthrough_lets_a_request_through(fake):
    # status=None queues a "handle normally" slot, so a later scripted error
    # can hit the second request of a sequence.
    fake.script("POST", "/api/claims", None, times=1)
    fake.script("POST", "/api/claims", 400, {"error": "invalid", "detail": "x"}, times=1)
    s, b = post(fake)
    assert s == 201 and b["claim"]["gpu"] == 4
    assert post(fake, gpu=5)[0] == 400
    assert post(fake, gpu=5)[0] == 201
