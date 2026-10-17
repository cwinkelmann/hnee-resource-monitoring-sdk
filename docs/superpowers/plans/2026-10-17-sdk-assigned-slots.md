# hrm SDK: Assigned Slots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach `hrm` the booking server's assigned-slots contract — a reservation that never fails for lack of capacity, waits for its own start, is told when an important booking takes its card, and gives the card back by truncation — then write two examples that use it.

**Architecture:** One new module `src/hrm/slots.py` holds `Slot`, the object a job works with: it wraps the rows the server assigned, waits for its start, parks the `/events` long-poll in a daemon thread, and releases. `client.py` gains three thin methods (`capacity`, `reserve`, `release_claim`) that speak the new routes and nothing more; `models.py` learns the v4 columns. The existing `book`/`cancel`/`hold`/`release` and `gate`/`Lease` stay exactly as they are — the old contract still works against the server, and nothing in this plan changes their behaviour.

**Tech Stack:** Python 3.9+, standard library only (`urllib.request`, `threading`, `json`). pytest with the repo's own `tests/fakeserver.py` in-memory server. PyTorch appears in one example and is never a package dependency.

**Spec:** `/Users/christian/work/ResourceMonitor/docs/superpowers/specs/2026-10-15-assigned-slots-design.md` — the server side of this contract, already implemented and reviewed on `feat/assigned-slots` in that repo. Its "SDK (`hrm`)" and "What a job looks like" sections are what this plan builds.

## Global Constraints

- **Stdlib only** in `src/hrm/`. `pyproject.toml` keeps `tzdata` (Windows) as its only dependency, with `pandas` and `pytest` as extras. An example may import PyTorch; the package may not.
- **Additive only.** `book`, `cancel`, `hold`, `release`, `gate` and `Lease` keep their current signatures and behaviour. Every existing test passes unchanged.
- **The SDK never kills anything.** It tells the job its card was taken; the job's own code decides to stop. The server writes rows, the client reads them.
- **Python 3.9 floor**, so no `X | Y` annotations, no `dict[str, int]` at runtime; use `typing.Optional`/`List`/`Dict` as the rest of the package does.
- **Errors stay in the existing hierarchy** (`HRMError` and its subclasses in `errors.py`); `error_for_status` maps HTTP codes and must keep doing so.
- The server's routes, exactly as implemented: `POST /api/claims` (optional `gpu`, `start`, `end`, `duration_minutes`, `auto_advance`, `not_later_than`, `count`, `notice_seconds`) → `201 {claim, claims, takes, adjusted_start, assigned_later, revoked}`; `POST /api/claims/{id}/release` `{end?: "now"|ISO}` → `200 {claim, advanced}`; `GET /api/capacity?gpu=&vram_gib=&duration_minutes=&count=` → `200 {free_mib, free_gib, earliest, gpu, gpus}`; `GET /api/claims/{id}/events` → `200 {event, effective_mib, lost_mib, by, truncated, cancelled, deadline}` / `204` after 60 s / `404` / `409` / `503`.
- **The write rate limit is 3600/h per IP.** A reservation costs one write and a release costs one, so a job is two.

## Naming: `reserve`, not `book`

The spec writes `hrm.book(vram_gib=19.5, hours=1)` and `hrm.release(booking)`. **Both names are already taken in this SDK with different meanings**: `book(days=[...], memory=...)` books whole Berlin calendar days and returns a `BookingResult`, and `release(gpu)` clears a card's quick hold. Overloading either would make the dangerous case ambiguous — `release(4)` meaning "clear GPU 4's hold" versus "give back booking 4".

This plan therefore adds `hrm.reserve(...) -> Slot` with `slot.release()`, and leaves the spec's names alone. The spec was written against the server, without the SDK's existing surface in front of it.

## Waiting: `slot.check()`, not an exception from another thread

The spec sketches `with hrm.watch(booking): train()` raising `Preempted` inside the block. Raising an exception into another thread is not something Python does reliably, and a `Preempted` arriving in the middle of a CUDA kernel is not a safe place to checkpoint anyway. So the watcher **sets a flag** and the job asks at a safe point — `slot.check()` between epochs — which is exactly how this SDK's existing `Lease.stop_requested` already works. An `on_taken` callback is available for code that wants to react immediately.

## Testing strategy: the real server for the contract, stubs for the rest

**Revised on the user's point, and it is the right one:** t14 runs the real booking server
(`http://192.168.188.88:8765`, a 16376 MiB RTX 4080 SUPER, confirmed serving
`/api/capacity`). Modelling assigned slots, advancing and revocation a second time inside
`tests/fakeserver.py` would be ~200 lines duplicating logic that already exists — and when
the model drifts from the server, the tests pass while the SDK breaks. That is exactly the
failure the ResourceMonitor acceptance run caught, where every unit test passed against
mocks while the feature was broken end to end.

So the split is by *what is being tested*:

| what | against | why |
|---|---|---|
| the contract — does a reservation with no start come back assigned, does an early release advance someone, does `/events` wake | **the live server** (`tests/test_live.py`, skipped unless `HRM_LIVE_URL` is set) | only the real thing can answer, and a model that disagrees is worse than no model |
| client-side logic — shape validation, parsing a body, `release()` swallowing 404/409, watcher threading, context-manager order | **the fake server's `script()`**, which already returns canned responses for any route | fast, offline, deterministic, no new domain modelling |

`tests/fakeserver.py` therefore gains **no** assigned-slots behaviour. It needs exactly one
fix: a scripted `204` must be sent with no body. Scripted responses are consulted before
`_route`, so a canned reply works for routes the fake has never heard of.

The 174 existing tests keep driving the fake as they do today, unchanged.

## Review Focus

1. **`hours` of 0, negative, non-numeric, or over 14 days** must raise `ValueError` client-side before any request is sent. [Task 4]
2. **A reservation cancelled under a waiting job** (someone cancels it in the dashboard) must make `wait_until_start()` raise, not spin until the heat death of the universe. [Task 5]
3. **`watch()` against a server that has no `/events` route** (an older deployment) must not take the job down: a 404 means "this server cannot notify me", and the job carries on unwatched with a warning. [Task 7]
4. **`release()` twice, or after the window already ended**, must be a no-op rather than an exception — a `finally:` block runs on every path, including ones where the slot is already gone. [Task 6]
5. **The watcher thread must not keep the process alive or outlive its block**: daemon thread, joined on exit, and a thread that dies on an unexpected error must not leave the job believing it is watched. [Task 7]

---

## File Structure

**Created:**
- `src/hrm/slots.py` — `Slot` and `Capacity`: everything a job holds and asks. Waiting, watching and releasing live here because they change together. ~210 lines.
- `tests/test_slots.py` — `Slot` against the fake server: assignment, waiting, releasing, advancing.
- `tests/test_watch.py` — the long-poll: woken, timed out, 404, cancelled, thread lifetime.
- `examples/train_gpu.py` — the runnable script: reserve → wait → train → release, with real PyTorch load and checkpoint/resume.
- `examples/notebook_session.ipynb` — hold a card while you work in a notebook.
- `examples/README.md` — one paragraph per example.

**Modified:**
- `src/hrm/models.py` — `Booking` gains `priority`, `auto_advance`, `group_id`, `notice_seconds`, `taken`.
- `src/hrm/client.py` — `capacity()`, `reserve()`, `release_claim()`, `watch_claim()`.
- `src/hrm/errors.py` — `Preempted`.
- `src/hrm/__init__.py` — `reserve`, `capacity`, `Slot`, `Capacity`, `Preempted` in `__all__`.
- `tests/fakeserver.py` — the new routes and the assigned-slot behaviour.
- `README.md` — a section on the new path, and the examples list.

**Out of scope, deliberately:** `gate`/`Lease` are not rewritten on top of `Slot`. They work, they are tested, and the lease-renewal pattern is still right for a job that wants a long window rather than a queue position. The README will say which to reach for.

---

## Task 1: `Booking` learns the v4 columns

**Files:**
- Modify: `src/hrm/models.py` (the `Booking` dataclass and `from_json`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Booking.priority: str` (default `"lendable"`), `Booking.auto_advance: bool` (default `False`), `Booking.group_id: Optional[str]`, `Booking.notice_seconds: Optional[int]`, `Booking.taken: List[Dict[str, Any]]`. Every later task reads these. `from_json` must accept a body with none of them (an older server) and a body with all of them.

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/test_models.py
def test_booking_reads_the_v4_columns():
    b = Booking.from_json({
        "id": 7, "user": "alice", "gpu": 4, "vram_mib": 20480,
        "start": "2026-10-06T12:00:00+00:00", "end": "2026-10-06T13:00:00+00:00",
        "note": None, "created_at": "2026-10-06T11:00:00+00:00",
        "created_ip": "192.0.2.10", "cancelled_at": None, "cancelled_ip": None,
        "kind": "calendar", "priority": "important", "auto_advance": True,
        "group_id": "abc123", "notice_seconds": 0,
        "taken": [{"start": "2026-10-06T12:30:00+00:00",
                   "end": "2026-10-06T13:00:00+00:00",
                   "vram_mib": 4096, "by": ["dorian.zwanzig"]}],
    })
    assert b.priority == "important"
    assert b.auto_advance is True
    assert b.group_id == "abc123"
    assert b.notice_seconds == 0
    assert b.taken[0]["by"] == ["dorian.zwanzig"]


def test_booking_from_an_older_server_gets_sane_defaults():
    b = Booking.from_json({
        "id": 1, "user": "alice", "gpu": 0, "vram_mib": 1024,
        "start": "2026-10-06T12:00:00+00:00", "end": "2026-10-06T13:00:00+00:00",
        "note": None, "created_at": "2026-10-06T11:00:00+00:00",
        "created_ip": None, "cancelled_at": None, "cancelled_ip": None,
    })
    assert b.priority == "lendable"
    assert b.auto_advance is False
    assert b.group_id is None
    assert b.notice_seconds is None
    assert b.taken == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_models.py -k v4_columns -v`
Expected: FAIL — `AttributeError: 'Booking' object has no attribute 'priority'`

- [ ] **Step 3: Write the implementation**

In `src/hrm/models.py`, add to the `Booking` dataclass after `kind: str`:

```python
    priority: str = "lendable"
    auto_advance: bool = False
    group_id: Optional[str] = None
    notice_seconds: Optional[int] = None
    taken: List[Dict[str, Any]] = field(default_factory=list)
```

`field` comes from `dataclasses`; add it to that import if it is not there. Then in
`from_json`, after `kind=...`:

```python
            priority=d.get("priority", "lendable"),
            auto_advance=bool(d.get("auto_advance", False)),
            group_id=d.get("group_id"),
            notice_seconds=d.get("notice_seconds"),
            taken=list(d.get("taken") or []),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS. Then `python -m pytest -q` — the whole suite stays green; defaults mean
every existing fixture still parses.

- [ ] **Step 5: Commit**

```bash
git add src/hrm/models.py tests/test_models.py
git commit -m "feat(models): Booking carries the v4 columns and its taken segments"
```

---

## Task 2: the live contract tests, and one fake-server fix

**Files:**
- Create: `tests/test_live.py`
- Modify: `tests/fakeserver.py` (a scripted 204 sends no body)
- Test: both of the above

**Interfaces:**
- Produces: the `live` fixture every later contract test uses, and `fake.script(..., 204)`
  working. Nothing in `src/` changes.

```python
# tests/test_live.py
@pytest.fixture(scope="module")
def live():
    """A client against a real booking server, or a skip.

    Set HRM_LIVE_URL (and HRM_LIVE_USER if your login differs) to run these:
        HRM_LIVE_URL=http://192.168.188.88:8765 HRM_LIVE_USER=christian pytest tests/test_live.py
    """
```

**How to run it.** These tests book real slots on a real box and release them again. They
use the smallest booking the server allows (1 GiB) for ten minutes, and every test releases
what it made in a `finally`. A run costs about a dozen writes against the 3600/h per-IP
budget.

- [ ] **Step 1: Write the failing tests**

```python
"""The contract, against a real booking server. Skipped unless HRM_LIVE_URL is set.

    HRM_LIVE_URL=http://192.168.188.88:8765 HRM_LIVE_USER=christian \
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
    whole = max(int(cap.free_gib), 2)
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
    assert victim.taken is not None, "the watcher was never told"
    assert victim.taken["event"] in ("taken", "cancelled")
    assert waited < 5.0
    with pytest.raises(hrm.Preempted):
        victim.check()


def test_a_full_card_still_grants_a_slot_later(client, slots):
    cap = client.capacity(memory=1, hours=1)
    whole = max(int(cap.free_gib), 2)
    first = client.reserve(memory=whole, hours=2, note="hrm live filler")
    slots.append(first)
    second = client.reserve(memory=whole, hours=1, gpu=first.gpu, note="hrm live queued")
    slots.append(second)
    assert second.assigned_later is True         # never NoCapacity
    assert second.start >= first.end - timedelta(minutes=1)
```

`victim.watcher_parked` is one more thing `Slot` must expose — a boolean that is True while
its `Watcher` has at least one long-poll in flight. Task 7 produces it as
`Watcher.watching`; `Slot.watcher_parked` is the one-line property that reads it, added in
Task 7 beside `watch()`.

```python
# appended to tests/test_fakeserver.py
def test_a_scripted_204_is_sent_without_a_body(fake):
    import urllib.request
    fake.script("GET", "/api/claims/1/events", 204, None)
    with urllib.request.urlopen(fake.url + "/api/claims/1/events", timeout=5) as r:
        assert r.status == 204
        assert r.read() == b""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_fakeserver.py -k scripted_204 -v`
Expected: FAIL — the handler sends `"null"` as the body, so `r.read()` is `b"null"`.

Run: `HRM_LIVE_URL=http://192.168.188.88:8765 HRM_LIVE_USER=christian PYTHONPATH=src python -m pytest tests/test_live.py -v`
Expected: FAIL — `AttributeError: 'Client' object has no attribute 'capacity'`. Everything
`tests/test_live.py` needs arrives in Tasks 3–8, so this file stays red until Task 8 and is
the real gate on the whole plan.

Run: `PYTHONPATH=src python -m pytest tests/test_live.py -q` (no env)
Expected: `6 skipped` — the file must be inert for anyone without a server.

- [ ] **Step 3: Write the implementation**

In `tests/fakeserver.py`'s handler, where the response is written:

```python
                data = b"" if payload is None else (
                    payload if isinstance(payload, str) else json.dumps(payload)).encode()
```

replacing the two lines that always encode a JSON payload. A 204 with a `Content-Length: 0`
is what the real server sends, and `Client._request` must read it as "nothing happened"
rather than invalid JSON (Task 7 makes it do that).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_fakeserver.py -v`
Expected: PASS, including the existing tests — no other route sends a `None` payload today.

Run: `PYTHONPATH=src python -m pytest tests/test_live.py -q`
Expected: `6 skipped`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_live.py tests/fakeserver.py tests/test_fakeserver.py
git commit -m "test(live): the contract against a real server, and a scripted 204 sends no body"
```

---

## Task 3: `Capacity` and `hrm.capacity()`

**Files:**
- Create: `src/hrm/slots.py`
- Modify: `src/hrm/client.py` (new `capacity` method), `src/hrm/__init__.py`
- Test: `tests/test_slots.py`

**Interfaces:**
- Consumes: the fake server's `/api/capacity` (Task 2).
- Produces:
```python
# slots.py
@dataclass(frozen=True)
class Capacity:
    free_mib: int
    free_gib: float
    earliest: Optional[datetime]      # None: nothing fits inside the server's horizon
    gpu: Optional[int]
    gpus: List[int]
    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Capacity"

# client.py
def capacity(self, memory: Optional[float] = None, *, hours: float = 1.0,
             gpu: Optional[int] = None, count: int = 1) -> Capacity

# __init__.py
def capacity(memory=None, *, hours=1.0, gpu=None, count=1) -> Capacity
```

- [ ] **Step 1: Write the failing tests**

```python
"""Slot, Capacity and the assigned-slots path, against the fake server."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import hrm
from hrm import Capacity
from hrm.errors import Conflict, InvalidRequest


@pytest.fixture
def client(fake):
    return hrm.Client(url=fake.url, user="cwinkelmann")


def test_capacity_reports_what_is_free_and_when_a_shape_fits(client):
    cap = client.capacity(memory=1, hours=1)
    assert isinstance(cap, Capacity)
    assert cap.free_mib > 0
    assert cap.free_gib == pytest.approx(cap.free_mib / 1024, abs=0.1)
    assert isinstance(cap.earliest, datetime)
    assert cap.earliest.tzinfo is not None
    assert cap.gpu in cap.gpus


def test_capacity_for_one_named_card_only_considers_that_card(client):
    cap = client.capacity(memory=1, hours=1, gpu=0)
    assert cap.gpus == [0]


def test_capacity_rejects_a_bad_shape_before_asking_the_server(client, fake):
    before = len(fake.requests)
    for bad in (0, -1, "lots"):
        with pytest.raises(ValueError):
            client.capacity(memory=bad, hours=1)
    for bad in (0, -1, 15 * 24):
        with pytest.raises(ValueError):
            client.capacity(memory=1, hours=bad)
    assert len(fake.requests) == before        # nothing was sent


def test_capacity_is_on_the_module_too(fake):
    hrm.connect(user="cwinkelmann", url=fake.url)
    assert hrm.capacity(memory=1, hours=1).free_mib > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -v`
Expected: FAIL — `ImportError: cannot import name 'Capacity' from 'hrm'`

- [ ] **Step 3: Write the implementation**

Create `src/hrm/slots.py`:

```python
"""A reserved slot: what a job holds, waits for, is told about, and gives back.

The booking server assigns the slot; this module is the job's side of that
contract. A reservation never fails for lack of capacity -- it comes back with a
start, which may be in the future and may move earlier -- so the job's shape is
always: reserve, wait for the start, run, release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .models import Booking, parse_time

MIB_PER_GIB = 1024
MAX_HOURS = 14 * 24


def _check_shape(memory: Optional[float], hours: float, count: int) -> None:
    """The client-side rules, checked before anything is sent: a bad shape is a
    programming error, not a server round trip."""
    if memory is not None and (isinstance(memory, bool)
                               or not isinstance(memory, (int, float))
                               or not memory > 0):
        raise ValueError("memory must be a number of GiB above 0, got %r" % (memory,))
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) \
            or not 0 < hours <= MAX_HOURS:
        raise ValueError("hours must be a number in (0, %d], got %r" % (MAX_HOURS, hours))
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 8:
        raise ValueError("count must be a whole number of GPUs, 1..8, got %r" % (count,))


@dataclass(frozen=True)
class Capacity:
    """What one card has free now, and when a given shape would next fit."""
    free_mib: int
    free_gib: float
    earliest: Optional[datetime]
    gpu: Optional[int]
    gpus: List[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Capacity":
        earliest = d.get("earliest")
        return cls(
            free_mib=int(d["free_mib"]),
            free_gib=float(d.get("free_gib", int(d["free_mib"]) / MIB_PER_GIB)),
            earliest=None if earliest is None else parse_time(earliest),
            gpu=d.get("gpu"),
            gpus=list(d.get("gpus") or []),
        )
```

In `client.py`, beside `bookings()`:

```python
    def capacity(self, memory: Optional[float] = None, *, hours: float = 1.0,
                 gpu: Optional[int] = None, count: int = 1) -> Capacity:
        """What is free now and when `memory` GiB would next fit for `hours`.

        Ask this instead of summing bookings yourself: the server owns the
        arithmetic, and clients that reimplemented it got it wrong.
        """
        _check_shape(memory, hours, count)
        query = ["duration_minutes=%d" % max(int(round(hours * 60)), 1),
                 "count=%d" % count]
        if memory is not None:
            query.append("vram_gib=%s" % memory)
        if gpu is not None:
            query.append("gpu=%d" % gpu)
        data = self._request("GET", "/api/capacity?" + "&".join(query))
        with _shape("/api/capacity"):
            return Capacity.from_json(data)
```

with `from .slots import Capacity, _check_shape` added to `client.py`'s imports.

In `__init__.py`, add `Capacity` to the `from .slots import ...` line (creating it) and to
`__all__`, plus the module-level function beside `bookings()`:

```python
def capacity(memory: Optional[float] = None, *, hours: float = 1.0,
             gpu: Optional[int] = None, count: int = 1) -> Capacity:
    return _client().capacity(memory, hours=hours, gpu=gpu, count=count)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS, 4 tests. Then `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/hrm/slots.py src/hrm/client.py src/hrm/__init__.py tests/test_slots.py
git commit -m "feat(slots): Capacity and hrm.capacity() -- ask the server, do not sum"
```

---

## Task 4: `hrm.reserve()` — a booking that never fails for lack of capacity

**Files:**
- Modify: `src/hrm/slots.py` (`Slot`), `src/hrm/client.py` (`reserve`), `src/hrm/__init__.py`
- Test: `tests/test_slots.py`

**Interfaces:**
- Consumes: `_check_shape` and `Capacity` (Task 3), `Booking` (Task 1).
- Produces:
```python
class Slot:
    def __init__(self, client, bookings: List[Booking], assigned_later: bool,
                 revoked: Optional[List[Dict[str, Any]]] = None) -> None
    bookings: List[Booking]
    assigned_later: bool
    revoked: List[Dict[str, Any]]
    booking: Booking            # bookings[0]
    id: int                     # bookings[0].id
    gpu: int                    # bookings[0].gpu
    gpus: List[int]
    start: datetime
    end: datetime
    memory_gib: float

# client.py
def reserve(self, memory: Optional[float] = None, *, hours: float = 1.0,
            gpu: Optional[int] = None, count: int = 1,
            auto_advance: bool = True, not_later_than: Optional[Union[datetime, str]] = None,
            notice_seconds: Optional[int] = None, important: bool = False,
            note: Optional[str] = None, user: Optional[str] = None) -> Slot
```
`auto_advance` defaults to **True**: a job that asked for the earliest slot wants it to move
earlier when a card frees up. `important=True` sends `priority="important"`.

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/test_slots.py
def test_a_reservation_on_a_free_card_starts_now(client, fake):
    slot = client.reserve(memory=1, hours=1, note="unit test")
    assert slot.assigned_later is False
    assert slot.start <= fake.now
    assert slot.end - slot.start == timedelta(hours=1)
    assert slot.memory_gib == pytest.approx(1.0, abs=0.01)
    assert slot.gpus == [slot.gpu]


def test_a_reservation_on_a_full_card_is_assigned_later_and_still_succeeds(client, fake):
    total = fake.gpu_total_gib(0)
    client.reserve(memory=total, hours=2, gpu=0)
    slot = client.reserve(memory=total, hours=1, gpu=0)
    assert slot.assigned_later is True        # never NoCapacity
    assert slot.start >= fake.now


def test_a_group_reservation_comes_back_as_one_slot_over_several_cards(client):
    slot = client.reserve(memory=1, hours=1, count=2)
    assert len(slot.bookings) == 2
    assert len(slot.gpus) == 2
    assert len({b.start for b in slot.bookings}) == 1
    assert slot.id == slot.bookings[0].id


def test_auto_advance_is_on_by_default_and_sent(client, fake):
    client.reserve(memory=1, hours=1)
    body = fake.requests_to("POST", "/api/claims")[-1].body
    assert body["auto_advance"] is True
    assert "start" not in body and "end" not in body
    assert body["duration_minutes"] == 60


def test_notice_and_important_reach_the_server(client, fake):
    client.reserve(memory=1, hours=1, notice_seconds=0, important=True)
    body = fake.requests_to("POST", "/api/claims")[-1].body
    assert body["notice_seconds"] == 0
    assert body["priority"] == "important"


def test_not_later_than_is_a_conflict_the_caller_can_read(client, fake):
    total = fake.gpu_total_gib(0)
    client.reserve(memory=total, hours=4, gpu=0)
    with pytest.raises(Conflict) as e:
        client.reserve(memory=total, hours=1, gpu=0,
                       not_later_than=fake.now + timedelta(minutes=30))
    assert e.value.status == 409


@pytest.mark.parametrize("kwargs", [
    {"memory": 0}, {"memory": -1}, {"memory": "lots"},
    {"hours": 0}, {"hours": -1}, {"hours": 15 * 24},
    {"count": 0}, {"count": 9}, {"count": 1.5},
])
def test_a_bad_shape_is_a_valueerror_before_any_request(client, fake, kwargs):
    before = len(fake.requests)
    args = {"memory": 1, "hours": 1}
    args.update(kwargs)
    with pytest.raises(ValueError):
        client.reserve(**args)
    assert len(fake.requests) == before


def test_reserve_is_on_the_module_too(fake):
    hrm.connect(user="cwinkelmann", url=fake.url)
    assert hrm.reserve(memory=1, hours=1).id > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -k reserv -v`
Expected: FAIL — `AttributeError: 'Client' object has no attribute 'reserve'`

- [ ] **Step 3: Write the implementation**

In `slots.py`:

```python
class Slot:
    """One reserved window, over one card or several.

    A slot is not a lock: it is the calendar row the server assigned, and the
    honour system does the rest. `start` may be in the future -- wait for it with
    wait_until_start() -- and may move earlier while you wait.
    """

    def __init__(self, client, bookings: List[Booking], assigned_later: bool,
                 revoked: Optional[List[Dict[str, Any]]] = None) -> None:
        if not bookings:
            raise ValueError("a slot needs at least one booking")
        self.client = client
        self.bookings = list(bookings)
        self.assigned_later = bool(assigned_later)
        self.revoked = list(revoked or [])

    def __repr__(self) -> str:
        return "Slot(ids=%r, gpus=%r, %s..%s)" % (
            [b.id for b in self.bookings], self.gpus,
            self.start.isoformat(), self.end.isoformat())

    @property
    def booking(self) -> Booking:
        return self.bookings[0]

    @property
    def id(self) -> int:
        return self.bookings[0].id

    @property
    def gpu(self) -> int:
        return self.bookings[0].gpu

    @property
    def gpus(self) -> List[int]:
        return [b.gpu for b in self.bookings]

    @property
    def start(self) -> datetime:
        return min(b.start for b in self.bookings)

    @property
    def end(self) -> datetime:
        return max(b.end for b in self.bookings)

    @property
    def memory_gib(self) -> float:
        return round(self.bookings[0].vram_mib / MIB_PER_GIB, 2)
```

In `client.py`:

```python
    def reserve(self, memory: Optional[float] = None, *, hours: float = 1.0,
                gpu: Optional[int] = None, count: int = 1,
                auto_advance: bool = True,
                not_later_than: Optional[Union[datetime, str]] = None,
                notice_seconds: Optional[int] = None, important: bool = False,
                note: Optional[str] = None, user: Optional[str] = None) -> Slot:
        """Reserve `memory` GiB for `hours`, starting as soon as it fits.

        This does not fail for lack of capacity: the server answers with a window,
        and `slot.assigned_later` says whether it starts now or later. Wait for it
        with `slot.wait_until_start()`, and give it back with `slot.release()`.
        """
        _check_shape(memory, hours, count)
        if notice_seconds is not None and (isinstance(notice_seconds, bool)
                                           or not isinstance(notice_seconds, int)
                                           or notice_seconds < 0):
            raise ValueError("notice_seconds must be a whole number of seconds, 0 or more")
        body = {"user": self.resolve_user(user), "count": count,
                "duration_minutes": max(int(round(hours * 60)), 1),
                "auto_advance": bool(auto_advance),
                "priority": "important" if important else "lendable"}
        if memory is not None:
            body["vram_gib"] = memory
        if gpu is not None:
            body["gpu"] = gpu
        if note is not None:
            body["note"] = note
        if notice_seconds is not None:
            body["notice_seconds"] = notice_seconds
        if not_later_than is not None:
            body["not_later_than"] = _iso(_to_datetime(not_later_than, "not_later_than"))
        data = self._request("POST", "/api/claims", body)
        with _shape("/api/claims"):
            rows = data.get("claims") or [data["claim"]]
            return Slot(self, [Booking.from_json(c) for c in rows],
                        bool(data.get("assigned_later", False)),
                        data.get("revoked"))
```

`_iso` is a one-liner beside `_to_datetime` in `client.py` if it does not exist:
`def _iso(t: datetime) -> str: return t.astimezone(timezone.utc).isoformat()`. Add `Slot`
to the `from .slots import ...` line.

In `__init__.py`, beside `book()`:

```python
def reserve(memory: Optional[float] = None, *, hours: float = 1.0,
            gpu: Optional[int] = None, count: int = 1, auto_advance: bool = True,
            not_later_than: Optional[Union[datetime, str]] = None,
            notice_seconds: Optional[int] = None, important: bool = False,
            note: Optional[str] = None, user: Optional[str] = None) -> Slot:
    return _client().reserve(memory, hours=hours, gpu=gpu, count=count,
                             auto_advance=auto_advance, not_later_than=not_later_than,
                             notice_seconds=notice_seconds, important=important,
                             note=note, user=user)
```

and add `"reserve"`, `"capacity"`, `"Slot"`, `"Capacity"` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS, 16 tests. Then `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/hrm/slots.py src/hrm/client.py src/hrm/__init__.py tests/test_slots.py
git commit -m "feat(slots): hrm.reserve() -- a booking that is always granted, maybe later"
```

---

## Task 5: `slot.wait_until_start()`

**Files:**
- Modify: `src/hrm/slots.py`
- Test: `tests/test_slots.py`

**Interfaces:**
- Consumes: `Slot` (Task 4), `Client.bookings()`.
- Produces:
```python
def refresh(self) -> "Slot"          # re-read this slot's rows from the server
def wait_until_start(self, poll: float = 10.0, timeout: Optional[float] = None,
                     sleep: Optional[Callable[[float], None]] = None) -> "Slot"
```
`wait_until_start` returns `self`, so `slot = hrm.reserve(...).wait_until_start()` reads
well. It raises `Conflict` if the slot is cancelled under it and `TimeoutError` if `timeout`
passes.

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/test_slots.py
def test_waiting_for_a_slot_that_already_started_returns_at_once(client):
    slot = client.reserve(memory=1, hours=1)
    slept = []
    assert slot.wait_until_start(sleep=slept.append) is slot
    assert slept == []


def test_waiting_polls_until_the_start_arrives(client, fake):
    total = fake.gpu_total_gib(0)
    client.reserve(memory=total, hours=2, gpu=0)
    slot = client.reserve(memory=total, hours=1, gpu=0)
    assert slot.assigned_later is True
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        fake.now = fake.now + timedelta(hours=1)    # time passes on the server
    slot.wait_until_start(poll=5, sleep=sleep)
    assert calls == [5, 5]                          # two hours at one hour a tick
    assert slot.start <= fake.now


def test_waiting_notices_the_slot_moved_earlier(client, fake):
    total = fake.gpu_total_gib(0)
    blocker = client.reserve(memory=total, hours=4, gpu=0)
    slot = client.reserve(memory=total, hours=1, gpu=0)
    far = slot.start

    def sleep(_seconds):
        fake.advance_claim(slot.id, fake.now)       # someone released early
    slot.wait_until_start(poll=5, sleep=sleep)
    assert slot.start < far


def test_a_cancelled_slot_stops_the_wait_with_a_conflict(client, fake):
    total = fake.gpu_total_gib(0)
    client.reserve(memory=total, hours=2, gpu=0)
    slot = client.reserve(memory=total, hours=1, gpu=0)
    client.cancel(slot.id)
    with pytest.raises(Conflict):
        slot.wait_until_start(poll=1, sleep=lambda _s: None)


def test_waiting_gives_up_after_the_timeout(client, fake):
    total = fake.gpu_total_gib(0)
    client.reserve(memory=total, hours=8, gpu=0)
    slot = client.reserve(memory=total, hours=1, gpu=0)
    ticks = [0.0]

    def sleep(seconds):
        ticks[0] += seconds
    with pytest.raises(TimeoutError):
        slot.wait_until_start(poll=5, timeout=12,
                              sleep=sleep)
    assert ticks[0] <= 15
```

`fake.advance_claim(claim_id, new_start)` is one more fake-server helper — add it in this
task since only this test needs it:

```python
    def advance_claim(self, claim_id: int, new_start: datetime) -> None:
        """Move a booking earlier, as an early release on the real server would."""
        for c in self.claims:
            if c["id"] == claim_id:
                length = _parse(c["end"]) - _parse(c["start"])
                c["start"] = _iso(new_start)
                c["end"] = _iso(new_start + length)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -k wait -v`
Expected: FAIL — `AttributeError: 'Slot' object has no attribute 'wait_until_start'`

- [ ] **Step 3: Write the implementation**

In `slots.py` (with `import time` and `from .errors import Conflict` at the top):

```python
    def refresh(self) -> "Slot":
        """Re-read this slot's rows. The server may have moved them earlier."""
        ids = {b.id for b in self.bookings}
        found = [b for b in self.client.bookings(include_cancelled=True) if b.id in ids]
        if found:
            self.bookings = sorted(found, key=lambda b: (b.start, b.id))
        return self

    @property
    def cancelled(self) -> bool:
        return all(b.cancelled_at is not None for b in self.bookings)

    def wait_until_start(self, poll: float = 10.0, timeout: Optional[float] = None,
                         sleep: Optional[Callable[[float], None]] = None) -> "Slot":
        """Block until the slot's window has begun, then return self.

        Re-reads the rows on every tick, so a slot that moves earlier shortens the
        wait. Raises Conflict if the slot is cancelled under us -- otherwise a job
        would wait for a window that no longer exists -- and TimeoutError if
        `timeout` seconds pass first.
        """
        rest = sleep or time.sleep
        waited = 0.0
        while True:
            if self.cancelled:
                raise Conflict("slot %d was cancelled while waiting for its start"
                               % self.id)
            if self.start <= self.client.now():
                return self
            if timeout is not None and waited >= timeout:
                raise TimeoutError("slot %d still starts at %s after %.0fs"
                                   % (self.id, self.start.isoformat(), waited))
            rest(poll)
            waited += poll
            self.refresh()
```

`Client.now()` is a one-line public reader for the injected clock, added in this task so
`Slot` does not reach into `client._clock`:

```python
    def now(self) -> datetime:
        """The client's idea of now (injectable for tests)."""
        return self._clock()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS, 21 tests. Then `python -m pytest -q`.

The fake server's clock and the client's are separate, so `test_waiting_polls_until_the_start_arrives`
must advance both: construct the client with `clock=lambda: fake.now` in the `client`
fixture so one `fake.now = ...` moves both. Update the fixture in this task:

```python
@pytest.fixture
def client(fake):
    return hrm.Client(url=fake.url, user="cwinkelmann", clock=lambda: fake.now)
```

- [ ] **Step 5: Commit**

```bash
git add src/hrm/slots.py src/hrm/client.py tests/fakeserver.py tests/test_slots.py
git commit -m "feat(slots): wait_until_start re-reads, so an advanced slot waits less"
```

---

## Task 6: `slot.release()` — give the card back honestly

**Files:**
- Modify: `src/hrm/slots.py`, `src/hrm/client.py` (`release_claim`)
- Test: `tests/test_slots.py`

**Interfaces:**
- Consumes: the fake server's `POST /api/claims/{id}/release` (Task 2).
- Produces:
```python
# client.py
def release_claim(self, booking_or_id: Union[Booking, int],
                  end: Optional[Union[datetime, str]] = None) -> Dict[str, Any]
#   -> {"claim": Booking, "advanced": [ {id, user, gpu, old_start, new_start}, ... ]}

# slots.py
def release(self, end: Optional[Union[datetime, str]] = None) -> List[Dict[str, Any]]
#   returns the advanced list, flattened across the slot's rows; always safe to call
```

**Review Focus item 4 lives here.** `release()` runs from a `finally:` block, so calling it
twice, after the window ended, or on a slot someone already cancelled must not raise. It
swallows `NotFound` and `Conflict` for exactly that reason, and nothing else.

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/test_slots.py
def test_releasing_a_started_slot_truncates_it_rather_than_cancelling(client, fake):
    slot = client.reserve(memory=1, hours=4)
    original_end = slot.end
    fake.now = fake.now + timedelta(hours=1)
    slot.release()
    slot.refresh()
    assert slot.booking.cancelled_at is None       # it ran; the energy ledger keeps it
    assert slot.end < original_end


def test_releasing_a_slot_that_has_not_started_cancels_it(client, fake):
    total = fake.gpu_total_gib(0)
    client.reserve(memory=total, hours=2, gpu=0)
    slot = client.reserve(memory=total, hours=1, gpu=0)
    slot.release()
    slot.refresh()
    assert slot.booking.cancelled_at is not None


def test_releasing_every_member_of_a_group(client):
    slot = client.reserve(memory=1, hours=1, count=2)
    slot.release()
    slot.refresh()
    assert all(b.cancelled_at is not None or b.end <= slot.client.now()
               for b in slot.bookings)


def test_release_is_safe_to_call_twice(client, fake):
    slot = client.reserve(memory=1, hours=4)
    fake.now = fake.now + timedelta(hours=1)
    slot.release()
    slot.release()                                 # from a finally: block, on every path


def test_release_is_safe_on_an_already_cancelled_slot(client):
    slot = client.reserve(memory=1, hours=4)
    client.cancel(slot.id)
    slot.release()                                 # 409 from the server, swallowed


def test_release_is_safe_when_the_booking_is_gone(client, fake):
    slot = client.reserve(memory=1, hours=4)
    fake.script("POST", "/api/claims/%d/release" % slot.id, 404, {"error": "not found"})
    slot.release()


def test_release_lets_a_real_error_through(client, fake):
    slot = client.reserve(memory=1, hours=4)
    fake.script("POST", "/api/claims/%d/release" % slot.id, 500, {"error": "internal"})
    with pytest.raises(hrm.ServerUnavailable):
        slot.release()


def test_release_reports_the_slots_it_advanced(client, fake):
    slot = client.reserve(memory=1, hours=4)
    fake.script("POST", "/api/claims/%d/release" % slot.id, 200,
                {"claim": None, "advanced": [{"id": 99, "user": "alice", "gpu": 0,
                                              "old_start": "2026-10-06T17:00:00+00:00",
                                              "new_start": "2026-10-06T13:00:00+00:00"}]})
    moved = slot.release()
    assert [m["id"] for m in moved] == [99]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -k releas -v`
Expected: FAIL — `AttributeError: 'Slot' object has no attribute 'release'`

- [ ] **Step 3: Write the implementation**

In `client.py`:

```python
    def release_claim(self, booking_or_id: Union[Booking, int],
                      end: Optional[Union[datetime, str]] = None) -> Dict[str, Any]:
        """Give a booking back: a started one has its end moved to `end` (default
        now), so the window it really used stays on the calendar; one that has not
        started is cancelled instead. Returns the server's reply with the booking
        parsed and the slots it pulled earlier."""
        booking_id = booking_or_id.id if isinstance(booking_or_id, Booking) \
            else int(booking_or_id)
        body = {} if end is None else {
            "end": _iso(_to_datetime(end, "end"))}
        data = self._request("POST", "/api/claims/%d/release" % booking_id, body)
        with _shape("/api/claims/<id>/release"):
            claim = data.get("claim")
            return {"claim": None if claim is None else Booking.from_json(claim),
                    "advanced": list(data.get("advanced") or [])}
```

In `slots.py` (with `from .errors import Conflict, NotFound` at the top):

```python
    def release(self, end: Optional[Union[datetime, str]] = None) -> List[Dict[str, Any]]:
        """Give every row of this slot back, and return the slots that moved earlier.

        Safe on every path: this is what a job's `finally:` calls, so a slot that is
        already cancelled, already over, or gone from the server is not an error
        worth taking a job down for. Anything else (a 500, an unreachable server)
        still raises.
        """
        moved: List[Dict[str, Any]] = []
        for booking in self.bookings:
            try:
                moved.extend(self.client.release_claim(booking.id, end)["advanced"])
            except (NotFound, Conflict):
                continue          # already cancelled, already over, or never existed
        return moved
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slots.py -v`
Expected: PASS, 29 tests. Then `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/hrm/slots.py src/hrm/client.py tests/test_slots.py
git commit -m "feat(slots): release truncates, and is safe from a finally block"
```

---

## Task 7: `slot.watch()` — the interrupt

**Files:**
- Modify: `src/hrm/slots.py` (`Watcher`, `Slot.watch`, `Slot.check`), `src/hrm/client.py` (`watch_claim`), `src/hrm/errors.py` (`Preempted`)
- Test: `tests/test_watch.py`

**Interfaces:**
- Consumes: the fake server's `GET /api/claims/{id}/events` (Task 2).
- Produces:
```python
# errors.py
class Preempted(HRMError):
    def __init__(self, detail: str = "", event: Optional[Dict[str, Any]] = None)
    event: Dict[str, Any]       # the server's payload: event, by, lost_mib, truncated, ...

# client.py
def watch_claim(self, booking_or_id, timeout: float = 90.0) -> Optional[Dict[str, Any]]
#   one long-poll: the event, or None when the server's hold expired (204)

# slots.py
def watch(self, on_taken: Optional[Callable[[Dict[str, Any]], None]] = None) -> "Watcher"
def check(self) -> None         # raises Preempted once the card has been taken
taken: Optional[Dict[str, Any]]  # the event, or None
```
`Watcher` is a context manager: entering starts one daemon thread per row, leaving stops it.

**Review Focus items 3 and 5 live here.** A server with no `/events` route answers 404: the
watcher logs a warning once and stops, leaving `slot.taken` None, and the job runs on
unwatched rather than dying. The threads are daemons, are joined on exit, and a thread that
raises sets a flag so `watching` stops claiming to be live.

- [ ] **Step 1: Write the failing tests**

```python
"""The watch channel: a parked long-poll tells a job its card was taken."""
from __future__ import annotations

import time
from datetime import timedelta

import pytest

import hrm
from hrm.errors import Preempted


@pytest.fixture
def client(fake):
    return hrm.Client(url=fake.url, user="cwinkelmann", clock=lambda: fake.now)


def _wait_for(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_watched_slot_learns_it_was_taken(client, fake):
    slot = client.reserve(memory=1, hours=1, notice_seconds=0)
    fake.queue_event(slot.id, {"event": "taken", "effective_mib": 0, "lost_mib": 1024,
                               "by": ["dorian.zwanzig"], "truncated": True,
                               "cancelled": False, "deadline": None})
    with slot.watch():
        assert _wait_for(lambda: slot.taken is not None)
    assert slot.taken["by"] == ["dorian.zwanzig"]


def test_check_raises_preempted_once_the_card_is_taken(client, fake):
    slot = client.reserve(memory=1, hours=1, notice_seconds=0)
    fake.queue_event(slot.id, {"event": "taken", "effective_mib": 0, "lost_mib": 1024,
                               "by": ["dorian.zwanzig"], "truncated": True,
                               "cancelled": False, "deadline": None})
    with slot.watch():
        assert _wait_for(lambda: slot.taken is not None)
        with pytest.raises(Preempted) as e:
            slot.check()
    assert e.value.event["event"] == "taken"


def test_check_is_quiet_while_the_slot_stands(client, fake):
    slot = client.reserve(memory=1, hours=1)
    fake.event_hold_s = 0.05
    with slot.watch():
        slot.check()
        slot.check()


def test_the_on_taken_callback_fires_immediately(client, fake):
    slot = client.reserve(memory=1, hours=1, notice_seconds=0)
    fake.queue_event(slot.id, {"event": "cancelled", "effective_mib": 0,
                               "lost_mib": 1024, "by": [], "truncated": False,
                               "cancelled": True, "deadline": None})
    seen = []
    with slot.watch(on_taken=seen.append):
        assert _wait_for(lambda: bool(seen))
    assert seen[0]["event"] == "cancelled"


def test_a_server_without_the_events_route_does_not_take_the_job_down(client, fake):
    slot = client.reserve(memory=1, hours=1)
    fake.script("GET", "/api/claims/%d/events" % slot.id, 404,
                {"error": "not found"}, times=None)
    with slot.watch() as watcher:
        assert _wait_for(lambda: not watcher.watching)
        slot.check()                       # unwatched, but the job carries on
    assert slot.taken is None


def test_the_hold_expiring_just_reconnects(client, fake):
    slot = client.reserve(memory=1, hours=1)
    fake.event_hold_s = 0.05
    with slot.watch():
        assert _wait_for(lambda: len(fake.requests_to(
            "GET", "/api/claims/%d/events" % slot.id)) >= 2)
    assert slot.taken is None


def test_the_watcher_thread_does_not_outlive_its_block(client, fake):
    import threading
    slot = client.reserve(memory=1, hours=1)
    fake.event_hold_s = 0.05
    before = threading.active_count()
    with slot.watch():
        assert _wait_for(lambda: threading.active_count() > before)
    assert _wait_for(lambda: threading.active_count() == before)


def test_a_group_slot_watches_every_row(client, fake):
    slot = client.reserve(memory=1, hours=1, count=2, notice_seconds=0)
    fake.event_hold_s = 0.05
    with slot.watch():
        for booking in slot.bookings:
            assert _wait_for(lambda b=booking: fake.requests_to(
                "GET", "/api/claims/%d/events" % b.id))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_watch.py -v`
Expected: FAIL — `ImportError: cannot import name 'Preempted' from 'hrm.errors'`

- [ ] **Step 3: Write the implementation**

In `errors.py`:

```python
class Preempted(HRMError):
    """An important booking took this slot's card. Checkpoint and stop.

    ``event`` is the server's payload: which card, how much was lost, by whom,
    and whether the row was truncated or cancelled.
    """

    def __init__(self, detail: str = "", event: Optional[dict] = None) -> None:
        super().__init__(detail)
        self.event = event or {}
```

In `client.py`:

```python
    def watch_claim(self, booking_or_id: Union[Booking, int],
                    timeout: float = 90.0) -> Optional[Dict[str, Any]]:
        """One long-poll on a booking: block until something happens to it.

        Returns the event, or None if the server's hold expired with nothing to
        say (HTTP 204) -- then call it again. The timeout must exceed the server's
        hold (60 s), or every call looks like a network failure.
        """
        booking_id = booking_or_id.id if isinstance(booking_or_id, Booking) \
            else int(booking_or_id)
        path = "/api/claims/%d/events" % booking_id
        was, self.timeout = self.timeout, timeout
        try:
            data = self._request("GET", path)
        finally:
            self.timeout = was
        return data or None
```

`_request` must not choke on a bodyless 204. In `_request`, after `raw = resp.read()`:

```python
            if not raw:
                return {}            # 204: nothing happened, and that is an answer
```

In `slots.py` (with `import logging`, `import threading` and
`from .errors import Conflict, NotFound, Preempted` at the top, and
`log = logging.getLogger("hrm.slots")`):

```python
class Watcher:
    """Parks one long-poll per row of a slot, in daemon threads.

    The threads only ever *set* things: the job asks at a safe point with
    Slot.check(). Raising into another thread is not something Python does
    reliably, and the middle of a CUDA kernel is not where anyone wants to
    checkpoint.
    """

    def __init__(self, slot: "Slot",
                 on_taken: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self.slot = slot
        self.on_taken = on_taken
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._live = 0
        self._lock = threading.Lock()

    @property
    def watching(self) -> bool:
        """True while at least one long-poll is parked. A server with no /events
        route leaves this False, and the job runs on unwatched."""
        with self._lock:
            return self._live > 0

    def __enter__(self) -> "Watcher":
        for booking in self.slot.bookings:
            thread = threading.Thread(target=self._loop, args=(booking.id,),
                                      name="hrm-watch-%d" % booking.id, daemon=True)
            with self._lock:
                self._live += 1
            self._threads.append(thread)
            thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []

    def _loop(self, booking_id: int) -> None:
        try:
            while not self._stop.is_set():
                try:
                    event = self.slot.client.watch_claim(booking_id)
                except NotFound:
                    log.warning("this server has no /events route for booking %d; "
                                "running unwatched", booking_id)
                    return
                except Conflict:
                    return                      # cancelled or over: nothing to watch
                except Exception as exc:        # a blip, a restart, a timeout
                    log.warning("watch on booking %d failed (%s); retrying",
                                booking_id, exc.__class__.__name__)
                    if self._stop.wait(5.0):
                        return
                    continue
                if event is None:
                    continue                    # 204: the hold expired, reconnect
                self.slot._record_taken(event)
                if self.on_taken is not None:
                    try:
                        self.on_taken(event)
                    except Exception:
                        log.exception("on_taken callback raised")
                return
        finally:
            with self._lock:
                self._live -= 1
```

and on `Slot`:

```python
    taken: Optional[Dict[str, Any]] = None      # set by a Watcher; read by check()

    def _record_taken(self, event: Dict[str, Any]) -> None:
        self.taken = event

    def watch(self, on_taken: Optional[Callable[[Dict[str, Any]], None]] = None
              ) -> "Watcher":
        """Park a long-poll per row, so the server can say in milliseconds that an
        important booking took the card.

            with slot.watch():
                for epoch in range(100):
                    slot.check()        # raises Preempted once it is taken
                    train_one_epoch()
        """
        return Watcher(self, on_taken)

    def check(self) -> None:
        """Raise Preempted if the card has been taken. Call it where stopping is
        safe -- between epochs, after a checkpoint."""
        if self.taken is not None:
            by = ", ".join(self.taken.get("by") or []) or "someone"
            raise Preempted("GPU %d was taken by %s" % (self.gpu, by), self.taken)
```

`taken` is declared on the class, so set it in `__init__` as `self.taken = None` too —
`Slot` is a plain class, not a frozen dataclass, so both are fine; the class-level line is
documentation and the instance line is the state.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_watch.py -v`
Expected: PASS, 8 tests. Then `python -m pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/hrm/slots.py src/hrm/client.py src/hrm/errors.py tests/test_watch.py
git commit -m "feat(slots): watch() parks a long-poll; check() raises Preempted"
```

---

## Task 8: the whole pattern in six lines

**Files:**
- Modify: `src/hrm/slots.py` (`Slot.__enter__`/`__exit__`), `src/hrm/__init__.py`, `README.md`
- Test: `tests/test_slots.py`, `tests/test_module_api.py`

**Interfaces:**
- Produces: `Slot` as a context manager — entering waits for the start and parks the
  watcher, leaving stops the watcher and releases the rows. This is the shape the spec's
  "What a job looks like" section describes:

```python
with hrm.reserve(memory=19.5, hours=1, notice_seconds=0) as slot:
    for epoch in range(epochs):
        slot.check()
        train_one_epoch()
        save_checkpoint()
```

- [ ] **Step 1: Write the failing tests**

```python
# appended to tests/test_slots.py
def test_the_context_manager_waits_parks_and_releases(client, fake):
    fake.event_hold_s = 0.05
    with client.reserve(memory=1, hours=4) as slot:
        assert slot.start <= client.now()          # it waited
        assert slot.watcher is not None and slot.watcher.watching is True
        booking_id = slot.id
    assert slot.watcher.watching is False          # and stopped watching
    found = [b for b in client.bookings(include_cancelled=True) if b.id == booking_id]
    assert found and (found[0].cancelled_at is not None
                      or found[0].end <= client.now())


def test_the_context_manager_releases_even_when_the_body_raises(client, fake):
    fake.event_hold_s = 0.05
    with pytest.raises(ZeroDivisionError):
        with client.reserve(memory=1, hours=4) as slot:
            booking_id = slot.id
            1 / 0
    found = [b for b in client.bookings(include_cancelled=True) if b.id == booking_id]
    assert found and (found[0].cancelled_at is not None
                      or found[0].end <= client.now())


def test_preempted_from_check_still_releases_the_slot(client, fake):
    slot = client.reserve(memory=1, hours=4, notice_seconds=0)
    fake.queue_event(slot.id, {"event": "taken", "effective_mib": 0, "lost_mib": 1024,
                               "by": ["dorian.zwanzig"], "truncated": True,
                               "cancelled": False, "deadline": None})
    with pytest.raises(hrm.Preempted):
        with slot:
            for _ in range(200):
                slot.check()
                time.sleep(0.02)
    slot.refresh()
    assert slot.booking.cancelled_at is not None or slot.end <= client.now()
```

```python
# appended to tests/test_module_api.py
def test_the_new_names_are_exported():
    import hrm
    for name in ("reserve", "capacity", "Slot", "Capacity", "Preempted"):
        assert name in hrm.__all__
        assert hasattr(hrm, name)
```

`import time` at the top of `tests/test_slots.py` for the third test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slots.py -k context_manager tests/test_module_api.py -k new_names -v`
Expected: FAIL — `TypeError: 'Slot' object does not support the context manager protocol`

- [ ] **Step 3: Write the implementation**

In `slots.py`, on `Slot` (and `self.watcher: Optional[Watcher] = None` in `__init__`):

```python
    def __enter__(self) -> "Slot":
        """Wait for the window, then park the watcher. The whole job pattern is
        `with hrm.reserve(...) as slot:` plus `slot.check()` where stopping is safe."""
        self.wait_until_start()
        self.watcher = Watcher(self).__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        try:
            if self.watcher is not None:
                self.watcher.stop()
        finally:
            self.release()
        return False
```

`Watcher` is defined above `Slot` in the file, or referenced lazily — put `Watcher` first,
since `Slot` is the one that uses it.

In `__init__.py`, extend the imports and `__all__`:

```python
from .errors import (Conflict, Forbidden, HRMError, InvalidRequest, NoCapacity,  # noqa: E402
                     NotFound, Preempted, RateLimited, ServerUnavailable, UnknownUser)
from .slots import Capacity, Slot, Watcher  # noqa: E402
```

with `"reserve", "capacity", "Slot", "Capacity", "Watcher", "Preempted"` added to `__all__`.

In `README.md`, a new section after the quickstart link:

```markdown
## Two ways to hold a GPU

**A job that wants the next free slot** — `reserve` never fails for lack of capacity. The
server assigns the earliest window that fits and moves it earlier if a card frees up:

```python
import hrm
hrm.connect(user="cwinkelmann")

with hrm.reserve(memory=19.5, hours=1, notice_seconds=0) as slot:
    for epoch in range(epochs):
        slot.check()            # raises Preempted if someone took the card
        train_one_epoch()
        save_checkpoint()
```

Entering waits for the slot's start, parks a long-poll so the server can say in
milliseconds that an important booking took the card, and releases the booking on the way
out — including when the body raises. `notice_seconds=0` promises you will be off the card
at once; the server only honours that promise while the watcher is actually connected, and
it can never buy you more notice than the box's `grace_minutes`.

**A session that wants a window** — `gate` books a lease and renews it while you work. Use
it for an interactive session or a job that must not be interrupted; use `reserve` for a
queue of jobs that should take their turn.

`hrm.capacity(memory=19.5, hours=1)` answers what is free now and when that shape next
fits. Ask it rather than summing bookings yourself.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest -q`
Expected: PASS, everything.

- [ ] **Step 5: Commit**

```bash
git add src/hrm/slots.py src/hrm/__init__.py README.md tests/test_slots.py tests/test_module_api.py
git commit -m "feat(slots): a slot is a context manager -- wait, watch, run, release"
```

---

## Task 9: `examples/train_gpu.py` — the runnable script

**Files:**
- Create: `examples/train_gpu.py`
- Test: `tests/test_examples.py`

**Interfaces:**
- Consumes: `hrm.reserve`, `hrm.capacity`, `Slot.check`, `Preempted` (Tasks 3–8).
- Produces: a script runnable as
  `python examples/train_gpu.py --memory 19.5 --hours 1 --out runs/demo`.

**What it must demonstrate**, because these are the things a real job gets wrong:

1. It asks `hrm.capacity()` first and prints what it found, so a user who picked an
   impossible size learns that before waiting.
2. It reserves with `notice_seconds=0` and `auto_advance=True`, prints whether the slot
   starts now or later, and waits.
3. It trains for real: a transformer sized from `--memory`, AMP, synthetic data, so the card
   is genuinely busy for hours.
4. It checkpoints every epoch and resumes from the last checkpoint, which is what makes
   preemption survivable.
5. It calls `slot.check()` once per epoch — the only place stopping is safe — and exits 0 on
   `Preempted` after writing a checkpoint, because being preempted is a normal outcome, not
   a failure.
6. The `with` block releases the booking on every path.

The script must not import torch at module scope: `tests/test_examples.py` imports it to
check its shape, and the test machine has no CUDA.

- [ ] **Step 1: Write the failing test**

```python
"""The examples are part of the contract: they must stay importable and honest."""
from __future__ import annotations

import importlib.util
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_gpu_imports_without_torch_installed():
    """Importing must not need CUDA or torch: the import happens inside main()."""
    module = _load("train_gpu")
    assert callable(module.main)
    src = (EXAMPLES / "train_gpu.py").read_text()
    head = src.split("def ", 1)[0]
    assert "import torch" not in head


def test_train_gpu_parses_its_arguments():
    module = _load("train_gpu")
    args = module.build_parser().parse_args(["--memory", "19.5", "--hours", "2"])
    assert args.memory == 19.5
    assert args.hours == 2.0
    assert args.out


def test_train_gpu_uses_the_assigned_slot_contract():
    src = (EXAMPLES / "train_gpu.py").read_text()
    for needed in ("hrm.reserve(", "slot.check()", "notice_seconds", "auto_advance",
                   "Preempted", "hrm.capacity("):
        assert needed in src, needed
    assert "hrm.book(" not in src        # the old day-based verb is not this path


def test_train_gpu_checkpoints_and_resumes():
    src = (EXAMPLES / "train_gpu.py").read_text()
    assert "torch.save" in src and "torch.load" in src
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_examples.py -v`
Expected: FAIL — `FileNotFoundError: .../examples/train_gpu.py`

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python3
"""Long-running GPU training that takes its turn on the shared box.

Reserve a slot, wait for it, train, checkpoint every epoch, and give the card
back. The booking never fails for lack of capacity: if the box is full you are
assigned a later window and this script waits for it, so you can start a hundred
of these and let the server decide the order.

    python examples/train_gpu.py --memory 19.5 --hours 1 --out runs/demo

Being preempted is a normal outcome, not a failure. The script books `lendable`
with `notice_seconds=0`, which says "an important booking may have this card at
once"; when that happens it writes a checkpoint and exits 0, and the same command
resumes from where it stopped.

Needs PyTorch (`pip install torch`) and a CUDA GPU; everything else is the hrm SDK.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import hrm

# A transformer block's parameters and activations scale with these; the defaults
# are sized so one block plus a batch fills roughly a GiB, and --memory picks how
# many blocks to stack.
D_MODEL = 1024
N_HEADS = 16
SEQ_LEN = 512
BATCH = 16
GIB_PER_LAYER = 0.55          # measured on an A6000 at these sizes, AMP on


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--memory", type=float, default=None,
                   help="GiB of VRAM to book (default: most of the card the server offers)")
    p.add_argument("--hours", type=float, default=1.0, help="how long to book for")
    p.add_argument("--gpu", type=int, default=None, help="pin to one GPU (default: any)")
    p.add_argument("--epochs", type=int, default=1000, help="epochs to run at most")
    p.add_argument("--steps", type=int, default=200, help="steps per epoch")
    p.add_argument("--out", type=Path, default=Path("runs/train_gpu"),
                   help="checkpoint directory")
    p.add_argument("--user", default=None, help="your account (default: HRM_USER or login)")
    p.add_argument("--url", default=None, help="booking server (default: HRM_URL)")
    p.add_argument("--important", action="store_true",
                   help="book as important: nobody can take this card from you")
    p.add_argument("--not-later-than", default=None,
                   help="give up if the earliest slot is after this ISO time")
    return p


def pick_memory(asked, gpu):
    """What to book: what was asked, or most of the biggest card the server has."""
    free = hrm.capacity(memory=1, hours=1, gpu=gpu)
    if asked is not None:
        if free.earliest is None:
            print("note: %.1f GiB does not fit anywhere in the next fortnight" % asked)
        return asked
    whole = max(free.free_gib, 1.0)
    memory = max(round(whole * 0.9, 1), 1.0)
    print("booking %.1f GiB of the %.1f GiB this box has free" % (memory, whole))
    return memory


def layers_for(memory_gib: float) -> int:
    """Stack enough blocks to use the booking. Honest about the arithmetic: this is
    a rule of thumb, and the point is to be busy, not to hit the number exactly."""
    return max(int(memory_gib / GIB_PER_LAYER), 2)


def make_scaler(torch):
    """AMP's scaler moved namespace in torch 2.4; accept both so the example runs
    on whatever is installed on the box."""
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def autocast(torch):
    try:
        return torch.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def build_model(torch, memory_gib: float):
    nn = torch.nn
    layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS,
                                       dim_feedforward=4 * D_MODEL,
                                       batch_first=True)
    model = nn.TransformerEncoder(layer, num_layers=layers_for(memory_gib))
    return nn.Sequential(model, nn.Linear(D_MODEL, D_MODEL))


def load_checkpoint(torch, path: Path, model, optimiser):
    """Resume if there is something to resume from; return the next epoch."""
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cuda")
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optimiser"])
    print("resuming from %s at epoch %d" % (path, state["epoch"] + 1))
    return state["epoch"] + 1


def save_checkpoint(torch, path: Path, model, optimiser, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                "epoch": epoch}, tmp)
    tmp.replace(path)          # atomic: a checkpoint is never half-written


def train_one_epoch(torch, model, optimiser, scaler, steps: int) -> float:
    model.train()
    total = 0.0
    for _ in range(steps):
        batch = torch.randn(BATCH, SEQ_LEN, D_MODEL, device="cuda")
        target = torch.randn(BATCH, SEQ_LEN, D_MODEL, device="cuda")
        optimiser.zero_grad(set_to_none=True)
        with autocast(torch):
            loss = torch.nn.functional.mse_loss(model(batch), target)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total += float(loss.detach())
    return total / max(steps, 1)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError:
        print("this example needs PyTorch: pip install torch", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("no CUDA GPU visible; run this on the box you booked", file=sys.stderr)
        return 1

    hrm.connect(user=args.user, url=args.url)
    memory = pick_memory(args.memory, args.gpu)

    slot = hrm.reserve(memory=memory, hours=args.hours, gpu=args.gpu,
                       important=args.important,
                       notice_seconds=None if args.important else 0,
                       not_later_than=args.not_later_than,
                       note="train_gpu.py")
    if slot.assigned_later:
        print("the box is busy: assigned %s on GPU %d — waiting (it may move earlier)"
              % (slot.start.astimezone().strftime("%a %d %b %H:%M"), slot.gpu))
    else:
        print("starting now on GPU %d" % slot.gpu)

    checkpoint = args.out / "checkpoint.pt"
    try:
        with slot:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
            print("GPU %d is mine until %s"
                  % (slot.gpu, slot.end.astimezone().strftime("%H:%M")))
            model = build_model(torch, memory).cuda()
            optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = make_scaler(torch)
            first = load_checkpoint(torch, checkpoint, model, optimiser)
            held = torch.cuda.memory_reserved() / (1024 ** 3)
            print("%d transformer blocks, %.1f GiB reserved on the card"
                  % (layers_for(memory), held))

            for epoch in range(first, args.epochs):
                slot.check()        # the one safe place to stop: before the next epoch
                began = time.monotonic()
                loss = train_one_epoch(torch, model, optimiser, scaler, args.steps)
                save_checkpoint(torch, checkpoint, model, optimiser, epoch)
                print("epoch %d  loss %.4f  %.1fs" % (epoch, loss,
                                                      time.monotonic() - began),
                      flush=True)
                if slot.end <= slot.client.now():
                    print("the booked window is over; stopping at epoch %d" % epoch)
                    break
    except hrm.Preempted as taken:
        print("preempted: %s" % taken)
        print("checkpoint is at %s; run the same command to carry on" % checkpoint)
        return 0
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_examples.py -v`
Expected: PASS, 4 tests. Then `python -m pytest -q`.

Then check it is a valid program and its help works, which the import test cannot show:

```bash
python -c "import ast; ast.parse(open('examples/train_gpu.py').read()); print('syntax ok')"
python examples/train_gpu.py --help
```

Expected: the usage text, with `--memory`, `--hours` and `--out` described. On a machine
with no GPU, `python examples/train_gpu.py --memory 1 --hours 1` must print
`no CUDA GPU visible` and exit 1 rather than raising.

- [ ] **Step 5: Commit**

```bash
git add examples/train_gpu.py tests/test_examples.py
git commit -m "docs(examples): train_gpu.py -- reserve, wait, train, checkpoint, release"
```

---

## Task 10: the notebook example and the examples index

**Files:**
- Create: `examples/notebook_session.ipynb`, `examples/README.md`
- Modify: `README.md` (the examples link), `tests/test_examples.py`
- Do **not** modify: `examples/quickstart.ipynb` (it has uncommitted edits from another session)

**Interfaces:**
- Consumes: `hrm.reserve`, `hrm.capacity`, `Slot.release`, `Slot.watch` (Tasks 3–8).
- Produces: a notebook whose cells run top to bottom against a real server.

**Why a second notebook rather than more cells in the quickstart.** The quickstart shows the
calendar verbs: usage, bookings, book, cancel. The interactive case is a different story: a
notebook's cells span wall-clock time, so a `with` block is the wrong shape — you acquire in
one cell, work in several, and release in the last. The realistic failure is forgetting the
last cell, so the notebook says what happens then.

- [ ] **Step 1: Write the failing test**

```python
# appended to tests/test_examples.py
import json


def _notebook(name):
    return json.loads((EXAMPLES / (name + ".ipynb")).read_text())


def test_the_notebook_session_example_is_valid_and_uses_the_new_verbs():
    nb = _notebook("notebook_session")
    assert nb["nbformat"] >= 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    for needed in ("hrm.connect(", "hrm.capacity(", "hrm.reserve(", "slot.release()"):
        assert needed in code, needed
    assert "import hrm" in code


def test_the_notebook_ships_without_stored_output():
    nb = _notebook("notebook_session")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def test_every_example_is_listed_in_the_examples_readme():
    listing = (EXAMPLES / "README.md").read_text()
    for path in sorted(EXAMPLES.iterdir()):
        if path.suffix in (".py", ".ipynb"):
            assert path.name in listing, path.name
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_examples.py -k notebook -v`
Expected: FAIL — `FileNotFoundError: .../examples/notebook_session.ipynb`

- [ ] **Step 3: Write the notebook and the index**

Write `examples/notebook_session.ipynb` with `nbformat: 4`, `nbformat_minor: 5`, a
`python3` kernelspec, every code cell carrying `"outputs": []` and
`"execution_count": null`, and these cells in order:

1. **markdown** — "# Hold a GPU while you work in this notebook", two sentences on the
   difference from `quickstart.ipynb`: that one books calendar days, this one takes a slot
   for the length of a session and gives it back.
2. **code** — `import hrm` / `hrm.connect(user="cwinkelmann")` / `print("hrm", hrm.__version__)`
3. **markdown** — "## What is free right now" — `capacity` is the one authoritative answer.
4. **code** — `cap = hrm.capacity(memory=8, hours=3)` then print `cap.free_gib`, `cap.gpu`
   and `cap.earliest`, with a line saying that `earliest` in the future means you will be
   queued rather than refused.
5. **markdown** — "## Take a slot" — explain `notice_seconds` and why an interactive session
   usually wants `important=True` instead: you are sitting at the keyboard and cannot
   checkpoint on command.
6. **code** —
   ```python
   slot = hrm.reserve(memory=8, hours=3, important=True, note="notebook session")
   print(slot)
   print("starts later:", slot.assigned_later)
   slot.wait_until_start()
   print("GPU", slot.gpu, "is yours until", slot.end.astimezone().strftime("%H:%M"))
   ```
7. **markdown** — "## Work" — the cell below is where your own code goes; the booking holds
   while you think.
8. **code** — a small torch cell guarded by a try/except ImportError that allocates a tensor
   on `slot.gpu` and prints the memory in use, so the booking corresponds to something real:
   ```python
   try:
       import torch
       x = torch.randn(8192, 8192, device="cuda:%d" % slot.gpu)
       print("%.1f GiB on GPU %d" % (torch.cuda.memory_allocated(slot.gpu) / 1024**3,
                                     slot.gpu))
   except ImportError:
       print("no torch here — this cell is where your own GPU work goes")
   ```
9. **markdown** — "## See it on the calendar" — your booking among everyone else's.
10. **code** — `df = hrm.bookings().to_frame()` / `df[df["id"].isin([b.id for b in slot.bookings])]`
11. **markdown** — "## Give it back" — the last cell, and the honest warning: if you forget
    it, the booking stands until its end and the card looks taken to everyone else; the
    monitor will notice the card is idle and say so in `#gpu-watch`.
12. **code** — `slot.release()` then `print("released:", slot.refresh().booking)`

Write `examples/README.md`:

```markdown
# Examples

| file | what it shows |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | the calendar verbs: usage, bookings, book whole days, cancel |
| [`notebook_session.ipynb`](notebook_session.ipynb) | hold a GPU while you work in a notebook, then give it back |
| [`train_gpu.py`](train_gpu.py) | a long training run that takes its turn: reserve, wait, train, checkpoint, release |

`train_gpu.py` is the one to copy for batch work. It books `lendable` with
`notice_seconds=0`, so a colleague who needs the card gets it at once and the job resumes
from its last checkpoint later — which is how 13,000 one-hour jobs share a box without
anyone having to queue by hand.

Both notebooks and the script read `HRM_URL` and `HRM_USER` from the environment, so the
same code runs against carrot, olive or the ThinkPad:

```bash
export HRM_URL=http://10.188.1.1:8765
export HRM_USER=cwinkelmann
```
```

In the top-level `README.md`, change the single quickstart link to point at
`examples/README.md` as well, leaving the surrounding sentence as it is.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_examples.py -v`
Expected: PASS, 7 tests.

Then confirm the notebook is machine-valid, since a hand-written `.ipynb` is easy to get
subtly wrong:

```bash
python -c "
import json; nb = json.load(open('examples/notebook_session.ipynb'))
assert nb['nbformat'] == 4 and nb['cells']
print(len(nb['cells']), 'cells, valid JSON')"
```

Then `python -m pytest -q` for the whole suite.

- [ ] **Step 5: Commit**

```bash
git add examples/notebook_session.ipynb examples/README.md README.md tests/test_examples.py
git commit -m "docs(examples): a notebook session, and an index of the examples"
```

---

## After the last task

- `python -m pytest -q` — the pre-existing suite plus roughly 60 new tests.
- The two examples are the SDK's half of `deploy/acceptance-slots.py` in the
  ResourceMonitor repo: that script is the protocol contract, these are what a person
  writes against it.
- `gate`/`Lease` are untouched and still right for a session that must not be interrupted.
  The README now says which to reach for.
