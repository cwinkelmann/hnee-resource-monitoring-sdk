# hrm

Python SDK for the HNEE GPU booking server: see what the shared GPU box is
doing, list bookings, and book or cancel GPU memory from Python or a notebook.
Pure Python (3.9+), no required dependencies (only `tzdata` on Windows); pandas is optional.

```python
import hrm

hrm.connect(user="alice")                    # your carrot account
hrm.usage().to_frame()                       # what every GPU is doing right now
result = hrm.book(days=["2026-10-06", "2026-10-07"], memory=48)
print(result)                                # booked GPU 4, 48 GiB, Tue 06 Oct 15:12 → Thu 08 Oct 00:00 (Europe/Berlin), id 12
hrm.cancel(result.bookings[0])
```

A full walk-through with real (anonymised) output is in
[`examples/quickstart.ipynb`](examples/quickstart.ipynb), with the rest listed in
[`examples/README.md`](examples/README.md).

## Install via pip directly from GitHub

```bash
pip install git+https://github.com/cwinkelmann/hnee-resource-monitoring-sdk
```

With pandas, for the `.to_frame()` helpers:

```bash
pip install "hnee-resource-monitoring[pandas] @ git+https://github.com/cwinkelmann/hnee-resource-monitoring-sdk"
```

On Windows, pip also pulls in `tzdata` (the IANA time-zone data) automatically.

In a notebook, use `%pip install ...` with the same argument. The package is
called `hnee-resource-monitoring`; you import it as `hrm`.

### Configure

Tell the SDK who you are once, either in code or in the environment:

```python
import hrm
hrm.connect(user="alice")                          # your carrot account name
hrm.connect(user="alice", url="http://host:8765")  # another server
```

```bash
export HRM_USER=alice
export HRM_URL=http://host:8765    # optional; defaults to the server on carrot
```

Without either, the SDK uses your local login name if it is a carrot account,
and raises `hrm.UnknownUser` otherwise. `hrm.users()` lists the accounts.
Reading (usage, bookings) never needs a user.

## List bookings

```python
bookings = hrm.bookings()            # active bookings, 7 days back to 14 days ahead
for b in bookings:
    print(b)                         # #12 GPU 4 · 48.0 GiB · alice · Tue 06 Oct 15:12 → Thu 08 Oct 00:00

hrm.bookings().to_frame()            # as a pandas DataFrame (needs the [pandas] extra)
hrm.bookings(days_ahead=3, days_back=1, include_cancelled=True)
```

Each item is an `hrm.Booking` with `id`, `user`, `gpu`, `vram_gib`, `start`,
`end` (timezone-aware datetimes), `note`, `kind` and `cancelled`.
`days_ahead` goes up to 14 and `days_back` up to 30.

What the GPUs are doing right now (memory in use, free memory after bookings,
utilisation, power, processes, active bookings):

```python
usage = hrm.usage()
usage.to_frame()
for gpu in usage.gpus:
    print(gpu.index, gpu.free_mib, [p.user for p in gpu.procs])
```

## Create a booking

```python
result = hrm.book(days=["2026-10-06", "2026-10-07"], memory=48)
print(result)       # booked GPU 4, 48 GiB, Tue 06 Oct 15:12 → Thu 08 Oct 00:00 (Europe/Berlin), id 12
result.gpu          # 4
result.ids          # [12]
```

- **`days`** are calendar days in Europe/Berlin (carrot's local time), as
  `"YYYY-MM-DD"` strings or `datetime.date` objects. Each day runs from 00:00
  to 24:00; if today is in the list, the booking starts now. Consecutive days
  become one booking; a gap starts a new one, so
  `days=["2026-10-06", "2026-10-07", "2026-10-09"]` creates two bookings on
  the same GPU. A day in the past raises `ValueError`.
- **`memory`** is GPU memory in GiB (`48`, `1.5` …, at least 1). Leave it out
  to book the whole card.
- **GPU auto-pick:** the SDK looks at the cards and the existing bookings and
  takes the lowest-numbered GPU that has `memory` GiB free for every day you
  asked for. If none has, it raises `hrm.NoCapacity` naming the best
  candidate. If someone else books the GPU in the meantime (HTTP 409), it
  tries the next candidate (three attempts in total). Choose a GPU yourself
  with `gpu=3`.
- Other options: `note="training run"`, or an exact time range instead of
  days: `hrm.book(start="2026-10-06 14:00", end="2026-10-06 18:00", memory=20)`
  (naive times are Europe/Berlin).

If one of several bookings fails, the ones already made are cancelled again.
If the server's answer is lost (a timeout, a dropped connection, or you
interrupt the cell), the SDK looks the booking up and cancels it too, then
re-raises the original error. Anything it could not cancel is named in a
`RuntimeWarning`, and in that case it does not try another GPU; check
`hrm.bookings()` and cancel what the warning names.

### Quick hold

To grab a whole card right now, until the next 09:00 (Europe/Berlin):

```python
hrm.hold(3)
```

## Remove a booking

```python
hrm.cancel(12)                       # by id
hrm.cancel(result.bookings[0])       # or pass the Booking
for b in result.bookings:            # everything one book() call created
    hrm.cancel(b)
```

Cancelling is idempotent: cancelling an already cancelled booking is fine.
`hrm.cancel` returns the booking with `cancelled` set to `True`.

A quick hold made with `hrm.hold(gpu)` ends with:

```python
hrm.release(3)                       # ends the active quick holds on GPU 3
```

## Gate a long-running job

A job that needs a GPU for hours or days does not need a scheduler. Let it wait
at the gate, hold a short lease, and renew that lease while it runs:

```python
with hrm.gate(memory=73, note="convnext training") as lease:
    print("got GPU", lease.gpu, "until", lease.deadline)
    while not lease.stop_requested:
        train_one_epoch()
        save_checkpoint()
```

`hrm.gate()` polls (reads only, so it costs none of the write budget) until one
GPU has `memory` GiB free for the whole lease window, books it, and renews the
booking shortly before it ends. On the way out — normally, through an exception
or on Ctrl-C — it cancels every booking it made, and only those.

| argument | default | meaning |
|---|---|---|
| `memory` | the whole card | GiB to book |
| `gpus` | any | restrict the search; the lowest fitting index wins |
| `lease_hours` | 6 | length of one lease, renewed while the job runs |
| `renew_before` | 1 | renew this many hours before the lease ends |
| `poll` | 60 | seconds between looks for a free GPU |
| `wait_timeout` | none | give up at the gate after this many seconds |
| `note`, `user` | — | as for `hrm.book()` |

Three things are worth knowing:

- **Your own bookings count.** If you already hold memory on a card, only what
  is missing is booked (and the window is split where your own booking ends), so
  a card you already hold part of is preferred over taking a second one. If your
  own booking already covers the need, nothing is booked at all.
- **A refused renewal asks, it does not kill.** When someone else has booked the
  next window, `lease.stop_requested` becomes `True` and a warning is logged; the
  current booking still runs to its end and your job is never signalled. Check
  the flag between iterations and checkpoint. `lease.wait_stop(timeout)` blocks
  until it is set.
- **Short leases limit the damage.** If the process is killed outright, nothing
  cancels the booking — it lingers until its end, which is why the default lease
  is six hours and renewed rather than booked for the whole run up front.

### Several slots on one card, and jobs of known length

Running a queue through N slots per GPU under one account needs two things.

First, `reuse_own=False`. By default the gate counts your own overlapping
bookings towards the need, which is right for a single job and wrong for a
slot: the second worker would see the first one's booking as "already yours"
and book nothing, so both would use the same memory. The gate warns when a
need is met this way.

```python
with hrm.gate(memory=19.5, gpus=[4], reuse_own=False, note="queue slot") as lease:
    ...
```

Slots must also fit the card: an H100 is 79.65 GiB, so four 20 GiB bookings are
refused (80 > 79.65) and 19.5 GiB is the practical quarter.

Second, `has_room_for()`. A worker must not claim a job the lease cannot
cover — after a refused renewal the lease may have less than an hour left:

```python
with hrm.gate(memory=19.5, gpus=[4], reuse_own=False, renew_before=2) as lease:
    while not lease.stop_requested and lease.has_room_for(1.0):
        run_one_job()          # an hour long; the lease outlasts it
```

Keep `renew_before` longer than one job (the default is 1 hour, so raise it for
hour-long jobs), or a job claimed just before a refused renewal outlives the
booking.

For a job that is not Python, the same thing from the shell:

```bash
python -m hrm --memory 73 --gpus 4,5 --hours 6 --note "training" \
    --stop-file /tmp/stop-training -- ./train.sh
```

It exits with the command's exit code, with 75 if it gave up at the gate, and
with 130 on Ctrl-C or `kill` (releasing the booking either way). The job is
never signalled; `--stop-file` is created when the lease cannot be renewed, for
the script to notice (`[ -e /tmp/stop-training ] && break`).

## Take the next free slot

`hrm.gate()` above holds a window and renews it. The other shape is a job that
wants its *turn*: `hrm.reserve()` never fails for lack of capacity — the server
assigns the earliest window that fits and moves it earlier if a card frees up.

```python
with hrm.reserve(memory=19.5, hours=1, notice_seconds=0) as slot:
    for epoch in range(epochs):
        slot.check()            # raises hrm.Preempted if someone took the card
        train_one_epoch()
        save_checkpoint()
```

Entering the block waits for the slot's start, then parks a long-poll
(`GET /api/claims/<id>/events`) so the server can say in milliseconds that an
important booking has taken the card. Leaving it stops the watcher and releases
the booking — on every path, including an exception.

| argument | default | meaning |
|---|---|---|
| `memory` | the whole card | GiB to reserve |
| `hours` | 1 | how long the window is |
| `gpu` | any | pin to one card; otherwise the server picks |
| `count` | 1 | reserve this many cards atomically: one group, one shared start |
| `auto_advance` | `True` | move the slot earlier when capacity frees up |
| `not_later_than` | none | 409 instead of a slot after this time |
| `notice_seconds` | none | warning you need before losing the card; 0 means "at once" |
| `important` | `False` | book as important: nobody can take this card from you |

Three things are worth knowing:

- **`notice_seconds` is a promise, and the server only honours it while you are
  listening.** It is read from the parked long-poll, so a job that has crashed
  gets the box's full `grace_minutes` instead — a stale promise cannot hand a
  card away. It can never buy you *more* notice than that grace either: a
  declared notice may only shorten the wait.
- **`slot.check()` is where you stop, not where you are stopped.** The watcher
  thread sets a flag; `check()` raises `hrm.Preempted` the next time you ask.
  Call it between epochs, after a checkpoint — the middle of a CUDA kernel is
  not a safe place to stop, and nothing here signals your process.
- **Being preempted is a normal outcome.** A job that checkpoints resumes from
  where it stopped; that is how a long queue of jobs shares a box with people
  who need a card now.

`hrm.capacity(memory=19.5, hours=1)` answers what is free now and when that
shape next fits. Ask it rather than summing bookings yourself — the server owns
that arithmetic, and clients that reimplemented it got it wrong.

Which to reach for: `gate` for a session or a job that must not be interrupted,
`reserve` for a queue of jobs that should take their turn.

## Errors

Every error is a subclass of `hrm.HRMError` and has `.status` (the HTTP status,
or `None`) and `.detail` (the server's explanation).

| exception | when |
|---|---|
| `hrm.InvalidRequest` | HTTP 400, 413, 415: the server rejected the request (bad times, too long, too much memory …) |
| `hrm.Forbidden` | HTTP 403 |
| `hrm.NotFound` | HTTP 404, e.g. cancelling an id that does not exist |
| `hrm.Conflict` | HTTP 409: the GPU does not have that much memory free for that time |
| `hrm.RateLimited` | HTTP 429: too many writes, see the limits below |
| `hrm.ServerUnavailable` | HTTP 503 (and other 5xx), server unreachable, timeout |
| `hrm.NoCapacity` | raised by the SDK: no GPU has enough free memory for your days |
| `hrm.UnknownUser` | raised by the SDK: no user configured and your login is not a carrot account |

```python
try:
    hrm.book(days=["2026-10-06"], memory=80)
except hrm.NoCapacity as e:
    print("no room:", e.detail)
```

## Honour system

There is no login. Anyone on the LAN can book or cancel in any name, so
please book only for yourself and cancel only your own bookings. Bookings are
reported (on the dashboard and in Slack alerts) but never enforced: nothing
stops a process from using a GPU someone else has booked.

## Server limits

- A booking lasts at most 14 days.
- A booking can't start in the past (5 minutes of slack).
- Memory is from 1 GiB up to the card's total.
- 3600 writes (book, reserve, cancel, hold, release) per hour per client IP; more raise
  `hrm.RateLimited`. The budget is shared by everyone behind the same IP, so it
  is a backstop against a runaway loop rather than a per-user quota.

The server is the authority: the SDK's planning only picks a candidate GPU,
and the server's answer always wins.

## Development

```bash
pip install -e ".[test,pandas]"
pytest -q
HRM_INTEGRATION=1 pytest tests/test_integration.py   # read-only checks against the live server
```

## See also

The booking server and dashboard:
[cwinkelmann/hnee-resource-monitoring](https://github.com/cwinkelmann/hnee-resource-monitoring).

## License

MIT, see [LICENSE](LICENSE).
