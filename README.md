# hrm

Python SDK for the HNEE GPU booking server: see what the shared GPU box is
doing, list bookings, and book or cancel GPU memory from Python or a notebook.
Pure Python (3.9+), no required dependencies; pandas is optional.

```python
import hrm

hrm.connect(user="alice")                    # your carrot account
hrm.usage().to_frame()                       # what every GPU is doing right now
result = hrm.book(days=["2026-10-06", "2026-10-07"], memory=48)
print(result)                                # booked GPU 4, 48 GiB, Tue 06 Oct 15:12 → Thu 08 Oct 00:00 (Europe/Berlin), id 12
hrm.cancel(result.bookings[0])
```

A full walk-through with real (anonymised) output is in
[`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## Install via pip directly from GitHub

```bash
pip install git+https://github.com/cwinkelmann/hnee-resource-monitoring-sdk
```

With pandas, for the `.to_frame()` helpers:

```bash
pip install "hnee-resource-monitoring[pandas] @ git+https://github.com/cwinkelmann/hnee-resource-monitoring-sdk"
```

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

If one of several bookings fails, the ones already made are cancelled again,
so you never end up with half a booking.

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
- 30 writes (book, cancel, hold, release) per hour per client IP; more raise
  `hrm.RateLimited`.

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
