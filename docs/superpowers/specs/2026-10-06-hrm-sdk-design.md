# hrm — Python SDK for the HNEE GPU booking server: design

**Date:** 2026-10-06. **Status:** approved in chat, section by section. The user asked for the code to be created in a local folder; nothing is pushed until they say so.
**Server:** the ResourceMonitor dashboard on carrot (`http://10.188.1.1:8765`), repo `cwinkelmann/hnee-resource-monitoring` (main). The SDK changes nothing on the server.

## Intent
- **What the user said:** build a booking SDK in `cwinkelmann/hnee-resource-monitoring-sdk`, with an example notebook that installs it, retrieves current usage and bookings, and places a booking simply: `hrm.book(days=["2026-10-06", "2026-10-07"], memory=48)` gives two days of 48 GB.
- **Decisions the user made:**
  - **GPU:** auto-picked, with optional `gpu=`.
  - **User:** set once via `hrm.connect(user=…)` / `HRM_USER`. It falls back to the local login only if that is a known carrot account.
  - **Implementation:** pure-Python, no required dependencies, with an optional `[pandas]` extra (option 1).
  - **Notebook and CI:** the notebook ships with live outputs, from a short booking that is cancelled again. A GitHub Actions workflow is included.
- **Success:** in a fresh notebook, `pip install git+https://github.com/cwinkelmann/hnee-resource-monitoring-sdk`, then `import hrm`. `usage()`, `bookings()`, `book(...)` and `cancel(...)` work against carrot.

## Constraints
- Python ≥ 3.9 (`zoneinfo`). **No runtime dependencies.** pandas is an optional extra, used only by `.to_frame()`.
- The default URL is `http://10.188.1.1:8765`, overridable with `HRM_URL` or `connect(url=…)`. **The repo is public: no credentials, no real user names or IPs in fixtures, docs or tests.** The fixtures are anonymised (alice, bob, carol, user04…, 192.0.2.10).
- Talk to the server exactly as it expects:
  - JSON bodies with `Content-Type: application/json`;
  - no `Origin` header;
  - a 10 s timeout;
  - `User-Agent: hrm-sdk/<version>`.
- The server is the authority. Client-side planning only picks a candidate GPU; a 409 from the server always wins.
- Honour system: there's no authentication. The README states this, and that bookings are reported but never enforced.
- Times: the server stores UTC ISO. The SDK's day semantics use **Europe/Berlin** (carrot's local time). Python objects carry aware datetimes.

## Server API (the contract; recorded, anonymised responses are in tests/fixtures)

| SDK call | HTTP |
|---|---|
| `users()` | `GET /api/users` returns `{"users": [str]}` |
| `usage()` | `GET /api/now` returns `{ts, age_s, stale, slack, gpus:[{gpu,total_mib,used_mib,util_pct,power_w,bookings:[claim],booked_mib,free_mib,procs:[{pid,user,name,used_mib}]}], alerts:[{kind,key,gpu,user,text,sent}]}` |
| `bookings(days_ahead, days_back)` | `GET /api/claims?days=1..14&back=1..30` returns `{now, claims:[claim]}` |
| book | `POST /api/claims` with `{user, gpu, vram_gib?, start, end, note?}`. `vram_gib` absent means the whole card; times are ISO with an offset. Returns 201 `{"claim": claim}`. |
| `cancel(id)` | `POST /api/claims/<id>/cancel` with `{}`. Returns 200 `{"claim": claim}`. Idempotent. |
| `hold(gpu)` / `release(gpu)` | `POST /api/claims/quick` with `{user | null, gpu}`. Returns 201 `{"claim": claim}` or 200 `{"claim": null}`. |

`claim` = `{id, user, gpu, vram_mib, vram_gib, start, end, note, created_at, created_ip, cancelled_at, cancelled_ip, kind}`.

Errors are `{"error": str, "detail"?: str}` with these statuses: 400 invalid, 403 forbidden, 404, 409 conflict, 413, 415, 429 too many requests, 503 no history yet / busy.

**Server limits:**
- a booking lasts at most 14 days;
- `start >= now - 5 min`;
- VRAM ranges from 1 GiB to the card total;
- 30 writes per hour per client IP;
- capacity: at every instant of `[start, end)`, the sum of active bookings on that GPU plus the new one must be ≤ the card's `total_mib`.

## Public API (`import hrm`)

```python
hrm.connect(user=None, url=None)              # sets the default client; env HRM_USER / HRM_URL
hrm.users() -> list[str]
hrm.usage() -> Usage                          # .gpus: list[Gpu]; .alerts; .taken_at; .stale; .to_frame()
hrm.bookings(days_ahead=14, days_back=7, include_cancelled=False) -> BookingList   # list subclass with .to_frame()
hrm.book(days=None, *, memory=None, gpu=None, start=None, end=None, note=None, user=None) -> BookingResult
hrm.cancel(booking_or_id) -> Booking
hrm.hold(gpu, user=None) -> Booking           # quick hold: whole card until next 09:00 Berlin
hrm.release(gpu) -> None
hrm.Client(url=..., user=...)                 # same methods; the module functions use a default Client
```

### `book()` semantics
- Pass **either** `days` **or** `start`/`end`, not both (`ValueError`).
- **`days`**: an iterable of `date` objects or `"YYYY-MM-DD"` strings. They're deduplicated and sorted, then grouped into **runs of consecutive days**. Each run becomes the window `[max(now, first_day 00:00 Berlin), (last_day + 1) 00:00 Berlin)`.
  - A day before today raises `ValueError("2026-10-05 is in the past")`.
  - A run longer than 14 days raises `ValueError`.
- **`start`/`end`**: a `datetime` or ISO string. A naive value is interpreted as Europe/Berlin.
- **`memory`**: GiB as int or float. `None` means the whole card. `<= 0` raises `ValueError`.
- **GPU choice:**
  - If `gpu` is given, use only that GPU.
  - Otherwise compute candidates from `usage()` (card totals) and `bookings(days_ahead=14, days_back=1)`, the active, uncancelled claims. For each GPU, `free = total - max over each window's boundary instants of the active booked sum`. A GPU qualifies if `free >= need` for **every** window, where `need = memory` in MiB or the whole `total` if `memory` is None.
  - Candidates are taken in ascending GPU index. With no candidate, raise `NoCapacity("no GPU has 48 GiB free for the whole window; best: GPU 3 with 39.6 GiB")`.
- **Booking:** for the chosen GPU, POST each run in order.
  - On a **409** for the first run, try the next candidate GPU, up to 2 retries in total, then raise `Conflict` carrying the server detail.
  - On any failure after run 1 succeeded, **cancel the runs already created** (best effort, errors swallowed), then re-raise. A run's 409 after the first run also retries the whole set on the next candidate after rolling back.
- **Returns** `BookingResult(gpu: int, bookings: list[Booking])`. `str()` gives `"booked GPU 4, 48 GiB, Tue 06 Oct 15:12 → Thu 08 Oct 00:00 (Europe/Berlin), id 12"`, one line per booking.

### User resolution (on the first write, or a call with `user=None`)
`user` argument → `connect(user=)` → `HRM_USER` → `getpass.getuser()` **if** it's in `users()`. Otherwise raise `UnknownUser("set hrm.connect(user=...) or HRM_USER; 'christian' is not a carrot account")`.

### Exceptions (all subclass `hrm.HRMError`)

| exception | cause |
|---|---|
| `InvalidRequest` | 400, 413, 415 |
| `Forbidden` | 403 |
| `NotFound` | 404 |
| `Conflict` | 409 |
| `RateLimited` | 429 |
| `ServerUnavailable` | 503, URLError, timeout |
| `NoCapacity` | raised by the client |
| `UnknownUser` | raised by the client |

Each carries `.status` and `.detail`, which is the server's `detail` or `error` text.

### Models (frozen dataclasses)
- `Process(pid, user, name, used_mib)`, with `user` None meaning unattributed.
- `Booking(id, user, gpu, vram_mib, start, end, note, created_at, created_ip, cancelled_at, cancelled_ip, kind)`, with aware datetimes plus properties `vram_gib`, `active_at(t)` and `cancelled`.
- `Gpu(index, total_mib, used_mib, util_pct, power_w, booked_mib, free_mib, procs, bookings)`.
- `Alert(kind, key, gpu, user, text, sent)`.
- `Usage(taken_at, age_s, stale, slack, gpus, alerts)`.
- `.to_frame()` on `Usage` and `BookingList` imports pandas lazily. Without pandas it raises `ImportError('pip install "hnee-resource-monitoring[pandas]"')`.

## Package and delivery
- Repo layout: `pyproject.toml` (setuptools, name `hnee-resource-monitoring`, package `hrm` under `src/`, extras `pandas`, `test=[pytest]`), `src/hrm/{__init__,client,models,planning,errors}.py`, `tests/`, `examples/quickstart.ipynb`, `README.md`, `.github/workflows/test.yml`.
- **Notebook:**
  1. `%pip install` from the git URL;
  2. `connect`;
  3. `usage().to_frame()`;
  4. `bookings().to_frame()`;
  5. a booking via `book(days=[today], memory=1, note="sdk quickstart")`;
  6. print the result;
  7. `cancel`;
  8. confirm it's cancelled.

  It's committed **with outputs from one live run**. The live run uses `memory=1` and a single day, and is cancelled in the last cell. The live run installs from the local path, because the repo is not pushed yet; the committed install cell shows the git URL.
- **Tests:**
  - pure planning tests;
  - client tests against a stdlib fake HTTP server serving the anonymised fixtures, with scripted error responses;
  - an integration test that's opt-in (`HRM_INTEGRATION=1`) and read-only.
- **CI:** GitHub Actions running pytest on Python 3.9 and 3.12. It only runs once pushed.
- **Local git only:** `main` gets a baseline commit and the work goes on `feat/sdk`, with **no push**.

## Out of scope
Async, a CLI, server changes, authentication and a PyPI release.
