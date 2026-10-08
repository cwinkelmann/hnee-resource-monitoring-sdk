# hrm.pool: a job queue that runs on lendable GPUs (design)

**Date:** 2026-10-08. **Status:** draft for review. The user chose "go with the defaults"; the
decisions below are those defaults, not answers given one by one.
**Depends on:** the booking server's lendable/important bookings (server main, PR #5). The
server changes nothing for this feature.

## Intent

- **What the user said:** queue many jobs (e.g. 60) and run them with 1 to N GPU workers. Each
  worker sits on a *lendable* booking. When someone takes that GPU with an important booking,
  the worker goes away and the pool carries on with the GPUs it still has.
- **Defaults chosen (correct any of them):**
  1. A job is a **shell command** (argv list), not a Python callable.
  2. **One GPU per job**. The pool uses only the GPUs it is given (`gpus=[4, 5, 6, 7]`); no
     default, so it never wanders onto another user's half of the box.
  3. **Jobs are shorter than the grace period** (server default 30 min). A taken worker drains:
     it finishes the job it is on and takes no new one. No checkpoint/resume protocol.
  4. It **runs on carrot** as one supervisor process, started by the user.
- **Success:** `hrm-pool run jobs.txt --gpus 4-7` on carrot works through 60 commands. It books
  GPUs as they come free, hands a GPU back without losing a job when an important booking takes
  it, and cancels every booking it made when the queue is empty or the user presses Ctrl-C.

## Constraints

- **Stdlib only**, like the rest of the SDK (sqlite3, subprocess, threading). POSIX only (it
  uses process groups); carrot is Linux.
- **The pool only ever signals its own child processes.** It never touches a process it did not
  start. Its bookings are lendable, so an important booking can take them with
  `grace_minutes` notice. Releasing them is the pool's job, not the server's.
- **The write budget.** The server allows 30 writes/hour per client IP, and every user on carrot
  shares one IP. The pool caps its own writes (book + cancel) at **10/hour** by default and
  prefers long leases over many short bookings.
- **Timing invariant:** `job_timeout + poll + margin ≤ grace_minutes`. With the defaults
  (20 min + 1 min + 2 min ≤ 30 min) a job started just before a take still ends before the take
  begins. `run()` reads `grace_minutes` from `/api/claims` and refuses to start if the invariant
  fails.
- **Public repo:** no real user names, IPs other than the default URL, or job paths in code,
  docs or fixtures.

## Approaches considered

- **A. One supervisor process, one thread per GPU worker, SQLite queue (recommended).** A single
  owner of all bookings, so the write budget and crash cleanup live in one place. Threads only
  wait on subprocesses and HTTP, so the GIL does not matter.
- **B. Independent worker processes that share the SQLite queue.** More robust if one worker
  dies, but each would book on its own: no shared write budget, and crash cleanup needs a
  separate reaper.
- **C. An existing tool (Ray, task-spooler, a Slurm-like scheduler).** None knows about the booking
  server, and all of them add dependencies to a zero-dependency SDK.

## Prerequisite SDK changes (small, land first)

The SDK predates lendable bookings. It needs:
- `Booking.priority` (`"lendable" | "important"`; rows without it read as `"important"`, as on
  the server) and `Booking.taken: list[Take]`, where
  `Take(start, end, vram_mib, by: list[str])` is parsed from the `taken` field of `/api/claims`.
- `book(..., priority="lendable")` sends `priority`. The default stays what the server's
  default is (lendable).
- `BookingList.grace_minutes`, from the `/api/claims` response.
- New fixtures recorded from olive, anonymised.

## Public API

```python
from hrm.pool import Pool

pool = Pool("runs/tiles.sqlite", gpus=[4, 5, 6, 7])   # the queue file; created if missing
pool.add([["python", "infer.py", "--tile", str(i)] for i in range(60)], cwd="~/proj")
pool.run()           # blocks until the queue is empty; Ctrl-C drains and cancels bookings
pool.status()        # counts per state, workers, bookings held
```

`Pool(path, gpus, *, max_workers=len(gpus), job_timeout="20m", lease="4h", max_attempts=2,
poll="60s", margin="2m", max_writes_per_hour=10, client=None)`

CLI (`python -m hrm.pool`, also the console script `hrm-pool`):
```
hrm-pool add  runs/tiles.sqlite -- python infer.py --tile 3
hrm-pool run  jobs.txt --gpus 4-7      # jobs.txt: one command per line (shlex), # comments
hrm-pool status runs/tiles.sqlite
```
`run jobs.txt` creates `jobs.sqlite` next to it and adds every line, unless that queue already
holds them; re-running it resumes.

## Components

| Module | Does | Depends on |
|---|---|---|
| `pool/queue.py` | SQLite queue: `jobs(id, argv json, cwd, state, attempts, gpu, booking_id, started_at, ended_at, exit_code)`, `bookings(id, gpu, start, end, state)`, `writes(ts)`. `BEGIN IMMEDIATE` per transition. | sqlite3 |
| `pool/plan.py` | **Pure** `decide(now, workers, queue_counts, claims, usage, cfg) -> list[Action]`. Actions: `Book(gpu, start, end)`, `Renew(gpu, start, end)`, `Drain(gpu, reason)`, `Cancel(booking_id)`. No I/O. | models only |
| `pool/runner.py` | The supervisor loop: every `poll`, fetch `/api/claims` + `/api/now` once, call `decide`, carry out the actions, start/stop worker threads. Workers pull a job, run it under `CUDA_VISIBLE_DEVICES=<gpu>`, enforce `job_timeout`. | queue, plan, Client |
| `pool/__main__.py` | CLI | runner |

The pure `decide()` is the same seam ResourceMonitor uses (`Snapshot` → rules). All scheduling
rules are tested with fixtures and a fake clock, without a GPU or a server.

## Rules in `decide()`

1. **Scale up.** Book a GPU when all of these hold:
   - jobs are pending and there are fewer workers than `min(max_workers, pending + running)`;
   - the GPU is in `gpus`, has no worker, and is not in a take cooldown;
   - the whole card is unbooked for `[now, now + lease)`;
   - `/api/now` shows less than 1 GiB used on it (unbooked use counts as busy);
   - the write budget allows it.

   The booking is lendable, covers the whole card, and is noted `hrm-pool <queue name>`.
2. **Drain on take.** If the worker's booking has a `taken` segment starting before
   `now + job_timeout + poll + margin`, the worker drains: no new job, and it cancels the booking
   once its current job ends. The GPU enters cooldown until that segment ends.
3. **Renew.** If jobs are pending and the booking ends within `job_timeout + poll`, book
   `[end, end + lease)` on the same GPU. If that fails (409 or budget), the worker drains.
4. **Scale down.** If no jobs are pending, idle workers cancel their booking and stop.
5. **Last resort.** If a job is still running at `take.start − margin` (only possible after
   clock skew or a missed poll), the pool sends SIGTERM to its process group and SIGKILL 30 s
   later. The job goes back to `pending` **without** counting an attempt.

## Job lifecycle and errors

`pending → running → done | failed`, with `running → pending` on drain-kill or pool restart.
- Exit 0 → `done`. Non-zero exit or `job_timeout` → attempt + 1, then `pending` again, or
  `failed` after `max_attempts`.
- Each job's stdout+stderr goes to `<queue>.logs/<id>.log`.
- Env: `CUDA_VISIBLE_DEVICES`, `HRM_POOL_JOB_ID`, `HRM_POOL_GPU`.
- **Jobs must be idempotent** (write to a temp path, rename when finished). This is documented
  as the one contract on the user's side.
- **Crash recovery:** an `flock` on `<queue>.lock` keeps a second pool off the same queue. On
  start, `running` jobs go back to `pending`, and bookings recorded as live are cancelled
  (idempotent on the server).
- **Server unreachable:** running jobs carry on, and no new bookings are made. If the server is
  still unreachable when a booking ends, the worker drains rather than running unbooked.
- **Ctrl-C / SIGTERM:** the first one drains (running jobs finish). A second one kills the
  pool's own children, requeues their jobs and cancels the bookings.

## Testing

- `plan.py`: table tests for each rule: scale up, no booking of a card in unbooked use, cooldown
  after a take, drain threshold, renewal, renewal refused → drain, write budget, scale down.
- `queue.py`: transitions, attempts, crash recovery, lock contention.
- `runner.py`: a fake Client plus `sleep`/`false` commands and an injected clock. Checks that a
  take drains without losing a job, that `job_timeout` kills only the pool's own child, and that
  every booking is cancelled on exit.
- **On olive (2×A6000) before carrot:** 10 × `sleep 120` jobs on both GPUs. Make an important
  booking from the dashboard, then check that the worker drains, the job is not lost, the take
  alert fires and the pool finishes on one GPU.

## Out of scope

Checkpoint/resume for long jobs, multi-GPU jobs (DDP), fractional-card workers, several hosts,
priorities between jobs, and any change to the booking server.
