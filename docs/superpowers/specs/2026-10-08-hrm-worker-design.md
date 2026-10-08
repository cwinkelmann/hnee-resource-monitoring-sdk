# hrm-worker: per-GPU job workers on lendable bookings (design)

**Date:** 2026-10-08. **Status:** draft for review. This replaces the `hrm.pool` draft (`d25b668`):
the user chose per-GPU containers with long bookings over a supervisor that scales and renews.
**Depends on:** the booking server's lendable/important bookings (server main, PR #5). The
server does not change.

## Intent

- **What the user said:** deploy a set of containers that each read one job at a time. Each
  container has one long-standing booking that covers many jobs. If someone deletes that
  booking, the container finishes its current iteration, exits and frees the GPU.
- **Defaults carried over (correct any of them):**
  - a job is a shell command;
  - one GPU per job;
  - one iteration (job) is shorter than the grace period;
  - it runs on carrot.
- **Success:**
  - `start-workers.sh --gpus 4-7` starts one container per free GPU, and they work through a
    queue of e.g. 60 jobs.
  - When an important booking takes a GPU, or someone cancels a worker's booking, that worker
    finishes its current job, cancels its booking and exits. No job is lost, and the other
    person gets the card by the time their booking starts.
  - When the queue is empty, every worker exits and no booking is left behind.

## Constraints

- **Stdlib only**, like the rest of the SDK. POSIX only; carrot is Linux.
- **"Deleted" includes "taken".** Other users rarely cancel a lendable booking. They make an
  important booking, the server derives a `taken` segment on ours, and our booking stays live.
  The worker treats both a cancellation and an upcoming take as the signal to stop.
- **Timing invariant:** `max_job < grace_minutes` (defaults 25 min < 30 min). The worker checks
  for a stop signal immediately before it claims each job. A take made at any later moment
  starts at least `grace_minutes` after it was made, so the job finishes first. `hrm-worker run`
  reads `grace_minutes` from `/api/claims` and refuses to start if the invariant fails.
- **The worker only ever signals its own child.** `max_job` is enforced by killing the worker's
  own job process. Nothing else is ever signalled.
- **Attribution:** the monitor attributes GPU memory by host UID. Containers must run their
  job as the booking user's host UID (`--user $(id -u):$(id -g)` on rootful Docker). Otherwise
  the job shows up as root/system or as an unknown UID, and carrot raises a "process on a booked
  GPU" alert against our own booking. **This is to be verified on olive** (see Testing).
- **Writes:** two per worker run (book, cancel). The server's 30 writes/hour limit per client
  IP is not a concern.
- **Public repo:** no real user names, no IPs other than the default URL, and no job paths in
  code, docs or fixtures.

## Prerequisite SDK changes (small, land first)

- `Booking.priority` (`"lendable" | "important"`; rows without it read as `"important"`, as on
  the server).
- `Booking.taken: list[Take]`, where `Take(start, end, vram_mib, by: list[str])` is parsed from
  the `taken` field of `/api/claims`.
- `book(..., priority="lendable")` sends `priority`.
- `BookingList.grace_minutes`, from the `/api/claims` response.
- New fixtures recorded from olive, anonymised.

## The queue: a directory

```
<queue>/todo/0042.sh       waiting
<queue>/running/gpu4-0042.sh   claimed by the worker on GPU 4
<queue>/done/0042.sh       exit 0
<queue>/failed/0042.sh     non-zero exit, or killed at max_job
<queue>/logs/0042.log      stdout + stderr of the job
```

- **Claim:** take the lowest-sorted file in `todo/` and `os.rename` it to
  `running/gpu<N>-<name>`. A rename within one filesystem is atomic. If another worker won the
  race, the rename fails with `FileNotFoundError` and the worker tries the next file. No lock and
  no database are needed, and `ls` shows the state.
- **A job** is a file run with `sh <file>`, with the queue directory's parent as the working
  directory. Its environment adds `HRM_JOB=<name>` and `HRM_GPU=<host index>`.
  `hrm-worker enqueue --queue DIR jobs.txt` writes one file per non-comment line.
- **Jobs must be idempotent** (write to a temp path, rename when finished), because a job
  interrupted by a crash or `docker stop` is run again. This is the one contract on the user's
  side.
- **No automatic retries.** To retry, run `mv failed/* todo/` (or `hrm-worker retry`).

## Components

| File | Does |
|---|---|
| `src/hrm/jobdir.py` | The directory queue: `enqueue(lines)`, `claim(gpu)`, `finish(job, ok)`, `requeue(gpu)` (moves `running/gpu<N>-*` back to `todo/`), `status()` |
| `src/hrm/worker.py` | **Pure** `stop_reason(booking, now, max_job) -> str or None`, plus the loop below |
| console script `hrm-worker` | `run`, `enqueue`, `requeue`, `retry`, `status` |
| `examples/worker/Dockerfile` | `FROM <your image>`, `pip install` the SDK, `ENTRYPOINT ["hrm-worker", "run"]` |
| `examples/worker/start-workers.sh` | One container per free GPU (below) |

`stop_reason` returns:
- `"cancelled"` if the booking was cancelled;
- `"taken by <users>"` if it has a `taken` segment with `start < now + max_job`;
- `"booking ends"` if `booking.end < now + max_job`;
- otherwise `None`.

It is the only scheduling rule, and its tests are a table.

## The worker loop (`hrm-worker run --queue DIR --gpu N`)

1. **Start.**
   - Check `max_job < grace_minutes`.
   - Check that `/api/now` shows less than 1 GiB in use on GPU N; unbooked use counts as busy.
   - Book GPU N: whole card, lendable, from now for `--hours` (default 168, server max 14 days),
     note `hrm-worker <queue name>`.
   - Any failure (409, busy card, server down) means exit 5 without running a job.
2. **Each iteration.**
   - `GET /api/claims` and find our booking.
   - If `stop_reason` gives a reason, exit 3 (cancelled/taken) or 4 (booking ends).
   - Otherwise claim a job. An empty queue means exit 0.
   - Run the job, with `max_job` enforced on the worker's own child (SIGTERM, SIGKILL 30 s
     later, then `failed/`).
3. **Exit (always, via `finally`):** cancel the booking (idempotent) and log the reason.

**Errors**
- **Server unreachable at the check:** retry 3 times over about a minute, then exit 5 rather than
  run without a booking. If the cancel then fails too, the booking is left until it ends. The
  monitor's idle alert points at it.
- **SIGTERM** (`docker stop`; use `-t 60`): terminate the child, move its job back to `todo/`,
  cancel the booking and exit.
- **Hard crash** (OOM-kill, host reboot): the job stays in `running/gpu<N>-*` and the booking
  stays live. `start-workers.sh` handles this (below).

## `start-workers.sh --gpus 4-7 --queue DIR --image IMG`

For each GPU N that has no running container `hrm-worker-gpu<N>`:
1. `hrm-worker requeue --gpu N`, which moves N's orphaned jobs back to `todo/`.
2. Run the container:

```
docker run -d --rm --name hrm-worker-gpu<N> --gpus device=<N> --network host \
  --user $(id -u):$(id -g) -e HRM_USER -e HRM_URL -v DIR:DIR \
  IMG --queue DIR --gpu <N>
```

`--network host` lets the container reach the booking server like the host does. `HRM_USER`
must be set, because the container's login is not a carrot account. GPUs that are booked or in
use make their worker exit 5 at once; that is the "only free GPUs" filter. Running the script
again later picks up GPUs that have come free. Nothing scales automatically.

The worker also runs without Docker: `CUDA_VISIBLE_DEVICES=4 hrm-worker run --queue DIR --gpu 4`.

## Testing

- `stop_reason`: a table covering:
  - cancelled;
  - a take before, at and after `now + max_job`;
  - a take on another booking (ignored);
  - the booking ending;
  - an important booking (never taken).
- `jobdir`: enqueue/claim/finish/requeue. Two processes racing for one job; exactly one wins.
- `worker` loop with a fake Client and `true`/`false`/`sleep` jobs, checking:
  - exit codes 0/3/4/5;
  - `max_job` kills only the worker's own child;
  - SIGTERM requeues the job;
  - the booking is cancelled on every exit path.
- **On olive (2×A6000) before carrot:**
  - Start two workers on 10 × `sleep 120` jobs.
  - Make an important booking on one GPU from the dashboard. That worker finishes its job and
    exits 3, the take alert fires, and the other worker finishes the queue.
  - Confirm the dashboard attributes the container's process to the booking user, not
    root/system. If it doesn't, fix the `--user` mapping before going any further.

## Out of scope

Automatic scale-up or lease renewal, checkpoint/resume for jobs longer than the grace period,
multi-GPU jobs, fractional-card workers, automatic retries, several hosts, and any change to the
booking server.
