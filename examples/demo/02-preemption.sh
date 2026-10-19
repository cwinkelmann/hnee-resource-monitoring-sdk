#!/usr/bin/env bash
# Demo 2: a colleague needs the card, and gets it.
#
# Starts the training example as a lendable job that promises no notice, lets it get a
# few epochs in, then books the card as important from a second process. The job is told
# in milliseconds, checkpoints, and exits 0 -- being preempted is a normal outcome. Then
# it resumes from where it stopped.
#
# Needs a GPU and PyTorch (examples/train_gpu.py). On a box without them, run 01-queue.sh,
# which needs neither.
#
#   ./02-preemption.sh
cd "$(dirname "$0")"
. ./common.sh
trap 'say "cleaning up"; [ -n "${JOB_PID:-}" ] && kill $JOB_PID 2>/dev/null; release_mine' EXIT INT TERM
require_sdk
require_server

GIB="${HRM_GIB:-6}"
OUT="${HRM_OUT:-$HOME/runs/hrm-demo-preempt}"
LOG="$(mktemp -t hrm-demo-job.XXXXXX)"
REVOKED_FILE="$(mktemp -t hrm-demo-revoked.XXXXXX)"
export REVOKED_FILE
JOB_PID=""

if ! "$HRM_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "this demo needs PyTorch and a CUDA GPU in $HRM_PY" >&2
    echo "01-queue.sh shows the booking half and needs neither" >&2
    exit 1
fi

# A booking is bookkeeping: the server will happily hand out a card that is physically
# full, because nothing enforces bookings. Check what is actually on it, or this demo
# dies of CUDA OOM half way through and looks like a bug in the SDK.
hrm_py '
import os, sys
import hrm
c = hrm.Client(url=os.environ["HRM_URL"], user=os.environ["HRM_USER"])
busy = [(g.index, p.user, p.name, p.used_mib)
        for g in c.usage().gpus for p in g.procs if p.used_mib > 512]
if busy:
    print("   someone is already using the card:")
    for index, user, name, mib in busy:
        print("      GPU %d: %s %s, %d MiB" % (index, user, name, mib))
    print("   this demo needs a free card -- it would run out of CUDA memory.")
    sys.exit(1)
' || exit 1

rm -rf "$OUT"

say "Start a long training run that accepts being interrupted"
note "lendable + notice_seconds=0: 'take this card whenever you need it'."
note "log: $LOG"
HRM_URL="$HRM_URL" HRM_USER="$HRM_USER" \
    "$HRM_PY" ../train_gpu.py --memory "$GIB" --hours 1 --steps 20 --epochs 500 \
    --out "$OUT" --note "$NOTE_TAG: training" > "$LOG" 2>&1 &
JOB_PID=$!

say "Let it get some epochs in"
for _ in $(seq 1 40); do
    if grep -q "^epoch 2 " "$LOG" 2>/dev/null; then break; fi
    sleep 2
done
tail -4 "$LOG"

say "Now someone needs the card: an important booking for most of it"
hrm_py "
import os, time
import hrm
hrm.connect(user=os.environ['HRM_USER'], url=os.environ['HRM_URL'])
# The card, not what is free: booking only the free part would fit beside the job and
# squeeze nobody, which is the mistake that makes this demo look like it does nothing.
# Floored to whole GiB, so 16376 MiB asks for 15 and not an impossible 16.
card = max(g.total_mib for g in hrm.usage().gpus) // 1024
began = time.monotonic()
taker = hrm.reserve(memory=card, hours=1, important=True,
                    note=os.environ['NOTE_TAG'] + ': urgent')
print('   booking %d placed in %.2fs, starts %s'
      % (taker.id, time.monotonic() - began,
         taker.start.astimezone().strftime('%H:%M:%S')))
import pathlib
for r in taker.revoked:
    print('   revoked booking %d of %s: %d MiB lost, truncated=%s'
          % (r['id'], r['user'], r['lost_mib'], r['truncated']))
# Hand the ids to the next step, so it reports this run's victims and not every
# booking a past demo left behind.
pathlib.Path(os.environ['REVOKED_FILE']).write_text(
    ' '.join(str(r['id']) for r in taker.revoked))
if not taker.revoked:
    print('   nothing was revoked -- was the job still waiting for its slot, or does')
    print('   what was booked still fit beside it?')
"

say "What the job did about it"
note "It stops at the next epoch boundary -- the one place a checkpoint is safe."
for _ in $(seq 1 60); do
    if ! kill -0 "$JOB_PID" 2>/dev/null; then break; fi
    sleep 1
done
if kill -0 "$JOB_PID" 2>/dev/null; then
    tail -4 "$LOG"
    note "it is STILL RUNNING, so it was not preempted -- see the step above"
    kill "$JOB_PID" 2>/dev/null       # our own child, and only ever our own
    JOB_PID=""
else
    wait "$JOB_PID" 2>/dev/null
    STATUS=$?
    JOB_PID=""
    tail -4 "$LOG"
    note "exit status: $STATUS (0 -- being preempted is not a failure)"
fi

say "The victim's row: truncated, not deleted"
note "The window it really used stays on the calendar and in the energy report."
hrm_py '
import os
import hrm
c = hrm.Client(url=os.environ["HRM_URL"], user=os.environ["HRM_USER"])
# Only what this run revoked -- a note match would list every past demo too.
wanted = {int(x) for x in open(os.environ["REVOKED_FILE"]).read().split()}
for b in c.bookings():
    if b.id in wanted:
        print("   booking %d  %s -> %s  cancelled=%s  priority=%s  %r"
              % (b.id, b.start.astimezone().strftime("%H:%M:%S"),
                 b.end.astimezone().strftime("%H:%M:%S"),
                 b.cancelled_at is not None, b.priority, b.note))
'

say "It can carry on where it stopped"
hrm_py "
import torch
state = torch.load('$OUT/checkpoint.pt', map_location='cpu')
print('   checkpoint at epoch %d (%d layers) -- the same command resumes from %d'
      % (state['epoch'], state['layers'], state['epoch'] + 1))
"

say "Done"
