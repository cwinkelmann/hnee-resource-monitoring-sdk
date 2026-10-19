#!/usr/bin/env bash
# Demo 1: a full box is not a refusal.
#
# Books the whole card, then books it again. The second booking is not refused -- it is
# given the earliest window that fits. Releasing the first one early pulls the second
# forward, which is the whole point of auto_advance.
#
#   ./01-queue.sh
#   HRM_URL=http://10.188.1.1:8765 HRM_USER=cwinkelmann ./01-queue.sh
cd "$(dirname "$0")"
. ./common.sh
trap 'say "cleaning up"; release_mine' EXIT INT TERM
require_sdk
require_server

GIB="${HRM_GIB:-$(card_gib)}"

say "What is free right now?"
note "hrm.capacity() is the server's own arithmetic. Do not sum bookings yourself."
hrm_py '
import os
import hrm
hrm.connect(user=os.environ["HRM_USER"], url=os.environ["HRM_URL"])
cap = hrm.capacity(memory=1, hours=1)
print("   %.1f GiB free on GPU %s; a 1 h slot would fit from %s"
      % (cap.free_gib, cap.gpu, cap.earliest.astimezone().strftime("%H:%M:%S")))
'

say "Book the whole card for two hours"
# The id comes back on stdout so the later steps act on this booking and not on a
# note match that could find one from an earlier run.
FIRST_ID=$(hrm_py "
import os
import hrm
hrm.connect(user=os.environ['HRM_USER'], url=os.environ['HRM_URL'])
slot = hrm.reserve(memory=$GIB, hours=2, note=os.environ['NOTE_TAG'] + ': first')
import sys
print('   booking %d on GPU %d, %s -> %s, assigned_later=%s'
      % (slot.id, slot.gpu, slot.start.astimezone().strftime('%H:%M:%S'),
         slot.end.astimezone().strftime('%H:%M:%S'), slot.assigned_later),
      file=sys.stderr)
print(slot.id)
")

say "Book the same card again -- the box is now full"
note "No NoCapacity, no retry loop: the server answers with a window in the future."
hrm_py "
import os
import hrm
hrm.connect(user=os.environ['HRM_USER'], url=os.environ['HRM_URL'])
slot = hrm.reserve(memory=$GIB, hours=1, auto_advance=True,
                   note=os.environ['NOTE_TAG'] + ': queued')
print('   booking %d on GPU %d, %s -> %s, assigned_later=%s'
      % (slot.id, slot.gpu, slot.start.astimezone().strftime('%H:%M:%S'),
         slot.end.astimezone().strftime('%H:%M:%S'), slot.assigned_later))
print('   it waits for the first one to end. A job would now call wait_until_start().')
"

say "Release the first booking early"
note "Its end moves to now -- the window it really used stays on the calendar -- and"
note "whoever is waiting for that card moves up, inside the same transaction."
FIRST_ID="$FIRST_ID" hrm_py '
import os
import hrm
import hrm.models
c = hrm.Client(url=os.environ["HRM_URL"], user=os.environ["HRM_USER"])
first_id = int(os.environ["FIRST_ID"])
out = c.release_claim(first_id)
print("   booking %d now ends %s (cancelled: %s)"
      % (first_id, out["claim"].end.astimezone().strftime("%H:%M:%S"),
         out["claim"].cancelled_at is not None))
def local(iso):
    # The server answers in UTC; everything else here is shown local, and mixing the
    # two makes an advance read as if time went backwards.
    return hrm.models.parse_time(iso).astimezone().strftime("%H:%M:%S")

for moved in out["advanced"]:
    print("   -> booking %d moved from %s to %s"
          % (moved["id"], local(moved["old_start"]), local(moved["new_start"])))
if not out["advanced"]:
    print("   -> nothing moved (was the second booking made with auto_advance?)")
'

say "Done"
