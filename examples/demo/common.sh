# Shared setup for the demo scripts. Source it, do not run it.
#
#   HRM_URL   the booking server            (default: carrot, as in hrm itself)
#   HRM_USER  your account on that host     (default: $USER)
#   HRM_PY    the python to use             (default: the hrm-train conda env, then python3)
#   HRM_GIB   VRAM per booking in GiB       (default: a third of the card, measured below)
set -u

: "${HRM_URL:=http://10.188.1.1:8765}"
: "${HRM_USER:=$USER}"
export HRM_URL HRM_USER

# The demo books real slots on a real box. Everything it makes carries this note, and
# every script releases its own on the way out -- including on Ctrl-C.
NOTE_TAG="hrm demo"
export NOTE_TAG

if [ -z "${HRM_PY:-}" ]; then
    for candidate in "$HOME/anaconda3/envs/hrm-train/bin/python" \
                     "$HOME/miniconda3/envs/hrm-train/bin/python" \
                     "$(command -v python3 || true)"; do
        if [ -x "$candidate" ]; then HRM_PY="$candidate"; break; fi
    done
fi
export HRM_PY

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

# Run a python snippet with hrm available. Keeping every snippet here rather than in a
# .py file means each script reads top to bottom as the story it is telling.
hrm_py() { "$HRM_PY" -c "$1" "${@:2}"; }

require_sdk() {
    if ! "$HRM_PY" -c "import hrm" 2>/dev/null; then
        echo "the hrm SDK is not importable by $HRM_PY" >&2
        echo "install it:  $HRM_PY -m pip install -e /path/to/hnee-resource-monitoring-sdk" >&2
        exit 1
    fi
}

require_server() {
    if ! hrm_py '
import os, sys
import hrm
try:
    hrm.Client(url=os.environ["HRM_URL"], user=os.environ["HRM_USER"]).usage()
except hrm.HRMError as exc:
    sys.exit("cannot reach %s: %s" % (os.environ["HRM_URL"], exc))
' ; then
        exit 1
    fi
}

# Release every booking this demo made, whatever happened. release() moves a started
# booking's end to now and cancels one that has not started, so the calendar keeps the
# window that was really used.
release_mine() {
    hrm_py '
import os
import hrm
c = hrm.Client(url=os.environ["HRM_URL"], user=os.environ["HRM_USER"])
tag = os.environ["NOTE_TAG"]
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
freed = []
for b in c.bookings():
    # Only what is still live: a booking already released by an earlier run has
    # cancelled_at None and an end in the past, and "releasing" it again says nothing.
    if (b.note or "").startswith(tag) and b.cancelled_at is None and b.end > now:
        try:
            c.release_claim(b.id)
            freed.append(b.id)
        except hrm.HRMError:
            pass
print("   released:", ", ".join(map(str, freed)) or "nothing to release")
'
}

card_gib() {
    hrm_py '
import os
import hrm
c = hrm.Client(url=os.environ["HRM_URL"], user=os.environ["HRM_USER"])
print(max(g.total_mib for g in c.usage().gpus) // 1024)
'
}
