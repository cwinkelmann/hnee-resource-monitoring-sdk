"""Child-process targets for the Slot.run tests.

They live in a real importable module because Slot.run uses the "spawn" start
method: the child re-imports the target by name, and a function defined inside a
test function cannot be pickled.
"""
from __future__ import annotations

import time
from pathlib import Path


def finishes_quickly(stop, seconds=0.2):
    """Does its work and returns, ignoring nothing."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop.wait(timeout=0.02):
            return
    return


def runs_until_stopped(stop, marker, seconds=30.0):
    """Writes `marker` when it is asked to stop -- the job's checkpoint, in miniature."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop.wait(timeout=0.02):
            Path(marker).write_text("stopped gracefully")
            return
    Path(marker).write_text("ran out of work")


def ignores_the_stop(stop, seconds=30.0):
    """A job that will not listen. The parent has to end it."""
    time.sleep(seconds)
