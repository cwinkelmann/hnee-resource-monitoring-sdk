from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture():
    def _load(name):
        return json.loads((FIXTURES / (name + ".json")).read_text())

    return _load


@pytest.fixture
def fake():
    """A running fake booking server (see tests/fakeserver.py)."""
    from fakeserver import FakeServer

    server = FakeServer().start()
    yield server
    server.stop()
