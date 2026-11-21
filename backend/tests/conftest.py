"""Shared fixtures for the API tests.

``seed_data`` writes two days of deterministic synthetic M1 bars into the same
``tmp_path`` the app factory uses, so a test can ask for stored data without
inventing a second root that the app would never read.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider

SYMBOL = "XAUUSD"
#: Monday 2026-01-05 through Wednesday 2026-01-07 (end inclusive, whole minutes).
SEED_START = datetime(2026, 1, 5, tzinfo=UTC)
SEED_END = datetime(2026, 1, 7, tzinfo=UTC)


@pytest.fixture
def seed_data(tmp_path):
    """Store two days of synthetic M1 for ``XAUUSD`` under ``tmp_path``."""
    frame = SyntheticProvider(seed=7).fetch(SYMBOL, SEED_START, SEED_END)
    return ParquetStore(tmp_path).write(SYMBOL, "M1", frame)
