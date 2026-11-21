"""Shared fixtures for the API tests.

``seed_data`` writes two days of deterministic synthetic M1 bars into the same
``tmp_path`` the app factory uses, so a test can ask for stored data without
inventing a second root that the app would never read.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider

SYMBOL = "XAUUSD"
#: Monday 2026-01-05 through Wednesday 2026-01-07 (end inclusive, whole minutes).
SEED_START = datetime(2026, 1, 5, tzinfo=UTC)
SEED_END = datetime(2026, 1, 7, tzinfo=UTC)

#: First stamp and length of the OHLC-only CSV fixture.
NOSPREAD_START = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
NOSPREAD_ROWS = 240


def write_nospread_csv(path: Path, rows: int = NOSPREAD_ROWS) -> Path:
    """Write a headerless six-column broker CSV with no bid/ask quotes.

    The walk is deterministic, ranges bracket the open/close and the volume is
    a whole number, so the file only exercises the missing-spread path.
    """
    rng = np.random.default_rng(11)
    stamp = NOSPREAD_START
    price = 2350.0
    lines: list[str] = []
    for _ in range(rows):
        open_ = price
        close = open_ + float(rng.normal(0.0, 0.15))
        high = max(open_, close) + abs(float(rng.normal(0.2, 0.05)))
        low = min(open_, close) - abs(float(rng.normal(0.2, 0.05)))
        volume = int(rng.integers(5, 60))
        lines.append(
            f"{stamp:%Y.%m.%d %H:%M},{open_:.2f},{high:.2f},{low:.2f},{close:.2f},{volume}"
        )
        price = close
        stamp += timedelta(minutes=1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def seed_data(tmp_path):
    """Store two days of synthetic M1 for ``XAUUSD`` under ``tmp_path``."""
    frame = SyntheticProvider(seed=7).fetch(SYMBOL, SEED_START, SEED_END)
    return ParquetStore(tmp_path).write(SYMBOL, "M1", frame)


@pytest.fixture
def nospread_csv(tmp_path):
    """A 6-column broker export: OHLC plus volume, no bid/ask quotes."""
    return write_nospread_csv(tmp_path / "nospread.csv")
