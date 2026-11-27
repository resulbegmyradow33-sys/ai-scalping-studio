"""Flow side-table reader: duplicate rule, columns and path safety."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from scalping.strategy.flow import flow_series, flow_thresholds


def _stamps(n: int = 3) -> list[datetime]:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    return [start + timedelta(minutes=i) for i in range(n)]


def test_duplicate_flow_stamps_keep_the_last_row(tmp_path: Path) -> None:
    stamps = _stamps()
    folder = tmp_path / "XAUUSD" / "M1"
    folder.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": [stamps[0], stamps[1], stamps[1], stamps[2]],
            "flow": [0.1, 0.2, 0.9, 0.3],
        }
    ).write_parquet(folder / "part.parquet")
    frame = pl.DataFrame({"ts": stamps})

    values = flow_series(tmp_path, "XAUUSD", "M1", frame)

    assert values.tolist() == [0.1, 0.9, 0.3]


def test_missing_columns_and_unsafe_symbols_are_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "XAUUSD" / "M1"
    folder.mkdir(parents=True)
    pl.DataFrame({"ts": _stamps(1), "other": [1.0]}).write_parquet(folder / "part.parquet")
    frame = pl.DataFrame({"ts": _stamps(1)})

    with pytest.raises(ValueError, match="missing columns"):
        flow_series(tmp_path, "XAUUSD", "M1", frame)
    with pytest.raises(ValueError, match="invalid flow symbol"):
        flow_series(tmp_path, "../XAUUSD", "M1", frame)
    with pytest.raises(ValueError, match="unsupported timeframe"):
        flow_series(tmp_path, "XAUUSD", "M7", frame)


def test_thresholds_are_causal_and_shifted() -> None:
    import numpy as np

    flow = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    thresholds = flow_thresholds(flow, window=2, quantile=0.5)

    assert np.isnan(thresholds[0])
    assert np.isnan(thresholds[1])
    assert thresholds[2] >= 1.0
    assert thresholds[3] >= thresholds[2]
