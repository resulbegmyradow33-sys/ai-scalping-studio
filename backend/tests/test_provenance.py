"""Dataset provenance sidecars."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scalping.data.provenance import (
    build_provenance,
    checksum,
    load_provenance,
    save_provenance,
)


def _frame() -> pl.DataFrame:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    return pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=i) for i in range(3)],
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "spread": [0.1, 0.1, 0.1],
            "volume": [1, 2, 3],
        }
    )


def test_checksum_is_stable_and_changes_with_data() -> None:
    frame = _frame()
    assert checksum(frame) == checksum(frame.clone())
    changed = frame.with_columns((pl.col("close") + 1).alias("close"))
    assert checksum(frame) != checksum(changed)


def test_round_trip_preserves_metadata(tmp_path) -> None:
    frame = _frame()
    provenance = build_provenance(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        source="unit-test",
        synthetic=True,
    )
    path = save_provenance(tmp_path, provenance)
    assert path.exists()
    loaded = load_provenance(tmp_path, "XAUUSD", "M1")
    assert loaded == provenance
    assert loaded.synthetic is True
    assert loaded.rows == 3
    assert loaded.dataset_id.startswith("XAUUSD:M1:")


def test_provenance_path_refuses_to_leave_the_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="single path segment"):
        load_provenance(tmp_path, "../outside", "M1")
    with pytest.raises(ValueError, match="single path segment"):
        load_provenance(tmp_path, "XAUUSD", "../../outside")
    with pytest.raises(ValueError, match="single path segment"):
        load_provenance(tmp_path, "/etc/passwd", "M1")
