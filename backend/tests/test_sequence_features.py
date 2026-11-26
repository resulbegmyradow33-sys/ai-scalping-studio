"""Optional sequence features: streaks, log returns and causality."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scalping.features.sequences import SEQUENCE_COLUMNS, sequence_features

START = datetime(2026, 1, 5, tzinfo=UTC)


def bars(closes: list[float]) -> pl.DataFrame:
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=i) for i in range(close.size)],
            "open": open_,
            "high": np.maximum(open_, close) + 0.05,
            "low": np.minimum(open_, close) - 0.05,
            "close": close,
            "bid_close": close,
            "ask_close": close + 0.2,
            "spread": np.full(close.size, 0.2),
            "volume": np.full(close.size, 10, dtype=np.int64),
        }
    )


def test_streaks_and_log_returns_follow_the_path() -> None:
    frame = bars([100, 101, 102, 101, 100, 99, 100])
    out = sequence_features(frame)

    assert out.columns == ["ts", *SEQUENCE_COLUMNS]
    assert out["up_streak"].to_list() == [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    assert out["down_streak"].to_list() == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0]
    assert out["log_ret_1"][0] == 0.0
    assert out["log_ret_1"][-1] == pytest.approx(math.log(100 / 99))
    assert out["log_ret_3"][3] == pytest.approx(math.log(101 / 100))


def test_features_are_causal() -> None:
    path = [100.0 + np.sin(i / 3.0) for i in range(120)]
    base = sequence_features(bars(path))

    mutated = path.copy()
    mutated[80:] = [value * 2 for value in mutated[80:]]
    changed = sequence_features(bars(mutated))

    assert_frame_equal(base.head(80), changed.head(80))


def test_flat_and_empty_frames_keep_the_schema() -> None:
    flat = sequence_features(bars([100.0] * 6))
    assert flat["up_streak"].to_list() == [0.0] * 6
    assert flat["down_streak"].to_list() == [0.0] * 6
    assert flat["log_ret_1"].to_list() == [0.0] * 6

    empty = sequence_features(bars([100.0]).head(0))
    assert empty.height == 0
    assert empty.columns == ["ts", *SEQUENCE_COLUMNS]
