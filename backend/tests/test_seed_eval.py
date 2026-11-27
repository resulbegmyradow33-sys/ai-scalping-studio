"""Seed ensemble evaluation: rows, weighting and guard rails."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.config import Settings
from scalping.features.build import FeatureFrame
from scalping.rl.seed_eval import evaluate_seed_ensemble


def frame(n: int = 300) -> FeatureFrame:
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    price = np.full(n, 1000.0)
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=np.column_stack([price, price + 0.001, price - 0.001, price]),
        spread=np.full(n, 0.20),
        atr=np.full(n, 5.0),
        valid_from=60,
        columns=["a", "b", "c"],
    )


HOLD = lambda obs: [1.0, 0.0]
BUY = lambda obs: [0.0, 1.0]


def test_members_and_ensemble_are_compared_on_the_same_slice() -> None:
    rows = evaluate_seed_ensemble({"hold": HOLD, "buy": BUY}, frame(), Settings())
    by_name = {row["name"]: row for row in rows}

    assert [row["name"] for row in rows] == ["hold", "buy", "ensemble"]
    assert by_name["hold"]["trades"] == 0
    assert by_name["buy"]["trades"] > 0
    assert by_name["ensemble"]["trades"] == 0  # equal weights tie -> HOLD (index 0)
    assert by_name["hold"]["delta_net_profit"] == 0.0


def test_weights_can_steer_the_ensemble() -> None:
    rows = evaluate_seed_ensemble(
        {"hold": HOLD, "buy": BUY}, frame(), Settings(), weights=[0.1, 0.9]
    )
    by_name = {row["name"]: row for row in rows}

    assert by_name["ensemble"]["trades"] > 0


def test_guard_rails() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_seed_ensemble({}, frame(), Settings())
    with pytest.raises(TypeError, match="callable"):
        evaluate_seed_ensemble({"bad": 1}, frame(), Settings())  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="match members"):
        evaluate_seed_ensemble({"hold": HOLD}, frame(), Settings(), weights=[0.5, 0.5])
