"""Reward decomposition: reconstruction, components and counterfactual totals."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.config import Settings
from scalping.features.build import FeatureFrame
from scalping.rl.reward import reward_breakdown


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


def hold(_obs) -> int:
    return 0


def always_buy(_obs) -> int:
    return 1


def test_hold_has_zero_components() -> None:
    report = reward_breakdown(hold, frame(), Settings())

    assert report["opened_steps"] == 0
    assert all(value == 0.0 for value in report["totals"].values())
    assert report["reconstruction_max_error"] == pytest.approx(0.0)
    assert set(report["variants"]) == {
        "all",
        "no_equity",
        "no_drawdown",
        "no_entry",
        "no_risk",
        "equity_only",
    }


def test_buy_steps_reconstruct_and_components_are_separated() -> None:
    report = reward_breakdown(always_buy, frame(), Settings())

    assert report["opened_steps"] > 0
    assert report["totals"]["entry"] < 0.0
    assert report["totals"]["risk"] > 0.0
    assert report["reconstruction_max_error"] < 1e-9
    assert report["variants"]["no_entry"]["mean"] != pytest.approx(
        report["variants"]["all"]["mean"]
    )
    assert report["variants"]["equity_only"]["mean"] != pytest.approx(
        report["variants"]["all"]["mean"]
    )


def test_policy_must_be_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        reward_breakdown(1, frame(), Settings())  # type: ignore[arg-type]
