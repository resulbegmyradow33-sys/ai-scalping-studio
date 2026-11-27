"""Risk-limit ablation: baseline, enabled-only variants and deltas."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.backtest.ablation import risk_ablation
from scalping.config import RiskConfig, Settings
from scalping.features.build import FeatureFrame


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


def always_buy(_obs) -> int:
    return 1


def test_only_enabled_limits_become_variants() -> None:
    settings = Settings(
        risk=RiskConfig(max_spread_points=100.0, weekly_loss_pct=0.02)
    )

    rows = risk_ablation(always_buy, frame(), settings)

    assert [row["variant"] for row in rows] == [
        "baseline",
        "disable:daily_loss_pct",
        "disable:weekly_loss_pct",
        "disable:max_spread_points",
    ]
    assert rows[0]["delta_net_profit"] == 0.0
    assert rows[1]["field"] == "daily_loss_pct"


def test_disabling_a_binding_limit_opens_trades() -> None:
    # spread 0.20 / point 0.001 = 200 points > cap 100: baseline cannot enter.
    settings = Settings(
        risk=RiskConfig(max_spread_points=100.0, daily_loss_pct=0.03)
    )

    rows = {row["variant"]: row for row in risk_ablation(always_buy, frame(), settings)}

    assert rows["baseline"]["trades"] == 0
    assert rows["disable:max_spread_points"]["trades"] > 0
    assert rows["disable:max_spread_points"]["delta_net_profit"] != 0.0


def test_no_enabled_limit_gives_baseline_only_and_bad_input_is_rejected() -> None:
    rows = risk_ablation(always_buy, frame(), Settings(risk=RiskConfig(daily_loss_pct=1.0)))
    assert [row["variant"] for row in rows] == ["baseline"]

    with pytest.raises(TypeError, match="callable"):
        risk_ablation(1, frame(), Settings())  # type: ignore[arg-type]
