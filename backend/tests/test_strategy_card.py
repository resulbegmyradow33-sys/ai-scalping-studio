"""The flow rule, strategy card and calibration gate."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.strategy import calibrate as calibrate_module
from scalping.strategy.calibrate import calibrate, summarise
from scalping.strategy.card import (
    PolicyCard,
    RuleMetrics,
    RuleParams,
    frame_digest,
    load_card,
    save_card,
)
from scalping.strategy.rule import FlowMomentumRule


def _timestamps(n: int) -> pl.Series:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=n - 1)
    return pl.datetime_range(start, end, interval="1h", eager=True).alias("ts")


def test_summarise_counts_and_compounds() -> None:
    metrics = summarise(np.array([1.0, -1.0, 2.0, 0.0]))
    assert metrics.trades == 4
    assert metrics.expectancy_r == pytest.approx(0.5)
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(3.0)
    assert 0.0 <= metrics.max_drawdown < 1.0


def test_summarise_empty_is_neutral() -> None:
    metrics = summarise(np.array([]))
    assert metrics.trades == 0
    assert metrics.expectancy_r == 0.0
    assert metrics.max_drawdown == 0.0


def test_rule_holds_when_flow_is_not_extreme() -> None:
    frame = pl.DataFrame({"ts": _timestamps(4)})
    rule = FlowMomentumRule.build(
        frame,
        flow=np.array([0.1, 0.1, 0.1, 0.1]),
        threshold=np.array([0.5, 0.5, 0.5, 0.5]),
        signal=np.array([1, 1, -1, 0]),
    )
    assert [rule.act(index) for index in range(4)] == [0, 0, 0, 0]
    assert rule.blocked_reason(0) == "no_flow"


def test_rule_enters_in_the_signal_direction() -> None:
    frame = pl.DataFrame({"ts": _timestamps(3)})
    rule = FlowMomentumRule.build(
        frame,
        flow=np.array([0.9, -0.9, 0.9]),
        threshold=np.array([0.5, 0.5, 0.5]),
        signal=np.array([1, -1, 0]),
    )
    assert [rule.act(index) for index in range(3)] == [1, 2, 0]
    assert rule.blocked_reason(0) is None
    assert rule.blocked_reason(2) is None


def test_card_round_trip(tmp_path) -> None:
    params = RuleParams(timeframe="H1", flow_quantile=0.8, flow_window=240)
    card = PolicyCard.build(
        params,
        RuleMetrics(trades=10, expectancy_r=0.1),
        RuleMetrics(trades=4, expectancy_r=0.2),
        "abc123",
    )
    path = tmp_path / "card.json"
    save_card(path, card)
    loaded = load_card(path)
    assert loaded == card
    assert json.loads(path.read_text())["params"]["timeframe"] == "H1"


def test_card_rejects_unknown_version(tmp_path) -> None:
    path = tmp_path / "card.json"
    path.write_text(
        json.dumps(
            {
                "version": 99,
                "strategy": "flow_momentum",
                "params": {},
                "digest": "x",
                "created_at": "now",
            }
        )
    )
    with pytest.raises(ValueError):
        load_card(path)


def test_frame_digest_changes_with_data() -> None:
    frame = pl.DataFrame(
        {"ts": _timestamps(5), "close": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    other = frame.with_columns((pl.col("close") + 1).alias("close"))
    assert frame_digest(frame) != frame_digest(other)
    assert frame_digest(frame) == frame_digest(frame.clone())


def _calibration_frame(n: int = 400) -> pl.DataFrame:
    return pl.DataFrame({"ts": _timestamps(n), "close": np.linspace(2000, 2100, n)})


def _momentum(ts: pl.Series, r: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": ts,
            "r_multiple": np.full(ts.len(), r),
            "sig_momentum": np.ones(ts.len(), dtype=np.int8),
        }
    )


def test_calibrate_promotes_a_positive_holdout(monkeypatch) -> None:
    frame = _calibration_frame()
    monkeypatch.setattr(
        calibrate_module,
        "_momentum_labels",
        lambda frame, params, settings: _momentum(frame["ts"], 0.5),
    )
    flow = np.linspace(-1.0, 1.0, frame.height)
    report = calibrate(
        frame,
        flow,
        risk_options=[(3.0, 6.0, 48)],
        quantiles=[0.8],
        windows=[24],
        min_train_trades=5,
        min_holdout_trades=5,
        holdout_fraction=0.2,
        base=Settings(),
    )
    assert report.promoted is True
    assert report.candidate.holdout.expectancy_r == pytest.approx(0.5)
    assert "positive" in report.reason


def test_calibrate_rejects_a_negative_holdout(monkeypatch) -> None:
    frame = _calibration_frame()
    monkeypatch.setattr(
        calibrate_module,
        "_momentum_labels",
        lambda frame, params, settings: _momentum(frame["ts"], -0.5),
    )
    flow = np.linspace(-1.0, 1.0, frame.height)
    report = calibrate(
        frame,
        flow,
        risk_options=[(3.0, 6.0, 48)],
        quantiles=[0.8],
        windows=[24],
        min_train_trades=5,
        min_holdout_trades=5,
        holdout_fraction=0.2,
        base=Settings(),
    )
    assert report.promoted is False
    assert "not positive" in report.reason


def test_calibrate_rejects_when_incumbent_is_better(monkeypatch) -> None:
    frame = _calibration_frame()
    calls = {"n": 0}

    def labels(frame: pl.DataFrame, params: RuleParams, settings: Settings) -> pl.DataFrame:
        calls["n"] += 1
        # The first call is the candidate grid, the second the incumbent.
        return _momentum(frame["ts"], 0.1 if calls["n"] == 1 else 0.9)

    monkeypatch.setattr(calibrate_module, "_momentum_labels", labels)
    flow = np.linspace(-1.0, 1.0, frame.height)
    report = calibrate(
        frame,
        flow,
        risk_options=[(3.0, 6.0, 48)],
        quantiles=[0.8],
        windows=[24],
        min_train_trades=5,
        min_holdout_trades=5,
        holdout_fraction=0.2,
        incumbent=RuleParams(timeframe="H1"),
    )
    assert report.promoted is False
    assert "does not beat" in report.reason


def test_rule_replays_inside_the_simulator() -> None:
    from scalping.backtest.engine import run_backtest
    from scalping.features.build import FeatureFrame

    n = 300
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    price = np.full(n, 1000.0)
    frame = FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts,
        ohlc=np.column_stack((price, price + 0.001, price - 0.001, price)),
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=60,
        columns=["a", "b", "c"],
    )
    stamps = pl.DataFrame({"ts": _timestamps(n)})
    rule = FlowMomentumRule.build(
        stamps,
        flow=np.ones(n),
        threshold=np.full(n, 0.5),
        signal=np.ones(n, dtype=np.int8),
    )
    rule.reset(60 + 59)
    result = run_backtest(
        lambda obs: rule.act(),
        frame,
        Settings(),
        policy_name="flow_momentum",
        window=60,
        max_steps=n + 1,
    )
    assert result.metrics.trades >= 1
    assert result.trades[0].strategy == "flow_momentum"
