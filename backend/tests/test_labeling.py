"""Offline counterfactual labeling and bandit context tests.

The load bearing properties: every label is priced by the same
:class:`~scalping.env.broker.PaperBroker` the environment uses, with the
environment's timing (signal bar's close quote), resolves inside the data
window, and the context matrix is strictly causal.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.data.base import conform_bars
from scalping.data.synthetic import SyntheticProvider
from scalping.env.simulator import Simulator
from scalping.features.build import build_features
from scalping.features.indicators import atr
from scalping.features.prepare import prepare_bars
from scalping.strategy import labeling
from scalping.strategy.labeling import CONTEXT_DIM, context_matrix, label_events
from scalping.strategy.signals import ALL_STRATEGIES, strategy_signals


def data(days=3):
    return SyntheticProvider(seed=21).fetch(
        "XAUUSD",
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 5 + days, tzinfo=UTC),
    )


def flat_frame(days=1):
    return data(days).with_columns(
        [
            pl.lit(2350.0).alias("open"),
            pl.lit(2350.0).alias("close"),
            pl.lit(2350.01).alias("high"),
            pl.lit(2349.99).alias("low"),
        ]
    )


def test_labels_have_required_columns_and_values():
    out = label_events(data(), Settings())
    assert {"ts", "strategy", "side", "entry", "exit", "bars", "r_multiple", "reason"} <= set(
        out.columns
    )
    assert out.height > 0
    assert out["bars"].max() <= 120
    assert set(out["reason"].unique().to_list()) <= {"sl", "tp", "timeout"}


def test_r_multiple_sign_matches_outcome():
    out = label_events(data(), Settings())
    for row in out.head(50).iter_rows(named=True):
        gain = (row["exit"] - row["entry"]) * row["side"]
        assert (row["r_multiple"] > 0) == (gain > 0)


def test_no_signals_returns_empty_frame():
    out = label_events(flat_frame(1), Settings())
    assert out.height == 0
    assert "r_multiple" in out.columns


def test_context_matrix_shape_and_finiteness():
    f = build_features(data())
    X = context_matrix(f)
    assert X.shape == (f.X.shape[0], CONTEXT_DIM)
    assert np.isfinite(X).all()


def test_context_is_causal():
    df = data()
    f = build_features(df)
    full = context_matrix(f)
    part = context_matrix(build_features(df.head(400)))
    assert np.allclose(full[:400], part, equal_nan=True)


def test_no_signals_keeps_the_declared_schema():
    out = label_events(flat_frame(1), Settings())
    assert out.columns == [
        "ts",
        "strategy",
        "side",
        "entry",
        "exit",
        "bars",
        "r_multiple",
        "reason",
    ]
    assert out.schema["ts"] == pl.Datetime("us", "UTC")
    assert out.schema["strategy"] == pl.String
    assert out.schema["side"] == pl.Int8
    assert out.schema["entry"] == pl.Float64
    assert out.schema["exit"] == pl.Float64
    assert out.schema["bars"] == pl.Int32
    assert out.schema["r_multiple"] == pl.Float64
    assert out.schema["reason"] == pl.String


def test_entry_is_the_signal_bar_quote_plus_slippage():
    settings = Settings()
    df = data(1)
    out = label_events(df, settings)
    frame = df.sort("ts")
    signals = strategy_signals(frame)
    atr14 = atr(frame, 14).to_numpy()
    positions = {stamp: index for index, stamp in enumerate(frame["ts"].to_list())}

    checked = 0
    for row in out.iter_rows(named=True):
        index = positions[row["ts"]]
        assert row["side"] in (-1, 1)
        assert signals[f"sig_{row['strategy']}"][index] == row["side"]
        quote_column = "ask_close" if row["side"] == 1 else "bid_close"
        quote = frame[quote_column][index]
        expected = quote + row["side"] * settings.costs.slippage_atr_frac * atr14[index]
        assert row["entry"] == pytest.approx(expected, rel=0.0, abs=1e-9)
        checked += 1
    assert checked > 0


def test_label_matches_a_one_step_simulator_run():
    settings = Settings()
    df = data(1)
    features = build_features(df)
    out = label_events(df, settings)
    frame = df.sort("ts")
    positions = {stamp: index for index, stamp in enumerate(frame["ts"].to_list())}
    signals = strategy_signals(frame)

    # The simulator's first decision bar is max(start, valid_from + WINDOW - 1).
    for row in out.iter_rows(named=True):
        index = positions[row["ts"]]
        if index < 119:
            continue
        if signals[f"sig_{row['strategy']}"][index] != row["side"]:
            continue
        break
    else:  # pragma: no cover - the fixed fixture always has one
        raise AssertionError("no label after the simulator's first decision bar")

    sim = Simulator(features, settings, start=index)
    sim.reset()
    result = sim.step(1 if row["side"] == 1 else 2)
    opened = result.info["opened"]
    assert opened is not None
    assert opened.entry_price == pytest.approx(row["entry"], rel=0.0, abs=1e-12)

    trade = result.info["trade"]
    steps = 0
    while trade is None:
        steps += 1
        assert steps <= 120
        trade = sim.step(0).info["trade"]
    assert trade.exit == pytest.approx(row["exit"], rel=0.0, abs=1e-12)
    assert trade.reason == row["reason"]
    assert trade.r_multiple == pytest.approx(row["r_multiple"], rel=0.0, abs=1e-12)


def test_every_label_resolves_inside_the_data():
    df = data(1)
    out = label_events(df, Settings())
    frame = df.sort("ts")
    positions = {stamp: index for index, stamp in enumerate(frame["ts"].to_list())}
    last = frame.height - 1
    for row in out.iter_rows(named=True):
        assert row["bars"] >= 1
        # The signal bar is the entry bar; the exit bar is ``bars`` later.
        assert positions[row["ts"]] + row["bars"] <= last


def test_labels_are_independent_of_the_frame_length():
    settings = Settings()
    df = data(2)
    full = label_events(df, settings)
    short = label_events(df.head(700), settings)
    positions = {stamp: index for index, stamp in enumerate(df.sort("ts")["ts"].to_list())}

    short_map = {(row["ts"], row["strategy"]): row for row in short.iter_rows(named=True)}
    full_map = {(row["ts"], row["strategy"]): row for row in full.iter_rows(named=True)}
    resolved = [
        row for row in full.iter_rows(named=True) if positions[row["ts"]] + row["bars"] < 700
    ]
    assert short_map
    assert len(resolved) > 0
    # A longer frame never invents a label that the prefix already resolved.
    assert set(short_map) <= set(full_map)
    for row in resolved:
        other = short_map[(row["ts"], row["strategy"])]
        assert other["entry"] == pytest.approx(row["entry"])
        assert other["exit"] == pytest.approx(row["exit"])
        assert other["r_multiple"] == pytest.approx(row["r_multiple"])
        assert other["bars"] == row["bars"]
        assert other["reason"] == row["reason"]


def test_missing_quote_columns_raise():
    with pytest.raises(ValueError, match="ask_close"):
        label_events(data(1).drop("ask_close"), Settings())


def test_null_quotes_are_filled_by_the_spread_model():
    settings = Settings()
    unquoted = data(1).with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias("ask_close"),
            pl.lit(None, dtype=pl.Float64).alias("spread"),
        ]
    )
    out = label_events(unquoted, settings)
    assert out.height > 0
    assert np.isfinite(out["entry"].to_numpy()).all()
    assert np.isfinite(out["exit"].to_numpy()).all()

    frame = conform_bars(unquoted).sort("ts")
    atr14 = atr(frame, 14).to_numpy()
    filled = prepare_bars(frame, settings.costs, atr14)
    assert filled["spread"].min() >= settings.costs.spread_floor
    positions = {stamp: index for index, stamp in enumerate(filled["ts"].to_list())}
    for row in out.iter_rows(named=True):
        index = positions[row["ts"]]
        slippage = settings.costs.slippage_atr_frac * atr14[index]
        quote_column = "ask_close" if row["side"] == 1 else "bid_close"
        expected = filled[quote_column][index] + row["side"] * slippage
        assert row["entry"] == pytest.approx(expected, rel=0.0, abs=1e-9)


def test_golden_case_matches_a_hand_computed_trade():
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    rows = [
        (100.0, 100.5, 99.5, 100.0),
        (100.5, 101.5, 100.5, 101.0),
        (102.5, 103.5, 102.5, 103.0),
        (102.0, 102.2, 101.0, 101.5),
        (104.2, 105.5, 104.0, 105.0),
        (105.0, 105.5, 104.5, 105.2),
    ]
    frame = pl.DataFrame(
        {
            "ts": [start + timedelta(minutes=index) for index in range(len(rows))],
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "bid_close": [row[3] for row in rows],
            "ask_close": [row[3] + 0.2 for row in rows],
            "spread": [0.2] * len(rows),
            "volume": [1] * len(rows),
        }
    )

    signal_ts = frame["ts"][3]
    out = label_events(frame, Settings())
    row = out.filter((pl.col("strategy") == "fvg") & (pl.col("ts") == signal_ts)).row(0, named=True)

    # By hand: bars 0..3 have true ranges 1.0, 1.5, 2.5 and 2.0 (ATR(14) with
    # fewer than 14 samples is the mean of what exists).
    atr_value = (1.0 + 1.5 + 2.5 + 2.0) / 4.0
    entry = 101.5 + 0.2 + 0.05 * atr_value  # bar 3 ask plus ATR slippage
    sl = entry - atr_value
    tp = entry + 2.0 * atr_value
    risk_lot = 10_000.0 * 0.005 / (atr_value * 100.0)
    cap_lot = 10_000.0 * 20.0 / (entry * 100.0)
    lot = math.floor(min(risk_lot, cap_lot) / 0.01 + 1e-9) * 0.01
    commission = 7.0 * lot
    r_multiple = ((tp - entry) * 100.0 * lot - commission) / (abs(entry - sl) * 100.0 * lot)

    assert row["side"] == 1
    assert row["entry"] == pytest.approx(entry, rel=0.0, abs=1e-9)
    assert row["exit"] == pytest.approx(tp, rel=0.0, abs=1e-9)
    assert row["bars"] == 1
    assert row["reason"] == "tp"
    assert row["r_multiple"] == pytest.approx(r_multiple, rel=0.0, abs=1e-9)
    assert row["r_multiple"] == pytest.approx(1.96, rel=0.0, abs=1e-9)


def test_context_matrix_of_an_empty_frame_is_empty():
    f = build_features(data(1).head(0))
    assert context_matrix(f).shape == (0, CONTEXT_DIM)


def test_context_success_is_neutral_before_any_exit():
    X = context_matrix(build_features(data(1)))
    assert np.allclose(X[0, -len(ALL_STRATEGIES) :], 0.5)


def test_context_with_no_labels_keeps_the_success_columns_neutral():
    f = build_features(data(1))
    empty = label_events(flat_frame(1), Settings())
    X = context_matrix(f, labels=empty)
    assert np.allclose(X[:, -len(ALL_STRATEGIES) :], 0.5)


def test_context_uses_precomputed_labels_when_given():
    df = data(1)
    features = build_features(df)
    labels = label_events(df, Settings())
    assert np.array_equal(context_matrix(features), context_matrix(features, labels=labels))


def test_success_rate_covers_the_trailing_240_bars():
    df = data(1)
    features = build_features(df)
    labels = label_events(df, Settings())
    X = context_matrix(features, labels=labels)
    positions = {stamp: index for index, stamp in enumerate(df.sort("ts")["ts"].to_list())}
    for moment in (300, 900):
        for arm, strategy in enumerate(ALL_STRATEGIES):
            exits = [
                row
                for row in labels.iter_rows(named=True)
                if row["strategy"] == strategy.value
                and moment - 239 <= positions[row["ts"]] + row["bars"] <= moment
            ]
            expected = (
                sum(1 for row in exits if row["r_multiple"] > 0) / len(exits) if exits else 0.5
            )
            assert X[moment, CONTEXT_DIM - len(ALL_STRATEGIES) + arm] == pytest.approx(
                expected, rel=0.0, abs=1e-12
            )


def test_atr_column_is_a_causal_z_score():
    features = build_features(data(1))
    X = context_matrix(features)
    values = features.X[:, features.columns.index("atr_norm")].astype(np.float64)
    moment = 500
    history = values[moment - 240 : moment]  # the 240 bars before this row
    expected = (values[moment] - history.mean()) / history.std(ddof=1)
    assert X[moment, 6] == pytest.approx(expected, rel=1e-6, abs=1e-9)
    assert abs(X[moment, 6]) > 1e-6  # standardised, not the raw ratio


def test_atr_z_score_is_zero_on_a_flat_series():
    X = context_matrix(build_features(flat_frame(1)))
    assert np.allclose(X[:, 6], 0.0)


def test_every_label_gets_a_fresh_broker(monkeypatch):
    """A label must not inherit a position, costs or bars from the one before."""
    created = []
    opened = []
    original = labeling.PaperBroker
    original_open = original.open

    def tracking(*args, **kwargs):
        broker = original(*args, **kwargs)
        created.append(broker)
        return broker

    def open_tracking(self, *args, **kwargs):
        opened.append(id(self))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(labeling, "PaperBroker", tracking)
    monkeypatch.setattr(original, "open", open_tracking)
    out = label_events(data(1), Settings())
    assert out.height > 0
    assert len({id(broker) for broker in created}) == len(created)
    # One broker per hypothetical trade; each is opened exactly once.
    assert len(opened) == len(created)
    assert len(set(opened)) == len(opened)
    assert out.height <= len(created)
