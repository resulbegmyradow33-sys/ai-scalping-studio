"""Offline counterfactual labeling and bandit context tests.

The load bearing properties: every label is produced by the same
:class:`~scalping.env.broker.PaperBroker` the backtest uses, resolves inside
the data window, and the context matrix is strictly causal.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.features.indicators import atr
from scalping.strategy import labeling
from scalping.strategy.labeling import CONTEXT_DIM, context_matrix, label_events
from scalping.strategy.signals import strategy_signals


def data(days=3):
    return SyntheticProvider(seed=21).fetch(
        "XAUUSD",
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 5 + days, tzinfo=UTC),
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
    flat = data(1).with_columns(
        [
            pl.lit(2350.0).alias("open"),
            pl.lit(2350.0).alias("close"),
            pl.lit(2350.01).alias("high"),
            pl.lit(2349.99).alias("low"),
        ]
    )
    out = label_events(flat, Settings())
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
    flat = data(1).with_columns(
        [
            pl.lit(2350.0).alias("open"),
            pl.lit(2350.0).alias("close"),
            pl.lit(2350.01).alias("high"),
            pl.lit(2349.99).alias("low"),
        ]
    )
    out = label_events(flat, Settings())
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


def test_entry_is_the_next_bar_quote_plus_slippage():
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
        entry_index = index + 1
        quote_column = "ask_close" if row["side"] == 1 else "bid_close"
        quote = frame[quote_column][entry_index]
        expected = quote + row["side"] * settings.costs.slippage_atr_frac * atr14[index]
        assert row["entry"] == pytest.approx(expected, rel=0.0, abs=1e-9)
        checked += 1
    assert checked > 0


def test_every_label_resolves_inside_the_data():
    df = data(1)
    out = label_events(df, Settings())
    frame = df.sort("ts")
    positions = {stamp: index for index, stamp in enumerate(frame["ts"].to_list())}
    last = frame.height - 1
    for row in out.iter_rows(named=True):
        assert row["bars"] >= 1
        # Entry bar is the signal bar plus one; the exit bar is ``bars`` later.
        assert positions[row["ts"]] + 1 + row["bars"] <= last


def test_labels_are_independent_of_the_frame_length():
    settings = Settings()
    df = data(2)
    full = label_events(df, settings)
    short = label_events(df.head(700), settings)
    positions = {stamp: index for index, stamp in enumerate(df.sort("ts")["ts"].to_list())}

    short_map = {(row["ts"], row["strategy"]): row for row in short.iter_rows(named=True)}
    full_map = {(row["ts"], row["strategy"]): row for row in full.iter_rows(named=True)}
    resolved = [
        row for row in full.iter_rows(named=True) if positions[row["ts"]] + 1 + row["bars"] < 700
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


def test_context_matrix_of_an_empty_frame_is_empty():
    f = build_features(data(1).head(0))
    assert context_matrix(f).shape == (0, CONTEXT_DIM)


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
