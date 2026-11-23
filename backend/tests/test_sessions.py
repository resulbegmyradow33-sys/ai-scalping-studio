"""Per-symbol sessions: parsing, boundaries and simulator enforcement."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from scalping.backtest.engine import data_hash, run_backtest, run_id
from scalping.config import Settings
from scalping.data.registry import parse_session, session_for
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame


def frame(start: str, n: int = 300, price: float = 1000.0) -> FeatureFrame:
    ts = np.datetime64(start, "us") + np.arange(n).astype("timedelta64[m]")
    ohlc = np.column_stack(
        [
            np.full(n, price),
            np.full(n, price + 0.001),
            np.full(n, price - 0.001),
            np.full(n, price),
        ]
    )
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, 0.5),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def test_weekly_window_boundaries() -> None:
    spec = parse_session("sun22:00-fri21:00 UTC")

    assert spec.is_open(datetime(2026, 1, 2, 20, 59, tzinfo=UTC))  # Friday
    assert not spec.is_open(datetime(2026, 1, 2, 21, 0, tzinfo=UTC))
    assert not spec.is_open(datetime(2026, 1, 3, 12, 0, tzinfo=UTC))  # Saturday
    assert not spec.is_open(datetime(2026, 1, 4, 21, 59, tzinfo=UTC))  # Sunday
    assert spec.is_open(datetime(2026, 1, 4, 22, 0, tzinfo=UTC))
    assert spec.is_open(datetime(2026, 1, 5, 0, 0, tzinfo=UTC))  # Monday
    # Naive timestamps are UTC, matching the rest of the data layer.
    assert not spec.is_open(datetime(2026, 1, 3, 12, 0))


def test_crypto_is_open_around_the_clock() -> None:
    btc = session_for("BTCUSD")
    assert btc.always_open is True
    assert btc.is_open(datetime(2026, 1, 3, 12, 0, tzinfo=UTC))


def test_session_lookup_is_per_symbol_and_cached() -> None:
    assert session_for("XAUUSD") is session_for("XAGUSD")  # same raw window
    assert session_for("XAUUSD") is not session_for("BTCUSD")
    assert session_for("btcusd").always_open is True


def test_bad_session_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported session"):
        parse_session("weekdays")
    with pytest.raises(ValueError, match="invalid session"):
        parse_session("mon25:00-fri21:00 UTC")


def test_simulator_blocks_entries_outside_the_session() -> None:
    weekend = frame("2026-01-03T00:00")  # Saturday

    guarded = Simulator(weekend, Settings(), session=session_for("XAUUSD"))
    guarded.reset(seed=1)
    _, _, _, _, info = guarded.step(1)
    assert info["blocked"] == "session_closed"
    assert info["position"] is None and info["opened"] is None

    free = Simulator(weekend, Settings())
    free.reset(seed=1)
    _, _, _, _, opened_info = free.step(1)
    assert opened_info["opened"] is not None


def test_backtest_uses_the_symbol_session() -> None:
    weekend = frame("2026-01-03T00:00")

    def always_buy(_obs: np.ndarray) -> int:
        return 1

    closed = run_backtest(always_buy, weekend, Settings(), symbol="XAUUSD")
    open_ = run_backtest(always_buy, weekend, Settings(), symbol="BTCUSD")

    assert closed.trades == []
    assert open_.trades  # 24x7 symbol may re-enter after a timeout exit
    assert all(trade.entry_ts.weekday() == 5 for trade in open_.trades)  # Saturday


def test_session_changes_the_run_id() -> None:
    digest = data_hash(frame("2026-01-05T00:00"))
    weekend_digest = data_hash(frame("2026-01-03T00:00"))

    assert run_id(Settings(), 0, digest, session="") != run_id(
        Settings(), 0, digest, session="24x7 UTC"
    )
    assert run_id(Settings(), 0, weekend_digest, session="24x7 UTC") != run_id(
        Settings(), 0, weekend_digest, session="sun22:00-fri21:00 UTC"
    )
