"""Entry-latency execution delay: queue BUY/SELL, fill on the next close."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.backtest.engine import data_hash, run_backtest, run_id
from scalping.config import RiskConfig, Settings
from scalping.env.simulator import Simulator
from scalping.env.trading_env import TradingEnv
from scalping.features.build import FeatureFrame

BUY, SELL, CLOSE, HOLD = 1, 2, 3, 0


def frame(n: int = 300, *, slope: float = 0.0, atr: float = 0.5) -> FeatureFrame:
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    price = 1000.0 + slope * np.arange(n, dtype=np.float64)
    ohlc = np.column_stack([price, price + 0.001, price - 0.001, price])
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, atr),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def sim(frame_: FeatureFrame, *, delay: int = 1, settings: Settings | None = None) -> Simulator:
    instance = Simulator(frame_, settings or Settings(), execution_delay=delay)
    instance.reset(seed=1)
    return instance


def test_delay_zero_matches_the_historical_loop() -> None:
    script = [BUY, HOLD, CLOSE, HOLD, SELL, HOLD]
    plain = sim(frame(), delay=0)
    explicit = sim(frame(), delay=0)

    for action in script:
        left = plain.step(action)
        right = explicit.step(action)
        np.testing.assert_array_equal(left.obs, right.obs)
        assert (left.reward, left.info["equity"], left.info["blocked"]) == (
            right.reward,
            right.info["equity"],
            right.info["blocked"],
        )
        assert plain.pending_action is None


def test_entry_fills_on_the_next_bar() -> None:
    instance = sim(frame())

    decision = instance.step(BUY)

    assert decision.info["position"] is None
    assert decision.info["opened"] is None
    assert instance.pending_action == BUY

    fill = instance.step(HOLD)

    assert instance.pending_action is None
    position = fill.info["position"]
    assert position is not None
    assert fill.info["opened"] == position
    assert position.entry_ts == instance._ts(120)


def test_fill_uses_the_execution_bar_price() -> None:
    instance = sim(frame(slope=-1.0, atr=5.0))
    decision_close = float(instance.features.ohlc[119, 3])
    fill_close = float(instance.features.ohlc[120, 3])

    instance.step(BUY)
    filled = instance.step(HOLD).info["opened"]

    assert filled is not None
    expected = fill_close + 0.20 + 0.05 * 5.0
    assert filled.entry_price == pytest.approx(expected)
    assert filled.entry_price != pytest.approx(decision_close + 0.20 + 0.05 * 5.0)


def test_close_stays_immediate_and_timeout_is_not_delayed() -> None:
    settings = Settings(risk=RiskConfig(max_bars_in_trade=3))
    instance = sim(frame(), settings=settings)
    instance.step(BUY)
    instance.step(HOLD)  # fill at 120

    closed = instance.step(CLOSE)

    assert closed.info["trade"] is not None
    assert closed.info["trade"].reason == "manual"
    assert instance.pending_action is None


def test_risk_gate_runs_on_the_fill_bar_and_can_clear() -> None:
    blocked = sim(frame())
    blocked.step(BUY)
    blocked._force_equity(9_500.0)
    result = blocked.step(HOLD)
    assert result.info["blocked"] == "daily_loss_limit"
    assert result.info["position"] is None
    assert blocked.pending_action is None

    cleared = sim(frame())
    cleared._force_equity(9_500.0)
    cleared.step(BUY)  # decision while daily loss is tripped
    cleared._force_equity(10_000.0)
    filled = cleared.step(HOLD)
    assert filled.info["opened"] is not None


def test_data_end_fills_and_bills_on_the_last_bar() -> None:
    instance = Simulator(frame(122), Settings(), start=119, end=121, execution_delay=1)
    instance.reset(seed=1)
    instance.step(BUY)

    result = instance.step(HOLD)

    trade = result.info["trade"]
    assert trade is not None
    assert trade.reason == "data_end"
    assert trade.entry_ts == trade.exit_ts == instance._ts(120)
    assert result.truncated is True


def test_terminal_step_drops_the_intent_and_reset_clears_it() -> None:
    short = Simulator(frame(120), Settings(), start=119, end=120, execution_delay=1)
    short.reset(seed=1)
    result = short.step(BUY)
    assert result.truncated is True
    assert short.pending_action is None
    assert result.info["opened"] is None

    instance = sim(frame())
    instance.step(BUY)
    assert instance.pending_action == BUY
    instance.reset(seed=1)
    assert instance.pending_action is None
    assert instance.step(HOLD).info["position"] is None


def test_hold_only_rollout_is_identical_but_the_run_id_changes() -> None:
    def hold(_obs: np.ndarray) -> int:
        return HOLD

    plain = run_backtest(hold, frame(), Settings(), policy_name="hold")
    delayed = run_backtest(hold, frame(), Settings(), policy_name="hold", execution_delay=1)

    np.testing.assert_array_equal(plain.equity, delayed.equity)
    assert plain.trades == delayed.trades == []
    assert plain.run_id != delayed.run_id

    digest = data_hash(frame())
    assert run_id(Settings(), 0, digest) == run_id(Settings(), 0, digest, execution_delay=0)
    assert run_id(Settings(), 0, digest, execution_delay=1) != run_id(Settings(), 0, digest)


def test_trading_env_forwards_the_delay() -> None:
    env = TradingEnv(frame(), Settings(), execution_delay=1)
    env.reset(seed=1)

    first = env.step(BUY)
    assert first.info["position"] is None
    assert env.sim.pending_action == BUY

    second = env.step(HOLD)
    assert second.info["opened"] is not None


@pytest.mark.parametrize("bad", [-1, 2])
def test_delay_range_is_validated(bad: int) -> None:
    with pytest.raises(ValueError, match="execution_delay"):
        Simulator(frame(), Settings(), execution_delay=bad)


def test_delay_rejects_bool() -> None:
    with pytest.raises(TypeError, match="execution_delay"):
        Simulator(frame(), Settings(), execution_delay=True)
