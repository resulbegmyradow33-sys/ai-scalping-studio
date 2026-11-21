"""Tests for the deterministic simulator and the Gymnasium wrapper.

These cover the accounting loop end to end: broker fills, risk gates, forced
exits, data-end billing, the observation window and the reward contract.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from scalping.config import RiskConfig, Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.simulator import Simulator
from scalping.env.trading_env import TradingEnv
from scalping.features.build import WARMUP, FeatureFrame, build_features


def make_env(days=2, max_bars: int | None = None, seed: int = 5):
    df = SyntheticProvider(seed=seed).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 5 + days, tzinfo=UTC)
    )
    settings = Settings()
    if max_bars is not None:
        settings = Settings(risk=RiskConfig(max_bars_in_trade=max_bars))
    return TradingEnv(build_features(df), settings)


def flat_frame(n: int = 300, price: float = 1000.0, spread: float = 0.20, atr_value: float = 0.5):
    """Synthetic frame whose bars never move, so no SL/TP can ever trigger.

    ATR 0.5 with a 0.002 bar range keeps both stops far outside every bar while
    still sizing a full lot, which makes the timeout and reward arithmetic
    independent of any price path or seed.
    """
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
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
        spread=np.full(n, spread),
        atr=np.full(n, atr_value),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def trend_frame(n: int = 300, slope: float = -1.0, start: float = 1000.0):
    """Frame with a constant per-bar price drift and stops too far to trigger."""
    price = start + slope * np.arange(n, dtype=np.float64)
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    ohlc = np.column_stack([price, price + 0.001, price - 0.001, price])
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, 5.0),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def test_spaces():
    env = make_env()
    assert env.action_space.n == 4
    assert env.observation_space.shape == (env.obs_dim,)


def test_reset_is_deterministic_given_seed():
    a, b = make_env(), make_env()
    assert np.allclose(a.reset(seed=1)[0], b.reset(seed=1)[0])


def test_reset_returns_obs_and_info():
    env = make_env()
    obs, info = env.reset(seed=1)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.obs_dim,)
    assert isinstance(info, dict)


def test_hold_does_not_trade():
    env = make_env()
    env.reset(seed=1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(0)
        assert info["position"] is None
        assert info["equity"] == pytest.approx(10_000.0)
        if term or trunc:
            break


def test_buy_opens_and_reward_is_finite():
    env = make_env()
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(1)
    assert info["position"] is not None
    assert not term and not trunc
    assert np.isfinite(r) and -5 <= r <= 5
    assert obs.shape == (env.obs_dim,)


def test_force_close_at_max_bars():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    for _ in range(121):
        _, _, _, _, info = env.step(0)
    assert info["position"] is None


def test_short_data_does_not_crash():
    df = SyntheticProvider(seed=5).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 5, 1, tzinfo=UTC)
    )
    env = TradingEnv(build_features(df), Settings())
    obs, _ = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,)


def test_no_bars_truncates_with_a_zero_observation():
    sim = Simulator(flat_frame(n=61), Settings())
    obs = sim.reset(seed=1)
    assert not obs.any()
    result = sim.step(0)
    assert result.truncated and not result.terminated
    assert result.info["reason"] == "no_bars"
    assert not result.obs.any()
    assert result.reward == 0.0


def test_equity_drop_terminates_episode():
    env = make_env()
    env.reset(seed=1)
    env.sim._force_equity(8_000)  # test hook on the Simulator
    _, _, term, _, _ = env.step(0)
    assert term


# --- risk gate, observation contract and accounting ------------------------


def test_buy_blocked_by_daily_loss_limit():
    env = make_env()
    env.reset(seed=1)
    env.sim._force_equity(9_500.0)  # 5% below the 10k day anchor, limit is 3%
    _, r, term, trunc, info = env.step(1)
    assert info["blocked"] == "daily_loss_limit"
    assert info["position"] is None
    assert np.isfinite(r)
    assert not term and not trunc


def test_observation_is_the_copied_feature_window():
    env = make_env()
    obs, _ = env.reset(seed=1)
    frame = env.sim.features
    index = env.sim.bar_index
    width = frame.X.shape[1]
    start = index - WARMUP + 1
    assert start >= frame.valid_from
    np.testing.assert_array_equal(obs[: WARMUP * width], frame.X[start : index + 1].reshape(-1))
    assert obs.dtype == np.float32
    assert not np.shares_memory(obs, frame.X), "observation must copy the window"


def test_observation_slides_one_bar_per_step():
    env = make_env()
    first, _ = env.reset(seed=1)
    width = env.sim.features.X.shape[1]
    second, *_ = env.step(0)
    np.testing.assert_array_equal(
        first[width : width + (WARMUP - 1) * width],
        second[: (WARMUP - 1) * width],
    )


def test_position_features_reflect_the_open_trade():
    env = make_env()
    env.reset(seed=1)
    obs, *_ = env.step(1)
    side, bars, unrealized, day_pnl, risk_used = obs[-5:]
    assert side == 1.0
    assert 0 <= bars <= 1
    assert 0 < risk_used < 0.1  # 0.5% of equity at risk, plus fill noise
    assert abs(day_pnl) < 0.01
    assert np.isfinite(unrealized)


def test_equity_invariant_while_a_position_is_open():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    checked = 0
    for _ in range(6):
        *_, info = env.step(0)
        if info["position"] is None:
            break
        account = env.sim.broker.state(env.sim.balance, env.sim.day_start_equity)
        assert abs(info["equity"] - account.equity) < 1e-9
        checked += 1
    assert checked >= 3, "need at least three steps with an open position"


def test_broker_atr_tracks_the_current_bar():
    env = make_env()
    env.reset(seed=1)
    assert env.sim.broker.atr == pytest.approx(env.sim.features.atr[env.sim.bar_index])
    env.step(0)
    assert env.sim.broker.atr == pytest.approx(env.sim.features.atr[env.sim.bar_index])


def test_non_finite_feature_raises():
    frame = flat_frame()
    frame.X[120, 0] = np.nan
    sim = Simulator(frame, Settings())
    sim.reset(seed=1)
    with pytest.raises(ValueError, match="non-finite"):
        sim.step(0)


def test_step_obs_is_isolated_from_env_state():
    env = make_env()
    env.reset(seed=1)
    obs, *_ = env.step(0)
    assert not np.shares_memory(obs, env._obs)
    obs[:] = np.nan
    assert np.isfinite(env._obs).all()


def test_closed_trade_pnl_lands_in_the_balance():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    for _ in range(200):
        *_, info = env.step(0)
        if info["trade"] is not None:
            assert env.sim.balance == pytest.approx(10_000.0 + info["trade"].pnl)
            assert info["equity"] == pytest.approx(env.sim.balance)
            assert info["position"] is None
            return
    pytest.fail("no trade closed within 200 bars")


def test_sell_entry_side_and_pnl_sign():
    frame = trend_frame(slope=-1.0)
    sim = Simulator(frame, Settings())
    sim.reset(seed=1)
    start = sim.bar_index
    obs, _, _, _, info = sim.step(2)
    position = info["position"]
    assert position is not None
    assert position.side == -1
    assert obs[-5] == -1.0, "short position must show side -1 in the observation"
    slippage = Settings().costs.slippage_atr_frac * frame.atr[start]
    assert position.entry_price == pytest.approx(frame.ohlc[start, 3] - slippage)
    assert position.tp < position.entry_price < position.sl
    _, _, _, _, info = sim.step(3)
    trade = info["trade"]
    assert trade is not None and trade.reason == "manual"
    assert trade.exit == pytest.approx(frame.ohlc[start + 1, 3] + frame.spread[start + 1])
    gross = (trade.entry - trade.exit) * 100.0 * trade.lot
    assert trade.pnl == pytest.approx(gross - Settings().costs.commission_per_lot * trade.lot)
    assert trade.exit < trade.entry
    assert trade.pnl > 0, "a short must profit when price falls by more than costs"
    assert sim.balance == pytest.approx(10_000.0 + trade.pnl)


def test_reward_arithmetic_for_an_opened_step():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    _, reward, _, _, info = sim.step(1)
    assert info["equity"] == pytest.approx(9_977.5)  # entry 1000.225, one lot
    # -0.225 delta, no drawdown penalty, -0.05 for opening, +0.02 risk bonus.
    assert reward == pytest.approx(-0.225 - 0.05 + 0.02)
    _, held, _, _, info = sim.step(0)
    assert info["position"] is not None
    assert held == pytest.approx(0.02), "flat price: only the risk bonus applies"


def test_small_max_bars_books_a_timeout_exit():
    sim = Simulator(flat_frame(), Settings(risk=RiskConfig(max_bars_in_trade=5)))
    sim.reset(seed=1)
    sim.step(1)
    trade = None
    closed_at = None
    for k in range(1, 10):
        _, _, _, _, info = sim.step(0)
        if info["trade"] is not None:
            trade = info["trade"]
            closed_at = k
            break
    assert closed_at == 4, "bars_in_trade starts at 1 on the entry step"
    assert trade is not None and trade.reason == "timeout"
    assert sim.balance == pytest.approx(10_000.0 + trade.pnl)


def test_buy_while_in_position_is_blocked():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    _, _, _, _, info = env.step(1)
    assert info["blocked"] == "max_positions"
    assert info["position"] is not None


def test_close_while_flat_is_blocked():
    env = make_env()
    env.reset(seed=1)
    _, _, _, _, info = env.step(3)
    assert info["blocked"] == "no_position"
    assert info["position"] is None


def test_force_close_reason_is_timeout_at_120_bars():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    _, _, _, _, info = sim.step(1)
    assert info["position"] is not None and info["trade"] is None
    trade = None
    closed_at = None
    for k in range(1, 121):
        _, _, _, _, info = sim.step(0)
        if info["trade"] is not None:
            trade = info["trade"]
            closed_at = k
            break
    assert closed_at == 119, "bars_in_trade reaches 120 after 119 holds"
    assert trade is not None and trade.reason == "timeout"
    assert sim.balance == pytest.approx(10_000.0 + trade.pnl)


def test_session_roll_moves_the_day_anchor():
    env = make_env()
    env.reset(seed=1)
    sim = env.sim
    sim._force_equity(9_900.0)
    assert sim.session == date(2026, 1, 4)  # 01:59 UTC anchors to the prior day
    assert sim.day_start_equity == pytest.approx(10_000.0)
    while sim.session == date(2026, 1, 4):
        sim.step(0)
    assert sim.session == date(2026, 1, 5)
    assert sim.day_start_equity == pytest.approx(9_900.0)


def test_data_end_truncates():
    env = make_env()
    sim = Simulator(
        env.sim.features, Settings(), start=env.sim.bar_index, end=env.sim.bar_index + 3
    )
    sim.reset(seed=1)
    for _ in range(10):
        result = sim.step(0)
        if result.truncated:
            break
    assert result.truncated and not result.terminated
    assert result.info["reason"] == "data_end"


def test_data_end_bills_a_position_entered_on_the_last_bar():
    sim = Simulator(flat_frame(), Settings(), start=119, end=121)
    sim.reset(seed=1)
    sim.step(0)  # move to the last bar
    result = sim.step(1)  # enter on the last bar
    assert result.truncated and result.info["reason"] == "data_end"
    assert result.info["position"] is None
    assert result.info["trade"] is not None
    assert result.info["trade"].reason == "data_end"
    assert sim.balance == pytest.approx(result.info["equity"])


def test_data_end_bills_a_position_held_into_the_last_bar():
    sim = Simulator(flat_frame(), Settings(), start=119, end=122)
    sim.reset(seed=1)
    sim.step(1)  # open on the first bar
    sim.step(0)  # hold into the last bar
    result = sim.step(0)  # last bar: no next bar to price the exit
    assert result.truncated and result.info["reason"] == "data_end"
    assert result.info["position"] is None
    assert result.info["trade"] is not None
    assert result.info["trade"].reason == "data_end"
    assert sim.balance == pytest.approx(result.info["equity"])


def test_manual_close_books_the_trade():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    _, _, _, _, info = env.step(3)
    assert info["position"] is None
    assert info["trade"] is not None and info["trade"].reason == "manual"
    assert env.sim.balance == pytest.approx(10_000.0 + info["trade"].pnl)


def test_lot_too_small_is_blocked():
    df = SyntheticProvider(seed=5).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 7, tzinfo=UTC)
    )
    env = TradingEnv(build_features(df), Settings(risk=RiskConfig(risk_pct=1e-9)))
    env.reset(seed=1)
    _, _, _, _, info = env.step(1)
    assert info["blocked"] == "lot_too_small"
    assert info["position"] is None


def test_invalid_action_is_rejected():
    env = make_env()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="action must be"):
        env.step(4)


def test_step_after_done_is_idempotent():
    env = make_env()
    env.reset(seed=1)
    env.sim._force_equity(8_000)
    _, _, term, trunc, _ = env.step(0)
    assert term and not trunc
    obs, reward, term2, trunc2, info = env.step(0)
    assert term2 and not trunc2 and reward == 0.0
    assert obs.shape == (env.obs_dim,)
    assert info["reason"] == "equity_depleted"


def test_force_equity_requires_a_flat_account():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    with pytest.raises(RuntimeError, match="flat account"):
        env.sim._force_equity(12_345)
    assert env.sim.broker.position is not None

    env.step(3)  # close first, then the hook is allowed
    assert env.sim.broker.position is None
    env.sim._force_equity(12_345)
    _, _, _, _, info = env.step(0)
    assert info["equity"] == pytest.approx(12_345.0)
    assert info["position"] is None


def test_strategy_selector_labels_the_entry():
    df = SyntheticProvider(seed=5).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 7, tzinfo=UTC)
    )
    env = TradingEnv(build_features(df), Settings(), strategy_selector=lambda obs: "breakout")
    env.reset(seed=1)
    _, _, _, _, info = env.step(1)
    assert info["strategy"] == "breakout"
    assert info["position"] is not None
    assert info["position"].strategy == "breakout"
