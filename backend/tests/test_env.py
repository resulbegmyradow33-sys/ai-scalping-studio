"""Tests for the deterministic simulator and the Gymnasium wrapper.

These cover the accounting loop end to end: broker fills, risk gates, forced
exits, the observation window and the reward/termination contract.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from scalping.config import RiskConfig, Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.simulator import Simulator
from scalping.env.trading_env import TradingEnv
from scalping.features.build import WARMUP, build_features


def make_env(days=2, max_bars: int | None = None, seed: int = 5):
    df = SyntheticProvider(seed=seed).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 5 + days, tzinfo=UTC)
    )
    settings = Settings()
    if max_bars is not None:
        settings = Settings(risk=RiskConfig(max_bars_in_trade=max_bars))
    return TradingEnv(build_features(df), settings)


def test_spaces():
    env = make_env()
    assert env.action_space.n == 4
    assert env.observation_space.shape == (env.obs_dim,)


def test_reset_is_deterministic_given_seed():
    a, b = make_env(), make_env()
    assert np.allclose(a.reset(seed=1), b.reset(seed=1))


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


def test_small_max_bars_books_a_timeout_exit():
    env = make_env(max_bars=5)
    env.reset(seed=1)
    env.step(1)
    for _ in range(4):
        *_, info = env.step(0)
    assert info["position"] is None
    assert info["trade"] is not None
    assert info["trade"].reason == "timeout"
    assert env.sim.balance == pytest.approx(10_000.0 + info["trade"].pnl)


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
    env = make_env(seed=19)
    env.reset(seed=19)
    _, _, _, _, info = env.step(1)
    assert info["position"] is not None and info["trade"] is None
    for _ in range(118):
        _, _, _, _, info = env.step(0)
        assert info["trade"] is None, "position must survive to the 120-bar limit"
    _, _, _, _, info = env.step(0)
    assert info["position"] is None
    assert info["trade"] is not None
    assert info["trade"].reason == "timeout"
    assert env.sim.balance == pytest.approx(10_000.0 + info["trade"].pnl)


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


def test_force_equity_closes_an_open_position():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    env.sim._force_equity(12_345)
    assert env.sim.broker.position is None
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
