"""Action masks: which of HOLD/BUY/SELL/CLOSE the loop can execute right now.

The mask must agree with the accounting loop's own rejections: CLOSE while flat,
a second entry while a position is open, entries under a blocking flow gate and
every action after the episode is done are masked out. The evaluation helper
maps a forbidden action to HOLD instead of sending a rejected order.
"""

from __future__ import annotations

import numpy as np
import pytest

from scalping.config import Settings
from scalping.env.masks import HOLD, masked_action
from scalping.env.simulator import N_ACTIONS, WINDOW, Simulator
from scalping.env.trading_env import TradingEnv
from scalping.features.build import FeatureFrame

HOLD_, BUY, SELL, CLOSE = 0, 1, 2, 3


def flat_frame(
    n: int = 300, price: float = 1000.0, spread: float = 0.20, atr_value: float = 0.5
) -> FeatureFrame:
    """A frame whose bars never move, so stops cannot trigger and masks stay stable.

    ATR 0.5 with a 0.002 bar range keeps both stops far outside every bar, so an
    open position survives until it is closed explicitly. ``valid_from=60`` puts
    the first decision bar at ``60 + WINDOW - 1``.
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


def first_bar(frame: FeatureFrame) -> int:
    return frame.valid_from + WINDOW - 1


def flow_sim(flow: np.ndarray, threshold: float = 0.5) -> Simulator:
    frame = flat_frame(n=len(flow))
    sim = Simulator(frame, Settings(), flow=flow, flow_thr=np.full(len(flow), threshold))
    sim.reset(seed=1)
    return sim


# --- shape and dtype -------------------------------------------------------


def test_mask_has_the_action_space_shape_and_boolean_dtype():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    mask = sim.action_masks()
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == np.bool_
    assert N_ACTIONS == 4


# --- position state --------------------------------------------------------


def test_flat_mask_allows_entries_and_hides_close():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    assert sim.broker.position is None
    assert sim.action_masks().tolist() == [True, True, True, False]


def test_long_mask_allows_only_hold_and_close():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    result = sim.step(BUY)
    position = result.info["position"]
    assert position is not None and int(position.side) == 1
    assert sim.action_masks().tolist() == [True, False, False, True]


def test_short_mask_allows_only_hold_and_close():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    result = sim.step(SELL)
    position = result.info["position"]
    assert position is not None and int(position.side) == -1
    assert sim.action_masks().tolist() == [True, False, False, True]


def test_mask_returns_to_flat_after_a_close():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    sim.step(BUY)
    closed = sim.step(CLOSE)
    assert closed.info["position"] is None and closed.info["trade"] is not None
    assert sim.action_masks().tolist() == [True, True, True, False]


# --- flow gate -------------------------------------------------------------


def test_flow_gate_masks_entries_while_flat():
    sim = flow_sim(np.zeros(300))
    assert sim.action_masks().tolist() == [True, False, False, False]
    # The same bar is rejected by the loop, so mask and step agree.
    assert sim.step(BUY).info["blocked"] == "no_flow"


def test_flow_gate_that_passes_keeps_entries_available():
    sim = flow_sim(np.full(300, 2.0))
    assert sim.action_masks().tolist() == [True, True, True, False]
    assert sim.step(BUY).info["position"] is not None


def test_non_finite_flow_masks_entries():
    flow = np.full(300, 2.0)
    flow[first_bar(flat_frame())] = np.nan
    sim = flow_sim(flow)
    assert sim.action_masks().tolist() == [True, False, False, False]


def test_flow_gate_does_not_mask_close_in_a_position():
    """A weak flow bar must not hide the exit of an already open position."""
    flow = np.zeros(300)
    flow[first_bar(flat_frame())] = 2.0
    sim = flow_sim(flow)
    assert sim.action_masks().tolist() == [True, True, True, False]
    assert sim.step(BUY).info["position"] is not None
    assert sim.action_masks().tolist() == [True, False, False, True]
    assert sim.step(CLOSE).info["position"] is None


# --- terminal states -------------------------------------------------------


def test_done_mask_allows_only_hold():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    sim._force_equity(8_000.0)  # 80% of the initial balance terminates the episode
    result = sim.step(HOLD_)
    assert result.terminated
    assert sim.action_masks().tolist() == [True, False, False, False]
    # Still only HOLD after a further (ignored) step.
    sim.step(BUY)
    assert sim.action_masks().tolist() == [True, False, False, False]


def test_no_bars_mask_allows_only_hold():
    sim = Simulator(flat_frame(n=61), Settings())
    sim.reset(seed=1)
    assert sim._no_bars
    assert sim.action_masks().tolist() == [True, False, False, False]


# --- purity ----------------------------------------------------------------


def test_mask_is_a_pure_function_of_the_state():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    position = sim.broker.position
    bar, balance, equity = sim.bar_index, sim.balance, sim._equity_value
    first, second = sim.action_masks(), sim.action_masks()
    np.testing.assert_array_equal(first, second)
    assert (sim.bar_index, sim.balance, sim._equity_value) == (bar, balance, equity)
    assert sim.broker.position is position


def test_masked_out_actions_are_the_ones_step_rejects():
    flat = Simulator(flat_frame(), Settings())
    flat.reset(seed=1)
    assert not flat.action_masks()[CLOSE]
    assert flat.step(CLOSE).info["blocked"] == "no_position"

    long = Simulator(flat_frame(), Settings())
    long.reset(seed=1)
    long.step(BUY)
    mask = long.action_masks()
    assert not mask[BUY] and not mask[SELL]
    assert long.step(SELL).info["blocked"] == "max_positions"


# --- gym wrapper -----------------------------------------------------------


def test_trading_env_delegates_the_mask():
    env = TradingEnv(flat_frame(), Settings())
    env.reset(seed=1)
    assert env.action_space.n == N_ACTIONS
    np.testing.assert_array_equal(env.action_masks(), env.sim.action_masks())
    assert env.action_masks().tolist() == [True, True, True, False]
    env.step(BUY)
    assert env.action_masks().tolist() == [True, False, False, True]


def test_trading_env_mask_is_a_copy():
    env = TradingEnv(flat_frame(), Settings())
    env.reset(seed=1)
    mask = env.action_masks()
    mask[:] = False
    assert env.action_masks().tolist() == [True, True, True, False]


def test_trading_env_rejects_a_malformed_mask(monkeypatch):
    env = TradingEnv(flat_frame(), Settings())
    env.reset(seed=1)
    monkeypatch.setattr(env.sim, "action_masks", lambda: np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        env.action_masks()

    monkeypatch.setattr(env.sim, "action_masks", lambda: np.ones(N_ACTIONS, dtype=np.int8))
    with pytest.raises(TypeError, match="boolean"):
        env.action_masks()


# --- masked_action ---------------------------------------------------------


def test_masked_action_keeps_allowed_actions():
    flat = np.array([True, True, True, False])
    assert masked_action(HOLD_, flat) == HOLD_
    assert masked_action(BUY, flat) == BUY
    assert masked_action(SELL, flat) == SELL


def test_masked_action_maps_forbidden_actions_to_hold():
    flat = np.array([True, True, True, False])
    assert masked_action(CLOSE, flat) == HOLD

    in_position = np.array([True, False, False, True])
    assert masked_action(BUY, in_position) == HOLD
    assert masked_action(SELL, in_position) == HOLD
    assert masked_action(CLOSE, in_position) == CLOSE


def test_masked_action_accepts_numpy_integers():
    mask = np.array([True, False, False, True])
    assert masked_action(np.int64(CLOSE), mask) == CLOSE
    assert masked_action(np.int64(BUY), mask) == HOLD


def test_masked_action_rejects_malformed_input():
    good = np.array([True, True, True, False])
    with pytest.raises(ValueError, match="shape"):
        masked_action(HOLD_, np.array([True, True, True]))
    with pytest.raises(TypeError, match="boolean"):
        masked_action(HOLD_, good.astype(np.int8))
    with pytest.raises(ValueError, match="allow HOLD"):
        masked_action(HOLD_, np.zeros(N_ACTIONS, dtype=bool))
    with pytest.raises(ValueError, match="0..3"):
        masked_action(4, good)
    with pytest.raises(ValueError, match="0..3"):
        masked_action(-1, good)
    with pytest.raises(TypeError, match="int"):
        masked_action(1.0, good)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        masked_action(True, good)


def test_hold_constant_matches_the_action_order():
    assert HOLD == HOLD_ == 0


def test_daily_loss_blocks_entries_in_the_mask_and_the_loop():
    sim = Simulator(flat_frame(), Settings())
    sim.reset(seed=1)
    sim.day_start_equity = 10_000.0
    sim.balance = 9_000.0  # 10% loss, beyond the 3% daily limit

    assert sim.action_masks().tolist() == [True, False, False, False]
    result = sim.step(BUY)
    assert result.info["blocked"] == "daily_loss_limit"
    assert result.info["position"] is None


def test_missing_atr_blocks_entries_in_the_mask():
    sim = Simulator(flat_frame(atr_value=0.0), Settings())
    sim.reset(seed=1)

    assert sim.action_masks().tolist() == [True, False, False, False]
    assert sim.step(BUY).info["blocked"] == "no_atr"


def test_lot_below_contract_minimum_is_masked_out():
    from scalping.config import ContractSpec

    settings = Settings(contract=ContractSpec(min_lot=10.0))
    sim = Simulator(flat_frame(), settings)
    sim.reset(seed=1)

    assert sim.action_masks().tolist() == [True, False, False, False]
    assert sim.step(BUY).info["blocked"] == "lot_too_small"


def test_session_close_is_reflected_in_the_mask():
    from scalping.data.registry import session_for

    frame = flat_frame()
    weekend = np.datetime64("2026-01-03T00:00", "us")  # Saturday
    frame = FeatureFrame(
        X=frame.X,
        ts=(weekend + np.arange(frame.ts.size)).astype("datetime64[m]").astype("datetime64[us]"),
        ohlc=frame.ohlc,
        spread=frame.spread,
        atr=frame.atr,
        valid_from=frame.valid_from,
        columns=frame.columns,
    )
    sim = Simulator(frame, Settings(), session=session_for("XAUUSD"))
    sim.reset(seed=1)
    assert sim.action_masks().tolist() == [True, False, False, False]
    assert sim.step(BUY).info["blocked"] == "session_closed"


def test_mask_never_hides_an_action_the_loop_accepts():
    for action in (HOLD_, BUY, SELL, CLOSE):
        sim = Simulator(flat_frame(), Settings())
        sim.reset(seed=1)
        allowed = bool(sim.action_masks()[action])
        result = sim.step(action)
        if allowed:
            assert result.info["blocked"] is None
        else:
            assert result.info["blocked"] is not None


def test_mask_never_allows_a_rejected_entry():
    blocked = Simulator(flat_frame(atr_value=0.0), Settings())
    blocked.reset(seed=1)
    mask = blocked.action_masks()
    assert not mask[BUY] and not mask[SELL]

    for action in (BUY, SELL, CLOSE):
        if not mask[action]:
            assert blocked.step(action).info["blocked"] is not None


def test_masked_evaluation_is_part_of_the_run_identity():
    from scalping.backtest.engine import data_hash, run_backtest, run_id

    frame = flat_frame()

    def always_buy(_obs: np.ndarray) -> int:
        return BUY

    plain = run_backtest(always_buy, frame, Settings(), policy_name="always_buy")
    masked = run_backtest(
        always_buy, frame, Settings(), policy_name="always_buy", mask_actions=True
    )

    assert plain.run_id != masked.run_id
    np.testing.assert_array_equal(plain.equity, masked.equity)
    assert len(plain.trades) == len(masked.trades)
    assert run_id(Settings(), 0, data_hash(frame)) != run_id(
        Settings(), 0, data_hash(frame), mask_actions=True
    )
