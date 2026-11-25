"""Tests for the Phase 11 risk kill switches and limits.

Every switch is exercised at its boundary (trip at the limit, not one tick
before), with its reset rule, and with the default ``0`` proving inert. The
simulator tests cover the state it owns: week anchor, peak equity, consecutive
losses, the timed pause latch and spread points.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from scalping.config import ContractSpec, RiskConfig, Settings
from scalping.env.risk import RiskEngine
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame
from scalping.types import AccountState, Position, Side

#: A session timestamp well inside the 17:00 UTC FX day (18:30 UTC).
TS = datetime(2026, 1, 5, 18, 30, tzinfo=UTC)


def engine(**kw: float) -> RiskEngine:
    return RiskEngine(RiskConfig(**kw), ContractSpec())


def st(
    equity: float = 10_000.0,
    *,
    day_start: float | None = None,
    week_start: float | None = None,
    peak: float | None = None,
    losses: int = 0,
    paused: datetime | None = None,
    spread: float | None = None,
    slippage: float | None = None,
    quality: bool = True,
    position: Position | None = None,
) -> AccountState:
    """A snapshot with the daily gate inert unless the test sets a base."""
    base = equity if day_start is None else day_start
    return AccountState(
        balance=equity,
        equity=equity,
        position=position,
        day_start_equity=base,
        day_pnl=equity - base,
        bars_in_trade=0,
        trades_closed=0,
        week_start_equity=week_start,
        peak_equity=peak,
        consecutive_losses=losses,
        paused_until=paused,
        spread_points=spread,
        slippage_points=slippage,
        data_quality_ok=quality,
    )


def long_position() -> Position:
    return Position(
        side=Side.LONG,
        lot=0.1,
        entry_price=2350.0,
        entry_ts=datetime(2026, 1, 5, 18, 0, tzinfo=UTC),
        sl=2349.0,
        tp=2352.0,
        strategy="test",
    )


def flat_frame(
    n: int = 300,
    *,
    price: float = 1000.0,
    spread: float = 0.20,
    atr_value: float = 0.5,
    start: datetime = datetime(2026, 1, 5, tzinfo=UTC),
    spreads: np.ndarray | None = None,
) -> FeatureFrame:
    """Flat bars: no SL/TP can trigger, so exits are exactly the ones asked for."""
    ts = np.datetime64(start.replace(tzinfo=None), "us") + np.arange(n).astype("timedelta64[m]")
    ohlc = np.column_stack(
        [
            np.full(n, price),
            np.full(n, price + 0.001),
            np.full(n, price - 0.001),
            np.full(n, price),
        ]
    )
    spread_arr = np.full(n, spread) if spreads is None else np.asarray(spreads, dtype=np.float64)
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=spread_arr,
        atr=np.full(n, atr_value),
        valid_from=60,
        columns=["a", "b", "c"],
    )


# --- defaults are inert ----------------------------------------------------


def test_default_limits_are_off():
    e = engine()
    # Every new field is at its extreme; with the limits at 0 none of it counts.
    account = st(
        equity=5_000.0,
        week_start=20_000.0,
        peak=20_000.0,
        losses=99,
        paused=TS + timedelta(days=1),
        spread=10_000.0,
        slippage=10_000.0,
        quality=False,
    )

    assert e.allow_new(account, TS) == (True, None)
    assert e.kill_switch_reason(account) is None


def _legacy_allow_new(account: AccountState) -> tuple[bool, str | None]:
    """The pre-Phase-11 gate, copied so the equivalence test has a reference."""
    if account.day_start_equity <= 0:
        return False, "lot_too_small"
    loss = (account.day_start_equity - account.equity) / account.day_start_equity
    if loss >= RiskConfig().daily_loss_pct:
        return False, "daily_loss_limit"
    if (1 if account.position is not None else 0) >= RiskConfig().max_positions:
        return False, "max_positions"
    return True, None


def test_default_decisions_match_the_pre_phase_11_engine():
    e = engine()
    sequence = [
        (st(), TS),
        (st(equity=9_701.0), TS),
        (st(equity=9_700.0), TS),
        (st(equity=0.0), TS),
        (st(equity=-500.0), TS),
        (st(equity=0.0, day_start=0.0), TS),
        (st(equity=100.0, day_start=-1.0), TS),
        (st(position=long_position()), TS),
        (st(equity=9_500.0, week_start=20_000.0, peak=20_000.0, losses=9, spread=9_999.0), TS),
        (st(equity=9_500.0, slippage=9_999.0, quality=False), TS),
        (st(equity=9_500.0), TS.replace(day=6)),
    ]
    for account, ts in sequence:
        expected = _legacy_allow_new(account)
        assert e.allow_new(account, ts) == expected
        assert e.allow_new_extended(account, ts) == expected


def test_zeroed_new_fields_reproduce_the_default_run():
    """An explicit all-zero config is the same run as the default config."""
    actions = [1, 0, 0, 3, 1, 0, 0, 0]
    default = Simulator(flat_frame(), Settings())
    zeroed = Simulator(
        flat_frame(),
        Settings(
            risk=RiskConfig(
                weekly_loss_pct=0.0,
                max_drawdown_pct=0.0,
                consecutive_loss_pause=0,
                max_spread_points=0.0,
                min_equity_frac=0.0,
                pause_bars=0,
                max_slippage_points=0.0,
                data_quality_block=False,
            )
        ),
    )
    default.reset(seed=1)
    zeroed.reset(seed=1)
    for action in actions:
        a = default.step(action)
        b = zeroed.step(action)
        assert a.reward == b.reward
        assert a.info["equity"] == b.info["equity"]
        assert a.info["blocked"] == b.info["blocked"]
        assert a.info["position"] == b.info["position"]
        assert a.info["trade"] == b.info["trade"]


# --- weekly loss -----------------------------------------------------------


def test_weekly_loss_trips_at_the_boundary_not_before():
    e = engine(weekly_loss_pct=0.05)
    assert e.allow_new(st(equity=9_500.0, week_start=10_000.0), TS) == (
        False,
        "weekly_loss_limit",
    )
    assert e.allow_new(st(equity=9_501.0, week_start=10_000.0), TS) == (True, None)
    assert e.kill_switch_reason(st(equity=9_500.0, week_start=10_000.0)) == "weekly_loss_limit"


def test_weekly_loss_without_an_anchor_is_inert():
    e = engine(weekly_loss_pct=0.05)
    assert e.allow_new(st(equity=1_000.0, week_start=None), TS) == (True, None)
    assert e.allow_new(st(equity=1_000.0, week_start=0.0), TS) == (True, None)


def test_week_anchor_rolls_at_the_monday_1700_utc_session():
    e = engine()
    # Jan 4 2026 is a Sunday, so the ISO week 1 session ends at Monday 17:00.
    assert e.week_anchor(datetime(2026, 1, 5, 16, 59, tzinfo=UTC)) == (2026, 1)
    assert e.week_anchor(datetime(2026, 1, 5, 17, 0, tzinfo=UTC)) == (2026, 2)
    assert e.week_anchor(datetime(2026, 1, 6, 0, 30, tzinfo=UTC)) == (2026, 2)


# --- drawdown --------------------------------------------------------------


def test_drawdown_trips_at_the_boundary_not_before():
    e = engine(max_drawdown_pct=0.10)
    assert e.allow_new(st(equity=9_000.0, peak=10_000.0), TS) == (False, "max_drawdown")
    assert e.allow_new(st(equity=9_001.0, peak=10_000.0), TS) == (True, None)
    assert e.kill_switch_reason(st(equity=8_999.0, peak=10_000.0)) == "max_drawdown"


def test_drawdown_without_a_peak_is_inert():
    e = engine(max_drawdown_pct=0.10)
    assert e.allow_new(st(equity=1_000.0, peak=None), TS) == (True, None)
    assert e.allow_new(st(equity=1_000.0, peak=0.0), TS) == (True, None)


# --- emergency equity floor ------------------------------------------------


def test_emergency_stop_is_strictly_below_the_fraction():
    e = engine(min_equity_frac=0.8)
    # 80% of the default 10,000 initial balance is the exact floor.
    assert e.allow_new(st(equity=8_000.0), TS) == (True, None)
    assert e.allow_new(st(equity=7_999.99), TS) == (False, "emergency_stop")
    assert e.kill_switch_reason(st(equity=7_999.99)) == "emergency_stop"


# --- consecutive losses ----------------------------------------------------


def test_consecutive_loss_pause_trips_at_the_count():
    e = engine(consecutive_loss_pause=3)
    assert e.allow_new(st(losses=2), TS) == (True, None)
    assert e.allow_new(st(losses=3), TS) == (False, "consecutive_loss_pause")


# --- spread ----------------------------------------------------------------


def test_spread_limit_trips_only_above_the_cap():
    e = engine(max_spread_points=20.0)
    assert e.allow_new(st(spread=20.0), TS) == (True, None)
    assert e.allow_new(st(spread=20.01), TS) == (False, "max_spread")
    assert e.allow_new(st(spread=None), TS) == (True, None)


# --- slippage --------------------------------------------------------------


def test_slippage_limit_trips_only_above_the_cap():
    e = engine(max_slippage_points=25.0)
    assert e.allow_new(st(slippage=25.0), TS) == (True, None)
    assert e.allow_new(st(slippage=25.01), TS) == (False, "slippage_limit")
    assert e.allow_new(st(slippage=0.0), TS) == (True, None)


def test_slippage_limit_without_an_estimate_is_inert():
    e = engine(max_slippage_points=1.0)
    assert e.allow_new(st(slippage=None), TS) == (True, None)


def test_slippage_limit_is_off_at_zero():
    e = engine()
    assert e.allow_new(st(slippage=1_000_000.0), TS) == (True, None)


# --- data quality ----------------------------------------------------------


def test_data_quality_block_only_when_enabled():
    enabled = engine(data_quality_block=True)
    assert enabled.allow_new(st(quality=True), TS) == (True, None)
    assert enabled.allow_new(st(quality=False), TS) == (False, "data_quality_block")

    disabled = engine()
    assert disabled.allow_new(st(quality=False), TS) == (True, None)


def test_data_quality_block_never_touches_the_equity_kill_switches():
    e = engine(data_quality_block=True)
    assert e.kill_switch_reason(st(quality=False)) is None
    equity = engine(data_quality_block=True, max_drawdown_pct=0.10)
    assert equity.allow_new(st(equity=8_000.0, peak=10_000.0, quality=False), TS) == (
        False,
        "max_drawdown",
    )


# --- pause latch -----------------------------------------------------------


def test_pause_latch_blocks_until_its_timestamp():
    e = engine(pause_bars=3)
    paused_until = TS + timedelta(minutes=3)
    assert e.allow_new(st(paused=paused_until), TS) == (False, "kill_switch_pause")
    assert e.allow_new(st(paused=paused_until), paused_until) == (True, None)


def test_pause_latch_is_inert_without_pause_bars():
    e = engine()
    assert e.allow_new(st(paused=TS + timedelta(days=1)), TS) == (True, None)


# --- configuration and bypass rules ---------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"weekly_loss_pct": -0.01},
        {"weekly_loss_pct": 1.01},
        {"weekly_loss_pct": float("nan")},
        {"max_drawdown_pct": -0.01},
        {"max_drawdown_pct": 1.01},
        {"min_equity_frac": -0.01},
        {"min_equity_frac": 1.01},
        {"max_spread_points": -1.0},
        {"max_spread_points": float("inf")},
        {"max_slippage_points": -0.5},
        {"max_slippage_points": float("nan")},
        {"consecutive_loss_pause": -1},
        {"pause_bars": -1},
    ],
)
def test_invalid_switch_config_is_rejected(kw):
    with pytest.raises(ValueError):
        engine(**kw)


def test_non_bool_data_quality_flag_is_rejected():
    with pytest.raises(TypeError):
        engine(data_quality_block=1)  # type: ignore[arg-type]


def test_entry_gate_has_no_override_parameter():
    assert list(inspect.signature(RiskEngine.allow_new).parameters) == ["self", "account", "ts"]
    assert list(inspect.signature(RiskEngine.allow_new_extended).parameters) == [
        "self",
        "account",
        "ts",
    ]


# --- simulator integration -------------------------------------------------


def test_weekly_switch_force_closes_the_open_position():
    sim = Simulator(flat_frame(), Settings(risk=RiskConfig(weekly_loss_pct=0.001)))
    sim.reset(seed=1)
    result = sim.step(1)
    trade = result.info["trade"]
    assert trade is not None and trade.reason == "weekly_loss_limit"
    assert result.info["position"] is None
    assert result.info["opened"] is not None, "the entry did fill before the switch closed it"
    assert sim.broker.position is None
    assert sim.balance == pytest.approx(10_000.0 + trade.pnl)
    # The loss is still beyond the weekly limit, so a new entry is blocked.
    assert sim.step(1).info["blocked"] == "weekly_loss_limit"


def test_drawdown_switch_force_closes_the_open_position():
    sim = Simulator(flat_frame(), Settings(risk=RiskConfig(max_drawdown_pct=0.001)))
    sim.reset(seed=1)
    result = sim.step(1)
    trade = result.info["trade"]
    assert trade is not None and trade.reason == "max_drawdown"
    assert result.info["position"] is None
    assert sim.step(1).info["blocked"] == "max_drawdown"


def test_emergency_switch_force_closes_the_open_position():
    # 0.999 of 10,000 leaves 9,990; the entry spread alone marks equity lower.
    sim = Simulator(flat_frame(), Settings(risk=RiskConfig(min_equity_frac=0.999)))
    sim.reset(seed=1)
    result = sim.step(1)
    trade = result.info["trade"]
    assert trade is not None and trade.reason == "emergency_stop"
    assert result.info["position"] is None


def test_weekly_anchor_rolls_with_the_session_boundary():
    sim = Simulator(flat_frame(n=1_200), Settings())
    sim.reset(seed=1)
    sim._force_equity(9_000.0)
    assert sim.week == (2026, 1)
    assert sim.week_start_equity == pytest.approx(10_000.0)

    roll_bar = None
    while sim.week == (2026, 1):
        roll_bar = sim.bar_index
        sim.step(0)

    assert sim.week == (2026, 2)
    assert roll_bar == 1_020, "17:00 UTC on Monday starts the new week"
    assert sim.week_start_equity == pytest.approx(9_000.0)


def test_weekly_switch_resets_when_the_week_anchor_rolls():
    sim = Simulator(
        flat_frame(n=1_200),
        Settings(risk=RiskConfig(weekly_loss_pct=0.05, daily_loss_pct=0.10)),
    )
    sim.reset(seed=1)
    sim._force_equity(9_400.0)  # 6% below the 10,000 week anchor
    while sim.bar_index < 1_019:
        sim.step(0)

    assert sim.week == (2026, 1)
    assert sim.step(1).info["blocked"] == "weekly_loss_limit"

    assert sim.week == (2026, 1)
    reopened = sim.step(1)  # bar 1020 rolls the week before the entry is gated
    assert sim.week == (2026, 2)
    assert sim.week_start_equity == pytest.approx(9_400.0)
    assert reopened.info["position"] is not None


def test_consecutive_loss_pause_blocks_then_expires_after_pause_bars():
    sim = Simulator(
        flat_frame(),
        Settings(risk=RiskConfig(consecutive_loss_pause=1, pause_bars=2)),
    )
    sim.reset(seed=1)
    sim.step(1)  # open at bar 119
    closed = sim.step(3)  # manual close at bar 120, a loss after costs
    trade = closed.info["trade"]
    assert trade is not None and trade.pnl < 0
    assert sim.consecutive_losses == 1
    assert sim.paused_until == datetime(2026, 1, 5, 2, 3, tzinfo=UTC)

    for _ in range(2):  # the two paused decision bars, 121 and 122
        blocked = sim.step(1)
        assert blocked.info["blocked"] == "consecutive_loss_pause"
        assert blocked.info["position"] is None

    reopened = sim.step(1)  # bar 123: the latch expired and the streak restarted
    assert reopened.info["position"] is not None
    assert sim.consecutive_losses == 0


def test_zero_pause_consecutive_block_lasts_until_the_session_roll():
    frame = flat_frame(n=1_700, start=datetime(2026, 1, 5, 15, 0, tzinfo=UTC))
    sim = Simulator(frame, Settings(risk=RiskConfig(consecutive_loss_pause=1)))
    sim.reset(seed=1)
    sim.step(1)  # open at 16:59
    closed = sim.step(3)  # close at 17:00; the roll resets the old streak first
    assert closed.info["trade"] is not None and closed.info["trade"].pnl < 0
    assert sim.consecutive_losses == 1
    assert sim.paused_until is None

    assert sim.step(1).info["blocked"] == "consecutive_loss_pause"
    while sim.session != date(2026, 1, 6):
        sim.step(0)
    assert sim.consecutive_losses == 0
    assert sim.step(1).info["position"] is not None


def test_spread_limit_blocks_entries_but_allows_hold_and_close():
    spreads = np.full(300, 0.20)
    spreads[120:] = 0.50  # 500 points from bar 120 on
    sim = Simulator(
        flat_frame(spreads=spreads),
        Settings(risk=RiskConfig(max_spread_points=300.0, max_positions=2)),
    )
    sim.reset(seed=1)
    sim.step(1)  # bar 119, spread 200 points: allowed
    assert sim.broker.position is not None
    assert sim.spread_points == pytest.approx(200.0)

    blocked = sim.step(1)  # bar 120, spread 500 points
    assert blocked.info["blocked"] == "max_spread"
    assert blocked.info["position"] is not None, "an open position is not force-closed"

    held = sim.step(0)
    assert held.info["blocked"] is None and held.info["position"] is not None

    closed = sim.step(3)
    assert closed.info["blocked"] is None
    assert closed.info["position"] is None
    assert closed.info["trade"] is not None


# --- slippage cap and data-quality block (Phase 11 remaining items) --------


def test_slippage_cap_allows_the_boundary_and_blocks_one_step_over():
    # flat_frame: ATR 0.5 price at slippage_atr_frac 0.05 is 0.025 price, and
    # the default point of 0.001 turns that into exactly 25 contract points.
    at_cap = Simulator(flat_frame(), Settings(risk=RiskConfig(max_slippage_points=25.0)))
    at_cap.reset(seed=1)
    assert at_cap.slippage_points == pytest.approx(25.0)
    assert at_cap.action_masks().tolist() == [True, True, True, False]
    assert at_cap.step(1).info["opened"] is not None

    over = Simulator(flat_frame(), Settings(risk=RiskConfig(max_slippage_points=24.99)))
    over.reset(seed=1)
    assert over.slippage_points == pytest.approx(25.0)
    assert over.action_masks().tolist() == [True, False, False, False]
    result = over.step(1)
    assert result.info["blocked"] == "slippage_limit"
    assert result.info["opened"] is None


def test_missing_atr_blocks_with_no_atr_by_default_and_the_quality_reason_when_on():
    frame = flat_frame()
    atr = frame.atr.copy()
    atr[119] = np.nan  # the first decision bar
    unusable = replace(frame, atr=atr)

    off = Simulator(unusable, Settings())
    off.reset(seed=1)
    assert off.data_quality_ok is False, "quality is reported even while the gate is off"
    assert off.action_masks().tolist() == [True, False, False, False]
    assert off.step(1).info["blocked"] == "no_atr"

    on = Simulator(unusable, Settings(risk=RiskConfig(data_quality_block=True)))
    on.reset(seed=1)
    assert on.action_masks().tolist() == [True, False, False, False]
    result = on.step(1)
    assert result.info["blocked"] == "data_quality_block"
    assert result.info["opened"] is None


def test_missing_spread_is_sanitised_by_default_and_blocked_when_the_flag_is_on():
    frame = flat_frame()
    spread = frame.spread.copy()
    spread[119] = np.nan
    unusable = replace(frame, spread=spread)

    off = Simulator(unusable, Settings())
    off.reset(seed=1)
    assert off.spread_points is None, "no usable spread to express in points"
    assert off.data_quality_ok is False
    assert off.step(1).info["opened"] is not None, "the legacy bar read spread 0"

    on = Simulator(unusable, Settings(risk=RiskConfig(data_quality_block=True)))
    on.reset(seed=1)
    assert on.action_masks().tolist() == [True, False, False, False]
    assert on.step(1).info["blocked"] == "data_quality_block"


def test_stale_bar_gap_blocks_the_entry_only_when_the_flag_is_on():
    frame = flat_frame()
    ts = frame.ts.copy()
    ts[119:] += np.timedelta64(15, "m")  # 16-minute gap before the decision bar
    stale = replace(frame, ts=ts)

    off = Simulator(stale, Settings())
    off.reset(seed=1)
    assert off.data_quality_ok is False
    assert off.step(1).info["opened"] is not None

    on = Simulator(stale, Settings(risk=RiskConfig(data_quality_block=True)))
    on.reset(seed=1)
    assert on.action_masks().tolist() == [True, False, False, False]
    result = on.step(1)
    assert result.info["blocked"] == "data_quality_block"
    assert result.info["opened"] is None


def test_stale_gap_threshold_follows_the_frame_cadence():
    frame = flat_frame(n=300)
    hourly_ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(300).astype("timedelta64[h]")
    hourly = replace(frame, ts=hourly_ts)

    normal = Simulator(hourly, Settings(risk=RiskConfig(data_quality_block=True)))
    normal.reset(seed=1)
    assert normal.data_quality_ok is True, "a 60-minute cadence is not a stale bar"
    assert normal.step(1).info["opened"] is not None

    ts = hourly.ts.copy()
    ts[150:] += np.timedelta64(1, "h")  # one whole H1 bar missing
    missing = replace(hourly, ts=ts)
    sim = Simulator(missing, Settings(risk=RiskConfig(data_quality_block=True)))
    sim.reset(seed=1)
    while sim.bar_index < 150:
        assert sim.step(0).info["blocked"] is None
    assert sim.data_quality_ok is False
    assert sim.step(1).info["blocked"] == "data_quality_block"


def test_new_gates_do_not_block_hold_or_close():
    frame = flat_frame()
    ts = frame.ts.copy()
    ts[120:] += np.timedelta64(15, "m")
    atr = frame.atr.copy()
    atr[120:] = 5.0  # 250 points of slippage, far over the 25-point cap
    degraded = replace(frame, ts=ts, atr=atr)

    sim = Simulator(
        degraded,
        Settings(risk=RiskConfig(max_slippage_points=25.0, data_quality_block=True)),
    )
    sim.reset(seed=1)
    assert sim.step(1).info["opened"] is not None, "bar 119 is clean"

    assert sim.bar_index == 120
    assert sim.data_quality_ok is False
    assert sim.slippage_points == pytest.approx(250.0)
    assert sim.action_masks().tolist() == [True, False, False, True]

    held = sim.step(0)
    assert held.info["blocked"] is None and held.info["position"] is not None

    closed = sim.step(3)
    assert closed.info["blocked"] is None
    trade = closed.info["trade"]
    assert trade is not None and trade.reason == "manual"
    assert closed.info["position"] is None


def test_equity_force_close_is_unaffected_by_the_new_gates():
    sim = Simulator(
        flat_frame(),
        Settings(
            risk=RiskConfig(
                weekly_loss_pct=0.001,
                max_slippage_points=25.0,
                data_quality_block=True,
            )
        ),
    )
    sim.reset(seed=1)
    result = sim.step(1)
    trade = result.info["trade"]
    assert trade is not None and trade.reason == "weekly_loss_limit"
    assert result.info["position"] is None


def test_unusable_bars_keep_default_off_parity():
    frame = flat_frame()
    spread = frame.spread.copy()
    spread[119:] = np.nan
    atr = frame.atr.copy()
    atr[121] = np.nan
    degraded = replace(frame, spread=spread, atr=atr)

    actions = [1, 0, 0, 3, 0, 0, 0]
    default = Simulator(degraded, Settings())
    zeroed = Simulator(
        degraded,
        Settings(risk=RiskConfig(max_slippage_points=0.0, data_quality_block=False)),
    )
    default.reset(seed=1)
    zeroed.reset(seed=1)
    for action in actions:
        a = default.step(action)
        b = zeroed.step(action)
        assert a.reward == b.reward
        assert a.info["equity"] == b.info["equity"]
        assert a.info["blocked"] == b.info["blocked"]
        assert a.info["position"] == b.info["position"]
        assert a.info["trade"] == b.info["trade"]


def test_the_new_gates_have_no_per_call_override():
    assert list(inspect.signature(RiskEngine.allow_new_extended).parameters) == [
        "self",
        "account",
        "ts",
    ]
    state_fields = {field.name: field for field in fields(AccountState)}
    assert state_fields["slippage_points"].default is None
    assert state_fields["data_quality_ok"].default is True

    simulator_params = inspect.signature(Simulator.__init__).parameters
    assert not any(
        "slippage" in name or "quality" in name for name in simulator_params
    ), "the gates come from frozen Settings, not from a call argument"
