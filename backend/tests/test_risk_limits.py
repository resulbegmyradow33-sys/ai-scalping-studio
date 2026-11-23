"""Tests for the Phase 11 risk kill switches and limits.

Every switch is exercised at its boundary (trip at the limit, not one tick
before), with its reset rule, and with the default ``0`` proving inert. The
simulator tests cover the state it owns: week anchor, peak equity, consecutive
losses, the timed pause latch and spread points.
"""

from __future__ import annotations

import inspect
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
        {"consecutive_loss_pause": -1},
        {"pause_bars": -1},
    ],
)
def test_invalid_switch_config_is_rejected(kw):
    with pytest.raises(ValueError):
        engine(**kw)


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
