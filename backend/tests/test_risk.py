"""Tests for the risk engine: dual-limit sizing, session gates and ATR stops."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from scalping.config import ContractSpec, RiskConfig
from scalping.env.risk import RiskEngine, SizingResult
from scalping.types import AccountState, Position, Side


def engine(**kw: float) -> RiskEngine:
    return RiskEngine(RiskConfig(**kw), ContractSpec())


def st(
    *,
    equity: float = 10_000.0,
    day_start: float = 10_000.0,
    position: Position | None = None,
    bars: int = 0,
) -> AccountState:
    return AccountState(
        balance=equity,
        equity=equity,
        position=position,
        day_start_equity=day_start,
        day_pnl=equity - day_start,
        bars_in_trade=bars,
        trades_closed=0,
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


# --- sizing ----------------------------------------------------------------


def test_risk_sizing_uses_half_percent():
    s = engine().size(equity=10_000, entry=2350.0, sl=2349.0)
    # risk $50 / $1 distance = 50oz = 0.5 lot
    assert s.lot == pytest.approx(0.5, abs=0.01) and s.reason is None
    assert s.effective_risk_pct == pytest.approx(0.005, rel=1e-12)


def test_notional_cap_binds_when_sl_is_tight():
    s = engine().size(equity=10_000, entry=2350.0, sl=2349.9)
    # risk-based -> 5.0 lot (500oz, $1.175M = 117x) -> capped at 20x = $200k -> 0.85 lot
    assert s.lot == pytest.approx(0.85, abs=0.01)
    assert s.reason == "notional_cap"
    assert s.effective_risk_pct < 0.005


def test_lot_too_small_blocks():
    s = engine().size(equity=50, entry=2350.0, sl=2349.0)
    assert s.lot == 0.0 and s.reason == "lot_too_small"
    assert s.effective_risk_pct == 0.0


def test_zero_equity_sizes_nothing():
    s = engine().size(equity=0.0, entry=2350.0, sl=2349.0)
    assert s == SizingResult(lot=0.0, reason="lot_too_small", effective_risk_pct=0.0)
    negative = engine().size(equity=-500.0, entry=2350.0, sl=2349.0)
    assert negative == SizingResult(lot=0.0, reason="lot_too_small", effective_risk_pct=0.0)


def test_lot_is_floored_to_the_step():
    # $50 / ($0.707 * 100) = 0.7072 lots of risk budget -> floor to 0.70, never up.
    s = engine().size(equity=10_000, entry=2350.0, sl=2349.293)
    assert s.lot == pytest.approx(0.70, abs=1e-9)
    assert s.lot * 100 == pytest.approx(70.0, abs=1e-9)
    assert s.effective_risk_pct == pytest.approx(0.004949, rel=1e-5)


def test_floor_lot_survives_binary_step_noise():
    # 0.29 / 0.01 is 28.999999999999996 in binary; the floor must not drop a step.
    e = engine()
    assert e._floor_lot(0.29) == pytest.approx(0.29, abs=1e-12)
    assert e._floor_lot(0.2949) == pytest.approx(0.29, abs=1e-12)


def test_notional_cap_boundary_at_exactly_twenty_x_does_not_bind():
    # entry 2000, 1 lot = $200k notional = exactly 20x equity; the risk lot is
    # also exactly 1.0 lot, so both limits meet at the boundary.
    s = engine().size(equity=10_000, entry=2000.0, sl=1999.5)
    assert s.lot == pytest.approx(1.0, abs=1e-9)
    assert s.reason is None
    assert s.effective_risk_pct == pytest.approx(0.005, rel=1e-12)


def test_notional_cap_binds_just_below_twenty_x():
    s = engine().size(equity=10_000, entry=2000.0, sl=1999.51)
    assert s.lot == pytest.approx(1.0, abs=1e-9)
    assert s.reason == "notional_cap"
    assert s.effective_risk_pct == pytest.approx(0.0049, rel=1e-6)


def test_size_rejects_degenerate_prices():
    e = engine()
    for entry, sl in ((0.0, -1.0), (-2350.0, -2351.0), (2350.0, 2350.0)):
        with pytest.raises(ValueError):
            e.size(equity=10_000, entry=entry, sl=sl)
    with pytest.raises(ValueError):
        e.size(equity=float("nan"), entry=2350.0, sl=2349.0)
    with pytest.raises(ValueError):
        e.size(equity=10_000, entry=float("inf"), sl=2349.0)


# --- session gate ----------------------------------------------------------


def test_daily_loss_limit_blocks_until_next_session():
    e = engine()
    ts = datetime(2026, 1, 5, 10, tzinfo=UTC)
    ok, reason = e.allow_new(st(equity=9_650, day_start=10_000), ts)
    assert (ok, reason) == (False, "daily_loss_limit")
    ok2, _ = e.allow_new(st(equity=9_650, day_start=9_650), ts.replace(day=6))
    assert ok2


def test_daily_loss_limit_triggers_at_the_exact_pct():
    e = engine()
    ts = datetime(2026, 1, 5, 10, tzinfo=UTC)
    assert e.allow_new(st(equity=9_700, day_start=10_000), ts) == (False, "daily_loss_limit")
    assert e.allow_new(st(equity=9_701, day_start=10_000), ts) == (True, None)


def test_allow_new_does_not_mutate_the_account():
    e = engine()
    ts = datetime(2026, 1, 5, 10, tzinfo=UTC)
    account = st(equity=9_650, day_start=10_000, position=None)
    before = tuple(getattr(account, f.name) for f in fields(account))
    e.allow_new(account, ts)
    e.allow_new(account, ts.replace(day=6))
    after = tuple(getattr(account, f.name) for f in fields(account))
    assert after == before
    assert account == st(equity=9_650, day_start=10_000)


def test_max_positions_blocks_and_clears():
    e = engine()
    ts = datetime(2026, 1, 5, 18, 30, tzinfo=UTC)
    assert e.allow_new(st(position=long_position()), ts) == (False, "max_positions")
    assert e.allow_new(st(position=None), ts) == (True, None)
    two = engine(max_positions=2)
    assert two.allow_new(st(position=long_position()), ts) == (True, None)


def test_allow_new_blocks_a_blown_account():
    e = engine()
    ts = datetime(2026, 1, 5, 10, tzinfo=UTC)
    # A zero equity on a real base is a 100% loss, not a sizing problem.
    assert e.allow_new(st(equity=0.0, day_start=10_000), ts) == (False, "daily_loss_limit")
    assert e.allow_new(st(equity=-500.0, day_start=10_000), ts) == (False, "daily_loss_limit")
    # No measurable base: nothing can be sized or compared.
    assert e.allow_new(st(equity=0.0, day_start=0.0), ts) == (False, "lot_too_small")
    assert e.allow_new(st(equity=100.0, day_start=-1.0), ts) == (False, "lot_too_small")


def test_session_anchor_uses_the_1700_utc_boundary():
    e = engine()
    assert e.session_anchor(datetime(2026, 1, 5, 16, 59, tzinfo=UTC)) == date(2026, 1, 4)
    assert e.session_anchor(datetime(2026, 1, 5, 17, 0, tzinfo=UTC)) == date(2026, 1, 5)
    # 00:30 UTC still belongs to the session that opened the previous afternoon.
    assert e.session_anchor(datetime(2026, 1, 6, 0, 30, tzinfo=UTC)) == date(2026, 1, 5)


def test_session_anchor_normalises_to_utc():
    e = engine()
    plus_three = timezone(timedelta(hours=3))
    assert e.session_anchor(datetime(2026, 1, 5, 20, 0, tzinfo=plus_three)) == date(2026, 1, 5)
    assert e.session_anchor(datetime(2026, 1, 5, 19, 59, tzinfo=plus_three)) == date(2026, 1, 4)
    with pytest.raises(ValueError):
        e.session_anchor(datetime(2026, 1, 5, 10, 0))


def test_allow_new_rejects_a_naive_timestamp():
    with pytest.raises(ValueError):
        engine().allow_new(st(), datetime(2026, 1, 5, 10, 0))


# --- exits -----------------------------------------------------------------


def test_force_close_after_max_bars():
    e = engine()
    assert not e.must_force_close(119)
    assert e.must_force_close(120)
    assert e.must_force_close(121)


def test_sl_tp_from_atr():
    sl, tp = engine().sl_tp(entry=2350.0, side=Side.LONG, atr=1.0)
    assert (sl, tp) == (2349.0, 2352.0)


def test_sl_tp_short_mirrors_long():
    sl, tp = engine().sl_tp(entry=2350.0, side=Side.SHORT, atr=1.0)
    assert (sl, tp) == (2351.0, 2348.0)
    sl2, tp2 = engine().sl_tp(entry=2350.0, side=Side.SHORT, atr=2.5)
    assert (sl2, tp2) == (2352.5, 2345.0)


def test_sl_tp_rejects_bad_inputs():
    e = engine()
    with pytest.raises(ValueError):
        e.sl_tp(entry=2350.0, side=Side.LONG, atr=0.0)
    with pytest.raises(ValueError):
        e.sl_tp(entry=2350.0, side=Side.LONG, atr=-1.0)
    with pytest.raises(ValueError):
        e.sl_tp(entry=2350.0, side=Side.LONG, atr=float("nan"))
    with pytest.raises(ValueError):
        e.sl_tp(entry=2350.0, side=Side.FLAT, atr=1.0)


# --- configuration ---------------------------------------------------------


def test_limits_are_fixed_at_construction():
    e = engine()
    with pytest.raises(AttributeError):
        e.risk = RiskConfig(risk_pct=0.5)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        e.contract = ContractSpec(size=1.0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        e.size(equity=10_000, entry=2350.0, sl=2349.0, risk_pct=0.5)  # type: ignore[call-arg]


def test_invalid_config_is_rejected_at_construction():
    for kw in (
        {"risk_pct": 0.0},
        {"risk_pct": -0.005},
        {"risk_pct": 1.5},
        {"daily_loss_pct": 0.0},
        {"max_notional_x": 0.0},
        {"max_notional_x": float("inf")},
        {"max_positions": 0},
        {"max_bars_in_trade": 0},
    ):
        with pytest.raises(ValueError):
            engine(**kw)
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(), ContractSpec(size=0.0))
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(), ContractSpec(min_lot=0.0))
    with pytest.raises(ValueError):
        RiskEngine(RiskConfig(), ContractSpec(lot_step=0.0))
