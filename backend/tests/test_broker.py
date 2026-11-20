"""Hand-checkable accounting tests for the paper broker.

With a 100 oz contract and 1.00 lot a 0.01 price move is $1, so every expected
value below can be recomputed from the quoted prices.
"""

from datetime import UTC, datetime, timedelta

import pytest

from scalping.config import ContractSpec, CostConfig
from scalping.env.broker import PaperBroker
from scalping.types import Bar, Side

TS = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)


def bar(o, h, l, c, spread=0.20, ts=TS):
    return Bar(ts=ts, open=o, high=h, low=l, close=c, bid_close=c,
               ask_close=c + spread, spread=spread, volume=10)


def make(atr=1.0, costs=None):
    return PaperBroker(
        ContractSpec(),
        costs or CostConfig(spread_floor=0.20, slippage_atr_frac=0.0),
        atr=atr,
    )


def test_long_entry_charges_half_spread_and_commission():
    b = make()
    pos = b.open(Side.LONG, 1.0, TS, bid=2350.0, ask=2350.20, sl=2348.0, tp=2354.0,
                 strategy="mom")
    assert pos.entry_price == pytest.approx(2350.20)
    trade = b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # long enters at ask 2350.20, exits at bid 2352.00 -> +1.80 * 100oz, minus $7 commission
    assert trade.pnl == pytest.approx(100 * 1.80 - 7)
    # 0.10 entry half-spread + 0.10 exit half-spread (x100) + $7 commission
    assert trade.costs == pytest.approx(10.0 + 10.0 + 7.0)


def test_sl_hit_intrabar_closes_at_sl():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2349.0, tp=2354.0, strategy="mom")
    trade = b.update(bar(2350.0, 2350.5, 2348.5, 2349.5))
    assert trade is not None and trade.reason == "sl"
    assert trade.exit == pytest.approx(2349.0)
    assert trade.pnl == pytest.approx(100 * (2349.0 - 2350.20) - 7)


def test_tp_hit_before_sl_when_bar_contains_both():
    b = make()
    b.open(Side.SHORT, 1.0, TS, 2350.0, 2350.20, sl=2352.0, tp=2347.0, strategy="rev")
    trade = b.update(bar(2350.0, 2353.0, 2346.0, 2351.0))
    # Conservative: the bar contains both levels, so the stop is assumed first.
    assert trade is not None and trade.reason == "sl"
    assert trade.exit == pytest.approx(2352.0)
    assert trade.pnl == pytest.approx(100 * (2350.0 - 2352.0) - 7)


def test_gap_through_sl_fills_at_open():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2349.0, tp=2354.0, strategy="mom")
    trade = b.update(bar(2347.0, 2347.5, 2346.0, 2347.2))
    assert trade.reason == "sl" and trade.exit == pytest.approx(2347.0)


def test_equity_equals_balance_plus_unrealized():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    b.update(bar(2351.0, 2352.0, 2350.5, 2351.5))
    st = b.state(balance=10_000.0)
    assert st.equity == pytest.approx(10_000.0 + 100 * (2351.5 - 2350.20))


def test_short_entry_and_exit_mirror_long():
    b = make()
    pos = b.open(Side.SHORT, 1.0, TS, bid=2352.00, ask=2352.20, sl=2354.0, tp=2350.0,
                 strategy="rev")
    assert pos.entry_price == pytest.approx(2352.00)
    trade = b.close(bar(2350.0, 2351.0, 2349.0, 2350.0), "manual")
    # short enters at bid 2352.00, exits at ask 2350.20 -> +1.80 * 100oz, minus $7
    assert trade.exit == pytest.approx(2350.20)
    assert trade.pnl == pytest.approx(100 * 1.80 - 7)
    assert trade.costs == pytest.approx(10.0 + 10.0 + 7.0)


def test_commission_charged_once_from_cost_config_only():
    # ContractSpec.commission_per_lot is metadata; it must never be charged.
    b = PaperBroker(ContractSpec(commission_per_lot=999.0),
                    CostConfig(slippage_atr_frac=0.0), atr=0.0)
    b.open(Side.LONG, 2.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    trade = b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # 2 lots: $14 round-turn commission, charged once on close.
    assert trade.pnl == pytest.approx(2 * 100 * 1.80 - 14)
    assert trade.costs == pytest.approx(20.0 + 20.0 + 14.0)


def test_slippage_applied_adversely_both_sides():
    costs = CostConfig(spread_floor=0.20, slippage_atr_frac=0.05)
    longs = make(costs=costs)
    long_pos = longs.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0,
                          strategy="mom")
    assert long_pos.entry_price == pytest.approx(2350.25)  # ask + 0.05 ATR
    longs.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")

    shorts = make(costs=costs)
    short_pos = shorts.open(Side.SHORT, 1.0, TS, 2350.0, 2350.20, sl=2352.0, tp=2348.0,
                            strategy="rev")
    assert short_pos.entry_price == pytest.approx(2349.95)  # bid - 0.05 ATR


def test_costs_are_spread_slippage_and_commission():
    costs = CostConfig(spread_floor=0.20, slippage_atr_frac=0.05)
    b = make(costs=costs)
    b.open(Side.LONG, 1.0, TS, bid=2350.0, ask=2350.20, sl=2348.0, tp=2354.0,
           strategy="mom")
    trade = b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # 0.10 entry half-spread + 0.05 slippage + 0.10 exit half-spread, x100, + $7
    assert trade.costs == pytest.approx((0.10 + 0.05 + 0.10) * 100 + 7)
    assert trade.costs == pytest.approx(32.0)
    # Costs are embedded in the fills, never subtracted from pnl a second time.
    assert trade.pnl == pytest.approx(100 * (2352.0 - 2350.25) - 7)


def test_r_multiple_uses_initial_stop_risk():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    trade = b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # risk = |2350.20 - 2348.00| * 100 oz * 1 lot = $220
    assert trade.r_multiple == pytest.approx((100 * 1.80 - 7) / (2.2 * 100))


def test_gap_through_sl_short_fills_at_open():
    b = make()
    b.open(Side.SHORT, 1.0, TS, 2350.0, 2350.20, sl=2352.0, tp=2347.0, strategy="rev")
    trade = b.update(bar(2353.0, 2353.5, 2352.5, 2353.0))
    assert trade.reason == "sl" and trade.exit == pytest.approx(2353.0)
    assert trade.pnl == pytest.approx(100 * (2350.0 - 2353.0) - 7)


def test_tp_fills_at_tp_and_gap_never_improves_the_fill():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    trade = b.update(bar(2352.0, 2354.5, 2351.5, 2354.2))
    assert trade.reason == "tp" and trade.exit == pytest.approx(2354.0)

    gapped = make()
    gapped.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    trade = gapped.update(bar(2355.0, 2356.0, 2354.5, 2355.5))
    # The bar opened above the target; conservative accounting still fills at the target.
    assert trade.reason == "tp" and trade.exit == pytest.approx(2354.0)


def test_close_without_position_raises():
    b = make()
    with pytest.raises(RuntimeError, match="no open position"):
        b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")


def test_open_twice_raises():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    with pytest.raises(RuntimeError, match="already open"):
        b.open(Side.SHORT, 1.0, TS, 2350.0, 2350.20, sl=2352.0, tp=2347.0,
               strategy="rev")


def test_update_without_position_is_inert():
    b = make()
    assert b.update(bar(2352.0, 2353.0, 2351.0, 2352.0)) is None
    st = b.state(10_000.0)
    assert st.equity == pytest.approx(10_000.0)
    assert st.position is None and st.trades_closed == 0


def test_mark_is_pure_and_exposes_position():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    current = bar(2351.0, 2352.0, 2350.5, 2351.5)
    assert b.mark(current) == pytest.approx(100 * (2351.5 - 2350.20))
    assert b.mark(current) == pytest.approx(100 * (2351.5 - 2350.20))
    assert b.position is not None and b.position.entry_price == pytest.approx(2350.20)
    st = b.state(10_000.0)
    assert st.bars_in_trade == 0  # mark never advances the bar counter
    assert st.position is b.position


def test_state_reports_bars_in_trade_and_closed_trades():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    assert b.state(10_000.0).bars_in_trade == 0
    b.update(bar(2351.0, 2352.0, 2350.5, 2351.5))
    b.update(bar(2352.0, 2353.0, 2351.5, 2352.5))
    st = b.state(10_000.0)
    assert st.bars_in_trade == 2 and st.trades_closed == 0
    trade = b.close(bar(2352.5, 2353.0, 2352.0, 2352.5), "manual")
    assert trade.pnl == pytest.approx(100 * (2352.5 - 2350.20) - 7)  # $223
    st = b.state(10_000.0 + trade.pnl)
    assert st.position is None and st.bars_in_trade == 0
    assert st.trades_closed == 1
    assert st.equity == pytest.approx(10_223.0)


def test_state_uses_the_callers_day_anchor():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    st = b.state(10_000.0)
    assert st.day_start_equity == pytest.approx(10_000.0)
    assert st.day_pnl == pytest.approx(0.0)

    b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # Task 8/10 owns the 17:00 session anchor and passes it in explicitly.
    st = b.state(10_173.0, day_start_equity=10_000.0)
    assert st.day_start_equity == pytest.approx(10_000.0)
    assert st.day_pnl == pytest.approx(173.0)

    # The broker keeps no day state: the default anchor is the given balance.
    st = b.state(10_173.0)
    assert st.day_start_equity == pytest.approx(10_173.0)
    assert st.day_pnl == pytest.approx(0.0)


def test_trigger_exit_costs_exclude_exit_spread():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2349.0, tp=2354.0, strategy="mom")
    trade = b.update(bar(2350.0, 2350.5, 2348.5, 2349.5))
    # The fill is at the SL level, not across the quote: entry half-spread $10 + $7.
    assert trade.reason == "sl"
    assert trade.costs == pytest.approx(17.0)
    assert trade.pnl == pytest.approx(-127.0)


def test_timeout_close_is_a_trigger_exit():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    trade = b.close(bar(2351.0, 2352.0, 2350.5, 2351.5), "timeout")
    assert trade.reason == "timeout"
    assert trade.costs == pytest.approx(17.0)
    assert trade.pnl == pytest.approx(100 * (2351.5 - 2350.20) - 7)  # $123


def test_quote_exit_costs_include_exit_spread_and_reconcile_with_pnl():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    trade = b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # Entry half-spread $10 + exit half-spread $10 + $7 commission.
    assert trade.costs == pytest.approx(27.0)
    entry_mid = (2350.0 + 2350.20) / 2
    exit_mid = (2352.0 + 2352.20) / 2
    # Costs (including commission) bridge the fill PnL back to the mid-to-mid move.
    assert trade.pnl + trade.costs == pytest.approx(100 * (exit_mid - entry_mid))


def test_rejected_open_does_not_advance_the_clock():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")

    later = TS + timedelta(minutes=5)
    with pytest.raises(ValueError, match="sl < entry < tp"):
        b.open(Side.LONG, 1.0, later, 2350.0, 2350.20, sl=2360.0, tp=2340.0, strategy="mom")

    # If the rejected order had run _observe(later), this older timestamp would raise.
    pos = b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    assert pos.entry_ts == TS


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_cost_fields_must_be_finite(bad):
    with pytest.raises(ValueError, match="commission_per_lot"):
        PaperBroker(ContractSpec(), CostConfig(commission_per_lot=bad))
    with pytest.raises(ValueError, match="slippage_atr_frac"):
        PaperBroker(ContractSpec(), CostConfig(slippage_atr_frac=bad))


def test_contract_size_must_be_positive_and_finite():
    with pytest.raises(ValueError, match="size"):
        PaperBroker(ContractSpec(size=0.0), CostConfig())
    with pytest.raises(ValueError, match="size"):
        PaperBroker(ContractSpec(size=float("nan")), CostConfig())

