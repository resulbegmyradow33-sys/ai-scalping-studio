"""Paper broker: the single source of truth for fills and trade accounting.

Accounting contract
-------------------
* Spread lives in the quoted prices. A LONG opens at ``ask + slippage`` and
  exits at ``bid``; a SHORT opens at ``bid - slippage`` and exits at ``ask``.
  The exit price is never adjusted for the spread a second time.
* SL and TP triggers fill at the trigger level; a bar that opens through the
  stop fills at its open. If one bar contains both levels, the stop wins.
* ``Trade.pnl`` is fill-to-fill PnL times contract size and lot, minus the
  round-turn commission. Spread and slippage are already embedded in the fill
  prices, so ``pnl`` must never be reduced by ``Trade.costs`` again.
* ``Trade.costs`` is a reporting breakdown only: entry half-spread + exit
  half-spread + slippage + commission, in account currency.
* The broker does not own the balance. ``state(balance)`` receives it from the
  caller (the simulator); ``update`` and ``close`` return trades whose ``pnl``
  the caller applies to its own balance.
* ``bars_in_trade`` counts ``update`` calls while a position is open. ``open``
  and any close reset it; ``mark`` never advances it.
* The day anchor resets when the UTC calendar date of the observed bars
  changes. ``day_start_equity`` is the equity reported by the first ``state``
  call on that date.
"""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite

from scalping.types import (
    AccountState,
    Bar,
    ContractSpec,
    CostConfig,
    Position,
    Side,
    Trade,
)

__all__ = ["PaperBroker"]


def _check_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


class PaperBroker:
    """Simulated single-position broker for XAUUSD-style CFD bars.

    One position at a time (``RiskConfig.max_positions`` defaults to 1).
    Prices passed to ``open`` are the executable quotes, so the half-spread is
    derived from ``ask - bid`` and charged once on entry.
    """

    def __init__(self, contract: ContractSpec, costs: CostConfig, atr: float = 0.0) -> None:
        _check_finite("atr", atr)
        if atr < 0:
            raise ValueError(f"atr must be >= 0, got {atr!r}")
        if costs.slippage_atr_frac < 0:
            raise ValueError("slippage_atr_frac must be >= 0")
        if costs.commission_per_lot < 0:
            raise ValueError("commission_per_lot must be >= 0")
        self.contract = contract
        self.costs = costs
        self.atr = float(atr)
        self._position: Position | None = None
        self._entry_half_spread = 0.0
        self._entry_slippage = 0.0
        self._unrealized = 0.0
        self._bars_in_trade = 0
        self._trades_closed = 0
        self._last_ts: datetime | None = None
        self._day: date | None = None
        self._state_day: date | None = None
        self._day_start_equity: float | None = None

    @property
    def position(self) -> Position | None:
        """The open position, or ``None`` when flat."""
        return self._position

    def open(
        self,
        side: Side,
        lot: float,
        ts: datetime,
        bid: float,
        ask: float,
        sl: float,
        tp: float,
        strategy: str,
    ) -> Position:
        """Open a position at the adverse side of the quote, plus slippage."""
        if self._position is not None:
            raise RuntimeError("a position is already open; close it first")
        side = Side(side)
        if side is Side.FLAT:
            raise ValueError("cannot open a FLAT position")
        _check_finite("lot", lot)
        if lot <= 0:
            raise ValueError(f"lot must be > 0, got {lot!r}")
        for name, value in (("bid", bid), ("ask", ask), ("sl", sl), ("tp", tp)):
            _check_finite(name, value)
        if bid > ask:
            raise ValueError(f"bid must be <= ask, got bid={bid!r} ask={ask!r}")
        self._observe(ts)

        slippage = self._slippage_per_unit()
        if side is Side.LONG:
            entry = ask + slippage
            if not sl < entry < tp:
                raise ValueError(
                    f"LONG needs sl < entry < tp, got sl={sl!r} entry={entry!r} tp={tp!r}"
                )
        else:
            entry = bid - slippage
            if not tp < entry < sl:
                raise ValueError(
                    f"SHORT needs tp < entry < sl, got tp={tp!r} entry={entry!r} sl={sl!r}"
                )

        position = Position(
            side=side,
            lot=lot,
            entry_price=entry,
            entry_ts=ts,
            sl=sl,
            tp=tp,
            strategy=strategy,
        )
        self._position = position
        self._entry_half_spread = (ask - bid) / 2.0
        self._entry_slippage = slippage
        self._unrealized = 0.0
        self._bars_in_trade = 0
        return position

    def mark(self, bar: Bar) -> float:
        """Return unrealized PnL at the bar's liquidating quote; no balance side effects."""
        self._observe(bar.ts)
        if self._position is None:
            self._unrealized = 0.0
            return 0.0
        self._unrealized = self._unrealized_at(bar)
        return self._unrealized

    def update(self, bar: Bar) -> Trade | None:
        """Advance one bar: fill SL/TP if touched, otherwise mark to market."""
        if self._position is None:
            self._observe(bar.ts)
            self._unrealized = 0.0
            return None

        self._observe(bar.ts)
        self._bars_in_trade += 1
        position = self._position
        if position.side is Side.LONG:
            if bar.low <= position.sl:
                exit_price = min(position.sl, bar.open)
                return self._settle(bar, exit_price, "sl")
            if bar.high >= position.tp:
                return self._settle(bar, position.tp, "tp")
        else:
            if bar.high >= position.sl:
                exit_price = max(position.sl, bar.open)
                return self._settle(bar, exit_price, "sl")
            if bar.low <= position.tp:
                return self._settle(bar, position.tp, "tp")

        self._unrealized = self._unrealized_at(bar)
        return None

    def close(self, bar: Bar, reason: str) -> Trade:
        """Close the open position at the bar's liquidating quote."""
        position = self._position
        if position is None:
            raise RuntimeError("cannot close: no open position")
        self._observe(bar.ts)
        if position.side is Side.LONG:
            exit_price = bar.bid_close
        else:
            exit_price = bar.ask_close
        return self._settle(bar, exit_price, reason)

    def state(self, balance: float) -> AccountState:
        """Snapshot equity for the caller-supplied balance."""
        _check_finite("balance", balance)
        equity = balance + self._unrealized
        if self._day_start_equity is None or self._state_day != self._day:
            self._state_day = self._day
            self._day_start_equity = equity
        day_start = self._day_start_equity
        return AccountState(
            balance=balance,
            equity=equity,
            position=self._position,
            day_start_equity=day_start,
            day_pnl=equity - day_start,
            bars_in_trade=self._bars_in_trade if self._position is not None else 0,
            trades_closed=self._trades_closed,
        )

    # --- internals ---------------------------------------------------------

    def _observe(self, ts: datetime) -> None:
        """Track bar order and the current accounting day."""
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(
                f"bars must arrive in non-decreasing time order: {ts!r} after {self._last_ts!r}"
            )
        self._last_ts = ts
        self._day = ts.date()

    def _slippage_per_unit(self) -> float:
        return self.costs.slippage_atr_frac * self.atr

    def _unrealized_at(self, bar: Bar) -> float:
        position = self._position
        if position is None:
            return 0.0
        price = bar.bid_close if position.side is Side.LONG else bar.ask_close
        size = self.contract.size * position.lot
        return (price - position.entry_price) * int(position.side) * size

    def _settle(self, bar: Bar, exit_price: float, reason: str) -> Trade:
        """Book the exit, charge the round-turn commission and clear the position."""
        position = self._position
        if position is None:
            raise RuntimeError("cannot settle: no open position")
        size = self.contract.size * position.lot
        gross = (exit_price - position.entry_price) * int(position.side) * size
        commission = self.costs.commission_per_lot * position.lot
        costs = (
            self._entry_half_spread * size
            + (bar.ask_close - bar.bid_close) / 2.0 * size
            + self._entry_slippage * size
            + commission
        )
        risk = abs(position.entry_price - position.sl) * size
        pnl = gross - commission
        trade = Trade(
            entry_ts=position.entry_ts,
            exit_ts=bar.ts,
            side=position.side,
            lot=position.lot,
            entry=position.entry_price,
            exit=exit_price,
            sl=position.sl,
            tp=position.tp,
            pnl=pnl,
            costs=costs,
            r_multiple=pnl / risk if risk > 0 else 0.0,
            reason=reason,
            strategy=position.strategy,
        )
        self._position = None
        self._unrealized = 0.0
        self._bars_in_trade = 0
        self._entry_half_spread = 0.0
        self._entry_slippage = 0.0
        self._trades_closed += 1
        return trade
