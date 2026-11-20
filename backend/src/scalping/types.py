"""Domain value objects shared by every backend module.

Only the standard library is used here so that this module stays importable
from anywhere (data, env, strategy, api) without pulling in heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

__all__ = [
    "AccountState",
    "Bar",
    "ContractSpec",
    "CostConfig",
    "Position",
    "RiskConfig",
    "Side",
    "Trade",
]


class Side(IntEnum):
    """Position direction. Values are signed so ``side * distance`` scales a price distance."""

    FLAT = 0
    LONG = 1
    SHORT = -1

    def __mul__(self, other: object) -> Side | float | int:  # type: ignore[override]
        """Scale the side by a number.

        ``side * 1``, ``side * -1`` and ``side * 0`` return a ``Side`` member, so
        sign flips stay typed. Any other int/float returns plain numeric
        arithmetic (``Side.SHORT * 2.0 == -2.0``). Non-numeric operands raise
        ``TypeError``; returning ``NotImplemented`` here would silently fall back
        to ``str`` repetition because ``Side`` subclasses ``int``.

        The wider return type than ``int.__mul__`` is intentional.
        """
        if not isinstance(other, (int, float)):
            raise TypeError(
                f"Side can only be multiplied by a number, got {type(other).__name__}"
            )
        product = int(self) * other
        if other in (0, 1, -1):
            return Side(int(product))
        return product

    def __rmul__(self, other: object) -> Side | float | int:  # type: ignore[override]
        return self * other


@dataclass(frozen=True, slots=True)
class Bar:
    """One finished candle, including the top-of-book quote that closed it."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    bid_close: float
    ask_close: float
    spread: float
    volume: int

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low must be <= high")
        if not self.low <= self.open <= self.high:
            raise ValueError("open must be within [low, high]")
        if not self.low <= self.close <= self.high:
            raise ValueError("close must be within [low, high]")
        if self.bid_close > self.ask_close:
            raise ValueError("bid_close must be <= ask_close")


@dataclass(frozen=True, slots=True)
class Position:
    side: Side
    lot: float
    entry_price: float
    entry_ts: datetime
    sl: float
    tp: float
    strategy: str


@dataclass(frozen=True, slots=True)
class Trade:
    entry_ts: datetime
    exit_ts: datetime
    side: Side
    lot: float
    entry: float
    exit: float
    sl: float
    tp: float
    pnl: float
    costs: float
    r_multiple: float
    reason: str
    strategy: str


@dataclass(frozen=True, slots=True)
class AccountState:
    balance: float
    equity: float
    position: Position | None
    day_start_equity: float
    day_pnl: float
    bars_in_trade: int
    trades_closed: int


# --- Static configuration dataclasses ---------------------------------------
# Defined here (not in config.py) so that config.py can own Settings/YAML
# loading while every module still imports these without touching YAML.


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """XAUUSD-style CFD contract metadata."""

    size: float = 100.0  # ounces per lot
    point: float = 0.001  # smallest quoted price increment
    min_lot: float = 0.01
    lot_step: float = 0.01
    commission_per_lot: float = 7.0  # contract metadata; charging uses CostConfig
    leverage: float = 20.0
    initial_balance: float = 10_000.0


@dataclass(frozen=True, slots=True)
class RiskConfig:
    risk_pct: float = 0.005  # fraction of equity risked per trade
    daily_loss_pct: float = 0.03
    max_notional_x: float = 20.0  # notional cap as a multiple of equity
    max_positions: int = 1
    max_bars_in_trade: int = 120


@dataclass(frozen=True, slots=True)
class CostConfig:
    spread_floor: float = 0.20
    spread_vol_mult: float = 0.10
    slippage_atr_frac: float = 0.05
    commission_per_lot: float = 7.0  # the single commission source charged for trades
