"""Symbol and timeframe registry.

The data layer was gold-shaped: one ``ContractSpec`` and a hard-coded M1 source.
This registry is the single place that answers "what is this instrument?" and
"which bar sizes do we support?", so the simulator, providers, API and future
models no longer encode XAUUSD assumptions.

The values are **defaults for research**, not broker-verified specifications.
A production adapter must replace them from the broker's contract details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from scalping.types import ContractSpec

__all__ = [
    "SYMBOLS",
    "TIMEFRAMES",
    "SessionSpec",
    "SymbolSpec",
    "contract_for",
    "get_symbol",
    "parse_session",
    "resample_rule",
    "session_for",
    "symbols",
    "timeframe_rule",
]


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Research defaults for one instrument."""

    symbol: str
    description: str
    asset_class: str
    contract_size: float
    point: float
    price_precision: int
    min_lot: float
    max_lot: float
    lot_step: float
    commission_per_lot: float
    typical_spread: float
    quote_currency: str
    sessions: str
    leverage: float
    dukascopy_symbol: str | None = None

    def __post_init__(self) -> None:
        for name in ("contract_size", "point", "min_lot", "max_lot", "lot_step", "leverage"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{self.symbol}.{name} must be positive, got {value!r}")
        if self.min_lot > self.max_lot:
            raise ValueError(f"{self.symbol}: min_lot exceeds max_lot")
        if self.price_precision < 0:
            raise ValueError(f"{self.symbol}: price_precision must be non-negative")

    def contract(self, initial_balance: float = 10_000.0) -> ContractSpec:
        """Map the research defaults onto the simulator's contract object."""
        return ContractSpec(
            size=self.contract_size,
            point=self.point,
            min_lot=self.min_lot,
            lot_step=self.lot_step,
            commission_per_lot=self.commission_per_lot,
            leverage=self.leverage,
            initial_balance=initial_balance,
        )


#: Supported bar sizes and the Polars resampling rule that creates them from M1.
TIMEFRAMES: Final[dict[str, str]] = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
}

_METALS_SESSION = "sun22:00-fri21:00 UTC"
_FX_SESSION = "sun22:00-fri21:00 UTC"
_INDEX_SESSION = "sun22:00-fri21:00 UTC"
_CRYPTO_SESSION = "24x7 UTC"

#: Research defaults; replace from broker contract details before trading.
SYMBOLS: Final[dict[str, SymbolSpec]] = {
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        description="Spot gold vs US dollar",
        asset_class="metal",
        contract_size=100.0,
        point=0.001,
        price_precision=2,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=7.0,
        typical_spread=0.20,
        quote_currency="USD",
        sessions=_METALS_SESSION,
        leverage=20.0,
        dukascopy_symbol="XAUUSD",
    ),
    "XAGUSD": SymbolSpec(
        symbol="XAGUSD",
        description="Spot silver vs US dollar",
        asset_class="metal",
        contract_size=5_000.0,
        point=0.001,
        price_precision=3,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=7.0,
        typical_spread=0.02,
        quote_currency="USD",
        sessions=_METALS_SESSION,
        leverage=20.0,
        dukascopy_symbol="XAGUSD",
    ),
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        description="Euro vs US dollar",
        asset_class="fx",
        contract_size=100_000.0,
        point=0.00001,
        price_precision=5,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=7.0,
        typical_spread=0.00012,
        quote_currency="USD",
        sessions=_FX_SESSION,
        leverage=30.0,
    ),
    "GBPUSD": SymbolSpec(
        symbol="GBPUSD",
        description="British pound vs US dollar",
        asset_class="fx",
        contract_size=100_000.0,
        point=0.00001,
        price_precision=5,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=7.0,
        typical_spread=0.00016,
        quote_currency="USD",
        sessions=_FX_SESSION,
        leverage=30.0,
    ),
    "USDJPY": SymbolSpec(
        symbol="USDJPY",
        description="US dollar vs Japanese yen",
        asset_class="fx",
        contract_size=100_000.0,
        point=0.001,
        price_precision=3,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=7.0,
        typical_spread=0.014,
        quote_currency="JPY",
        sessions=_FX_SESSION,
        leverage=30.0,
    ),
    "AUDUSD": SymbolSpec(
        symbol="AUDUSD",
        description="Australian dollar vs US dollar",
        asset_class="fx",
        contract_size=100_000.0,
        point=0.00001,
        price_precision=5,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=7.0,
        typical_spread=0.00014,
        quote_currency="USD",
        sessions=_FX_SESSION,
        leverage=30.0,
    ),
    "NAS100": SymbolSpec(
        symbol="NAS100",
        description="US Nasdaq 100 index CFD",
        asset_class="index",
        contract_size=1.0,
        point=0.1,
        price_precision=1,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=0.0,
        typical_spread=1.0,
        quote_currency="USD",
        sessions=_INDEX_SESSION,
        leverage=20.0,
    ),
    "US30": SymbolSpec(
        symbol="US30",
        description="Dow Jones 30 index CFD",
        asset_class="index",
        contract_size=1.0,
        point=0.1,
        price_precision=1,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
        commission_per_lot=0.0,
        typical_spread=2.0,
        quote_currency="USD",
        sessions=_INDEX_SESSION,
        leverage=20.0,
    ),
    "USOIL": SymbolSpec(
        symbol="USOIL",
        description="WTI crude oil CFD",
        asset_class="commodity",
        contract_size=1_000.0,
        point=0.01,
        price_precision=2,
        min_lot=0.01,
        max_lot=50.0,
        lot_step=0.01,
        commission_per_lot=0.0,
        typical_spread=0.03,
        quote_currency="USD",
        sessions=_METALS_SESSION,
        leverage=10.0,
    ),
    "BTCUSD": SymbolSpec(
        symbol="BTCUSD",
        description="Bitcoin vs US dollar",
        asset_class="crypto",
        contract_size=1.0,
        point=0.01,
        price_precision=2,
        min_lot=0.01,
        max_lot=10.0,
        lot_step=0.01,
        commission_per_lot=0.0,
        typical_spread=5.0,
        quote_currency="USD",
        sessions=_CRYPTO_SESSION,
        leverage=2.0,
    ),
}


#: Minutes in one ISO week (Monday 00:00 UTC to the next Monday).
_WEEK_MINUTES: Final[int] = 7 * 24 * 60
_WEEKDAY_INDEX: Final[dict[str, int]] = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}
_SESSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(mon|tue|wed|thu|fri|sat|sun)(\d{2}):(\d{2})"
    r"-(mon|tue|wed|thu|fri|sat|sun)(\d{2}):(\d{2}) UTC"
)


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """A weekly UTC trading window; ``always_open`` covers 24x7 instruments."""

    label: str
    open_weekday: int = 0
    open_minute: int = 0
    close_weekday: int = 4
    close_minute: int = 0
    always_open: bool = False

    def is_open(self, ts: datetime) -> bool:
        """True when ``ts`` (naive means UTC) falls inside the weekly window."""
        if self.always_open:
            return True
        moment = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
        minute = moment.weekday() * 24 * 60 + moment.hour * 60 + moment.minute
        start = self.open_weekday * 24 * 60 + self.open_minute
        end = self.close_weekday * 24 * 60 + self.close_minute
        if end <= start:
            end += _WEEK_MINUTES
        return ((minute - start) % _WEEK_MINUTES) < (end - start)


def parse_session(text: str) -> SessionSpec:
    """Parse the registry's session strings (``sun22:00-fri21:00 UTC``)."""
    cleaned = text.strip()
    if cleaned.lower() in {"24x7 utc", "24x7"}:
        return SessionSpec(label=cleaned, always_open=True)
    match = _SESSION_PATTERN.fullmatch(cleaned)
    if match is None:
        raise ValueError(f"unsupported session window {text!r}")
    open_day, open_hour, open_minute, close_day, close_hour, close_minute = match.groups()
    open_h, open_m, close_h, close_m = (
        int(open_hour),
        int(open_minute),
        int(close_hour),
        int(close_minute),
    )
    if open_h > 23 or close_h > 23 or open_m > 59 or close_m > 59:
        raise ValueError(f"invalid session window {text!r}")
    return SessionSpec(
        label=cleaned,
        open_weekday=_WEEKDAY_INDEX[open_day],
        open_minute=open_h * 60 + open_m,
        close_weekday=_WEEKDAY_INDEX[close_day],
        close_minute=close_h * 60 + close_m,
    )


_SESSION_CACHE: dict[str, SessionSpec] = {}


def session_for(symbol: str) -> SessionSpec:
    """Weekly session for a registered symbol, parsed once and cached."""
    raw = get_symbol(symbol).sessions
    cached = _SESSION_CACHE.get(raw)
    if cached is None:
        cached = parse_session(raw)
        _SESSION_CACHE[raw] = cached
    return cached


def symbols() -> tuple[str, ...]:
    """Registered symbols in their canonical order."""
    return tuple(SYMBOLS)


def get_symbol(symbol: str) -> SymbolSpec:
    """Look up ``symbol`` case-insensitively; unknown symbols raise."""
    key = symbol.strip().upper()
    try:
        return SYMBOLS[key]
    except KeyError:
        raise ValueError(
            f"unknown symbol {symbol!r}; registered: {', '.join(sorted(SYMBOLS))}"
        ) from None


def contract_for(symbol: str, initial_balance: float = 10_000.0) -> ContractSpec:
    """ContractSpec for ``symbol`` with the account's starting balance."""
    return get_symbol(symbol).contract(initial_balance)


def timeframe_rule(timeframe: str) -> str:
    """Polars resampling rule for a supported timeframe."""
    key = timeframe.strip().upper()
    try:
        return TIMEFRAMES[key]
    except KeyError:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; supported: {', '.join(TIMEFRAMES)}"
        ) from None


#: Alias used by callers that already say "resample".
resample_rule = timeframe_rule
