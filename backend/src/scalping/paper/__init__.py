"""Paper trading: a persistent ledger and a restart-safe session runner.

The package is a tested library, not a service. It records what a simulated
strategy did — signals, orders, fills, positions, trades and equity — in a local
SQLite file and can resume that record after a restart. It has no network code,
no credentials and no path to a real broker; the default mode of the project is
paper trading only.

:class:`~scalping.paper.ledger.PaperLedger` is the storage layer;
:class:`~scalping.paper.runner.PaperSession` drives a
:class:`~scalping.env.simulator.Simulator` bar by bar and writes through it.
:class:`~scalping.paper.mt5.Mt5DemoAdapter` is a disabled skeleton that refuses
every operation; it is not an integration and real-money execution is out of
scope.
"""

from __future__ import annotations

from scalping.paper.ledger import (
    SCHEMA_VERSION,
    DuplicateError,
    EquityPoint,
    FillRecord,
    LedgerError,
    OrderRecord,
    PaperLedger,
    PositionRecord,
    ResumeState,
    SessionRecord,
    SignalRecord,
    TradeRecord,
)
from scalping.paper.mt5 import AdapterDisabled, Mt5DemoAdapter
from scalping.paper.runner import (
    ACTION_CODES,
    ACTION_NAMES,
    PaperRunResult,
    PaperSession,
    ResumeError,
)

__all__ = [
    "ACTION_CODES",
    "ACTION_NAMES",
    "SCHEMA_VERSION",
    "AdapterDisabled",
    "DuplicateError",
    "EquityPoint",
    "FillRecord",
    "LedgerError",
    "Mt5DemoAdapter",
    "OrderRecord",
    "PaperLedger",
    "PaperRunResult",
    "PaperSession",
    "PositionRecord",
    "ResumeError",
    "ResumeState",
    "SessionRecord",
    "SignalRecord",
    "TradeRecord",
]
