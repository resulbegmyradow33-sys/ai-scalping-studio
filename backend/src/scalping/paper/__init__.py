"""Paper trading: a persistent ledger and a restart-safe session runner.

The package is a tested library, not a service. It records what a simulated
strategy did — signals, orders, fills, positions, trades and equity — in a local
SQLite file and can resume that record after a restart. It has no network code,
no credentials and no path to a real broker; the default mode of the project is
paper trading only.

:class:`~scalping.paper.ledger.PaperLedger` is the storage layer;
:class:`~scalping.paper.runner.PaperSession` drives a
:class:`~scalping.env.simulator.Simulator` bar by bar and writes through it.
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
    "DuplicateError",
    "EquityPoint",
    "FillRecord",
    "LedgerError",
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
