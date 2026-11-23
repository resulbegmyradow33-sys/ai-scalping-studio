"""Paper-trading API: session records and the history behind them.

Two rules shape this module.

**The ledger path is configuration.** It lives on ``app.state`` (installed by
the app factory from ``data_root`` or an explicit ``paper_ledger`` argument) and
no request field can move it. A session id is an opaque token validated by
:data:`~scalping.api.schemas.PaperSessionId`; it is used as a SQL parameter and
never joined to a path.

**Reads never write.** Every GET opens its own ``mode=ro`` connection through
:meth:`~scalping.paper.ledger.PaperLedger.open_reader`, runs plain queries in
autocommit and closes. It does not create the file, does not run a migration and
does not take a write transaction, so a GET keeps working while the session
runner holds ``BEGIN IMMEDIATE`` (WAL readers see the last committed snapshot).

Writes (open, close) open a normal ledger, which migrates idempotently inside a
short ``BEGIN IMMEDIATE``. Nothing here starts a runner, schedules a bar or
talks to a broker; a session record is a row the future scheduler will drive.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request

from scalping.api.schemas import (
    PaperEquityPage,
    PaperEquityPoint,
    PaperOrder,
    PaperOrderPage,
    PaperPosition,
    PaperPositionPage,
    PaperSessionCreate,
    PaperSessionCreated,
    PaperSessionId,
    PaperSessionItem,
    PaperSessionPage,
    PaperSessionSummary,
    PaperTrade,
    PaperTradePage,
)
from scalping.paper.ledger import (
    DuplicateError,
    EquityPoint,
    OrderRecord,
    PaperLedger,
    PositionRecord,
    SessionRecord,
    TradeRecord,
)

__all__ = ["PAGE_DEFAULT", "PAGE_MAX", "router"]

router = APIRouter(prefix="/api/paper", tags=["paper"])

#: A page is bounded on both ends: enough for a table view, small enough that a
#: hostile ``limit`` cannot ask the server to materialise a whole session.
PAGE_DEFAULT: Final[int] = 100
PAGE_MAX: Final[int] = 1000


# ------------------------------------------------------------------ connections


@contextmanager
def _reader(request: Request) -> Iterator[PaperLedger | None]:
    """A read-only view, or ``None`` when there is no ledger yet.

    ``None`` covers both a missing file and a file whose schema is still being
    created by the first writer, so a GET that races a POST answers with an
    empty page instead of a 500.
    """
    path = Path(request.app.state.paper_ledger_path)
    if not path.exists():
        yield None
        return
    book = PaperLedger.open_reader(path)
    try:
        yield book if book.has_schema() else None
    finally:
        book.close()


@contextmanager
def _writer(request: Request) -> Iterator[PaperLedger]:
    """A normal ledger for one short write; the file is created on first use."""
    book = PaperLedger(Path(request.app.state.paper_ledger_path))
    try:
        yield book
    finally:
        book.close()


# --------------------------------------------------------------------- mapping


def _not_found(session_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"unknown paper session: {session_id}")


def _session_item(record: SessionRecord) -> PaperSessionItem:
    return PaperSessionItem(
        session_id=record.session_id,
        symbol=record.symbol,
        timeframe=record.timeframe,
        strategy=record.strategy,
        model_version=record.model_version,
        status=record.status,
        started_at=record.started_at,
        ended_at=record.ended_at,
    )


def _position(record: PositionRecord) -> PaperPosition:
    return PaperPosition(
        position_id=record.position_id,
        side=record.side,
        lot=record.lot,
        entry_ts=record.entry_ts,
        entry_price=record.entry_price,
        sl=record.sl,
        tp=record.tp,
        status=record.status,
        closed_ts=record.closed_ts,
        closed_price=record.closed_price,
    )


def _order(record: OrderRecord) -> PaperOrder:
    return PaperOrder(
        order_id=record.order_id,
        signal_id=record.signal_id,
        ts=record.ts,
        side=record.side,
        lot=record.lot,
        requested_price=record.requested_price,
        status=record.status,
        rejection_reason=record.rejection_reason,
    )


def _trade(record: TradeRecord) -> PaperTrade:
    return PaperTrade(
        trade_id=record.trade_id,
        side=record.side,
        entry_ts=record.entry_ts,
        entry_price=record.entry_price,
        exit_ts=record.exit_ts,
        exit_price=record.exit_price,
        lot=record.lot,
        pnl=record.pnl,
        r_multiple=record.r_multiple,
        exit_reason=record.exit_reason,
    )


def _equity_point(record: EquityPoint) -> PaperEquityPoint:
    return PaperEquityPoint(ts=record.ts, balance=record.balance, equity=record.equity)


def _summary(book: PaperLedger, session_id: str) -> PaperSessionSummary:
    """Read the session's account state in one place; the id is known to exist."""
    record = book.session(session_id)
    if record is None:  # pragma: no cover - callers check first
        raise _not_found(session_id)
    state = book.resume(session_id)
    return PaperSessionSummary(
        session_id=record.session_id,
        symbol=record.symbol,
        timeframe=record.timeframe,
        strategy=record.strategy,
        model_version=record.model_version,
        status=record.status,
        started_at=record.started_at,
        ended_at=record.ended_at,
        balance=state.balance,
        equity=state.equity,
        open_position=None if state.position is None else _position(state.position),
        trade_count=book.count_trades(session_id),
        last_bar_ts=state.last_bar_ts,
        config=record.config,
    )


# ---------------------------------------------------------------------- routes


@router.post("/sessions", response_model=PaperSessionCreated, status_code=201)
def open_session(body: PaperSessionCreate, request: Request) -> PaperSessionCreated:
    """Record a new running session; the server generates the id.

    The body carries no path and no id, so a caller cannot choose where the
    ledger lives or collide with another session's row. ``config`` is stored
    under ``extra`` with ``{"origin": "api"}``; no runner is started here.
    """
    session_id = uuid4().hex
    config = {"origin": "api", "extra": dict(body.config or {})}
    try:
        with _writer(request) as book:
            record = book.open_session(
                session_id,
                symbol=body.symbol,
                timeframe=body.timeframe,
                strategy=body.strategy,
                model_version=body.model_version,
                config=config,
            )
    except DuplicateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PaperSessionCreated(
        session_id=record.session_id,
        status=record.status,
        started_at=record.started_at,
    )


@router.get("/sessions", response_model=PaperSessionPage)
def list_sessions(
    request: Request,
    limit: int = Query(default=PAGE_DEFAULT, ge=1, le=PAGE_MAX),
    offset: int = Query(default=0, ge=0),
) -> PaperSessionPage:
    """Sessions newest first; an absent ledger is an empty page, not a 404."""
    with _reader(request) as book:
        if book is None:
            return PaperSessionPage(items=[], total=0, limit=limit, offset=offset)
        records, total = book.session_page(limit=limit, offset=offset)
    return PaperSessionPage(
        items=[_session_item(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}", response_model=PaperSessionSummary)
def session_summary(session_id: PaperSessionId, request: Request) -> PaperSessionSummary:
    """Balance, equity, the open position, the trade count and the last bar."""
    with _reader(request) as book:
        if book is None:
            raise _not_found(session_id)
        return _summary(book, session_id)


@router.post("/sessions/{session_id}/close", response_model=PaperSessionSummary)
def close_session(session_id: PaperSessionId, request: Request) -> PaperSessionSummary:
    """Close a running session.

    Closing is idempotent in effect: the ledger's ``UPDATE`` is guarded by
    ``status = 'running'``, so a second close can never move ``ended_at``. The
    API still answers 409 for a session that is already closed, so a caller can
    tell a retry from a first close.
    """
    with _writer(request) as book:
        record = book.session(session_id)
        if record is None:
            raise _not_found(session_id)
        if record.status != "running":
            raise HTTPException(
                status_code=409,
                detail=f"paper session {session_id} is already {record.status}",
            )
        book.close_session(session_id)
        return _summary(book, session_id)


@router.get("/sessions/{session_id}/positions", response_model=PaperPositionPage)
def session_positions(
    session_id: PaperSessionId,
    request: Request,
    limit: int = Query(default=PAGE_DEFAULT, ge=1, le=PAGE_MAX),
    offset: int = Query(default=0, ge=0),
) -> PaperPositionPage:
    """Positions oldest first, including the closed ones."""
    with _reader(request) as book:
        if book is None or book.session(session_id) is None:
            raise _not_found(session_id)
        records, total = book.position_page(session_id, limit=limit, offset=offset)
    return PaperPositionPage(
        items=[_position(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}/orders", response_model=PaperOrderPage)
def session_orders(
    session_id: PaperSessionId,
    request: Request,
    limit: int = Query(default=PAGE_DEFAULT, ge=1, le=PAGE_MAX),
    offset: int = Query(default=0, ge=0),
) -> PaperOrderPage:
    """Order attempts oldest first, rejections included with their reason."""
    with _reader(request) as book:
        if book is None or book.session(session_id) is None:
            raise _not_found(session_id)
        records, total = book.order_page(session_id, limit=limit, offset=offset)
    return PaperOrderPage(
        items=[_order(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}/history", response_model=PaperTradePage)
def session_history(
    session_id: PaperSessionId,
    request: Request,
    limit: int = Query(default=PAGE_DEFAULT, ge=1, le=PAGE_MAX),
    offset: int = Query(default=0, ge=0),
) -> PaperTradePage:
    """Closed trades oldest first, exactly as the broker settled them."""
    with _reader(request) as book:
        if book is None or book.session(session_id) is None:
            raise _not_found(session_id)
        records, total = book.trade_page(session_id, limit=limit, offset=offset)
    return PaperTradePage(
        items=[_trade(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}/equity", response_model=PaperEquityPage)
def session_equity(
    session_id: PaperSessionId,
    request: Request,
    limit: int = Query(default=PAGE_DEFAULT, ge=1, le=PAGE_MAX),
    offset: int = Query(default=0, ge=0),
) -> PaperEquityPage:
    """The newest equity window in chronological order; ``offset`` pages back.

    A chart wants the tail of the curve, so ``offset=0`` is the last ``limit``
    marks. Walk ``offset`` upwards to page into older history; ``total`` says
    how many marks the session has.
    """
    with _reader(request) as book:
        if book is None or book.session(session_id) is None:
            raise _not_found(session_id)
        records, total = book.equity_page(session_id, limit=limit, offset=offset)
    return PaperEquityPage(
        items=[_equity_point(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )
