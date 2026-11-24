"""Tests for the persistent paper ledger and the restart-safe session runner.

The ledger is the durable record, so the tests cover both halves of that claim:
every record type survives a reopen, and the runner rebuilds the same simulator
state from the recorded actions before it appends anything. The deterministic
replay test compares two complete databases row by row, which is the strongest
statement the slice can make: same bars, same policy, same ledger.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from scalping.config import ContractSpec, CostConfig, Settings
from scalping.env.broker import PaperBroker
from scalping.features.build import FeatureFrame
from scalping.paper import (
    SCHEMA_VERSION,
    DuplicateError,
    LedgerError,
    PaperLedger,
    PaperSession,
    ResumeError,
    provenance_config,
    session_provenance,
)
from scalping.types import Bar, Side

BASE = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- helpers


def flat_frame(
    n: int = 300,
    price: float = 1000.0,
    spread: float = 0.20,
    atr_value: float = 0.5,
) -> FeatureFrame:
    """A frame whose bars never move, so only a timeout can close a trade.

    ATR 0.5 with a 0.002 bar range keeps both stops far outside every bar while
    still sizing a full lot, which makes the run arithmetic independent of any
    price path or seed.
    """
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    ohlc = np.column_stack(
        [
            np.full(n, price),
            np.full(n, price + 0.001),
            np.full(n, price - 0.001),
            np.full(n, price),
        ]
    )
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, spread),
        atr=np.full(n, atr_value),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def buy_when_flat(obs: np.ndarray) -> int:
    """Open a long whenever the position feature says the account is flat."""
    return 1 if obs[-5] == 0 else 0


def dump(ledger: PaperLedger) -> dict[str, list[tuple[object, ...]]]:
    """Every row of every table, in insertion order, for byte-level comparison."""
    tables = ("sessions", "signals", "orders", "fills", "positions", "trades", "equity_curve")
    return {
        table: [tuple(row) for row in ledger.connection.execute(f"SELECT * FROM {table}")]
        for table in tables
    }


def make_session(
    ledger: PaperLedger,
    *,
    session_id: str = "s1",
    started_at: datetime = BASE,
    bars: int = 0,
) -> PaperSession:
    session = PaperSession(
        ledger,
        flat_frame(),
        Settings(),
        buy_when_flat,
        session_id=session_id,
        started_at=started_at,
        strategy="flow_momentum",
        model_version="m3",
    )
    if bars:
        session.run(bars)
    return session


# ----------------------------------------------------------------------------- schema


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "paper.db"
    with PaperLedger(path) as book:
        assert book.schema_version == SCHEMA_VERSION
        assert book.init_schema() == SCHEMA_VERSION
        assert book.init_schema() == SCHEMA_VERSION
        rows = book.connection.execute("SELECT version FROM schema_version").fetchall()
        assert [int(row["version"]) for row in rows] == [SCHEMA_VERSION]
        tables = {
            str(row["name"])
            for row in book.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "sessions",
        "signals",
        "orders",
        "fills",
        "positions",
        "trades",
        "equity_curve",
        "schema_version",
    } <= tables


def test_pragmas_are_set(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        assert int(book.connection.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        assert int(book.connection.execute("PRAGMA busy_timeout").fetchone()[0]) == 5000
        assert str(book.connection.execute("PRAGMA journal_mode").fetchone()[0]) == "wal"


# ------------------------------------------------------------------------- round trip


def test_every_record_type_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "paper.db"
    book = PaperLedger(path)
    opened = book.open_session(
        "s1",
        symbol="XAUUSD",
        timeframe="M1",
        strategy="flow_momentum",
        model_version="m3",
        started_at=BASE,
        config={"seed": 1},
    )
    assert opened.status == "running"
    assert opened.config == {"seed": 1}

    signal_id = book.record_signal(
        "s1",
        BASE,
        "buy",
        model_version="m3",
        probabilities={"buy": 0.6, "hold": 0.4},
    )
    order_id = book.record_order(
        "s1",
        signal_id=signal_id,
        ts=BASE,
        side=Side.LONG,
        lot=0.5,
        requested_price=2000.2,
        status="filled",
    )
    fill_id = book.record_fill(
        order_id,
        ts=BASE,
        price=2000.225,
        lot=0.5,
        spread=0.2,
        slippage=0.025,
        commission=3.5,
        sl=1999.725,
        tp=2001.225,
    )
    exit_ts = BASE + timedelta(minutes=5)
    trade_id = book.record_trade(
        "s1",
        side=Side.LONG,
        entry_ts=BASE,
        entry_price=2000.225,
        exit_ts=exit_ts,
        exit_price=2001.0,
        lot=0.5,
        pnl=35.25,
        r_multiple=1.5,
        exit_reason="tp",
    )
    mark_id = book.record_equity("s1", exit_ts, 10_035.25, 10_035.25)
    closed = book.close_session("s1", ended_at=exit_ts)
    book.close()

    assert closed.status == "closed"
    assert closed.ended_at == exit_ts

    with PaperLedger(path) as again:
        session = again.session("s1")
        assert session is not None
        assert session.started_at == BASE
        assert session.model_version == "m3"
        assert session.config == {"seed": 1}

        signals = again.signals("s1")
        assert len(signals) == 1
        assert signals[0].signal_id == signal_id
        assert signals[0].ts == BASE
        assert signals[0].action == "buy"
        assert signals[0].reason is None
        assert signals[0].model_version == "m3"
        assert signals[0].probabilities_json == '{"buy":0.6,"hold":0.4}'

        orders = again.orders("s1")
        assert len(orders) == 1
        assert orders[0].order_id == order_id
        assert orders[0].signal_id == signal_id
        assert orders[0].side == "long"
        assert orders[0].lot == pytest.approx(0.5)
        assert orders[0].requested_price == pytest.approx(2000.2)
        assert orders[0].status == "filled"
        assert orders[0].rejection_reason is None

        fills = again.fills("s1")
        assert len(fills) == 1
        assert fills[0].fill_id == fill_id
        assert fills[0].order_id == order_id
        assert fills[0].price == pytest.approx(2000.225)
        assert fills[0].spread == pytest.approx(0.2)
        assert fills[0].slippage == pytest.approx(0.025)
        assert fills[0].commission == pytest.approx(3.5)

        positions = again.positions("s1")
        assert len(positions) == 1
        assert positions[0].side == "long"
        assert positions[0].sl == pytest.approx(1999.725)
        assert positions[0].tp == pytest.approx(2001.225)
        assert positions[0].status == "closed"
        assert positions[0].closed_ts == exit_ts
        assert positions[0].closed_price == pytest.approx(2001.0)

        trades = again.trades("s1")
        assert len(trades) == 1
        assert trades[0].trade_id == trade_id
        assert trades[0].entry_ts == BASE
        assert trades[0].exit_ts == exit_ts
        assert trades[0].exit_reason == "tp"
        assert trades[0].pnl == pytest.approx(35.25)

        curve = again.equity("s1")
        assert len(curve) == 1
        assert curve[0].id == mark_id
        assert curve[0].ts == exit_ts
        assert curve[0].balance == pytest.approx(10_035.25)
        assert curve[0].equity == pytest.approx(10_035.25)


# -------------------------------------------------------------------------- rejections


def test_rejection_is_stored_with_its_reason(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        signal_id = book.record_signal("s1", BASE, "buy")
        book.record_order(
            "s1",
            signal_id=signal_id,
            ts=BASE,
            side=Side.LONG,
            lot=0.0,
            requested_price=2000.2,
            status="rejected",
            rejection_reason="no_signal",
        )
        orders = book.orders("s1")
        assert orders[0].status == "rejected"
        assert orders[0].rejection_reason == "no_signal"


def test_rejection_without_a_reason_is_refused(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        signal_id = book.record_signal("s1", BASE, "buy")
        with pytest.raises(ValueError, match="rejection_reason"):
            book.record_order(
                "s1",
                signal_id=signal_id,
                ts=BASE,
                side=Side.LONG,
                lot=0.0,
                status="rejected",
            )
        assert book.orders("s1") == []


# ----------------------------------------------------------------------------- resume


def test_resume_after_reopen_returns_position_and_last_ts(tmp_path: Path) -> None:
    path = tmp_path / "paper.db"
    book = PaperLedger(path)
    book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s", started_at=BASE)
    signal_id = book.record_signal("s1", BASE, "buy")
    order_id = book.record_order(
        "s1", signal_id=signal_id, ts=BASE, side=Side.LONG, lot=1.0, requested_price=2000.2
    )
    book.record_fill(
        order_id,
        ts=BASE,
        price=2000.225,
        lot=1.0,
        spread=0.2,
        slippage=0.025,
        commission=7.0,
        sl=1999.725,
        tp=2001.225,
    )
    later = BASE + timedelta(minutes=3)
    book.record_equity("s1", later, 10_000.0, 9_995.0)
    book.close()

    with PaperLedger(path) as again:
        state = again.resume("s1")
        assert state.status == "running"
        assert state.balance == pytest.approx(10_000.0)
        assert state.equity == pytest.approx(9_995.0)
        assert state.last_bar_ts == later
        assert state.pending_orders == ()
        assert state.position is not None
        assert state.position.side == "long"
        assert state.position.lot == pytest.approx(1.0)
        assert state.position.entry_price == pytest.approx(2000.225)
        assert state.position.sl == pytest.approx(1999.725)
        assert state.position.tp == pytest.approx(2001.225)
        assert state.position.status == "open"


def test_resume_before_any_mark_has_no_balance(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        state = book.resume("s1")
        assert state.balance is None
        assert state.equity is None
        assert state.position is None
        assert state.last_bar_ts is None


def test_resume_unknown_session_raises(tmp_path: Path) -> None:
    with (
        PaperLedger(tmp_path / "paper.db") as book,
        pytest.raises(LedgerError, match="unknown session"),
    ):
        book.resume("nope")


# ------------------------------------------------------------------------ constraints


def test_duplicate_session_id_is_rejected(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        with pytest.raises(DuplicateError):
            book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        assert book.session("s1") is not None


def test_session_provenance_is_stored_and_survives_a_reopen(tmp_path: Path) -> None:
    path = tmp_path / "paper.db"
    config = {
        "origin": "test",
        "provenance": provenance_config("dukascopy", False),
    }
    with PaperLedger(path) as book:
        record = book.open_session(
            "s1", symbol="XAUUSD", timeframe="M1", strategy="s", config=config
        )
        assert session_provenance(record.config) == ("dukascopy", False)
    with PaperLedger(path) as book:
        reopened = book.session("s1")
        assert reopened is not None
        assert session_provenance(reopened.config) == ("dukascopy", False)


def test_unknown_session_provenance_is_none_not_false(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        record = book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        assert session_provenance(record.config) == (None, None)
        assert session_provenance({}) == (None, None)
        assert provenance_config(None, None) == {}


def test_mistyped_session_provenance_is_refused_before_the_insert(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        with pytest.raises(ValueError, match="invalid provenance"):
            book.open_session(
                "bad",
                symbol="XAUUSD",
                timeframe="M1",
                strategy="s",
                config={"provenance": {"synthetic": "true"}},
            )
        assert book.session("bad") is None
        with pytest.raises(ValueError, match="invalid provenance"):
            book.open_session(
                "bad-source",
                symbol="XAUUSD",
                timeframe="M1",
                strategy="s",
                config={"source": "  "},
            )
        assert book.session("bad-source") is None


def test_provenance_config_rejects_a_mistyped_flag() -> None:
    with pytest.raises(TypeError, match="synthetic"):
        provenance_config("synthetic", "true")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        provenance_config("   ", True)


def test_conflicting_provenance_spellings_are_refused() -> None:
    with pytest.raises(ValueError, match="different values"):
        session_provenance({"synthetic": False, "extra": {"synthetic": True}})
    with pytest.raises(ValueError, match="different values"):
        session_provenance({"source": "csv", "provenance": {"source": "synthetic"}})
    assert session_provenance({"source": "csv", "provenance": {"source": "csv"}}) == ("csv", None)


@pytest.mark.parametrize(
    "config",
    [
        {"source": "csv", "synthetic": False},
        {"provenance": {"source": "csv", "synthetic": False}},
        {"data": {"source": "csv", "synthetic": False}},
        {"extra": {"source": "csv", "synthetic": False}},
        {"extra": {"provenance": {"source": "csv", "synthetic": False}}},
        {"extra": {"data": {"source": "csv", "synthetic": False}}},
    ],
)
def test_every_documented_provenance_spelling_is_read(config: dict[str, object]) -> None:
    """``docs/orchestrator.md`` promises all six spellings keep reading.

    A session config written before the canonical ``extra.provenance`` block
    still has to report its source and flag, so an upgrade does not turn known
    evidence into unknown.
    """
    assert session_provenance(config) == ("csv", False)


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        with pytest.raises(LedgerError, match="unknown session"):
            book.record_signal("missing", BASE, "hold")
        with pytest.raises(LedgerError, match="unknown signal"):
            book.record_order(
                "s1", signal_id=999, ts=BASE, side=Side.LONG, lot=0.1, status="filled"
            )
        with pytest.raises(LedgerError, match="unknown order"):
            book.record_fill(999, ts=BASE, price=2000.0, lot=0.1)
        assert book.signals("s1") == []


def test_fill_on_a_rejected_order_is_refused(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        signal_id = book.record_signal("s1", BASE, "buy")
        order_id = book.record_order(
            "s1",
            signal_id=signal_id,
            ts=BASE,
            side=Side.LONG,
            lot=0.0,
            status="rejected",
            rejection_reason="no_signal",
        )
        with pytest.raises(LedgerError, match="only a filled order"):
            book.record_fill(order_id, ts=BASE, price=2000.0, lot=0.1)
        assert book.fills("s1") == []


def test_second_open_position_is_refused(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="s")
        first = book.record_signal("s1", BASE, "buy")
        first_order = book.record_order("s1", signal_id=first, ts=BASE, side=Side.LONG, lot=0.1)
        book.record_fill(
            first_order,
            ts=BASE,
            price=2000.0,
            lot=0.1,
            sl=1999.0,
            tp=2001.0,
        )
        second = book.record_signal("s1", BASE + timedelta(minutes=1), "buy")
        second_order = book.record_order(
            "s1", signal_id=second, ts=BASE + timedelta(minutes=1), side=Side.LONG, lot=0.1
        )
        with pytest.raises(LedgerError, match="already has an open position"):
            book.record_fill(
                second_order,
                ts=BASE + timedelta(minutes=1),
                price=2000.0,
                lot=0.1,
                sl=1999.0,
                tp=2001.0,
            )


# ------------------------------------------------------------------------- P&L algebra


@pytest.mark.parametrize(
    ("side", "entry", "exit_price", "expected"),
    [
        (Side.LONG, 2000.2, 2001.4, 56.5),
        (Side.SHORT, 2000.0, 2001.6, -83.5),
    ],
)
def test_trade_pnl_invariant(
    tmp_path: Path,
    side: Side,
    entry: float,
    exit_price: float,
    expected: float,
) -> None:
    """``pnl == (exit - entry) * side * lot * contract_size - commission``."""
    contract = ContractSpec(size=100.0)
    costs = CostConfig(slippage_atr_frac=0.0, commission_per_lot=7.0)
    lot = 0.5
    broker = PaperBroker(contract, costs, atr=1.0)
    broker.open(
        side,
        lot,
        BASE,
        bid=2000.0,
        ask=2000.2,
        sl=1990.0 if side is Side.LONG else 2010.0,
        tp=2010.0 if side is Side.LONG else 1990.0,
        strategy="t",
    )
    bar = Bar(
        ts=BASE + timedelta(minutes=1),
        open=2001.0,
        high=2002.0,
        low=2000.5,
        close=2001.5,
        bid_close=2001.4,
        ask_close=2001.6,
        spread=0.2,
        volume=1,
    )
    trade = broker.close(bar, "manual")
    commission = costs.commission_per_lot * lot
    expected_pnl = (trade.exit - trade.entry) * int(side) * lot * contract.size - commission
    assert trade.pnl == pytest.approx(expected_pnl)
    assert trade.pnl == pytest.approx(expected)
    assert trade.entry == pytest.approx(entry)
    assert trade.exit == pytest.approx(exit_price)

    with PaperLedger(tmp_path / "paper.db") as book:
        book.open_session("s1", symbol="XAUUSD", timeframe="M1", strategy="t")
        book.record_trade(
            "s1",
            side=trade.side,
            entry_ts=trade.entry_ts,
            entry_price=trade.entry,
            exit_ts=trade.exit_ts,
            exit_price=trade.exit,
            lot=trade.lot,
            pnl=trade.pnl,
            r_multiple=trade.r_multiple,
            exit_reason=trade.reason,
        )
        stored = book.trades("s1")[0]
        recomputed = (stored.exit_price - stored.entry_price) * int(
            side
        ) * stored.lot * contract.size - commission
        assert stored.pnl == pytest.approx(recomputed)


# ---------------------------------------------------------------------- replay / resume


def test_deterministic_replay_produces_identical_ledgers(tmp_path: Path) -> None:
    first_path = tmp_path / "first.db"
    second_path = tmp_path / "second.db"
    for path in (first_path, second_path):
        with PaperLedger(path) as book:
            session = make_session(book, started_at=BASE)
            result = session.run(500)
            assert result.done is True
            assert result.reason == "data_end"

    with PaperLedger(first_path) as first, PaperLedger(second_path) as second:
        left = dump(first)
        right = dump(second)
    assert left == right
    assert len(left["signals"]) > 0
    assert len(left["orders"]) > 0
    assert len(left["fills"]) > 0
    assert len(left["trades"]) > 0
    assert len(left["equity_curve"]) > 0


def test_runner_resumes_after_reopen_without_duplicating_history(tmp_path: Path) -> None:
    straight_path = tmp_path / "straight.db"
    split_path = tmp_path / "split.db"

    with PaperLedger(straight_path) as book:
        make_session(book, started_at=BASE).run(500)

    with PaperLedger(split_path) as book:
        make_session(book, started_at=BASE).run(60)
    with PaperLedger(split_path) as book:
        resumed = make_session(book, started_at=BASE)
        state = resumed.resume()
        assert state.position is not None  # the 60th bar is inside the trade
        result = resumed.run(500)
        assert result.done is True

    with PaperLedger(straight_path) as straight, PaperLedger(split_path) as split:
        assert dump(straight) == dump(split)


def test_run_after_resume_continues_from_the_last_bar(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        session = make_session(book)
        first = session.run(10)
        assert first.steps == 10
        assert first.last_bar_ts is not None
        second = session.run(5)
        assert second.steps == 5
        assert second.last_bar_ts is not None
        assert second.last_bar_ts > first.last_bar_ts
        signals = book.signals("s1")
        assert len(signals) == 15
        assert signals[-1].ts + timedelta(minutes=1) == second.last_bar_ts


def test_resume_refuses_a_closed_session(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        session = make_session(book)
        session.run(5)
        session.close()
        with pytest.raises(LedgerError, match="closed"):
            session.run(1)
    with PaperLedger(tmp_path / "paper.db") as book:
        resumed = make_session(book)
        with pytest.raises(ResumeError, match="closed"):
            resumed.resume()


def test_equity_curve_reconciles_with_trades(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        session = make_session(book)
        result = session.run(500)
        assert result.done is True
        trades = book.trades("s1")
        assert trades
        curve = book.equity("s1")
        final = curve[-1]
        expected = 10_000.0 + sum(trade.pnl for trade in trades)
        assert final.balance == pytest.approx(expected)
        assert final.equity == pytest.approx(expected)


def test_short_frame_records_nothing_and_reports_no_bars(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        session = PaperSession(
            book,
            flat_frame(n=10),
            Settings(),
            buy_when_flat,
            session_id="s1",
        )
        result = session.run(5)
        assert result.steps == 0
        assert result.done is True
        assert result.reason == "no_bars"
        assert book.signals("s1") == []


# ---------------------------------------------------------------- no real-money path


def test_paper_package_has_no_network_or_broker_imports() -> None:
    """The library must stay a local writer; a network import is a red flag."""
    package = Path(__file__).resolve().parents[1] / "src" / "scalping" / "paper"
    forbidden = (
        "socket",
        "requests",
        "httpx",
        "urllib",
        "http.client",
        "MetaTrader5",
        "mt5",
        "subprocess",
    )
    for path in sorted(package.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert f"import {name}" not in source, f"{path.name} imports {name}"
            assert f"from {name}" not in source, f"{path.name} imports from {name}"


def test_sqlite_integrity_holds_after_a_run(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as book:
        make_session(book).run(500)
        rows = book.connection.execute("PRAGMA integrity_check").fetchall()
        assert [str(row[0]) for row in rows] == ["ok"]
        with pytest.raises(sqlite3.IntegrityError):
            book.connection.execute(
                "INSERT INTO orders (session_id, ts, side, lot, status)"
                " VALUES ('s1', 'x', 'long', 0.1, 'rejected')"
            )
