"""Tests for the autonomous DSL-driven paper runner.

The runner is the first paper path that does not need a hand-written policy: a
validated ``StrategySpec`` is the only strategy input. These tests pin the
properties that make that safe and useful:

* an invalid spec is refused before any session row exists;
* the same store slice, spec and seed produce byte-identical ledgers;
* a resumed session equals one uninterrupted run;
* every signal carries the canonical spec hash and a non-empty reason;
* a NO TRADE spec records decisions but opens nothing;
* the simulator's risk gate is recorded, never caught or bypassed;
* the spec's sizing node cannot reach the broker;
* the module holds no network, credential or real-money path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from scalping.config import RiskConfig, Settings
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.paper import PaperLedger
from scalping.paper.autonomous import AutonomousPaperRunner
from scalping.strategy.dsl import (
    DSL_SPEC_VERSION,
    AtrMultiple,
    Compare,
    Const,
    DslValidationError,
    EntryRule,
    ExitRule,
    HoldingLimit,
    NoTrade,
    StrategySpec,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
END = datetime(2026, 1, 7, tzinfo=UTC)
BASE = datetime(2026, 1, 5, 10, tzinfo=UTC)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def bars_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two deterministic synthetic days, shared read-only by the tests."""
    root = tmp_path_factory.mktemp("autonomous-bars")
    bars = SyntheticProvider(seed=7).fetch(SYMBOL, START, END)
    ParquetStore(root).write(SYMBOL, TIMEFRAME, bars)
    return root


@pytest.fixture
def falling_root(tmp_path: Path) -> Path:
    """A falling series so a long entry must lose, whatever the lot size."""
    root = tmp_path / "falling"
    ParquetStore(root).write(SYMBOL, TIMEFRAME, falling_bars())
    return root


# ----------------------------------------------------------------------------- specs


def trading_spec(*, sizing: AtrMultiple | None = None) -> StrategySpec:
    """Long every bar when flat; close on the holding limit.

    The entry cannot be satisfied while a position is open, so the simulator
    rejects those attempts with ``max_positions``. That is the intended paper
    path: the DSL states intent, the simulator's risk engine decides execution.
    """
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(10),
        sizing=sizing,
        max_lookback=2,
    )


def no_trade_spec() -> StrategySpec:
    """Every bar is NO TRADE: entries are true but the no-trade gate wins."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        no_trade=NoTrade(Compare(Const(1.0), ">=", Const(0.0))),
        max_lookback=1,
    )


def falling_bars(n: int = 400, start: float = 2000.0, step: float = 0.5) -> pl.DataFrame:
    """Deterministic M1 bars whose close falls ``step`` every minute."""
    stamps = [START + timedelta(minutes=index) for index in range(n)]
    closes = [start - step * index for index in range(n)]
    opens = [closes[0], *closes[:-1]]
    spread = 0.20
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": opens,
            "high": [max(open_, close) + 0.05 for open_, close in zip(opens, closes, strict=True)],
            "low": [min(open_, close) - 0.05 for open_, close in zip(opens, closes, strict=True)],
            "close": closes,
            "bid_close": closes,
            "ask_close": [close + spread for close in closes],
            "spread": [spread] * n,
            "volume": [100] * n,
        }
    )


# ---------------------------------------------------------------------------- helpers


def make_runner(
    ledger: PaperLedger,
    spec: StrategySpec,
    root: Path,
    *,
    session_id: str = "auto-test",
    settings: Settings | None = None,
    cache: bool = True,
) -> AutonomousPaperRunner:
    return AutonomousPaperRunner(
        ledger,
        spec,
        SYMBOL,
        TIMEFRAME,
        Settings() if settings is None else settings,
        session_id=session_id,
        cache=cache,
        store=ParquetStore(root),
        started_at=BASE,
    )


def dump(ledger: PaperLedger) -> dict[str, list[tuple[object, ...]]]:
    """Every row of every table, for byte-level comparison across ledgers."""
    tables = ("sessions", "signals", "orders", "fills", "positions", "trades", "equity_curve")
    return {
        table: [tuple(row) for row in ledger.connection.execute(f"SELECT * FROM {table}")]
        for table in tables
    }


# ----------------------------------------------------------------------------- tests


def test_invalid_spec_is_rejected_before_any_state(tmp_path: Path) -> None:
    """The validator runs first: no session row, no cache directory, no read."""
    empty_root = tmp_path / "empty-data"
    bad = StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        max_lookback=0,  # outside the validated range
    )
    with PaperLedger(tmp_path / "paper.db") as ledger:
        with pytest.raises(DslValidationError, match="max_lookback"):
            AutonomousPaperRunner(
                ledger,
                bad,
                SYMBOL,
                TIMEFRAME,
                Settings(),
                session_id="rejected",
                data_root=empty_root,
            )
        assert ledger.session("rejected") is None
        count = ledger.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert int(count) == 0
    assert not (empty_root / ".feature-cache").exists()


def test_deterministic_replay_writes_identical_ledgers(tmp_path: Path, bars_root: Path) -> None:
    """Same store slice, spec and seed -> the same database, row for row."""
    spec = trading_spec()
    snapshots: list[dict[str, list[tuple[object, ...]]]] = []
    for name in ("first", "second"):
        with PaperLedger(tmp_path / f"{name}.db") as ledger:
            runner = make_runner(ledger, spec, bars_root, session_id="det")
            result = runner.run(150)
            assert result.steps == 150
            assert ledger.count_trades("det") > 0
            snapshots.append(dump(ledger))
    assert snapshots[0] == snapshots[1]


def test_resume_matches_a_single_full_run(tmp_path: Path, bars_root: Path) -> None:
    """60 bars + a reopen + 440 bars is the same session as one 500-bar run."""
    spec = trading_spec()
    with PaperLedger(tmp_path / "full.db") as ledger:
        runner = make_runner(ledger, spec, bars_root, session_id="resume")
        assert runner.run(500).steps == 500
        full = dump(ledger)

    path = tmp_path / "resumed.db"
    with PaperLedger(path) as ledger:
        make_runner(ledger, spec, bars_root, session_id="resume").run(60)

    with PaperLedger(path) as ledger:
        resumed_runner = make_runner(ledger, spec, bars_root, session_id="resume")
        state = resumed_runner.resume()
        assert state.status == "running"
        assert state.last_bar_ts is not None
        assert resumed_runner.run(440).steps == 440
        resumed = dump(ledger)

    assert full == resumed


def test_signals_carry_the_spec_hash_and_a_reason(tmp_path: Path, bars_root: Path) -> None:
    spec = trading_spec()
    path = tmp_path / "attribution.db"
    with PaperLedger(path) as ledger:
        runner = make_runner(ledger, spec, bars_root, session_id="attr")
        runner.run(80)
        expected = spec.canonical_hash()
        assert runner.spec_hash == expected

        signals = ledger.signals("attr")
        assert signals
        for signal in signals:
            assert signal.model_version == expected
            assert signal.reason is not None and signal.reason.strip()
            assert signal.probabilities_json is None

        session = ledger.session("attr")
        assert session is not None
        assert session.strategy == "dsl"
        assert session.model_version == expected
        dsl = session.config["extra"]["dsl"]
        assert dsl["spec_hash"] == expected
        assert dsl["spec_version"] == DSL_SPEC_VERSION
        assert dsl["spec"] == spec.canonical_dict()

    # Attribution is durable, not a view over in-memory state.
    with PaperLedger.open_reader(path) as reader:
        reopened = reader.session("attr")
        assert reopened is not None
        assert reopened.model_version == spec.canonical_hash()
        assert reopened.config["extra"]["dsl"]["spec_hash"] == spec.canonical_hash()
        assert all(
            signal.model_version == spec.canonical_hash() for signal in reader.signals("attr")
        )


def test_flat_no_trade_spec_records_signals_without_trades(tmp_path: Path, bars_root: Path) -> None:
    with PaperLedger(tmp_path / "flat.db") as ledger:
        runner = make_runner(ledger, no_trade_spec(), bars_root, session_id="flat")
        runner.run(100)
        signals = ledger.signals("flat")
        assert len(signals) == 100
        assert {signal.action for signal in signals} == {"hold"}
        assert {signal.reason for signal in signals} == {"no_trade"}
        assert ledger.orders("flat") == []
        assert ledger.trades("flat") == []
        curve = ledger.equity("flat")
        assert len(curve) == 100
        initial = Settings().contract.initial_balance
        assert all(point.balance == initial and point.equity == initial for point in curve)


def test_daily_loss_gate_is_recorded_and_not_bypassed(tmp_path: Path, falling_root: Path) -> None:
    """A losing trade trips the gate; every later entry is refused and logged."""
    settings = Settings(risk=RiskConfig(daily_loss_pct=0.001))
    with PaperLedger(tmp_path / "risk.db") as ledger:
        runner = make_runner(
            ledger, trading_spec(), falling_root, session_id="risk", settings=settings
        )
        result = runner.run(200)
        assert result.done is False

        signals = ledger.signals("risk")
        blocked = [
            index
            for index, signal in enumerate(signals)
            if signal.reason is not None and "daily_loss_limit" in signal.reason
        ]
        assert blocked, "the risk gate never reported a block"
        first = blocked[0]
        # The gate keeps refusing after the loss; the runner neither clears nor
        # catches it, and the simulator keeps running.
        assert all(
            signal.reason is not None and "daily_loss_limit" in signal.reason
            for signal in signals[first:]
        )
        assert ledger.count_trades("risk") == 1
        assert ledger.trades("risk")[0].pnl < 0

        orders = ledger.orders("risk")
        filled = [order for order in orders if order.status == "filled"]
        rejected = [order for order in orders if order.status == "rejected"]
        assert len(filled) == 1
        assert rejected
        assert all(order.rejection_reason == "daily_loss_limit" for order in rejected)


def test_sizing_node_cannot_change_the_risk_size(tmp_path: Path, bars_root: Path) -> None:
    """A spec may request size, but the simulator's RiskEngine owns it."""
    plain = trading_spec()
    oversized = trading_spec(sizing=AtrMultiple(1_000.0))
    assert plain.canonical_hash() != oversized.canonical_hash()

    with PaperLedger(tmp_path / "plain.db") as ledger:
        make_runner(ledger, plain, bars_root, session_id="plain").run(80)
        plain_fills = [(fill.lot, fill.price) for fill in ledger.fills("plain")]

    with PaperLedger(tmp_path / "sized.db") as ledger:
        make_runner(ledger, oversized, bars_root, session_id="sized").run(80)
        sized_fills = [(fill.lot, fill.price) for fill in ledger.fills("sized")]

    assert plain_fills
    assert plain_fills == sized_fills


def test_max_bars_caps_total_decisions_across_calls(tmp_path: Path, bars_root: Path) -> None:
    with PaperLedger(tmp_path / "cap.db") as ledger:
        runner = make_runner(ledger, trading_spec(), bars_root, session_id="cap")
        assert runner.run(max_bars=70).steps == 70
        assert runner.summary().bars_processed == 70
        # The budget is already spent; a second call cannot add a bar.
        assert runner.run(max_bars=70).steps == 0
        assert runner.summary().bars_processed == 70


def test_summary_reports_the_durable_session(tmp_path: Path, bars_root: Path) -> None:
    spec = trading_spec()
    with PaperLedger(tmp_path / "summary.db") as ledger:
        runner = make_runner(ledger, spec, bars_root, session_id="summary")
        runner.run(120)
        summary = runner.summary()
        trades = ledger.trades("summary")
        assert summary.session_id == "summary"
        assert summary.spec_hash == spec.canonical_hash()
        assert summary.status == "running"
        assert summary.bars_processed == 120
        assert summary.trades == len(trades)
        assert summary.realized_pnl == pytest.approx(sum(trade.pnl for trade in trades))
        assert summary.balance is not None
        assert summary.equity is not None
        assert summary.last_bar_ts is not None
        assert summary.position is None  # the frame ends on a flat account here


def test_autonomous_module_has_no_network_or_real_money_path() -> None:
    """A source scan, independent of the ledger tests: this file stays local."""
    path = Path(__file__).resolve().parents[1] / "src" / "scalping" / "paper" / "autonomous.py"
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "socket",
        "requests",
        "httpx",
        "urllib",
        "http.client",
        "MetaTrader5",
        "mt5",
        "subprocess",
        "ccxt",
        "websocket",
    )
    for name in forbidden:
        assert f"import {name}" not in source, f"autonomous.py imports {name}"
        assert f"from {name}" not in source, f"autonomous.py imports from {name}"
    # No second writer: every ledger row goes through PaperLedger's methods.
    assert "INSERT INTO" not in source
    assert "UPDATE " not in source
    assert ".connection" not in source
    # No order-routing vocabulary that would suggest a live path.
    assert "order_send" not in source
    assert "Real-money" in source and "out of scope" in source
