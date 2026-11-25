"""Tests for continuous paper operation and matured-outcome analysis.

These tests pin the properties that make an incremental paper run safe:

* no new bars means no work and no new rows;
* appending bars advances exactly the appended count (one bar stays
  provisional so no ``data_end`` close can poison a later replay);
* a restart resumes the same session without duplicating a signal or trade;
* an overlapping store write adds no duplicate signal;
* a gap or an out-of-order/duplicate bar is refused with a typed error before
  anything is written;
* only closed trades enter a matured batch; an open position is counted, not
  included;
* the outcome fingerprint is stable for the same batch and changes when a new
  trade matures;
* future bars do not change earlier decisions (no look-ahead).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from scalping.config import RiskConfig, Settings
from scalping.data.provenance import build_provenance, provenance_path, save_provenance
from scalping.data.store import ParquetStore
from scalping.orchestrator import JobStore, Orchestrator, scan_once
from scalping.orchestrator.paper_triggers import REASON_SYNTHETIC, PaperOutcomeTriggerSource
from scalping.paper import PaperLedger
from scalping.paper.continuous import (
    BarGapError,
    BarOrderError,
    ContinuousError,
    ContinuousPaperRunner,
    OutcomeBatch,
    SessionMismatchError,
    matured_outcomes,
    outcome_fingerprint,
)
from scalping.paper.ledger import session_provenance
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
BASE = datetime(2026, 1, 5, 10, tzinfo=UTC)  # a Monday inside the metals session
SETTINGS = Settings(risk=RiskConfig(daily_loss_pct=0.9))
SPREAD = 0.20


# --------------------------------------------------------------------------- fixtures


def holding_spec(bars: int = 5) -> StrategySpec:
    """Long every bar when flat; close when the holding limit is reached."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(bars),
        max_lookback=2,
    )


def make_bars(count: int, *, start_price: float = 2000.0) -> pl.DataFrame:
    """Deterministic, continuous M1 bars with a saw-tooth close.

    The range gives ATR > 0 so the simulator can size stops and entries, and
    the whole frame stays inside one open session window for the gap tests.
    """
    stamps = [BASE + timedelta(minutes=index) for index in range(count)]
    closes = [start_price + 0.5 * ((index % 8) - 4) for index in range(count)]
    opens = [closes[0], *closes[:-1]]
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": opens,
            "high": [max(open_, close) + 0.20 for open_, close in zip(opens, closes, strict=True)],
            "low": [min(open_, close) - 0.20 for open_, close in zip(opens, closes, strict=True)],
            "close": closes,
            "bid_close": closes,
            "ask_close": [close + SPREAD for close in closes],
            "spread": [SPREAD] * count,
            "volume": [100] * count,
        }
    )


def write(root: Path, bars: pl.DataFrame) -> int:
    return ParquetStore(root).write(SYMBOL, TIMEFRAME, bars)


def make_runner(
    ledger: PaperLedger,
    root: Path,
    *,
    spec: StrategySpec | None = None,
    scope: str | None = None,
    cache: bool = True,
) -> ContinuousPaperRunner:
    return ContinuousPaperRunner(
        ledger,
        holding_spec() if spec is None else spec,
        SYMBOL,
        TIMEFRAME,
        SETTINGS,
        store_root=root,
        cache=cache,
        scope=scope,
    )


def signal_tuples(ledger: PaperLedger, session_id: str) -> list[tuple[object, ...]]:
    return [
        (signal.ts, signal.action, signal.reason, signal.model_version)
        for signal in ledger.signals(session_id)
    ]


# ----------------------------------------------------------------------------- tests


def test_empty_store_does_no_work(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, tmp_path / "empty")
        summary = runner.advance()
        assert summary.status == "new"
        assert summary.new_bars == 0
        assert summary.new_signals == 0
        assert summary.new_trades == 0
        assert summary.last_bar_ts is None
        assert ledger.session(runner.session_id) is None


def test_no_new_bars_does_no_work(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write(root, make_bars(200))
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        first = runner.advance()
        assert first.new_bars > 0
        assert first.new_signals == first.new_bars
        assert first.status == "running"
        equity_before = len(ledger.equity(runner.session_id))

        second = runner.advance()
        assert second.new_bars == 0
        assert second.new_signals == 0
        assert second.new_trades == 0
        assert second.bars_processed == first.bars_processed
        assert second.last_bar_ts == first.last_bar_ts
        assert len(ledger.equity(runner.session_id)) == equity_before


def test_append_processes_only_bars_after_the_cursor(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write(root, make_bars(200))
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        first = runner.advance()
        cursor = first.last_bar_ts
        assert cursor is not None
        before = signal_tuples(ledger, runner.session_id)
        assert len(before) == first.bars_processed

        # A refetch that contains the old bars plus 20 new ones.
        assert write(root, make_bars(220)) == 20
        second = runner.advance()

        assert second.new_bars == 20
        assert second.new_signals == 20
        assert second.bars_processed == first.bars_processed + 20
        after = signal_tuples(ledger, runner.session_id)
        assert after[: len(before)] == before
        new_signals = after[len(before) :]
        assert all(signal[0] > cursor for signal in new_signals)
        timestamps = [signal[0] for signal in after]
        assert len(set(timestamps)) == len(timestamps)


def test_restart_midway_resumes_without_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "data"
    path = tmp_path / "paper.db"
    write(root, make_bars(200))

    with PaperLedger(path) as ledger:
        runner = make_runner(ledger, root)
        first = runner.advance()
        before = signal_tuples(ledger, runner.session_id)
        assert first.new_bars == len(before)

    # New bars arrive while the process is down.
    write(root, make_bars(250))

    with PaperLedger(path) as ledger:
        runner = make_runner(ledger, root)
        second = runner.advance()
        after = signal_tuples(ledger, runner.session_id)

        assert second.new_bars == 50
        assert second.new_signals == 50
        assert second.bars_processed == first.bars_processed + 50
        assert after[: len(before)] == before
        assert len({signal[0] for signal in after}) == len(after)

        natural_keys = [
            (trade.entry_ts, trade.exit_ts, trade.side, trade.pnl)
            for trade in ledger.trades(runner.session_id)
        ]
        assert len(set(natural_keys)) == len(natural_keys)


def test_overlapping_store_slice_adds_no_duplicate_signal(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write(root, make_bars(200))
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        first = runner.advance()

        # The same 200 bars are written again together with 20 new ones.
        assert write(root, make_bars(220)) == 20
        second = runner.advance()

        assert second.new_signals == 20
        timestamps = [signal.ts for signal in ledger.signals(runner.session_id)]
        assert len(timestamps) == first.bars_processed + 20
        assert len(set(timestamps)) == len(timestamps)


def test_gap_is_refused_before_any_write(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bars = make_bars(200)
    gapped = pl.concat([bars[:150], bars[151:]])
    write(root, gapped)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        with pytest.raises(BarGapError, match="missing bar"):
            runner.advance()
        assert ledger.session(runner.session_id) is None
        assert ledger.signals(runner.session_id) == []


def test_gap_after_progress_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write(root, make_bars(200))
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        first = runner.advance()
        before = len(ledger.signals(runner.session_id))

        appended = make_bars(220)[200:]
        holed = pl.concat([appended[:10], appended[11:]])  # drop minute 210
        write(root, holed)

        with pytest.raises(BarGapError, match="missing bar"):
            runner.advance()
        assert len(ledger.signals(runner.session_id)) == before
        assert first.new_bars > 0


def test_out_of_order_and_duplicate_bars_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    bars = make_bars(200)
    write(root, bars)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        swapped = pl.concat([bars[:100], bars[101:102], bars[100:101], bars[102:]])
        monkeypatch.setattr(runner._store, "read", lambda *args, **kwargs: swapped)
        with pytest.raises(BarOrderError, match="not sorted"):
            runner.advance()

        duplicated = pl.concat([bars, bars[-1:]])
        monkeypatch.setattr(runner._store, "read", lambda *args, **kwargs: duplicated)
        with pytest.raises(BarOrderError, match="duplicate"):
            runner.advance()
        assert ledger.session(runner.session_id) is None


def test_session_mismatch_is_refused(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, tmp_path / "empty")
        ledger.open_session(
            runner.session_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="a-different-spec",
        )
        with pytest.raises(SessionMismatchError, match="records strategy"):
            runner.advance()


def test_matured_outcomes_exclude_open_positions(tmp_path: Path) -> None:
    session_id = "outcomes"
    entry = BASE
    with PaperLedger(tmp_path / "paper.db") as ledger:
        ledger.open_session(
            session_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="hash-1",
            started_at=BASE,
            config={"origin": "test", "data": {"dataset_id": "dataset-1"}},
        )
        ledger.record_trade(
            session_id,
            side="long",
            entry_ts=entry,
            entry_price=2000.0,
            exit_ts=entry + timedelta(minutes=5),
            exit_price=2001.0,
            lot=0.10,
            pnl=10.0,
            r_multiple=1.5,
            exit_reason="take_profit",
        )
        order_id = ledger.record_order(
            session_id,
            signal_id=None,
            ts=entry + timedelta(minutes=10),
            side="short",
            lot=0.10,
            requested_price=2000.0,
            status="filled",
        )
        ledger.record_fill(
            order_id,
            ts=entry + timedelta(minutes=10),
            price=2000.0,
            lot=0.10,
            sl=2005.0,
            tp=1990.0,
        )

        batch = matured_outcomes(ledger, session_id)
        assert isinstance(batch, OutcomeBatch)
        assert batch.closed_trades == 1
        assert batch.open_trades == 1
        assert batch.strategy_hash == "hash-1"
        assert batch.dataset_id == "dataset-1"
        row = batch.rows[0]
        assert row.side == "long"
        assert row.lot == pytest.approx(0.10)
        assert row.pnl == pytest.approx(10.0)
        assert row.r_multiple == pytest.approx(1.5)
        assert row.exit_reason == "take_profit"
        assert row.entry_ts == entry
        assert row.strategy_hash == "hash-1"
        assert row.dataset_id == "dataset-1"

        payload = batch.to_dict()
        assert payload["closed_trades"] == 1
        assert payload["open_trades"] == 1
        assert len(payload["outcomes"]) == 1
        assert json.loads(json.dumps(payload)) == payload
        assert json.loads(batch.to_json()) == payload


def test_outcome_fingerprint_is_stable_and_tracks_new_trades(tmp_path: Path) -> None:
    session_id = "fingerprint"
    with PaperLedger(tmp_path / "paper.db") as ledger:
        ledger.open_session(
            session_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="hash-1",
            started_at=BASE,
        )
        ledger.record_trade(
            session_id,
            side="long",
            entry_ts=BASE,
            entry_price=2000.0,
            exit_ts=BASE + timedelta(minutes=5),
            exit_price=2001.0,
            lot=0.10,
            pnl=10.0,
            r_multiple=1.0,
            exit_reason="take_profit",
        )
        first = outcome_fingerprint(matured_outcomes(ledger, session_id))
        second = outcome_fingerprint(matured_outcomes(ledger, session_id))
        assert first == second

        ledger.record_trade(
            session_id,
            side="short",
            entry_ts=BASE + timedelta(minutes=10),
            entry_price=2000.0,
            exit_ts=BASE + timedelta(minutes=15),
            exit_price=2002.0,
            lot=0.10,
            pnl=-20.0,
            r_multiple=-1.5,
            exit_reason="stop_loss",
        )
        third = outcome_fingerprint(matured_outcomes(ledger, session_id))
        assert third != first


def test_future_bars_do_not_change_earlier_signals(tmp_path: Path) -> None:
    short_root = tmp_path / "short"
    long_root = tmp_path / "long"
    write(short_root, make_bars(200))
    write(long_root, make_bars(260))

    with PaperLedger(tmp_path / "short.db") as ledger:
        runner = make_runner(ledger, short_root)
        summary = runner.advance()
        short_signals = signal_tuples(ledger, runner.session_id)
        cutoff = summary.last_bar_ts
        assert cutoff is not None

    # A fresh session over a longer store must decide the same bars the same way.
    with PaperLedger(tmp_path / "long.db") as ledger:
        runner = make_runner(ledger, long_root)
        runner.advance()
        long_signals = signal_tuples(ledger, runner.session_id)

    overlapping = [signal for signal in long_signals if signal[0] <= cutoff]
    assert overlapping == short_signals

    # The incremental ledger also keeps its prefix when the future arrives.
    with PaperLedger(tmp_path / "short.db") as ledger:
        write(short_root, make_bars(260))
        runner = make_runner(ledger, short_root)
        runner.advance()
        extended = signal_tuples(ledger, runner.session_id)
    assert extended[: len(short_signals)] == short_signals


def record_provenance(
    root: Path,
    bars: pl.DataFrame,
    *,
    source: str,
    synthetic: bool,
) -> None:
    """Write the ingest sidecar a real dataset would carry."""
    save_provenance(
        root,
        build_provenance(
            bars, symbol=SYMBOL, timeframe=TIMEFRAME, source=source, synthetic=synthetic
        ),
    )


def test_synthetic_dataset_is_flagged_from_session_to_trigger(tmp_path: Path) -> None:
    """A demo run is marked synthetic end to end, and the trigger refuses it."""
    root = tmp_path / "data"
    bars = make_bars(200)
    write(root, bars)
    record_provenance(root, bars, source="provider:synthetic", synthetic=True)
    ledger_path = tmp_path / "paper.db"

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root)
        summary = runner.advance()
        assert summary.trades > 0
        record = ledger.session(runner.session_id)
        assert record is not None
        assert session_provenance(record.config) == ("provider:synthetic", True)
        batch = matured_outcomes(ledger, runner.session_id)

    assert batch.strategy_hash is not None
    assert batch.dataset_id is not None
    assert batch.source == "provider:synthetic"
    assert batch.synthetic is True
    assert batch.rows
    assert all(row.source == "provider:synthetic" and row.synthetic is True for row in batch.rows)
    assert batch.to_dict()["synthetic"] is True

    store = JobStore(tmp_path / "jobs.db")
    source = PaperOutcomeTriggerSource(ledger_path, store=store)
    refused = scan_once(Orchestrator(store), [source])
    assert not refused.queued
    cursor = store.get_state(source.cursor_key)
    assert cursor is not None
    assert REASON_SYNTHETIC in cursor
    store.close()

    store = JobStore(tmp_path / "jobs.db")
    allowed = scan_once(
        Orchestrator(store),
        [PaperOutcomeTriggerSource(ledger_path, store=store, allow_synthetic=True)],
    )
    assert len(allowed.queued) == 1
    assert allowed.queued[0].payload["synthetic"] is True
    assert allowed.queued[0].payload["source"] == "provider:synthetic"
    store.close()


def test_real_and_unknown_provenance_are_distinguished(tmp_path: Path) -> None:
    bars = make_bars(200)
    real_root = tmp_path / "real"
    write(real_root, bars)
    record_provenance(real_root, bars, source="dukascopy", synthetic=False)

    ledger_path = tmp_path / "real.db"
    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, real_root)
        runner.advance()
        record = ledger.session(runner.session_id)
        assert record is not None
        assert session_provenance(record.config) == ("dukascopy", False)
        real_batch = matured_outcomes(ledger, runner.session_id)
    assert real_batch.synthetic is False
    assert real_batch.source == "dukascopy"

    empty_root = tmp_path / "empty"
    write(empty_root, bars)
    unknown_path = tmp_path / "unknown.db"
    with PaperLedger(unknown_path) as ledger:
        runner = make_runner(ledger, empty_root)
        runner.advance()
        record = ledger.session(runner.session_id)
        assert record is not None
        assert session_provenance(record.config) == (None, None)
        unknown_batch = matured_outcomes(ledger, runner.session_id)
    assert unknown_batch.synthetic is None
    assert unknown_batch.source is None
    assert all(row.synthetic is None and row.source is None for row in unknown_batch.rows)


def test_provenance_survives_a_ledger_reopen(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bars = make_bars(200)
    write(root, bars)
    record_provenance(root, bars, source="synthetic", synthetic=True)
    ledger_path = tmp_path / "paper.db"

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root)
        runner.advance()
        session_id = runner.session_id

    with PaperLedger(ledger_path) as ledger:
        record = ledger.session(session_id)
        assert record is not None
        assert session_provenance(record.config) == ("synthetic", True)
        batch = matured_outcomes(ledger, session_id)
        assert batch.synthetic is True
        assert batch.rows


def test_session_created_before_provenance_still_resumes(tmp_path: Path) -> None:
    """A pre-provenance session (unknown) keeps running; unknown never becomes real.

    ``docs/paper-trading.md`` provenance rule 3: a session recorded before the
    sidecar existed still resumes, and its unknown pair is not upgraded when a
    later ingest writes a real sidecar. The known-change refusal is tested
    separately; this pins the unknown side of the same rule.
    """
    root = tmp_path / "data"
    bars = make_bars(200)
    write(root, bars)
    ledger_path = tmp_path / "paper.db"

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root)
        runner.advance()
        session_id = runner.session_id
        record = ledger.session(session_id)
        assert record is not None
        assert session_provenance(record.config) == (None, None)

    # A later ingest records a real (non-synthetic) source. The session was
    # created before the record existed, so this is not a known change.
    record_provenance(root, bars, source="dukascopy", synthetic=False)
    write(root, make_bars(220))

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root)
        runner.advance()
        record = ledger.session(session_id)
        assert record is not None
        assert session_provenance(record.config) == (None, None)
        batch = matured_outcomes(ledger, session_id)

    assert batch.source is None
    assert batch.synthetic is None
    assert all(row.source is None and row.synthetic is None for row in batch.rows)


def test_outcome_fingerprint_changes_with_provenance(tmp_path: Path) -> None:
    def fingerprint(name: str, provenance: dict[str, object]) -> str:
        path = tmp_path / f"{name}.db"
        with PaperLedger(path) as ledger:
            ledger.open_session(
                "fp",
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                strategy="dsl",
                model_version="hash-1",
                started_at=BASE,
                config={
                    "origin": "test",
                    "data": {"dataset_id": "dataset-1"},
                    "provenance": provenance,
                },
            )
            ledger.record_trade(
                "fp",
                side="long",
                entry_ts=BASE,
                entry_price=2000.0,
                exit_ts=BASE + timedelta(minutes=5),
                exit_price=2001.0,
                lot=0.10,
                pnl=10.0,
                r_multiple=1.0,
                exit_reason="take_profit",
            )
            return outcome_fingerprint(matured_outcomes(ledger, "fp"))

    real = fingerprint("real", {"source": "dukascopy", "synthetic": False})
    same = fingerprint("same", {"source": "dukascopy", "synthetic": False})
    demo = fingerprint("demo", {"source": "dukascopy", "synthetic": True})
    unknown = fingerprint("unknown", {})

    assert real == same
    assert len({real, demo, unknown}) == 3


def test_changed_provenance_refuses_resume(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bars = make_bars(200)
    write(root, bars)
    record_provenance(root, bars, source="synthetic", synthetic=True)
    ledger_path = tmp_path / "paper.db"

    with PaperLedger(ledger_path) as ledger:
        make_runner(ledger, root).advance()

    # A re-ingest rewrites the sidecar: the evidence identity behind the
    # recorded trades changed, so the session must not continue.
    record_provenance(root, bars, source="dukascopy", synthetic=False)
    with (
        PaperLedger(ledger_path) as ledger,
        pytest.raises(SessionMismatchError, match="data source"),
    ):
        make_runner(ledger, root).advance()


def test_unreadable_provenance_sidecar_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    write(root, make_bars(200))
    sidecar = provenance_path(root, SYMBOL, TIMEFRAME)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text("{not json", encoding="utf-8")

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        with pytest.raises(ContinuousError, match="unreadable provenance"):
            runner.advance()
        assert ledger.session(runner.session_id) is None


def test_continuous_module_has_no_network_or_legacy_fallback() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "scalping" / "paper" / "continuous.py"
    source = path.read_text(encoding="utf-8")
    for name in (
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
    ):
        assert f"import {name}" not in source, f"continuous.py imports {name}"
        assert f"from {name}" not in source, f"continuous.py imports from {name}"
    assert "INSERT INTO" not in source
    assert "UPDATE " not in source
    assert ".connection" not in source
    assert "flow_momentum" not in source
    assert "order_send" not in source
