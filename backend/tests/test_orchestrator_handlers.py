"""Job handlers wired to the real pipeline modules.

Every test uses a tiny synthetic Parquet store and temporary roots; nothing
reads the repository's real `data/`, discovery archive, knowledge index or paper
ledger. The properties pinned here are the ones a queue depends on:

* an invalid payload fails once and permanently instead of retrying forever;
* repeating the same logical job creates no second document, dataset, trial or
  paper session;
* discovery records a decision and can never record a promotion;
* a scope without a governed champion is NO TRADE and writes nothing else;
* outcome analysis counts closed trades only and never promotes an open
  position into a matured result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.champion import scope_for
from scalping.discovery.protocols import TrialResult
from scalping.knowledge.index import KnowledgeIndex
from scalping.orchestrator import JobStore, Orchestrator
from scalping.orchestrator.handlers import (
    DEFAULT_HANDLER_KINDS,
    KIND_DATASET_BUILD,
    KIND_DISCOVERY_RUN,
    KIND_KNOWLEDGE_INGEST,
    KIND_PAPER_ADVANCE,
    KIND_PAPER_ANALYSE,
    HandlerContext,
    register_default_handlers,
)
from scalping.paper.ledger import PaperLedger
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
START = datetime(2026, 1, 5, tzinfo=UTC)
END = START + timedelta(days=1)


# --------------------------------------------------------------------------- fixtures


def synthetic_bars():
    return SyntheticProvider(seed=3).fetch(SYMBOL, START, END)


def trading_spec() -> StrategySpec:
    """Long every flat bar; close on the holding limit. A deterministic DSL spec."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(10),
        max_lookback=2,
    )


@dataclass
class Harness:
    """One job store, one orchestrator and the context its handlers captured."""

    store: JobStore
    orchestrator: Orchestrator
    context: HandlerContext

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dedupe: str,
        max_retries: int = 0,
    ):
        return self.store.enqueue(
            kind,
            dedupe_key=dedupe,
            payload=payload,
            max_retries=max_retries,
        )

    def run(self, kind: str, payload: dict[str, Any], *, dedupe: str, max_retries: int = 0):
        job, _ = self.enqueue(kind, payload, dedupe=dedupe, max_retries=max_retries)
        outcome = self.orchestrator.run_once()
        row = self.store.get(job.id)
        assert row is not None
        return job, outcome, row


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    data_root = tmp_path / "data"
    ParquetStore(data_root).write(SYMBOL, TIMEFRAME, synthetic_bars())
    context = HandlerContext(
        data_root=data_root,
        runs_root=tmp_path / "runs",
        archive_root=tmp_path / "archive",
        use_feature_cache=False,
    )
    store = JobStore(tmp_path / "orchestrator.sqlite")
    orchestrator = Orchestrator(store)
    register_default_handlers(orchestrator, context)
    return Harness(store=store, orchestrator=orchestrator, context=context)


def data_of(row: Any) -> dict[str, Any]:
    assert row.result is not None and row.result["ok"] is True
    payload = row.result["data"]
    # The store already serialised it; this proves the contract after a read.
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    return payload


def events_of(store: JobStore, job_id: int) -> list[dict[str, Any]]:
    return [
        json.loads(event.detail) if event.detail else {}
        for event in store.events(job_id)
        if event.detail
    ]


def seed_champion(context: HandlerContext) -> StrategySpec:
    """Archive and promote one spec for the test scope, as a reviewed path would."""
    spec = trading_spec()
    scope = scope_for(SYMBOL, TIMEFRAME)
    dataset_id = f"validation:{SYMBOL}:{TIMEFRAME}:h1:test"
    path = context.discovery_archive_path
    with DiscoveryArchive(path) as archive:
        record = archive.add_candidate(spec, code_version="test")
        archive.record_trial(
            record.candidate_id,
            TrialResult(
                metrics={"expectancy_r": 0.5, "net_profit": 3.0, "trades": 5.0},
                accepted=True,
            ),
            dataset_id=dataset_id,
            code_version="test",
            seed=0,
        )
        archive.promote(record.candidate_id, scope=scope, reason="unit test promotion")
        champion = archive.champion(scope)
    assert champion is not None and champion.candidate.candidate_id == spec.canonical_hash()
    return spec


# ------------------------------------------------------------------------ registry


def test_register_default_handlers_names_every_kind(harness: Harness) -> None:
    assert set(harness.orchestrator.handlers) == set(DEFAULT_HANDLER_KINDS)
    assert DEFAULT_HANDLER_KINDS == (
        KIND_KNOWLEDGE_INGEST,
        KIND_DATASET_BUILD,
        KIND_DISCOVERY_RUN,
        KIND_PAPER_ADVANCE,
        KIND_PAPER_ANALYSE,
    )


# -------------------------------------------------------------------------- ingest


def test_ingest_indexes_a_document_and_is_checksum_idempotent(
    harness: Harness, tmp_path: Path
) -> None:
    document = tmp_path / "research" / "note.md"
    document.parent.mkdir()
    document.write_text(
        "# Session study\n\nWhen the price closes above the session high, the trend continues.\n",
        encoding="utf-8",
    )

    _, outcome, row = harness.run(
        KIND_KNOWLEDGE_INGEST,
        {"document_path": str(document), "market_scope": SYMBOL, "timeframe_scope": TIMEFRAME},
        dedupe="ingest-1",
    )
    assert outcome.status == "succeeded"
    first = data_of(row)
    assert first["status"] == "indexed"
    assert first["sections"] >= 1
    assert first["drafts"] == 1
    assert first["duplicate"] is False

    # A second job with the same bytes and a different dedupe key must not
    # index or extract anything again.
    _, outcome, row = harness.run(
        KIND_KNOWLEDGE_INGEST,
        {"document_path": str(document), "market_scope": SYMBOL, "timeframe_scope": TIMEFRAME},
        dedupe="ingest-2",
    )
    assert outcome.status == "succeeded"
    second = data_of(row)
    assert second["duplicate"] is True
    assert second["drafts"] == 0
    assert second["skipped"] >= 1, "the already-stored draft must be reported as skipped"
    assert second["doc_id"] == first["doc_id"]

    with KnowledgeIndex(harness.context.knowledge_index_path) as index:
        assert len(index.list_documents()) == 1
        assert len(index.sections(first["doc_id"])) == first["sections"]
        assert len(index.list_hypotheses()) == 1


def test_ingest_unsupported_format_is_permanent(harness: Harness, tmp_path: Path) -> None:
    document = tmp_path / "scan.pdf"
    document.write_bytes(b"%PDF-1.4 not really a document")

    _, outcome, row = harness.run(
        KIND_KNOWLEDGE_INGEST,
        {"document_path": str(document)},
        dedupe="ingest-pdf",
        max_retries=3,
    )
    assert outcome.status == "failed"
    assert row.attempts == 1
    assert "status='unsupported'" in (row.error or "")
    permanent = [item for item in events_of(harness.store, row.id) if item.get("permanent")]
    assert permanent, "the queue must record that the failure was permanent"


# ------------------------------------------------------------------------- dataset


def test_build_dataset_writes_a_manifest_and_repeats_identically(
    harness: Harness,
) -> None:
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "horizon": 1}

    _, outcome, row = harness.run(KIND_DATASET_BUILD, payload, dedupe="dataset-1")
    assert outcome.status == "succeeded"
    first = data_of(row)
    assert first["dataset_id"].startswith(f"{SYMBOL}:{TIMEFRAME}:h1:")
    assert first["rows"] > 0
    assert first["splits"]["validation"]["rows"] > 0
    manifest_path = Path(first["manifest_path"])
    assert manifest_path.is_file()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["checksum"] == first["checksum"]
    assert written["split_counts"]["validation"] == first["splits"]["validation"]["rows"]

    _, outcome, row = harness.run(KIND_DATASET_BUILD, payload, dedupe="dataset-2")
    assert outcome.status == "succeeded"
    second = data_of(row)
    assert second["dataset_id"] == first["dataset_id"]
    assert second["checksum"] == first["checksum"]
    assert second["manifest_path"] == first["manifest_path"]


def test_build_dataset_refuses_an_unknown_symbol(harness: Harness) -> None:
    _, outcome, row = harness.run(
        KIND_DATASET_BUILD,
        {"symbol": "EURUSD", "timeframe": TIMEFRAME},
        dedupe="dataset-unknown",
        max_retries=3,
    )
    assert outcome.status == "failed"
    assert row.attempts == 1
    assert "no stored bars for EURUSD M1" in (row.error or "")
    assert not harness.context.datasets_root.exists()


# ----------------------------------------------------------------------- discovery


def test_discovery_records_a_decision_and_never_promotes(harness: Harness, tmp_path: Path) -> None:
    payload = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "max_trials": 2,
        "population": 3,
        "generations": 1,
        "patience": 1,
        "seed": 0,
        "stress": [2.0],
        "gate": {},
    }
    _, outcome, row = harness.run(KIND_DISCOVERY_RUN, payload, dedupe="discover-1")

    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["decision"] in {"no_trade", "no_change", "rejected"}
    assert data["promoted"] is False
    assert data["trials"] >= 1
    assert data["archive"] == str(harness.context.discovery_archive_path)
    assert Path(data["archive"]).is_file()

    with DiscoveryArchive(data["archive"]) as archive:
        decisions = [record.decision for record in archive.decisions()]
        assert decisions
        assert "promoted" not in decisions
        assert archive.champion() is None


def test_discovery_requires_an_explicit_gate(harness: Harness) -> None:
    _, outcome, row = harness.run(
        KIND_DISCOVERY_RUN,
        {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "max_trials": 1,
            "seed": 0,
            "stress": [2.0],
        },
        dedupe="discover-no-gate",
        max_retries=3,
    )
    assert outcome.status == "failed"
    assert row.attempts == 1
    assert "'gate' is required" in (row.error or "")


# --------------------------------------------------------------------------- paper


def test_paper_advance_without_a_champion_is_no_trade_and_writes_nothing(
    harness: Harness,
) -> None:
    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE,
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 20},
        dedupe="paper-no-champion",
    )

    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["action"] == "no_trade"
    assert data["reason_class"] == "no_champion"
    assert data["touched"] is False
    assert data["champion"] is None
    assert data["reason"]
    assert not harness.context.paper_ledger_path.exists()
    assert not harness.context.discovery_archive_path.exists()


def test_paper_advance_runs_the_governed_champion_and_dedupes(
    harness: Harness,
) -> None:
    spec = seed_champion(harness.context)
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 40}

    job, outcome, row = harness.run(
        KIND_PAPER_ADVANCE, payload, dedupe=f"paper-{SYMBOL}-{TIMEFRAME}-40"
    )
    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["action"] == "advanced"
    assert data["spec_hash"] == spec.canonical_hash()
    assert data["bars_this_call"] == 40
    assert data["bars_processed"] == 40
    assert Path(data["ledger"]).is_file()

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        session = ledger.session(data["session_id"])
        assert session is not None
        assert session.model_version == spec.canonical_hash()
        assert len(ledger.signals(data["session_id"])) == 40

    # Re-enqueueing the same logical advance is deduplicated by the orchestrator:
    # no second job, no second 40-bar step.
    duplicate, created = harness.enqueue(
        KIND_PAPER_ADVANCE, payload, dedupe=f"paper-{SYMBOL}-{TIMEFRAME}-40"
    )
    assert created is False
    assert duplicate.id == job.id
    assert harness.store.get(job.id) is not None

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        assert len(ledger.signals(data["session_id"])) == 40


def test_paper_advance_with_an_invalid_champion_payload_is_no_trade(
    harness: Harness,
) -> None:
    spec = seed_champion(harness.context)
    with DiscoveryArchive(harness.context.discovery_archive_path) as archive:
        archive.connection.execute(
            "UPDATE candidates SET canonical_json = ? WHERE candidate_id = ?",
            ('{"broken":true}', spec.canonical_hash()),
        )

    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE,
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10},
        dedupe="paper-invalid-champion",
    )
    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["action"] == "no_trade"
    assert data["reason_class"] == "invalid_champion"
    assert not harness.context.paper_ledger_path.exists()


# ------------------------------------------------------------------------ analysis


def _seed_ledger_with_two_closed_trades_and_one_open_position(
    context: HandlerContext,
) -> str:
    spec_hash = "a" * 64
    entry = START + timedelta(minutes=10)
    with PaperLedger(context.paper_ledger_path) as ledger:
        ledger.open_session(
            "paper-1",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version=spec_hash,
            config={"extra": {"origin": "autonomous", "dsl": {"spec_hash": spec_hash}}},
        )
        ledger.record_signal("paper-1", entry, "BUY", reason="entry", model_version=spec_hash)
        ledger.record_signal(
            "paper-1",
            entry + timedelta(minutes=4),
            "CLOSE",
            reason="exit",
            model_version=spec_hash,
        )
        ledger.record_trade(
            "paper-1",
            side="long",
            entry_ts=entry,
            entry_price=1000.0,
            exit_ts=entry + timedelta(minutes=2),
            exit_price=1001.0,
            lot=0.1,
            pnl=12.5,
            r_multiple=1.25,
            exit_reason="take_profit",
        )
        ledger.record_trade(
            "paper-1",
            side="short",
            entry_ts=entry + timedelta(minutes=3),
            entry_price=1001.0,
            exit_ts=entry + timedelta(minutes=4),
            exit_price=1002.0,
            lot=0.1,
            pnl=-5.0,
            r_multiple=-0.5,
            exit_reason="stop_loss",
        )
        # An open position is not a trade. It must be reported, not summed.
        ledger.connection.execute(
            """
            INSERT INTO positions (
                session_id, order_id, side, lot, entry_ts, entry_price,
                sl, tp, status, closed_ts, closed_price
            ) VALUES ('paper-1', NULL, 'long', 0.1, ?, 1002.0, 990.0, 1020.0, 'open', NULL, NULL)
            """,
            (entry.isoformat(),),
        )
    return spec_hash


def test_analyse_outcomes_counts_closed_trades_only(harness: Harness) -> None:
    spec_hash = _seed_ledger_with_two_closed_trades_and_one_open_position(harness.context)

    _, outcome, row = harness.run(KIND_PAPER_ANALYSE, {}, dedupe="analyse-1")
    assert outcome.status == "succeeded"
    data = data_of(row)

    assert data["trades"] == 2
    assert data["open_positions_excluded"] == 1
    assert data["wins"] == 1 and data["losses"] == 1 and data["breakeven"] == 0
    assert data["win_rate"] == pytest.approx(0.5)
    assert data["pnl_total"] == pytest.approx(7.5)
    assert data["pnl_mean"] == pytest.approx(3.75)
    assert data["r_total"] == pytest.approx(0.75)
    assert data["r_mean"] == pytest.approx(0.375)
    assert data["strategy_hashes"] == [spec_hash]

    assert len(data["sessions"]) == 1
    session = data["sessions"][0]
    assert session["session_id"] == "paper-1"
    assert session["spec_hash"] == spec_hash
    assert session["config_spec_hash"] == spec_hash
    assert session["closed_trades"] == 2
    assert session["open_positions_excluded"] == 1
    assert session["data_attribution"] == "bar_fingerprint"
    assert session["data_fingerprint"] in data["data_fingerprints"]


def test_analyse_outcomes_refuses_to_count_an_open_trade_as_matured(
    harness: Harness,
) -> None:
    _seed_ledger_with_two_closed_trades_and_one_open_position(harness.context)

    _, outcome, row = harness.run(
        KIND_PAPER_ANALYSE,
        {"include_open": True},
        dedupe="analyse-include-open",
        max_retries=3,
    )
    assert outcome.status == "failed"
    assert row.attempts == 1
    assert "unknown field" in (row.error or "")


def test_analyse_outcomes_before_any_paper_run_is_an_empty_success(
    harness: Harness,
) -> None:
    _, outcome, row = harness.run(KIND_PAPER_ANALYSE, {}, dedupe="analyse-empty")
    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["status"] == "no_ledger"
    assert data["trades"] == 0
    assert data["pnl_total"] == 0.0
    assert data["sessions"] == []


# --------------------------------------------------------- invalid payload policy


@pytest.mark.parametrize(
    ("name", "kind", "payload"),
    [
        ("ingest-missing", KIND_KNOWLEDGE_INGEST, {}),
        (
            "ingest-missing-file",
            KIND_KNOWLEDGE_INGEST,
            {"document_path": "/definitely/not/here.txt"},
        ),
        ("dataset-missing-timeframe", KIND_DATASET_BUILD, {"symbol": SYMBOL}),
        (
            "dataset-traversal-symbol",
            KIND_DATASET_BUILD,
            {"symbol": "../etc", "timeframe": TIMEFRAME},
        ),
        (
            "paper-zero-bars",
            KIND_PAPER_ADVANCE,
            {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 0},
        ),
        (
            "paper-boolean-bars",
            KIND_PAPER_ADVANCE,
            {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": True},
        ),
        ("analysis-unknown-field", KIND_PAPER_ANALYSE, {"session": "paper-1"}),
    ],
)
def test_invalid_payloads_fail_once_and_permanently(
    harness: Harness, name: str, kind: str, payload: dict[str, Any]
) -> None:
    _, outcome, row = harness.run(kind, payload, dedupe=f"invalid-{name}", max_retries=3)
    assert outcome.status == "failed"
    assert row.attempts == 1, "a permanent failure must not spend a retry"
    assert "invalid payload" in (row.error or "")
    assert [item for item in events_of(harness.store, row.id) if item.get("permanent")]
