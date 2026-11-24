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
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from promotion_evidence import evidence_for
from promotion_support import promote_for_tests, registered_test_policies
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.champion import scope_for
from scalping.discovery.promotion import PROMOTION_V1, EvidenceIdentity
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
from scalping.paper.autonomous import AutonomousPaperRunner
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
DATASET_CHECKSUM = "c1" * 32
SETTINGS_DIGEST = "c2" * 32
TEST_POLICY = replace(
    PROMOTION_V1,
    policy_id="handler-test-policy",
    require_locked_oos=False,
)


@pytest.fixture(autouse=True)
def _register_handler_test_policy():
    with registered_test_policies(TEST_POLICY):
        yield


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


def make_harness(tmp_path: Path, bars: pl.DataFrame | None = None) -> Harness:
    """One harness over a temp store seeded with ``bars`` (one day by default)."""
    data_root = tmp_path / "data"
    ParquetStore(data_root).write(SYMBOL, TIMEFRAME, synthetic_bars() if bars is None else bars)
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


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return make_harness(tmp_path)


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
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
        )
        # Seed through the evidence API: the paper provider only accepts an
        # evidence-mode champion. This is a test policy, not PROMOTION_V1.
        evidence = evidence_for(
            spec_hash=record.candidate_id,
            dataset_id=dataset_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            code_version="test",
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
            expectancy_r=0.5,
        )
        promote_for_tests(
            archive,
            record.candidate_id,
            evidence,
            TEST_POLICY,
            scope=scope,
            reason="unit test promotion",
            expected_identity=EvidenceIdentity(
                code_version="test",
                dataset_checksum=DATASET_CHECKSUM,
                settings_digest=SETTINGS_DIGEST,
            ),
        )
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
    assert data["mode"] == "continuous"
    assert data["reason_class"] == "no_champion"
    assert data["touched"] is False
    assert data["champion"] is None
    assert data["reason"]
    assert not harness.context.paper_ledger_path.exists()
    assert not harness.context.discovery_archive_path.exists()


def test_paper_advance_with_a_non_evidence_champion_is_no_trade(
    harness: Harness,
) -> None:
    seed_champion(harness.context)
    with DiscoveryArchive(harness.context.discovery_archive_path) as archive:
        archive.connection.execute(
            """
            UPDATE champions SET mode = 'exploratory', policy_id = NULL,
                policy_version = NULL, policy_digest = NULL, evidence_checksum = NULL
            """
        )

    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE,
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 20},
        dedupe="paper-non-evidence-champion",
    )

    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["action"] == "no_trade"
    assert data["mode"] == "continuous"
    assert data["reason_class"] == "no_evidence_champion"
    assert data["touched"] is False
    assert not harness.context.paper_ledger_path.exists()


def test_paper_advance_advances_the_governed_champion_continuously(
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
    assert data["mode"] == "continuous"
    assert data["spec_hash"] == spec.canonical_hash()
    assert data["status"] == "running"
    assert data["bars_this_call"] == data["bars_advanced"] == data["new_signals"] > 0
    assert data["bars_processed"] == data["bars_this_call"]
    assert data["last_bar_ts"] is not None
    assert data["session_id"].startswith(f"auto-{SYMBOL}-{TIMEFRAME}-")
    assert Path(data["ledger"]).is_file()

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        session = ledger.session(data["session_id"])
        assert session is not None
        assert session.model_version == spec.canonical_hash()
        assert session.config["extra"]["origin"] == "continuous"
        assert session.config["extra"]["scope"] == scope_for(SYMBOL, TIMEFRAME)
        signals = ledger.signals(data["session_id"])
        assert len(signals) == data["bars_processed"]
        assert len({signal.ts for signal in signals}) == len(signals)

    # Re-enqueueing the same logical advance is deduplicated by the orchestrator:
    # no second job, no second advance.
    duplicate, created = harness.enqueue(
        KIND_PAPER_ADVANCE, payload, dedupe=f"paper-{SYMBOL}-{TIMEFRAME}-40"
    )
    assert created is False
    assert duplicate.id == job.id
    assert harness.store.get(job.id) is not None

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        assert len(ledger.signals(data["session_id"])) == data["bars_processed"]


def test_paper_advance_second_scan_with_no_new_bars_is_a_no_op(harness: Harness) -> None:
    spec = seed_champion(harness.context)
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10}

    _, first_outcome, first_row = harness.run(KIND_PAPER_ADVANCE, payload, dedupe="paper-noop-1")
    assert first_outcome.status == "succeeded"
    first = data_of(first_row)

    # A different dedupe key forces the handler to run again. The ledger cursor
    # is the only state: no new bars means no new signal and no new trade.
    _, second_outcome, second_row = harness.run(KIND_PAPER_ADVANCE, payload, dedupe="paper-noop-2")
    assert second_outcome.status == "succeeded"
    second = data_of(second_row)
    assert second["action"] == "advanced"
    assert second["session_id"] == first["session_id"]
    assert second["spec_hash"] == spec.canonical_hash()
    assert second["bars_this_call"] == 0
    assert second["new_signals"] == 0
    assert second["bars_processed"] == first["bars_processed"]
    assert second["last_bar_ts"] == first["last_bar_ts"]
    # Only the provisional tail is newer than the cursor: nothing is decidable.
    assert second["done_reason"] in {"no_new_bars", "awaiting_closed_bar"}

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        assert len(ledger.signals(first["session_id"])) == first["bars_processed"]


def test_paper_advance_appends_only_the_newly_stored_bars(tmp_path: Path) -> None:
    bars = synthetic_bars()
    harness = make_harness(tmp_path, bars.head(1000))
    seed_champion(harness.context)
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10}

    _, first_outcome, first_row = harness.run(KIND_PAPER_ADVANCE, payload, dedupe="paper-append-1")
    assert first_outcome.status == "succeeded"
    first = data_of(first_row)
    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        before = [
            (signal.ts, signal.action, signal.reason)
            for signal in ledger.signals(first["session_id"])
        ]
    assert len(before) == first["bars_processed"]

    # The feed appends 40 bars; one stays provisional, so exactly 40 decisions run.
    assert (
        ParquetStore(harness.context.data_root).write(SYMBOL, TIMEFRAME, bars.slice(1000, 40)) == 40
    )
    _, second_outcome, second_row = harness.run(
        KIND_PAPER_ADVANCE, payload, dedupe="paper-append-2"
    )
    assert second_outcome.status == "succeeded"
    second = data_of(second_row)
    assert second["session_id"] == first["session_id"]
    assert second["bars_this_call"] == 40
    assert second["new_signals"] == 40
    assert second["bars_processed"] == first["bars_processed"] + 40

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        after = [
            (signal.ts, signal.action, signal.reason)
            for signal in ledger.signals(second["session_id"])
        ]
    assert after[: len(before)] == before
    assert all(item[0] > before[-1][0] for item in after[len(before) :])
    assert len({item[0] for item in after}) == len(after)


def test_paper_advance_survives_a_process_restart(tmp_path: Path) -> None:
    bars = synthetic_bars()
    harness = make_harness(tmp_path, bars.head(1000))
    seed_champion(harness.context)
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10}

    _, first_outcome, first_row = harness.run(KIND_PAPER_ADVANCE, payload, dedupe="paper-restart-1")
    assert first_outcome.status == "succeeded"
    first = data_of(first_row)
    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        before = [signal.ts for signal in ledger.signals(first["session_id"])]

    # Simulate a process restart: the job store and the ledger are reopened and
    # 20 bars arrived while nothing was running.
    harness.store.close()
    assert (
        ParquetStore(harness.context.data_root).write(SYMBOL, TIMEFRAME, bars.slice(1000, 20)) == 20
    )
    store = JobStore(tmp_path / "orchestrator.sqlite")
    orchestrator = Orchestrator(store)
    register_default_handlers(orchestrator, harness.context)
    job, _ = store.enqueue(
        KIND_PAPER_ADVANCE,
        dedupe_key="paper-restart-2",
        payload=payload,
    )
    outcome = orchestrator.run_once()
    row = store.get(job.id)
    assert row is not None
    assert outcome.status == "succeeded"
    second = data_of(row)

    assert second["session_id"] == first["session_id"]
    assert second["bars_this_call"] == 20
    assert second["bars_processed"] == first["bars_processed"] + 20
    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        after = [signal.ts for signal in ledger.signals(second["session_id"])]
    assert after[: len(before)] == before
    assert len(after) == len(set(after))
    store.close()


def test_paper_advance_refuses_a_bar_gap_with_a_typed_reason(tmp_path: Path) -> None:
    bars = synthetic_bars()
    harness = make_harness(tmp_path, bars.head(1000))
    seed_champion(harness.context)
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10}

    _, first_outcome, first_row = harness.run(KIND_PAPER_ADVANCE, payload, dedupe="paper-gap-1")
    assert first_outcome.status == "succeeded"
    first = data_of(first_row)

    appended = bars.slice(1000, 40)
    holed = pl.concat([appended.head(10), appended.slice(11)])
    assert ParquetStore(harness.context.data_root).write(SYMBOL, TIMEFRAME, holed) == 39

    _, gap_outcome, gap_row = harness.run(
        KIND_PAPER_ADVANCE, payload, dedupe="paper-gap-2", max_retries=0
    )
    assert gap_outcome.status == "failed"
    assert "bar_gap" in (gap_row.error or "")

    # A gap can be repaired by a later ingestion, so the same refusal is a
    # bounded retry rather than a permanent failure.
    _, retry_outcome, retry_row = harness.run(
        KIND_PAPER_ADVANCE, payload, dedupe="paper-gap-3", max_retries=1
    )
    assert retry_outcome.status == "retry"
    assert "bar_gap" in (retry_row.error or "")

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        assert len(ledger.signals(first["session_id"])) == first["bars_processed"]


def test_paper_advance_refuses_duplicate_bars_permanently(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_champion(harness.context)
    original = ParquetStore.read

    def duplicate_tail(self: ParquetStore, symbol: str, timeframe: str, *args: Any, **kwargs: Any):
        frame = original(self, symbol, timeframe, *args, **kwargs)
        return pl.concat([frame, frame.tail(1)]) if not frame.is_empty() else frame

    monkeypatch.setattr(ParquetStore, "read", duplicate_tail)

    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE,
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10},
        dedupe="paper-duplicate-bars",
        max_retries=2,
    )
    assert outcome.status == "failed"
    assert row.attempts == 1, "a corrupt store must not spend the retry budget"
    assert "bar_order" in (row.error or "")
    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        assert ledger.session_page(limit=10, offset=0)[1] == 0


def test_paper_advance_resumes_a_legacy_slice_session(harness: Harness) -> None:
    """A Phase-12 slice session has no cursor column; its last signal is the cursor."""
    spec = seed_champion(harness.context)
    with PaperLedger(harness.context.paper_ledger_path) as ledger:
        legacy = AutonomousPaperRunner(
            ledger,
            spec,
            SYMBOL,
            TIMEFRAME,
            harness.context.settings,
            cache=False,
            data_root=harness.context.data_root,
        )
        legacy.run(bars=40)
        legacy_session_id = legacy.session_id
        before = [
            (signal.ts, signal.action, signal.reason)
            for signal in ledger.signals(legacy_session_id)
        ]
        assert len(before) == 40
        record = ledger.session(legacy_session_id)
        assert record is not None
        assert record.config["extra"]["origin"] == "autonomous"

    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE,
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10},
        dedupe="paper-legacy-resume",
    )
    assert outcome.status == "succeeded"
    data = data_of(row)
    assert data["mode"] == "continuous"
    assert data["session_id"] == legacy_session_id
    assert data["bars_processed"] > 40

    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        after = [
            (signal.ts, signal.action, signal.reason)
            for signal in ledger.signals(legacy_session_id)
        ]
        record = ledger.session(legacy_session_id)
        assert record is not None
        # Resumed, not recreated: the legacy session row and its history stay put.
        assert record.config["extra"]["origin"] == "autonomous"
    assert after[: len(before)] == before
    assert len({item[0] for item in after}) == len(after)


def test_paper_advance_refuses_an_incompatible_legacy_session(harness: Harness) -> None:
    spec = seed_champion(harness.context)
    session_id = f"auto-{SYMBOL}-{TIMEFRAME}-{spec.canonical_hash()[:16]}"
    with PaperLedger(harness.context.paper_ledger_path) as ledger:
        ledger.open_session(
            session_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="b" * 64,
        )

    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE,
        {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10},
        dedupe="paper-legacy-conflict",
        max_retries=2,
    )
    assert outcome.status == "failed"
    assert row.attempts == 1, "a session conflict must not spend the retry budget"
    assert "session_conflict" in (row.error or "")
    with PaperLedger.open_reader(harness.context.paper_ledger_path) as ledger:
        assert ledger.signals(session_id) == []


def test_paper_advance_refuses_an_unreplayable_session(tmp_path: Path) -> None:
    """A ledger mark the simulator cannot reproduce is a conflict, not a retry."""
    bars = synthetic_bars()
    harness = make_harness(tmp_path, bars.head(1000))
    seed_champion(harness.context)
    payload = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 10}

    _, first_outcome, first_row = harness.run(KIND_PAPER_ADVANCE, payload, dedupe="paper-replay-1")
    assert first_outcome.status == "succeeded"
    first = data_of(first_row)

    # Tamper with the newest account mark: a replay of the same signals can no
    # longer land on the recorded balance.
    with PaperLedger(harness.context.paper_ledger_path) as ledger:
        last = ledger.equity(first["session_id"])[-1]
        ledger.record_equity(
            first["session_id"], last.ts + timedelta(minutes=1), 999_999.0, 999_999.0
        )
    assert (
        ParquetStore(harness.context.data_root).write(SYMBOL, TIMEFRAME, bars.slice(1000, 20)) == 20
    )

    _, outcome, row = harness.run(
        KIND_PAPER_ADVANCE, payload, dedupe="paper-replay-2", max_retries=2
    )
    assert outcome.status == "failed"
    assert row.attempts == 1
    assert "session_conflict" in (row.error or "")


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
