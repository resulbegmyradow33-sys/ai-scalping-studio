"""Approved proposals → ``discovery.run`` triggers: eligibility, dedupe, limits.

The properties pinned here decide whether an owner-approved experiment proposal
may become queue work:

* only ``approved`` proposals emit; proposed, rejected and converted records are
  skipped and no status changes;
* the emitted payload validates against the real ``discovery.run`` handler;
* the cursor dedupes a repeat scan and a job-store restart, and the store key
  dedupes a crash between enqueue and acknowledgement;
* a changed discovery template is a different payload hash and therefore a
  different experiment;
* a missing or unknown scope, a missing data requirement and a locked holdout
  hint are refused with one logged reason, not repeated every scan;
* scanning executes no discovery, opens no archive and imports no discovery or
  holdout module;
* the optional ``knowledge.proposals`` handler records a snapshot and returns
  ``queue_discovery`` with ``queued: false`` / ``executed: false``; a
  non-approved snapshot returns ``no_action``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scalping.knowledge import (
    DocumentIngestor,
    ExperimentProposal,
    KnowledgeIndex,
    ProposalStore,
    extract_hypotheses,
    propose_from_hypothesis,
)
from scalping.orchestrator import JobStore, Orchestrator, proposal_handlers, scan_once
from scalping.orchestrator.handlers import (
    KIND_DISCOVERY_RUN as HANDLER_DISCOVERY_RUN,
)
from scalping.orchestrator.handlers import (
    MAX_DISCOVERY_GENERATIONS as HANDLER_MAX_GENERATIONS,
)
from scalping.orchestrator.handlers import (
    MAX_DISCOVERY_POPULATION as HANDLER_MAX_POPULATION,
)
from scalping.orchestrator.handlers import (
    MAX_DISCOVERY_TRIALS as HANDLER_MAX_TRIALS,
)
from scalping.orchestrator.handlers import DiscoveryRequest
from scalping.orchestrator.proposal_handlers import (
    EVENT_NOT_QUEUED,
    EVENT_QUEUED,
    EVENT_RECEIVED,
    KIND_DISCOVERY_RUN,
    KIND_KNOWLEDGE_PROPOSALS,
    MAX_DISCOVERY_GENERATIONS,
    MAX_DISCOVERY_POPULATION,
    MAX_DISCOVERY_TRIALS,
    PROPOSAL_MESSAGE_FIELDS,
    REASON_HOLDOUT_DATA,
    REASON_MISSING_DATA,
    REASON_MISSING_SCOPE,
    REASON_UNKNOWN_SCOPE,
    DiscoveryTemplate,
    ProposalQueueTriggerSource,
    discovery_payload_hash,
    make_proposal_handler,
    proposal_message,
    register_proposal_handler,
)
from scalping.registry.gate import GateCriteria

SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
TEXT = """# Claims

When volatility expands, momentum signals become less reliable. A signed order-flow imbalance predicts short-term continuation. The more the spread widens, the lower the edge.

## Liquidity

Sweeps of session lows lead to reversals. A session breakout after a quiet Asian range tends to continue.
"""


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def index(tmp_path: Path) -> KnowledgeIndex:
    knowledge = KnowledgeIndex(tmp_path / "knowledge.sqlite3")
    yield knowledge
    knowledge.close()


@pytest.fixture
def proposals(tmp_path: Path) -> ProposalStore:
    store = ProposalStore(tmp_path / "proposals.sqlite3")
    yield store
    store.close()


@pytest.fixture
def jobs(tmp_path: Path) -> JobStore:
    store = JobStore(tmp_path / "jobs.sqlite3")
    yield store
    store.close()


def _proposals(
    index: KnowledgeIndex, store: ProposalStore, tmp_path: Path
) -> list[ExperimentProposal]:
    source = tmp_path / "notes.md"
    source.write_text(TEXT, encoding="utf-8")
    document = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    index.add_document(document)
    hypotheses = list(extract_hypotheses(index, document.doc_id).created)
    assert len(hypotheses) >= 4
    return [
        store.create(
            propose_from_hypothesis(
                hypothesis,
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                index=index,
            ),
            index=index,
        )
        for hypothesis in hypotheses
    ]


def _template(**changes: Any) -> DiscoveryTemplate:
    base: dict[str, Any] = {
        "max_trials": 25,
        "seed": 7,
        "gate": GateCriteria(min_trades=10, max_drawdown=0.25),
        "stress": (1.5, 2.0),
        "population": 8,
        "generations": 3,
        "patience": 2,
        "horizon": 2,
    }
    base.update(changes)
    return DiscoveryTemplate(**base)


def _events(store: JobStore, name: str) -> list[dict[str, Any]]:
    return [
        json.loads(event.detail)
        for event in store.events()
        if event.event == name and event.detail is not None
    ]


def _trigger_jobs(store: JobStore) -> list:
    return store.list_jobs(kind=KIND_DISCOVERY_RUN)


def _force_dataset_hint(store: ProposalStore, proposal_id: str, dataset_id: str) -> None:
    """Simulate a row written outside the store API.

    The schema ties ``dataset_id`` to ``converted``, so an approved row cannot
    carry one through ``ProposalStore``. The source still validates the field
    defensively; this helper proves a corrupt/out-of-contract row is refused
    instead of being forwarded.
    """
    connection = store.connection
    connection.execute("PRAGMA ignore_check_constraints = ON")
    try:
        connection.execute(
            "UPDATE proposals SET dataset_id = ? WHERE proposal_id = ?",
            (dataset_id, proposal_id),
        )
    finally:
        connection.execute("PRAGMA ignore_check_constraints = OFF")


def _result_data(store: JobStore, job_id: int) -> dict[str, Any]:
    row = store.get(job_id)
    assert row is not None and row.status == "succeeded"
    assert row.result is not None and row.result["ok"] is True
    return row.result["data"]


# ----------------------------------------------------------------------------- tests


def test_only_approved_proposals_emit_and_nothing_changes_status(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    records = _proposals(index, proposals, tmp_path)
    approved, rejected, converted, proposed = records

    proposals.approve(approved.proposal_id, "scope and data are clear", "owner")
    proposals.reject(rejected.proposal_id, "not falsifiable", "owner")
    proposals.approve(converted.proposal_id, "ready", "owner")
    proposals.link_candidate(
        converted.proposal_id, "a" * 64, "validation:XAUUSD:M15:abc", actor="owner"
    )
    # `proposed` is left untouched.

    orchestrator = Orchestrator(jobs)
    source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template())
    outcome = scan_once(orchestrator, [source])

    assert len(outcome.queued) == 1
    assert not outcome.duplicates
    job = outcome.queued[0]
    assert job.kind == KIND_DISCOVERY_RUN
    assert job.payload["symbol"] == SYMBOL
    assert job.payload["timeframe"] == TIMEFRAME
    assert job.payload["seed"] == 7
    assert job.dedupe_key.startswith(f"{KIND_DISCOVERY_RUN}:proposal:{approved.proposal_id}:")

    queued_events = _events(jobs, EVENT_QUEUED)
    assert [event["proposal_id"] for event in queued_events] == [approved.proposal_id]
    assert not any(
        event["proposal_id"] in {rejected.proposal_id, converted.proposal_id, proposed.proposal_id}
        for event in queued_events
    )
    assert not _events(jobs, EVENT_NOT_QUEUED)

    # Scan never mutates governance.
    assert proposals.get(approved.proposal_id).status == "approved"  # type: ignore[union-attr]
    assert proposals.get(approved.proposal_id).converted_candidate_hash is None  # type: ignore[union-attr]
    assert proposals.get(rejected.proposal_id).status == "rejected"  # type: ignore[union-attr]
    assert proposals.get(converted.proposal_id).status == "converted"  # type: ignore[union-attr]
    assert proposals.get(proposed.proposal_id).status == "proposed"  # type: ignore[union-attr]

    # Conversion is only reachable through the explicit path, then the proposal
    # leaves the approved queue and no second trigger appears.
    converted_now = proposals.link_candidate(
        approved.proposal_id, "b" * 64, "validation:XAUUSD:M15:def", actor="owner"
    )
    assert converted_now.status == "converted"
    later = scan_once(orchestrator, [source])
    assert not later.queued
    assert len(_trigger_jobs(jobs)) == 1


def test_emitted_payload_is_valid_for_the_real_discovery_handler(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "ready", "owner")

    template = _template()
    source = ProposalQueueTriggerSource(proposals, store=jobs, template=template)
    triggers = source.scan()
    assert len(triggers) == 1

    request = DiscoveryRequest.from_payload(triggers[0].payload)
    assert request.symbol == SYMBOL
    assert request.timeframe == TIMEFRAME
    assert request.max_trials == template.max_trials
    assert request.seed == template.seed
    assert request.stress == template.stress
    assert request.population == template.population
    assert request.generations == template.generations
    assert request.patience == template.patience
    assert request.horizon == template.horizon
    assert request.gate == template.gate
    # The payload is the handler's contract, so it must not smuggle fields the
    # handler rejects; provenance lives in the dedupe key and audit event.
    assert set(triggers[0].payload) == {
        "symbol",
        "timeframe",
        "max_trials",
        "seed",
        "stress",
        "gate",
        "population",
        "generations",
        "patience",
        "horizon",
    }


def test_constants_and_kind_match_the_handler_module() -> None:
    assert KIND_DISCOVERY_RUN == HANDLER_DISCOVERY_RUN
    assert MAX_DISCOVERY_TRIALS == HANDLER_MAX_TRIALS
    assert MAX_DISCOVERY_POPULATION == HANDLER_MAX_POPULATION
    assert MAX_DISCOVERY_GENERATIONS == HANDLER_MAX_GENERATIONS


def test_scan_executes_no_discovery_and_creates_no_archive(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "ready", "owner")

    source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template())
    triggers = source.scan()
    assert len(triggers) == 1
    # A bare scan must not enqueue its own trigger; only the caller does that.
    assert jobs.list_jobs() == []

    orchestrator = Orchestrator(jobs)
    outcome = scan_once(orchestrator, [source])
    assert len(outcome.queued) == 1
    row = jobs.get(outcome.queued[0].id)
    assert row is not None and row.status == "queued"
    assert row.result is None
    assert not (tmp_path / "discovery").exists()
    assert not (tmp_path / "runs").exists()
    assert not list(tmp_path.rglob("discovery.sqlite"))
    assert proposals.get(proposal.proposal_id).status == "approved"  # type: ignore[union-attr]


def test_module_imports_no_discovery_or_holdout_path() -> None:
    source_path = Path(proposal_handlers.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name == "scalping.discovery" or name.startswith("scalping.discovery.") for name in imported
    )
    text = source_path.read_text(encoding="utf-8")
    assert "run_discovery" not in text
    assert "attach_holdout" not in text


def test_dedupe_survives_scanner_and_store_restart(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "ready", "owner")

    orchestrator = Orchestrator(jobs)
    source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template())

    # Crash simulation: the trigger is enqueued but the cursor was never
    # acknowledged. The store key makes the next scan a duplicate, not a
    # second job.
    unacknowledged = source.scan()
    assert len(unacknowledged) == 1
    jobs.enqueue(
        unacknowledged[0].kind,
        dedupe_key=unacknowledged[0].dedupe_key,
        payload=unacknowledged[0].payload,
    )
    second = scan_once(orchestrator, [source])
    assert not second.queued
    assert len(second.duplicates) == 1

    third = scan_once(orchestrator, [source])
    assert not third.queued
    assert not third.duplicates
    assert len(_trigger_jobs(jobs)) == 1

    jobs.close()
    reopened = JobStore(tmp_path / "jobs.sqlite3")
    try:
        reopened_orchestrator = Orchestrator(reopened)
        reopened_source = ProposalQueueTriggerSource(
            proposals, store=reopened, template=_template()
        )
        after_restart = scan_once(reopened_orchestrator, [reopened_source])
        assert not after_restart.queued
        assert not after_restart.duplicates
        assert len(_trigger_jobs(reopened)) == 1
    finally:
        reopened.close()


def test_changed_template_is_a_different_payload_hash_and_job(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "ready", "owner")

    orchestrator = Orchestrator(jobs)
    first_source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template())
    first = scan_once(orchestrator, [first_source])
    assert len(first.queued) == 1
    first_hash = discovery_payload_hash(first.queued[0].payload)

    second_source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template(seed=8))
    second = scan_once(orchestrator, [second_source])
    assert len(second.queued) == 1
    assert second.queued[0].dedupe_key != first.queued[0].dedupe_key
    assert discovery_payload_hash(second.queued[0].payload) != first_hash
    assert len(_trigger_jobs(jobs)) == 2


def test_missing_scope_missing_data_and_holdout_are_refused_once(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    records = _proposals(index, proposals, tmp_path)
    blank_scope, unknown_scope, no_data, holdout = records
    for record in records:
        proposals.approve(record.proposal_id, "ready", "owner")

    with proposals.transaction() as connection:
        connection.execute(
            "UPDATE proposals SET symbol = '' WHERE proposal_id = ?", (blank_scope.proposal_id,)
        )
        connection.execute(
            "UPDATE proposals SET timeframe = 'M2' WHERE proposal_id = ?",
            (unknown_scope.proposal_id,),
        )
        connection.execute(
            "UPDATE proposals SET required_columns_json = '[]' WHERE proposal_id = ?",
            (no_data.proposal_id,),
        )
    _force_dataset_hint(proposals, holdout.proposal_id, "final-holdout:XAUUSD:M15")

    orchestrator = Orchestrator(jobs)
    source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template())
    outcome = scan_once(orchestrator, [source])
    assert not outcome.queued

    refusals = {event["proposal_id"]: event["reason"] for event in _events(jobs, EVENT_NOT_QUEUED)}
    assert refusals[blank_scope.proposal_id] == REASON_MISSING_SCOPE
    assert refusals[unknown_scope.proposal_id] == REASON_UNKNOWN_SCOPE
    assert refusals[no_data.proposal_id] == REASON_MISSING_DATA
    assert refusals[holdout.proposal_id] == REASON_HOLDOUT_DATA

    # A repeat scan is quiet: the reason is recorded only when it changes.
    second = scan_once(orchestrator, [source])
    assert not second.queued
    assert len(_events(jobs, EVENT_NOT_QUEUED)) == 4
    assert _trigger_jobs(jobs) == []


def test_provenance_is_recorded_in_the_audit_log_and_cursor(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "scope and data are clear", "owner")
    _force_dataset_hint(proposals, proposal.proposal_id, "validation:XAUUSD:M15:abc")

    orchestrator = Orchestrator(jobs)
    source = ProposalQueueTriggerSource(proposals, store=jobs, template=_template())
    outcome = scan_once(orchestrator, [source])
    assert len(outcome.queued) == 1
    job = outcome.queued[0]
    payload_hash = discovery_payload_hash(job.payload)

    event = _events(jobs, EVENT_QUEUED)[0]
    assert event == {
        "proposal_id": proposal.proposal_id,
        "hypothesis_id": proposal.hypothesis_id,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "status": "approved",
        "dataset_id": "validation:XAUUSD:M15:abc",
        "required_columns": list(proposal.required_columns),
        "required_groups": list(proposal.required_groups),
        "payload_hash": payload_hash,
        "dedupe_key": job.dedupe_key,
        "approval_actor": "owner",
        "approval_reason": "scope and data are clear",
    }

    cursor = json.loads(jobs.get_state(source.cursor_key) or "{}")
    identity = f"{proposal.proposal_id}:{payload_hash}"
    assert cursor["emitted"] == {identity: proposal.proposal_id}
    assert cursor["pending_ack"] == {}
    assert cursor["not_queued"] == {}


def test_optional_handler_records_and_proposes_without_executing(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "ready", "owner")
    current = proposals.get(proposal.proposal_id)
    assert current is not None
    payload = proposal_message(current, payload_hash="c" * 64)
    assert set(payload) == PROPOSAL_MESSAGE_FIELDS

    orchestrator = Orchestrator(jobs)
    assert register_proposal_handler(orchestrator) == KIND_KNOWLEDGE_PROPOSALS
    job, created = jobs.enqueue(
        KIND_KNOWLEDGE_PROPOSALS,
        dedupe_key=f"proposal:{proposal.proposal_id}",
        payload=payload,
    )
    assert created

    outcome = orchestrator.run_once()
    assert outcome.status == "succeeded"
    data = _result_data(jobs, job.id)
    assert data["action"] == "queue_discovery"
    assert data["queued"] is False
    assert data["executed"] is False
    assert data["proposal_id"] == proposal.proposal_id
    assert data["hypothesis_id"] == proposal.hypothesis_id
    assert data["status"] == "approved"
    assert data["dataset_id"] is None
    assert data["payload_hash"] == "c" * 64
    assert data["approval_actor"] == "owner"
    assert len(data["source_refs"]) == len(proposal.source_refs)

    assert len(jobs.list_jobs()) == 1  # no discovery job was created
    assert not (tmp_path / "discovery").exists()
    received = _events(jobs, EVENT_RECEIVED)
    assert received == [
        {
            "proposal_id": proposal.proposal_id,
            "hypothesis_id": proposal.hypothesis_id,
            "status": "approved",
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "dataset_id": None,
            "payload_hash": "c" * 64,
        }
    ]
    assert proposals.get(proposal.proposal_id).status == "approved"  # type: ignore[union-attr]


def test_optional_handler_does_not_propose_a_non_approved_snapshot(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    record = _proposals(index, proposals, tmp_path)[3]
    orchestrator = Orchestrator(jobs)
    register_proposal_handler(orchestrator)
    job, _ = jobs.enqueue(
        KIND_KNOWLEDGE_PROPOSALS,
        dedupe_key=f"proposal:{record.proposal_id}",
        payload=proposal_message(record),
    )

    outcome = orchestrator.run_once()
    assert outcome.status == "succeeded"
    data = _result_data(jobs, job.id)
    assert data["action"] == "no_action"
    assert data["status"] == "proposed"
    assert data["queued"] is False
    assert data["executed"] is False
    assert len(jobs.list_jobs()) == 1


def test_optional_handler_refuses_a_holdout_dataset_hint(
    index: KnowledgeIndex,
    proposals: ProposalStore,
    jobs: JobStore,
    tmp_path: Path,
) -> None:
    proposal = _proposals(index, proposals, tmp_path)[0]
    proposals.approve(proposal.proposal_id, "ready", "owner")
    current = proposals.get(proposal.proposal_id)
    assert current is not None
    snapshot = replace(current, dataset_id="locked-holdout:XAUUSD:M15")

    orchestrator = Orchestrator(jobs)
    register_proposal_handler(orchestrator)
    job, _ = jobs.enqueue(
        KIND_KNOWLEDGE_PROPOSALS,
        dedupe_key=f"proposal:{proposal.proposal_id}",
        payload=proposal_message(snapshot),
    )
    orchestrator.run_once()
    data = _result_data(jobs, job.id)
    assert data["action"] == "no_action"
    assert "holdout" in data["reason"]
    assert data["executed"] is False


def test_invalid_handler_payload_fails_once_without_retry(
    jobs: JobStore,
) -> None:
    orchestrator = Orchestrator(jobs)
    register_proposal_handler(orchestrator)
    job, _ = jobs.enqueue(
        KIND_KNOWLEDGE_PROPOSALS,
        dedupe_key="proposal:bad",
        payload={"proposal_id": "p", "hypothesis_id": "h"},
        max_retries=2,
    )
    outcome = orchestrator.run_once()
    assert outcome.status == "failed"
    row = jobs.get(job.id)
    assert row is not None and row.status == "failed"
    assert row.attempts == 1


def test_make_proposal_handler_is_the_registered_callable(
    jobs: JobStore,
) -> None:
    orchestrator = Orchestrator(jobs)
    handler = make_proposal_handler()
    orchestrator.register(KIND_KNOWLEDGE_PROPOSALS, handler)
    assert orchestrator.handlers[KIND_KNOWLEDGE_PROPOSALS] is handler
