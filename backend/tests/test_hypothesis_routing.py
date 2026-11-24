"""Hypothesis → experiment-proposal records: routing, approval gate and conversion."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from scalping.knowledge import (
    PROPOSAL_GROUPS,
    PROPOSAL_SCHEMA_VERSION,
    Approval,
    DocumentIngestor,
    DocumentRecord,
    ExperimentProposal,
    Hypothesis,
    KnowledgeIndex,
    KnowledgeIndexError,
    ProposalStore,
    RoutingError,
    SourceRef,
    create_hypothesis,
    extract_hypotheses,
    proposals_for_queue,
    propose_from_hypothesis,
    set_status,
)

TEXT = """# Claims

When volatility expands, momentum signals become less reliable. A signed order-flow imbalance predicts short-term continuation. The more the spread widens, the lower the edge.

## Liquidity

Sweeps of session lows lead to reversals.
"""


@pytest.fixture
def index(tmp_path: Path) -> KnowledgeIndex:
    return KnowledgeIndex(tmp_path / "knowledge.sqlite3")


def _document(index: KnowledgeIndex, tmp_path: Path, text: str = TEXT) -> DocumentRecord:
    source = tmp_path / "notes.md"
    source.write_text(text, encoding="utf-8")
    record = DocumentIngestor(tmp_path / "knowledge").ingest(source)
    index.add_document(record)
    return record


def _hypotheses(index: KnowledgeIndex, tmp_path: Path) -> list[Hypothesis]:
    record = _document(index, tmp_path)
    return list(extract_hypotheses(index, record.doc_id).created)


def _store(tmp_path: Path) -> ProposalStore:
    return ProposalStore(tmp_path / "proposals.sqlite3")


def _proposal(
    hypothesis: Hypothesis, index: KnowledgeIndex, store: ProposalStore
) -> ExperimentProposal:
    return store.create(
        propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M15", index=index)
    )


def test_proposal_from_extracted_hypothesis_is_a_text_record(
    index: KnowledgeIndex, tmp_path: Path
) -> None:
    hypothesis = _hypotheses(index, tmp_path)[0]
    store = _store(tmp_path)

    proposal = propose_from_hypothesis(hypothesis, symbol="xauusd", timeframe="m15", index=index)

    assert proposal.status == "proposed"
    assert proposal.symbol == "XAUUSD"
    assert proposal.timeframe == "M15"
    assert proposal.hypothesis_id == hypothesis.hypothesis_id
    assert proposal.hypothesis_kind in {"claim", "observation"}
    assert proposal.source_refs == hypothesis.source_refs
    assert proposal.dsl_observation == hypothesis.observed
    assert proposal.dsl_condition == hypothesis.proposed_condition
    assert proposal.dsl_expected_effect == hypothesis.expected_effect
    assert proposal.falsification_test == hypothesis.falsification_test
    assert proposal.approval is None
    assert proposal.converted_candidate_hash is None
    assert proposal.dataset_id is None
    assert len(proposal.proposal_id) == 64
    assert "price" in proposal.required_groups
    assert "volatility" in proposal.required_groups
    assert "spread" in proposal.required_columns

    stored = store.create(proposal)
    assert stored == store.get(proposal.proposal_id)
    assert stored.to_dict()["status"] == "proposed"


def test_missing_sources_are_refused(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypothesis = create_hypothesis(
        index,
        "Flow predicts continuation.",
        "Observed in a section.",
        "a flow imbalance",
        "continuation",
        "XAUUSD",
        "M15",
        "Reject when continuation is absent.",
        source_refs=(),
    )

    with pytest.raises(RoutingError) as excinfo:
        propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M15", index=index)
    assert excinfo.value.code == "missing_source"


def test_unresolvable_source_is_refused(index: KnowledgeIndex, tmp_path: Path) -> None:
    _hypotheses(index, tmp_path)
    fabricated = Hypothesis(
        hypothesis_id="not-stored",
        statement="When volatility expands, momentum weakens.",
        observed="A section.",
        proposed_condition="volatility expands",
        expected_effect="momentum weakens",
        market_scope="XAUUSD",
        timeframe_scope="M15",
        assumptions=(),
        falsification_test="Reject if absent.",
        source_refs=(SourceRef(doc_id="0" * 64, section_ordinal=0, quote="quote"),),
        status="proposed",
        experiment_refs=(),
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(RoutingError) as excinfo:
        propose_from_hypothesis(fabricated, symbol="XAUUSD", timeframe="M15", index=index)
    assert excinfo.value.code == "missing_source"


def test_supported_and_rejected_hypotheses_are_refused(
    index: KnowledgeIndex, tmp_path: Path
) -> None:
    first, second, *_ = _hypotheses(index, tmp_path)
    set_status(index, first.hypothesis_id, "testing")
    supported = set_status(index, first.hypothesis_id, "supported")
    rejected = set_status(index, second.hypothesis_id, "rejected")
    assert supported.status == "supported"
    assert rejected.status == "rejected"

    for hypothesis in (supported, rejected):
        with pytest.raises(RoutingError) as excinfo:
            propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M15", index=index)
        assert excinfo.value.code == "bad_status"


def test_approve_and_reject_are_audited(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypotheses = _hypotheses(index, tmp_path)
    store = _store(tmp_path)
    approved = _proposal(hypotheses[0], index, store)
    rejected = _proposal(hypotheses[1], index, store)

    approved = store.approve(approved.proposal_id, "scope and data are clear", "owner")
    rejected = store.reject(rejected.proposal_id, "not testable with stored data", "owner")

    assert approved.status == "approved"
    assert approved.approval is not None
    assert approved.approval.actor == "owner"
    assert approved.approval.reason == "scope and data are clear"
    assert approved.approval.decided_at
    assert approved.approval.time == approved.approval.decided_at

    assert rejected.status == "rejected"
    assert rejected.approval is not None
    assert rejected.approval.reason == "not testable with stored data"

    with pytest.raises(RoutingError) as excinfo:
        store.approve(rejected.proposal_id, "changed my mind", "owner")
    assert excinfo.value.code == "bad_status"

    with pytest.raises(ValueError, match="must not be empty"):
        store.approve(approved.proposal_id, "   ", "owner")


def test_only_proposed_records_can_be_created(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypothesis = _hypotheses(index, tmp_path)[0]
    store = _store(tmp_path)
    proposal = propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M15", index=index)
    forged = replace(
        proposal,
        status="approved",
        approval=Approval(actor="owner", reason="forged", decided_at="2026-01-01T00:00:00+00:00"),
    )

    with pytest.raises(RoutingError) as excinfo:
        store.create(forged)
    assert excinfo.value.code == "bad_status"


def test_candidate_link_is_one_way_and_one_time(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypothesis = _hypotheses(index, tmp_path)[0]
    store = _store(tmp_path)
    proposal = _proposal(hypothesis, index, store)

    with pytest.raises(RoutingError) as excinfo:
        store.link_candidate(proposal.proposal_id, "a" * 64, "validation:XAUUSD:M15:abc")
    assert excinfo.value.code == "not_approved"

    store.approve(proposal.proposal_id, "approved for a bounded run", "owner")
    converted = store.link_candidate(proposal.proposal_id, "a" * 64, "validation:XAUUSD:M15:abc")

    assert converted.status == "converted"
    assert converted.converted_candidate_hash == "a" * 64
    assert converted.dataset_id == "validation:XAUUSD:M15:abc"
    assert converted.approval is not None
    assert converted.approval.actor == "owner"

    with pytest.raises(RoutingError) as excinfo:
        store.link_candidate(proposal.proposal_id, "b" * 64, "validation:XAUUSD:M15:def")
    assert excinfo.value.code == "already_converted"

    with pytest.raises(RoutingError) as excinfo:
        store.approve(proposal.proposal_id, "again", "owner")
    assert excinfo.value.code == "already_converted"


def test_queue_returns_only_approved_proposals(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypotheses = _hypotheses(index, tmp_path)
    store = _store(tmp_path)
    approved = _proposal(hypotheses[0], index, store)
    rejected = _proposal(hypotheses[1], index, store)
    converted = _proposal(hypotheses[2], index, store)
    _proposal(hypotheses[3], index, store)

    store.approve(approved.proposal_id, "ready", "owner")
    store.reject(rejected.proposal_id, "not useful", "owner")
    store.approve(converted.proposal_id, "ready", "owner")
    store.link_candidate(converted.proposal_id, "c" * 64, "validation:XAUUSD:M15:ccc")

    queue = proposals_for_queue(store, limit=10)

    assert [item.proposal_id for item in queue] == [approved.proposal_id]
    assert proposals_for_queue(store, limit=1) == queue
    with pytest.raises(ValueError, match="limit"):
        proposals_for_queue(store, limit=0)


def test_unknown_symbol_or_timeframe_is_refused(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypothesis = _hypotheses(index, tmp_path)[0]

    with pytest.raises(RoutingError) as excinfo:
        propose_from_hypothesis(hypothesis, symbol="DOGEUSD", timeframe="M15", index=index)
    assert excinfo.value.code == "unknown_scope"

    with pytest.raises(RoutingError) as excinfo:
        propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M2", index=index)
    assert excinfo.value.code == "unknown_scope"

    with pytest.raises(RoutingError) as excinfo:
        propose_from_hypothesis(
            hypothesis, symbol="XAUUSD", timeframe="M15", index=index, hypothesis_kind="opinion"
        )
    assert excinfo.value.code == "unsupported_kind"


def test_proposal_id_is_deterministic_and_survives_reopen(
    index: KnowledgeIndex, tmp_path: Path
) -> None:
    hypothesis = _hypotheses(index, tmp_path)[0]
    store = _store(tmp_path)
    first = propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M15", index=index)
    second = propose_from_hypothesis(hypothesis, symbol="XAUUSD", timeframe="M15", index=index)
    assert first.proposal_id == second.proposal_id
    assert first.to_dict()["proposal_id"] == second.to_dict()["proposal_id"]

    stored = store.create(first)
    assert store.create(second).proposal_id == stored.proposal_id
    approved = store.approve(stored.proposal_id, "reproducible", "owner")
    assert store.schema_version == PROPOSAL_SCHEMA_VERSION
    store.close()

    reopened = _store(tmp_path)
    recovered = reopened.get(approved.proposal_id)
    assert recovered == approved
    assert recovered is not None
    assert recovered.source_refs == first.source_refs
    assert recovered.approval == approved.approval
    assert [item.proposal_id for item in proposals_for_queue(reopened, 10)] == [
        approved.proposal_id
    ]
    reopened.close()


def test_unknown_and_closed_store_operations_raise(index: KnowledgeIndex, tmp_path: Path) -> None:
    hypothesis = _hypotheses(index, tmp_path)[0]
    store = _store(tmp_path)
    proposal = _proposal(hypothesis, index, store)

    with pytest.raises(RoutingError) as excinfo:
        store.approve("0" * 64, "reason", "owner")
    assert excinfo.value.code == "unknown_proposal"

    store.close()
    with pytest.raises(KnowledgeIndexError, match="closed"):
        store.get(proposal.proposal_id)


def test_proposal_groups_match_the_live_feature_registry() -> None:
    from scalping.features.registry import FEATURE_GROUPS

    assert set(PROPOSAL_GROUPS) == set(FEATURE_GROUPS)


def test_routing_module_has_no_strategy_order_or_network_import() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / "src" / "scalping" / "knowledge" / "routing.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)

    forbidden_prefixes = (
        "scalping.strategy",
        "scalping.discovery",
        "scalping.api",
        "scalping.paper",
        "scalping.env",
        "scalping.backtest",
        "scalping.orchestrator",
        "scalping.registry",
    )
    forbidden_modules = {
        "socket",
        "subprocess",
        "urllib",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "telnetlib",
    }
    for module in modules:
        assert not module.startswith(forbidden_prefixes), module
        assert module.split(".")[0] not in forbidden_modules, module

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "eval" not in names
    assert "exec" not in names
    assert "compile" not in names
