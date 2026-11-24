"""Matured paper outcomes as a research trigger: eligibility, dedupe, proposal.

The properties pinned here decide whether a completed paper trade may start the
next research cycle:

* a new fingerprint enqueues exactly one ``paper.outcomes`` job;
* repeating the scan, across a store restart, does not duplicate it;
* the dedupe key alone survives a crash between enqueue and acknowledgement;
* an open position is never part of the payload;
* a session without a strategy hash or a dataset id is refused, with a logged
  reason, and a repeat scan does not repeat the log line;
* an empty batch or a missing ledger is a no-op;
* the handler records the event and returns a proposal; it enqueues nothing and
  imports no discovery or training code;
* closing one more trade changes the fingerprint and emits a second trigger.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scalping.orchestrator import (
    HandlerResult,
    JobStore,
    JobStoreReader,
    Orchestrator,
    RetryPolicy,
    ScanOutcome,
    scan_once,
)
from scalping.orchestrator.paper_triggers import (
    EVENT_NOT_QUEUED,
    EVENT_RECEIVED,
    EVENT_REFUSED,
    EVENT_TRUNCATED,
    KIND_PAPER_OUTCOMES,
    MAX_SCAN_PROPOSALS_LIMIT,
    MAX_SCAN_SESSIONS_LIMIT,
    PAPER_OUTCOME_FIELDS,
    REASON_EMPTY_BATCH,
    REASON_NO_DATASET_ID,
    REASON_NO_LEDGER,
    REASON_NO_STRATEGY_HASH,
    REASON_SYNTHETIC,
    REASON_TRUNCATED,
    REFUSAL_SYNTHETIC,
    PaperOutcomeTriggerSource,
    make_paper_outcome_handler,
    register_paper_outcome_handler,
)
from scalping.paper.ledger import PaperLedger

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
BASE = datetime(2026, 1, 5, 10, tzinfo=UTC)


# --------------------------------------------------------------------------- fixtures


def seed_session(
    ledger: PaperLedger,
    session_id: str,
    *,
    model_version: str | None = "hash-1",
    dataset_id: str | None = "dataset-1",
    closed: int = 1,
    open_positions: int = 0,
    pnl: float = 10.0,
    r_multiple: float = 1.0,
    started_at: datetime = BASE,
    exit_offset: timedelta = timedelta(minutes=5),
    extra_config: dict[str, object] | None = None,
) -> None:
    """One paper session with the requested number of closed and open trades."""
    config: dict[str, object] = {}
    if dataset_id is not None:
        config = {"origin": "test", "data": {"dataset_id": dataset_id}}
    if extra_config:
        config = {**config, **extra_config}
    ledger.open_session(
        session_id,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        strategy="dsl",
        model_version=model_version,
        started_at=started_at,
        config=config,
    )
    for index in range(closed):
        entry = started_at + timedelta(minutes=10 * index)
        ledger.record_trade(
            session_id,
            side="long",
            entry_ts=entry,
            entry_price=2000.0,
            exit_ts=entry + exit_offset,
            exit_price=2001.0,
            lot=0.10,
            pnl=pnl,
            r_multiple=r_multiple,
            exit_reason="take_profit",
        )
    for index in range(open_positions):
        entry = started_at + timedelta(minutes=30 + 10 * index)
        order_id = ledger.record_order(
            session_id,
            signal_id=None,
            ts=entry,
            side="short",
            lot=0.10,
            requested_price=2000.0,
            status="filled",
        )
        ledger.record_fill(
            order_id,
            ts=entry,
            price=2000.0,
            lot=0.10,
            sl=2005.0,
            tp=1990.0,
        )


def not_queued_events(store: JobStore) -> list[dict[str, object]]:
    return [
        json.loads(event.detail or "{}")
        for event in store.events()
        if event.event == EVENT_NOT_QUEUED
    ]


def cursor_document(store: JobStore, source: PaperOutcomeTriggerSource) -> dict[str, object]:
    raw = store.get_state(source.cursor_key)
    assert raw is not None
    document = json.loads(raw)
    assert isinstance(document, dict)
    return document


def trigger_jobs(store: JobStore) -> list:
    return store.list_jobs(kind=KIND_PAPER_OUTCOMES)


# ----------------------------------------------------------------------------- tests


def test_new_fingerprint_emits_one_trigger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "complete", pnl=12.5, r_multiple=1.25)
        seed_session(ledger, "empty", closed=0, open_positions=0)

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(ledger_path, store=store)

    outcome = scan_once(orchestrator, [source])
    assert isinstance(outcome, ScanOutcome)
    assert outcome.sources == (source.name,)
    assert len(outcome.queued) == 1
    assert not outcome.duplicates

    job = outcome.queued[0]
    assert job.kind == KIND_PAPER_OUTCOMES
    assert set(job.payload) == PAPER_OUTCOME_FIELDS
    assert job.payload["session_id"] == "complete"
    assert job.payload["trades"] == 1
    assert job.payload["pnl"] == pytest.approx(12.5)
    assert job.payload["r_total"] == pytest.approx(1.25)
    assert job.payload["strategy_hash"] == "hash-1"
    assert job.payload["dataset_id"] == "dataset-1"
    assert job.dedupe_key == f"{KIND_PAPER_OUTCOMES}:{job.payload['fingerprint']}"

    assert any(event.event == "trigger_scan" for event in store.events())
    store.close()


def test_repeat_scan_dedupes_through_the_cursor_and_store_key(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "complete")

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(ledger_path, store=store)

    # Crash simulation: enqueue the trigger but never acknowledge the scan.
    first = source.scan()
    assert len(first) == 1
    store.enqueue(
        first[0].kind,
        dedupe_key=first[0].dedupe_key,
        payload=first[0].payload,
    )

    second = scan_once(orchestrator, [source])
    assert not second.queued
    assert len(second.duplicates) == 1

    # The cursor is now persisted, so a third scan produces no trigger at all.
    third = scan_once(orchestrator, [source])
    assert not third.queued
    assert not third.duplicates
    assert len(trigger_jobs(store)) == 1
    store.close()

    # Same ledger and same job store after a process restart: still one job.
    reopened = JobStore(tmp_path / "jobs.db")
    reopened_orchestrator = Orchestrator(reopened)
    reopened_source = PaperOutcomeTriggerSource(ledger_path, store=reopened)
    after_restart = scan_once(reopened_orchestrator, [reopened_source])
    assert not after_restart.queued
    assert not after_restart.duplicates
    assert len(trigger_jobs(reopened)) == 1
    reopened.close()


def test_open_position_is_counted_but_never_in_the_payload(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "mixed", closed=1, open_positions=1)
        seed_session(ledger, "open_only", closed=0, open_positions=1)

    store = JobStore(tmp_path / "jobs.db")
    outcome = scan_once(
        Orchestrator(store),
        [PaperOutcomeTriggerSource(ledger_path, store=store)],
    )

    assert len(outcome.queued) == 1
    job = outcome.queued[0]
    assert job.payload["session_id"] == "mixed"
    assert job.payload["trades"] == 1
    assert job.payload["pnl"] == pytest.approx(10.0)
    assert "open_trades" not in job.payload

    reasons = not_queued_events(store)
    assert [item["session_id"] for item in reasons] == ["open_only"]
    assert reasons[0]["reason"] == REASON_EMPTY_BATCH
    store.close()


def test_missing_strategy_hash_is_refused_with_a_logged_reason(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "unattributed", model_version=None)

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(ledger_path, store=store)

    outcome = scan_once(orchestrator, [source])
    assert not outcome.queued
    assert not trigger_jobs(store)
    reasons = not_queued_events(store)
    assert len(reasons) == 1
    assert reasons[0]["session_id"] == "unattributed"
    assert reasons[0]["reason"] == REASON_NO_STRATEGY_HASH
    assert "strategy hash" in str(reasons[0]["detail"])

    # The same refusal is not logged again on the next tick.
    scan_once(orchestrator, [source])
    assert len(not_queued_events(store)) == 1
    store.close()


def test_missing_dataset_id_is_refused_with_a_logged_reason(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "unscoped", dataset_id=None)

    store = JobStore(tmp_path / "jobs.db")
    outcome = scan_once(
        Orchestrator(store),
        [PaperOutcomeTriggerSource(ledger_path, store=store)],
    )

    assert not outcome.queued
    reasons = not_queued_events(store)
    assert len(reasons) == 1
    assert reasons[0]["reason"] == REASON_NO_DATASET_ID
    assert "dataset id" in str(reasons[0]["detail"])
    store.close()


def test_empty_batch_and_missing_ledger_are_no_ops(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "no_trades", closed=0)

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)

    empty = scan_once(orchestrator, [PaperOutcomeTriggerSource(ledger_path, store=store)])
    assert not empty.queued
    assert not trigger_jobs(store)
    assert [item["reason"] for item in not_queued_events(store)] == [REASON_EMPTY_BATCH]

    missing = scan_once(
        orchestrator,
        [PaperOutcomeTriggerSource(tmp_path / "missing.db", store=store)],
    )
    assert not missing.queued
    assert any(item["reason"] == REASON_NO_LEDGER for item in not_queued_events(store))
    store.close()


def test_handler_returns_a_proposal_and_executes_nothing(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    assert register_paper_outcome_handler(orchestrator) == KIND_PAPER_OUTCOMES
    payload = {
        "session_id": "complete",
        "fingerprint": "a" * 64,
        "trades": 2,
        "pnl": 15.0,
        "r_total": 1.5,
        "strategy_hash": "hash-1",
        "dataset_id": "dataset-1",
    }
    job, created = store.enqueue(KIND_PAPER_OUTCOMES, dedupe_key="paper-1", payload=payload)
    assert created

    outcomes = orchestrator.run_until_idle()
    assert [item.status for item in outcomes] == ["succeeded"]

    finished = store.get(job.id)
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.result is not None
    data = finished.result["data"]
    assert data["action"] == "queue_discovery"
    assert data["queued"] is False
    assert data["executed"] is False
    assert data["session_id"] == "complete"
    assert data["fingerprint"] == "a" * 64
    assert data["trades"] == 2
    assert data["strategy_hash"] == "hash-1"
    assert data["dataset_id"] == "dataset-1"

    # The event is recorded with the job, and the handler created no second job.
    assert any(event.event == EVENT_RECEIVED for event in store.events(job_id=job.id))
    assert store.stats().total == 1
    store.close()


def test_handler_policy_callback_can_decline_without_executing(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def policy(payload) -> bool:
        seen.append(dict(payload))
        return False

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(
        store,
        handlers={KIND_PAPER_OUTCOMES: make_paper_outcome_handler(should_queue=policy)},
    )
    job, _ = store.enqueue(
        KIND_PAPER_OUTCOMES,
        dedupe_key="paper-2",
        payload={
            "session_id": "complete",
            "fingerprint": "b" * 64,
            "trades": 1,
            "pnl": -5.0,
            "r_total": -0.5,
            "strategy_hash": "hash-1",
            "dataset_id": "dataset-1",
        },
    )
    orchestrator.run_until_idle()

    finished = store.get(job.id)
    assert finished is not None and finished.result is not None
    data = finished.result["data"]
    assert data["action"] == "no_action"
    assert data["queued"] is False
    assert data["executed"] is False
    assert seen and seen[0]["fingerprint"] == "b" * 64
    assert store.stats().total == 1
    store.close()


def test_fingerprint_changes_when_a_new_trade_closes(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "growing", closed=1)

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(ledger_path, store=store)

    first = scan_once(orchestrator, [source])
    assert len(first.queued) == 1
    first_fingerprint = first.queued[0].payload["fingerprint"]

    with PaperLedger(ledger_path) as ledger:
        ledger.record_trade(
            "growing",
            side="short",
            entry_ts=BASE + timedelta(minutes=20),
            entry_price=2000.0,
            exit_ts=BASE + timedelta(minutes=25),
            exit_price=2002.0,
            lot=0.10,
            pnl=-20.0,
            r_multiple=-1.5,
            exit_reason="stop_loss",
        )

    second = scan_once(orchestrator, [source])
    assert len(second.queued) == 1
    second_job = second.queued[0]
    assert second_job.payload["fingerprint"] != first_fingerprint
    assert second_job.payload["trades"] == 2
    assert second_job.payload["pnl"] == pytest.approx(-10.0)
    assert second_job.payload["r_total"] == pytest.approx(-0.5)
    assert len(trigger_jobs(store)) == 2
    store.close()


def test_module_contains_no_discovery_training_or_order_path() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "scalping"
        / "orchestrator"
        / "paper_triggers.py"
    )
    text = source_path.read_text(encoding="utf-8")
    for forbidden in (
        "scalping.discovery",
        "run_discovery",
        "AutonomousPaperRunner",
        "torch",
        "stable_baselines3",
    ):
        assert forbidden not in text


# ------------------------------------------------------- coverage and truncation


def test_activity_order_does_not_starve_a_busy_older_session(tmp_path: Path) -> None:
    """A session that keeps closing trades is read before a newer quiet one."""
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(
            ledger,
            "busy",
            started_at=BASE,
            exit_offset=timedelta(days=10),
        )
        seed_session(
            ledger,
            "quiet",
            started_at=BASE + timedelta(days=1),
            exit_offset=timedelta(minutes=5),
        )

    store = JobStore(tmp_path / "jobs.db")
    source = PaperOutcomeTriggerSource(ledger_path, store=store, max_sessions=1)
    outcome = scan_once(Orchestrator(store), [source])

    assert [job.payload["session_id"] for job in outcome.queued] == ["busy"]
    store.close()


def test_rotation_visits_every_session_beyond_the_cap(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        for index, name in enumerate(("s1", "s2", "s3")):
            seed_session(
                ledger,
                name,
                exit_offset=timedelta(minutes=5 + index),
            )

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(ledger_path, store=store, max_sessions=1)

    visited: list[str] = []
    for _ in range(3):
        outcome = scan_once(orchestrator, [source])
        assert len(outcome.queued) == 1
        visited.append(str(outcome.queued[0].payload["session_id"]))

    assert sorted(visited) == ["s1", "s2", "s3"]
    # A full cycle later nothing is emitted again: coverage is not duplication.
    fourth = scan_once(orchestrator, [source])
    assert not fourth.queued
    assert len(trigger_jobs(store)) == 3
    store.close()


def test_deferred_sessions_are_recorded_not_dropped(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "first", exit_offset=timedelta(minutes=10))
        seed_session(ledger, "second", exit_offset=timedelta(minutes=5))

    store = JobStore(tmp_path / "jobs.db")
    outcome = scan_once(
        Orchestrator(store),
        [PaperOutcomeTriggerSource(ledger_path, store=store, max_sessions=1)],
    )
    assert len(outcome.queued) == 1

    summaries = [event for event in store.events() if event.event == EVENT_TRUNCATED]
    assert len(summaries) == 1
    summary = json.loads(summaries[0].detail or "{}")
    assert summary["deferred"] == 1
    assert summary["scanned"] == 1
    assert summary["total"] == 2

    truncated = [item for item in not_queued_events(store) if item["reason"] == REASON_TRUNCATED]
    assert [item["session_id"] for item in truncated] == ["second"]
    store.close()


def test_a_proposed_session_is_never_recorded_as_truncated(tmp_path: Path) -> None:
    """Deferred means "not read yet", not "refused"; a queued session is neither.

    A later scan may leave an already-proposed session outside its window. That
    must not write a stale `truncated` refusal, and visiting it again must clear
    any note left by an older cursor.
    """
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        for index, name in enumerate(("s1", "s2", "s3")):
            seed_session(ledger, name, exit_offset=timedelta(minutes=5 + index))

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(ledger_path, store=store, max_sessions=1)

    queued: set[str] = set()
    for _ in range(3):
        outcome = scan_once(orchestrator, [source])
        assert len(outcome.queued) == 1
        queued.add(str(outcome.queued[0].payload["session_id"]))
    assert queued == {"s1", "s2", "s3"}

    # All three triggers exist, so no session is a refusal any more.
    marker = store.events()[-1].id
    scan_once(orchestrator, [source])
    assert cursor_document(store, source)["not_queued"] == {}
    # The historical lines stay; no new false refusal is written.
    new_events = store.events(since_id=marker)
    assert not [event for event in new_events if event.event == EVENT_NOT_QUEUED]
    store.close()


def test_unacknowledged_trigger_is_reemitted_at_the_start_of_the_next_scan(
    tmp_path: Path,
) -> None:
    """A partial ack must not wait for the rotation to wrap around."""
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        for index, name in enumerate(("s1", "s2", "s3")):
            seed_session(ledger, name, exit_offset=timedelta(minutes=5 + index))

    store = JobStore(tmp_path / "jobs.db")
    source = PaperOutcomeTriggerSource(ledger_path, store=store, max_sessions=1)

    first = source.scan()
    assert len(first) == 1
    unacknowledged = first[0].payload["fingerprint"]
    source.seen([])

    document = cursor_document(store, source)
    assert document["pending_ack"] == {unacknowledged: first[0].payload["session_id"]}
    fingerprints = document["fingerprints"]
    assert isinstance(fingerprints, dict)
    assert unacknowledged not in fingerprints

    # max_sessions=1 would otherwise point the rotation at a different session.
    second = source.scan()
    assert second[0].payload["fingerprint"] == unacknowledged
    source.seen(second)
    assert cursor_document(store, source)["pending_ack"] == {}
    store.close()


def test_partial_acknowledgement_reemits_only_the_unacked_trigger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "alpha", exit_offset=timedelta(minutes=10))
        seed_session(ledger, "beta", exit_offset=timedelta(minutes=5))

    store = JobStore(tmp_path / "jobs.db")
    source = PaperOutcomeTriggerSource(ledger_path, store=store)

    first = source.scan()
    assert len(first) == 2
    acknowledged = first[0]
    source.seen([acknowledged])

    second = source.scan()
    assert len(second) == 1
    assert second[0].payload["fingerprint"] != acknowledged.payload["fingerprint"]
    assert second[0].payload["session_id"] == "beta"

    # Acknowledging the rest converges: one cursor entry per emitted trigger.
    source.seen(second)
    assert source.scan() == []
    store.close()


# ------------------------------------------------------------- failure handling


def test_transient_handler_failure_retries_before_the_batch_is_burned(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(ledger, "complete")

    calls = 0

    def flaky(job, context):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("provider hiccup")
        return HandlerResult.success({"retried": True})

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(
        store,
        handlers={KIND_PAPER_OUTCOMES: flaky},
        retry=RetryPolicy(base_seconds=1),
    )
    source = PaperOutcomeTriggerSource(ledger_path, store=store)
    now = datetime(2026, 1, 5, 12, tzinfo=UTC)

    first = scan_once(orchestrator, [source], now=now)
    assert len(first.queued) == 1
    assert first.queued[0].max_retries >= 1

    retried = orchestrator.run_once(now=now)
    assert retried.status == "retry"
    finished = orchestrator.run_once(now=now + timedelta(seconds=2))
    assert finished.status == "succeeded"
    assert finished.job is not None and finished.job.attempts == 2
    assert len(trigger_jobs(store)) == 1

    # The batch is not scanned again: the job consumed it only once.
    again = scan_once(orchestrator, [source], now=now + timedelta(seconds=5))
    assert not again.queued
    store.close()


# ------------------------------------------------------------------ provenance


def test_payload_carries_provenance_and_unknown_is_not_false(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(
            ledger,
            "known",
            started_at=BASE + timedelta(days=1),
            extra_config={
                "provenance": {"source": "provider:csv", "synthetic": False},
            },
        )
        seed_session(ledger, "unknown", started_at=BASE)

    store = JobStore(tmp_path / "jobs.db")
    outcome = scan_once(
        Orchestrator(store),
        [PaperOutcomeTriggerSource(ledger_path, store=store)],
    )

    by_session = {job.payload["session_id"]: job.payload for job in outcome.queued}
    assert by_session["known"]["source"] == "provider:csv"
    assert by_session["known"]["synthetic"] is False
    assert by_session["unknown"]["source"] is None
    # Unknown is None, never an invented False: a demo session must be refusable.
    assert by_session["unknown"]["synthetic"] is None
    assert set(by_session["known"]) == PAPER_OUTCOME_FIELDS
    store.close()


def test_synthetic_batch_is_refused_and_logged_by_default(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(
            ledger,
            "demo",
            extra_config={"provenance": {"source": "synthetic", "synthetic": True}},
        )

    store = JobStore(tmp_path / "jobs.db")
    source = PaperOutcomeTriggerSource(ledger_path, store=store)
    outcome = scan_once(Orchestrator(store), [source])

    assert not outcome.queued
    assert trigger_jobs(store) == []
    assert cursor_document(store, source)["not_queued"] == {"demo": REASON_SYNTHETIC}
    refused = [
        event
        for event in store.events(limit=50)
        if event.event == EVENT_NOT_QUEUED and event.detail is not None
    ]
    assert any(json.loads(event.detail)["reason"] == REASON_SYNTHETIC for event in refused)
    store.close()


def test_allow_synthetic_opt_in_emits_the_batch_with_its_flag(tmp_path: Path) -> None:
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        seed_session(
            ledger,
            "demo",
            extra_config={"provenance": {"source": "synthetic", "synthetic": True}},
        )

    store = JobStore(tmp_path / "jobs.db")
    outcome = scan_once(
        Orchestrator(store),
        [PaperOutcomeTriggerSource(ledger_path, store=store, allow_synthetic=True)],
    )

    assert len(outcome.queued) == 1
    payload = outcome.queued[0].payload
    assert payload["synthetic"] is True
    assert payload["source"] == "synthetic"
    store.close()


def test_handler_refuses_a_synthetic_batch_by_default(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    register_paper_outcome_handler(orchestrator)
    job, _ = store.enqueue(
        KIND_PAPER_OUTCOMES,
        dedupe_key="demo-synthetic",
        payload={
            "session_id": "demo",
            "fingerprint": "d" * 64,
            "trades": 1,
            "pnl": 1.0,
            "r_total": 0.1,
            "strategy_hash": "hash-1",
            "dataset_id": "dataset-1",
            "source": "synthetic",
            "synthetic": True,
        },
    )
    orchestrator.run_until_idle()

    finished = store.get(job.id)
    assert finished is not None and finished.result is not None
    data = finished.result["data"]
    assert data["action"] == "no_action"
    assert data["refused"] == REFUSAL_SYNTHETIC
    assert data["queued"] is False and data["executed"] is False
    assert any(event.event == EVENT_REFUSED for event in store.events(limit=50))
    store.close()


def test_handler_opt_in_lets_a_synthetic_batch_reach_the_policy(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    seen: list[bool | None] = []

    def policy(payload: object) -> bool:
        assert isinstance(payload, dict)
        seen.append(payload["synthetic"])
        return True

    register_paper_outcome_handler(orchestrator, should_queue=policy, allow_synthetic=True)
    job, _ = store.enqueue(
        KIND_PAPER_OUTCOMES,
        dedupe_key="demo-synthetic-optin",
        payload={
            "session_id": "demo",
            "fingerprint": "e" * 64,
            "trades": 1,
            "pnl": 1.0,
            "r_total": 0.1,
            "strategy_hash": "hash-1",
            "dataset_id": "dataset-1",
            "source": "synthetic",
            "synthetic": True,
        },
    )
    orchestrator.run_until_idle()

    assert seen == [True]
    finished = store.get(job.id)
    assert finished is not None and finished.result is not None
    assert finished.result["data"]["action"] == "queue_discovery"
    store.close()


def test_handler_forwards_provenance_to_the_proposal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    register_paper_outcome_handler(orchestrator, allow_synthetic=True)
    payload = {
        "session_id": "demo",
        "fingerprint": "c" * 64,
        "trades": 1,
        "pnl": 1.0,
        "r_total": 0.1,
        "strategy_hash": "hash-1",
        "dataset_id": "dataset-1",
        "source": "provider:synthetic",
        "synthetic": True,
    }
    job, _ = store.enqueue(KIND_PAPER_OUTCOMES, dedupe_key="demo-1", payload=payload)
    orchestrator.run_until_idle()

    finished = store.get(job.id)
    assert finished is not None and finished.result is not None
    data = finished.result["data"]
    assert data["source"] == "provider:synthetic"
    assert data["synthetic"] is True
    store.close()


# --------------------------------------------------------------- contract guards


def test_handlers_get_a_read_only_store_facade(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def inspect(job, context):  # type: ignore[no-untyped-def]
        observed["reader"] = isinstance(context.store, JobStoreReader)
        observed["has_enqueue"] = hasattr(context.store, "enqueue")
        observed["has_set_state"] = hasattr(context.store, "set_state")
        observed["has_cancel"] = hasattr(context.store, "cancel")
        observed["has_private_cancel"] = hasattr(context.store, "_cancel")
        observed["row"] = context.store.get(job.id) is not None
        try:
            context.store.enqueue("x", dedupe_key="y")  # type: ignore[attr-defined]
        except AttributeError:
            observed["blocked"] = True
        return HandlerResult.success({"checked": True})

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store, handlers={"inspect": inspect})
    job, _ = store.enqueue("inspect", dedupe_key="inspect-1", payload={})
    assert orchestrator.run_once().status == "succeeded"

    assert observed == {
        "reader": True,
        "has_enqueue": False,
        "has_set_state": False,
        "has_cancel": False,
        "has_private_cancel": False,
        "row": True,
        "blocked": True,
    }
    assert store.get(job.id).status == "succeeded"  # type: ignore[union-attr]
    store.close()


def test_handler_cancel_is_scoped_to_its_own_job(tmp_path: Path) -> None:
    """`context.cancel()` works for the running job and has no reader surface."""

    def inspect(job, context):  # type: ignore[no-untyped-def]
        # The reader cannot reach another job's row for a cancel.
        assert not hasattr(context.store, "cancel")
        assert not hasattr(context.store, "_cancel")
        context.cancel("owner stopped this job")
        return HandlerResult.success({"cancelled": True})

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store, handlers={"inspect": inspect})
    other, _ = store.enqueue("other", dedupe_key="other-1", payload={})
    job, _ = store.enqueue("inspect", dedupe_key="inspect-2", payload={}, priority=1)

    outcome = orchestrator.run_once()
    assert outcome.status == "cancelled"
    current = store.get(job.id)
    assert current is not None and current.status == "cancelled"
    untouched = store.get(other.id)
    assert untouched is not None and untouched.status == "queued"
    store.close()


def test_kind_injection_is_rejected(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    with pytest.raises(ValueError, match="no registered handler"):
        PaperOutcomeTriggerSource(tmp_path / "paper.db", store=store, kind="paper.other")
    store.close()


def test_scan_bounds_above_the_limit_are_rejected(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    with pytest.raises(ValueError, match="max_sessions"):
        PaperOutcomeTriggerSource(
            tmp_path / "paper.db", store=store, max_sessions=MAX_SCAN_SESSIONS_LIMIT + 1
        )
    with pytest.raises(ValueError, match="max_proposals"):
        PaperOutcomeTriggerSource(
            tmp_path / "paper.db", store=store, max_proposals=MAX_SCAN_PROPOSALS_LIMIT + 1
        )
    store.close()


def test_since_and_proposal_cap_bound_a_first_scan(tmp_path: Path) -> None:
    """A long historical ledger cannot propose an unbounded backlog at once."""
    ledger_path = tmp_path / "paper.db"
    with PaperLedger(ledger_path) as ledger:
        for index in range(6):
            seed_session(
                ledger,
                f"old-{index}",
                started_at=BASE + timedelta(days=index),
                exit_offset=timedelta(minutes=5 + index),
            )
        for index in range(6):
            seed_session(
                ledger,
                f"new-{index}",
                started_at=BASE + timedelta(days=30, minutes=index),
                exit_offset=timedelta(minutes=5 + index),
            )

    store = JobStore(tmp_path / "jobs.db")
    orchestrator = Orchestrator(store)
    source = PaperOutcomeTriggerSource(
        ledger_path,
        store=store,
        since=BASE + timedelta(days=10),
        max_proposals=2,
    )

    per_scan = [len(scan_once(orchestrator, [source]).queued) for _ in range(4)]
    assert per_scan == [2, 2, 2, 0]

    sessions = {str(job.payload["session_id"]) for job in trigger_jobs(store)}
    assert sessions == {f"new-{index}" for index in range(6)}
    assert not any(name.startswith("old-") for name in sessions)
    store.close()
