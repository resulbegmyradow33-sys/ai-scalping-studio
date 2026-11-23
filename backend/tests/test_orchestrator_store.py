"""Job queue: idempotency, ordering, dependencies, transitions and recovery."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scalping.orchestrator.store import (
    JOB_STATUSES,
    SCHEMA_VERSION,
    JobStateError,
    JobStore,
    StoreError,
    UnknownJobError,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "orchestrator.sqlite3")


def _enqueue(
    store: JobStore,
    kind: str,
    key: str,
    *,
    priority: int = 0,
    max_retries: int = 0,
    depends_on: Sequence[int] = (),
    scheduled_at: datetime | None = None,
) -> tuple[int, bool]:
    job, created = store.enqueue(
        kind,
        dedupe_key=key,
        priority=priority,
        max_retries=max_retries,
        depends_on=depends_on,
        scheduled_at=scheduled_at,
        now=T0,
    )
    return job.id, created


def _claimed(store: JobStore, *, now: datetime = T0, **kwargs: object) -> int | None:
    job = store.claim_next(now=now, **kwargs)
    return None if job is None else job.id


def test_schema_is_created_once_and_versioned(tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.sqlite3"
    first = JobStore(path)
    assert first.schema_version == SCHEMA_VERSION
    tables = {
        str(row["name"])
        for row in first.connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"jobs", "events", "state", "schema_version"} <= tables
    first.close()

    second = JobStore(path)
    assert second.schema_version == SCHEMA_VERSION
    versions = [row["version"] for row in second.connection.execute("SELECT * FROM schema_version")]
    assert versions == [SCHEMA_VERSION]
    second.close()


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.sqlite3"
    JobStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM schema_version")
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION + 1, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="newer than supported"):
        JobStore(path)


def test_enqueue_is_idempotent_by_dedupe_key(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job, created = store.enqueue("ingest", dedupe_key="ingest:a", payload={"path": "a.csv"}, now=T0)
    assert created is True
    again, created_again = store.enqueue(
        "ingest", dedupe_key="ingest:a", payload={"path": "different.csv"}, now=T0
    )
    assert created_again is False
    assert again.id == job.id
    assert again.payload == {"path": "a.csv"}
    assert [row.id for row in store.list_jobs()] == [job.id]
    assert [row.event for row in store.events(job.id)] == ["enqueued", "deduplicated"]
    assert store.stats(now=T0).counts["queued"] == 1


def test_enqueue_rejects_unknown_dependency(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(UnknownJobError, match="does not exist"):
        store.enqueue("train", dedupe_key="train:x", depends_on=[99], now=T0)


def test_priority_order_and_tie_break(tmp_path: Path) -> None:
    store = _store(tmp_path)
    low, _ = _enqueue(store, "a", "a:1")
    high, _ = _enqueue(store, "a", "a:2", priority=5)
    tie, _ = _enqueue(store, "a", "a:3", priority=5)

    assert _claimed(store) == high
    assert _claimed(store) == tie
    assert _claimed(store) == low
    assert _claimed(store) is None


def test_scheduled_at_is_respected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job, _ = store.enqueue(
        "train", dedupe_key="train:later", scheduled_at=T0 + timedelta(hours=1), now=T0
    )
    assert _claimed(store) is None
    claimed = store.claim_next(now=T0 + timedelta(hours=2))
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.started_at == (T0 + timedelta(hours=2)).isoformat(timespec="microseconds")


def test_dependency_must_succeed_before_child_is_claimed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent, _ = _enqueue(store, "ingest", "ingest:p")
    child, _ = _enqueue(store, "dataset", "dataset:c", depends_on=[parent], priority=9)

    assert _claimed(store) == parent
    assert _claimed(store) is None  # the child waits for its dependency
    store.complete(parent, {"rows": 10}, now=T0)
    assert _claimed(store) == child


def test_dependency_failure_is_reported_not_retried(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent, _ = _enqueue(store, "ingest", "ingest:p")
    child, _ = _enqueue(store, "dataset", "dataset:c", depends_on=[parent])
    _claimed(store)
    store.fail(parent, "bad csv", now=T0)

    assert _claimed(store) is None
    blocked = store.blocked_jobs()
    assert [item.job.id for item in blocked] == [child]
    assert blocked[0].blocking == ((parent, "failed"),)
    queued = store.get(child)
    assert queued is not None and queued.status == "queued"


def test_second_connection_cannot_claim_the_same_job(tmp_path: Path) -> None:
    first = _store(tmp_path)
    second = JobStore(tmp_path / "orchestrator.sqlite3")
    job, _ = first.enqueue("train", dedupe_key="train:once", now=T0)
    assert _claimed(first) == job.id
    assert _claimed(second) is None
    second.close()


def test_complete_stores_result_and_requires_running(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:1")
    with pytest.raises(JobStateError, match="only"):
        store.complete(job_id, {"ok": True}, now=T0)
    _claimed(store)
    finished = store.complete(job_id, {"ok": True, "data": {"auc": 0.55}}, now=T0)
    assert finished.status == "succeeded"
    assert finished.result == {"ok": True, "data": {"auc": 0.55}}
    assert finished.finished_at == T0.isoformat(timespec="microseconds")
    with pytest.raises(JobStateError):
        store.complete(job_id, {}, now=T0)


def test_retry_requeues_with_backoff_and_respects_the_bound(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:retry", max_retries=1)
    _claimed(store)
    retried = store.retry(job_id, "transient", delay_seconds=30, now=T0)
    assert retried.status == "queued"
    assert retried.attempts == 1
    assert retried.error == "transient"
    assert retried.started_at is None
    assert retried.scheduled_at == (T0 + timedelta(seconds=30)).isoformat(timespec="microseconds")
    assert _claimed(store) is None

    later = T0 + timedelta(seconds=31)
    assert _claimed(store, now=later) == job_id
    with pytest.raises(JobStateError, match="used 2 attempt"):
        store.retry(job_id, "again", delay_seconds=1, now=later)
    failed = store.fail(job_id, "again", now=later)
    assert failed.status == "failed"
    assert failed.attempts == 2


def test_retry_rejects_negative_or_nonfinite_delay(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:delay", max_retries=2)
    _claimed(store)
    with pytest.raises(ValueError, match=">= 0"):
        store.retry(job_id, "bad", delay_seconds=-1, now=T0)
    with pytest.raises(ValueError, match="finite"):
        store.retry(job_id, "bad", delay_seconds=float("inf"), now=T0)


def test_claim_clears_the_previous_attempt_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:clean", max_retries=2)
    _claimed(store)
    store.retry(job_id, "boom", delay_seconds=0, now=T0)
    claimed = store.claim_next(now=T0)
    assert claimed is not None and claimed.error is None


def test_defer_returns_the_job_without_spending_an_attempt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:defer")
    _claimed(store)
    deferred = store.defer(job_id, "RAM 95% used", delay_seconds=15, now=T0)
    assert deferred.status == "queued"
    assert deferred.attempts == 0
    assert deferred.error == "RAM 95% used"
    assert deferred.started_at is None
    assert deferred.scheduled_at == (T0 + timedelta(seconds=15)).isoformat(timespec="microseconds")
    assert [row.event for row in store.events(job_id)] == ["enqueued", "started", "deferred"]
    assert _claimed(store) is None  # still backing off
    assert _claimed(store, now=T0 + timedelta(seconds=15)) == job_id
    claimed = store.get(job_id)
    assert claimed is not None and claimed.attempts == 1


def test_defer_requires_a_running_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:defer-queued")
    with pytest.raises(JobStateError, match="only"):
        store.defer(job_id, "not yet", delay_seconds=1, now=T0)


def test_cancel_queued_and_running(tmp_path: Path) -> None:
    store = _store(tmp_path)
    queued, _ = _enqueue(store, "train", "train:q")
    cancelled = store.cancel(queued, "owner", now=T0)
    assert cancelled.status == "cancelled"
    assert cancelled.error == "cancelled: owner"
    assert _claimed(store) is None
    assert store.cancel(queued, now=T0).status == "cancelled"  # idempotent

    running, _ = _enqueue(store, "train", "train:r")
    assert _claimed(store) == running
    assert store.cancel(running, now=T0).status == "cancelled"
    with pytest.raises(JobStateError):
        store.fail(running, "late result", now=T0)


def test_cancel_refuses_a_finished_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:done")
    _claimed(store)
    store.complete(job_id, now=T0)
    with pytest.raises(JobStateError, match="cannot be cancelled"):
        store.cancel(job_id, now=T0)


def test_skip_requires_a_queued_job_and_stores_the_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id, _ = _enqueue(store, "train", "train:skip")
    skipped = store.skip(job_id, "dependency 4 is failed", now=T0)
    assert skipped.status == "skipped"
    assert skipped.error == "skipped: dependency 4 is failed"
    with pytest.raises(JobStateError):
        store.skip(job_id, "again", now=T0)


def test_recover_interrupted_requeues_while_attempts_remain(tmp_path: Path) -> None:
    store = _store(tmp_path)
    retryable, _ = _enqueue(store, "train", "train:recover", max_retries=1)
    exhausted, _ = _enqueue(store, "train", "train:exhausted", max_retries=0)
    assert _claimed(store) == retryable
    assert _claimed(store) == exhausted

    recovered = {item.job.id: item for item in store.recover_interrupted(now=T0)}
    assert recovered[retryable].requeued is True
    assert recovered[retryable].job.status == "queued"
    assert recovered[exhausted].requeued is False
    assert recovered[exhausted].job.status == "interrupted"
    exhausted_job = store.get(exhausted)
    assert exhausted_job is not None and exhausted_job.finished_at is not None
    assert [row.event for row in store.events(exhausted)] == [
        "enqueued",
        "started",
        "interrupted",
    ]
    assert [row.event for row in store.events(retryable)] == [
        "enqueued",
        "started",
        "interrupted",
        "requeued",
    ]

    # The requeued job can run again; the second attempt has no retry left.
    assert _claimed(store) == retryable
    second_recovery = store.recover_interrupted(now=T0)
    assert [item.requeued for item in second_recovery] == [False]
    again = store.get(retryable)
    assert again is not None and again.status == "interrupted"
    assert store.recover_interrupted(now=T0) == []


def test_one_heavy_job_at_a_time(tmp_path: Path) -> None:
    store = _store(tmp_path)
    heavy_a, _ = _enqueue(store, "train", "train:a", priority=9)
    heavy_b, _ = _enqueue(store, "train", "train:b", priority=8)
    light, _ = _enqueue(store, "ingest", "ingest:c")

    assert _claimed(store, heavy_kinds=("train",)) == heavy_a
    assert _claimed(store, heavy_kinds=("train",)) == light
    assert _claimed(store, heavy_kinds=("train",)) is None
    assert store.count_running(kinds=("train",)) == 1
    store.complete(heavy_a, now=T0)
    assert _claimed(store, heavy_kinds=("train",)) == heavy_b


def test_exclude_kinds_keeps_research_queued(tmp_path: Path) -> None:
    store = _store(tmp_path)
    research, _ = _enqueue(store, "research.hypothesis", "research:1", priority=9)
    ingest, _ = _enqueue(store, "ingest", "ingest:1")
    assert _claimed(store, exclude_kinds=("research.hypothesis",)) == ingest
    queued = store.get(research)
    assert queued is not None and queued.status == "queued"
    assert store.has_due(kinds=("research.hypothesis",), now=T0) is True


def test_stats_and_event_filters(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _enqueue(store, "train", "train:1")
    second, _ = _enqueue(store, "train", "train:2")
    store.cancel(second, now=T0)
    stats = store.stats(now=T0)
    assert stats.total == 2
    assert stats.counts["queued"] == 1
    assert stats.counts["cancelled"] == 1
    assert set(stats.counts) == set(JOB_STATUSES)
    assert stats.due == 1
    assert stats.oldest_queued_at is not None
    assert stats.events == 3  # two enqueues and one cancellation
    assert [row.id for row in store.events(second)] == [2, 3]
    assert [row.id for row in store.events(since_id=2)] == [3]
    assert [row.id for row in store.events(limit=1)] == [1]


def test_state_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get_state("missing") is None
    store.set_state("cursor", '{"files":{}}', now=T0)
    store.set_state("cursor", '{"files":{"a":1}}', now=T0)
    assert store.get_state("cursor") == '{"files":{"a":1}}'
    store.clear_state("cursor")
    assert store.get_state("cursor") is None


def test_append_event_validates_the_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    event = store.append_event("paused", detail="owner", now=T0)
    assert event.job_id is None
    with pytest.raises(UnknownJobError):
        store.append_event("started", job_id=42, now=T0)


def test_validation_rejects_bad_input(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="kind"):
        store.enqueue("   ", dedupe_key="x", now=T0)
    with pytest.raises(ValueError, match="status"):
        store.list_jobs(status="unknown")
    with pytest.raises(TypeError, match="priority"):
        store.enqueue("a", dedupe_key="x", priority=True, now=T0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ISO-8601"):
        store.enqueue(  # type: ignore[arg-type]
            "a", dedupe_key="y", scheduled_at="tomorrow", now=T0
        )
