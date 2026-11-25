"""Cancel, retry and defer semantics: terminal states, ceilings and drains.

These are contract tests, not happy-path smoke tests. Cancellation is checked
against the *row* after the handler returned (the code kept running; its result
must not be banked). Retry and deferral ceilings are proven by exhausting them,
not by trusting a single successful transition. A pause is proven to take effect
between loop iterations, which is also why it cannot interrupt a running handler.

`docs/orchestrator.md` records what this file proves and what it does not.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scalping.orchestrator import (
    HandlerResult,
    Job,
    JobContext,
    JobStateError,
    JobStore,
    Orchestrator,
    PermanentJobError,
    RetryPolicy,
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
    max_retries: int = 0,
    max_deferrals: int = 1440,
    depends_on: Sequence[int] = (),
    payload: Mapping[str, object] | None = None,
    now: datetime = T0,
) -> Job:
    job, _ = store.enqueue(
        kind,
        dedupe_key=key,
        max_retries=max_retries,
        max_deferrals=max_deferrals,
        depends_on=depends_on,
        payload=payload,
        now=now,
    )
    return job


def _event_names(store: JobStore, job_id: int) -> list[str]:
    return [row.event for row in store.events(job_id)]


def _event_detail(store: JobStore, job_id: int, event: str) -> dict[str, object]:
    matches = [row for row in store.events(job_id) if row.event == event]
    assert len(matches) == 1, f"expected one {event} event, got {len(matches)}"
    assert matches[0].detail is not None
    value = json.loads(matches[0].detail)
    assert isinstance(value, dict)
    return value


class _Recorder:
    """Records which jobs actually reached a handler."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, job: Job, context: JobContext) -> dict[str, object]:
        self.calls.append(job.id)
        return {"job": job.id}


class _AlwaysBusy:
    """A guard that refuses every start with the same measured reason."""

    def __init__(self, reason: str = "RAM 95% used") -> None:
        self.reason = reason
        self.calls = 0

    def can_start(self, kind: str) -> tuple[bool, str]:
        self.calls += 1
        return False, self.reason


# ------------------------------------------------------------------ cancellation


def test_cancel_queued_job_is_never_claimed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cancelled = _enqueue(store, "work", "work:cancelled")
    survivor = _enqueue(store, "work", "work:survivor")
    recorder = _Recorder()
    orchestrator = Orchestrator(store, handlers={"work": recorder})

    first = orchestrator.cancel(cancelled.id, "owner stopped it")
    assert first.status == "cancelled"
    assert first.error == "cancelled: owner stopped it"
    assert first.finished_at is not None
    assert first.result is None
    assert _event_names(store, cancelled.id) == ["enqueued", "cancelled"]
    assert _event_detail(store, cancelled.id, "cancelled") == {
        "previous_status": "queued",
        "reason": "owner stopped it",
    }

    # Cancelling twice is a no-op: no second transition and no second audit line.
    again = orchestrator.cancel(cancelled.id, "again")
    assert again.status == "cancelled"
    assert _event_names(store, cancelled.id) == ["enqueued", "cancelled"]

    assert orchestrator.run_once(now=T0).status == "succeeded"
    assert recorder.calls == [survivor.id]
    assert orchestrator.run_once(now=T0).status == "idle"
    cancelled_row = store.get(cancelled.id)
    assert cancelled_row is not None and cancelled_row.status == "cancelled"
    survivor_row = store.get(survivor.id)
    assert survivor_row is not None and survivor_row.status == "succeeded"
    assert store.count_running() == 0


def test_cancel_while_running_discards_the_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = _enqueue(store, "work", "work:running")
    orchestrator: Orchestrator

    def cancel_from_outside(job: Job, context: JobContext) -> Mapping[str, object]:
        # The cancel arrives from the owner's side while the handler is executing.
        orchestrator.cancel(job.id, "owner asked")
        return {"would_have_been_banked": True}

    orchestrator = Orchestrator(store, handlers={"work": cancel_from_outside})
    outcome = orchestrator.run_once(now=T0)

    assert outcome.status == "cancelled"
    assert outcome.job is not None and outcome.job.status == "cancelled"
    assert "result was discarded" in outcome.detail
    current = store.get(job.id)
    assert current is not None
    assert current.result is None
    assert current.attempts == 1
    assert _event_names(store, job.id) == [
        "enqueued",
        "started",
        "cancelled",
        "cancelled_while_running",
    ]
    assert _event_detail(store, job.id, "cancelled")["previous_status"] == "running"
    assert _event_detail(store, job.id, "cancelled_while_running") == {
        "status": "cancelled",
        "result_discarded": True,
    }
    # The discarded result is not retried later either.
    assert orchestrator.run_once(now=T0 + timedelta(days=1)).status == "idle"


def test_cancel_while_running_discards_a_failure_too(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = _enqueue(store, "work", "work:running-failure", max_retries=2)
    orchestrator: Orchestrator

    def cancel_then_raise(job: Job, context: JobContext) -> None:
        orchestrator.cancel(job.id, "owner asked")
        raise RuntimeError("late failure")

    orchestrator = Orchestrator(store, handlers={"work": cancel_then_raise})
    outcome = orchestrator.run_once(now=T0)

    assert outcome.status == "cancelled"
    current = store.get(job.id)
    assert current is not None
    assert current.status == "cancelled"
    assert current.result is None
    events = _event_names(store, job.id)
    assert "retry" not in events
    assert "failed" not in events
    assert events[-1] == "cancelled_while_running"
    # The cancellation survives the handler's exception; nothing is requeued.
    assert orchestrator.run_once(now=T0 + timedelta(days=1)).status == "idle"


def test_cancel_terminal_job_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def succeed(job: Job, context: JobContext) -> dict[str, object]:
        return {"ok": True}

    def fail(job: Job, context: JobContext) -> HandlerResult:
        return HandlerResult.failure("cannot succeed")

    orchestrator = Orchestrator(store, handlers={"work.ok": succeed, "work.bad": fail})
    succeeded = _enqueue(store, "work.ok", "work:done")
    failed = _enqueue(store, "work.bad", "work:failed")
    assert [item.status for item in orchestrator.run_until_idle(now=T0)] == [
        "succeeded",
        "failed",
    ]

    for job_id in (succeeded.id, failed.id):
        before = store.get(job_id)
        assert before is not None
        with pytest.raises(JobStateError, match="cannot be cancelled"):
            orchestrator.cancel(job_id, "too late")
        after = store.get(job_id)
        assert after is not None
        assert after.status == before.status
        assert after.result == before.result
        assert "cancelled" not in _event_names(store, job_id)


def test_cancel_unknown_job_raises_and_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    orchestrator = Orchestrator(store, handlers={})
    before = len(store.events())

    with pytest.raises(UnknownJobError, match="does not exist"):
        orchestrator.cancel(9999, "typo")

    assert len(store.events()) == before
    assert store.list_jobs() == []


# ------------------------------------------------------------------- retry bounds


def test_retry_backoff_is_capped_then_terminal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = _enqueue(store, "train", "train:flaky", max_retries=2)

    def flaky(job: Job, context: JobContext) -> HandlerResult:
        return HandlerResult.failure("flaky failure")

    orchestrator = Orchestrator(
        store,
        handlers={"train": flaky},
        retry=RetryPolicy(base_seconds=5, factor=3, max_seconds=12),
    )

    first = orchestrator.run_once(now=T0)
    assert first.status == "retry"
    assert first.job is not None
    assert first.job.attempts == 1
    assert first.job.scheduled_at == (T0 + timedelta(seconds=5)).isoformat(timespec="microseconds")

    # Nothing runs before the scheduled instant.
    assert orchestrator.run_once(now=T0 + timedelta(seconds=4)).status == "idle"

    second = orchestrator.run_once(now=T0 + timedelta(seconds=5))
    assert second.status == "retry"
    assert second.job is not None
    assert second.job.attempts == 2
    # 5 * 3 = 15, capped at max_seconds=12.
    assert second.job.scheduled_at == (T0 + timedelta(seconds=17)).isoformat(
        timespec="microseconds"
    )

    third = orchestrator.run_once(now=T0 + timedelta(seconds=17))
    assert third.status == "failed"
    current = store.get(job.id)
    assert current is not None
    assert current.status == "failed"
    assert current.attempts == 3
    assert current.error == "flaky failure"

    delays = [
        json.loads(row.detail or "{}")["delay_seconds"]
        for row in store.events(job.id)
        if row.event == "retry"
    ]
    assert delays == [5.0, 12.0]
    # The exhausted job is terminal, not merely overdue.
    assert orchestrator.run_once(now=T0 + timedelta(days=30)).status == "idle"


def test_store_retry_refuses_a_spent_budget(tmp_path: Path) -> None:
    """The store's own cap, independent of the orchestrator's decision."""
    store = _store(tmp_path)
    job = _enqueue(store, "train", "train:budget", max_retries=1)

    claimed = store.claim_next(now=T0)
    assert claimed is not None and claimed.id == job.id and claimed.attempts == 1
    requeued = store.retry(job.id, "transient", delay_seconds=0, now=T0)
    assert requeued.status == "queued" and requeued.attempts == 1

    claimed = store.claim_next(now=T0)
    assert claimed is not None and claimed.attempts == 2
    with pytest.raises(JobStateError, match="used 2 attempt"):
        store.retry(job.id, "again", delay_seconds=1, now=T0)

    still_running = store.get(job.id)
    assert still_running is not None and still_running.status == "running"
    assert _event_names(store, job.id).count("retry") == 1
    assert store.fail(job.id, "again", now=T0).status == "failed"


@pytest.mark.parametrize("mode", ["raise", "result"])
def test_permanent_failure_never_spends_the_retry_budget(tmp_path: Path, mode: str) -> None:
    store = _store(tmp_path)
    job = _enqueue(store, "train", f"train:permanent-{mode}", max_retries=3)

    def permanent_raise(job: Job, context: JobContext) -> None:
        raise PermanentJobError("no usable rows")

    def permanent_result(job: Job, context: JobContext) -> HandlerResult:
        return HandlerResult.failure("no usable rows", permanent=True)

    handlers = {"train": permanent_raise if mode == "raise" else permanent_result}
    orchestrator = Orchestrator(store, handlers=handlers)

    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "failed"
    current = store.get(job.id)
    assert current is not None
    assert current.status == "failed"
    assert current.attempts == 1
    assert current.error is not None and "no usable rows" in current.error
    assert "retry" not in _event_names(store, job.id)
    assert orchestrator.run_once(now=T0 + timedelta(days=1)).status == "idle"
    unchanged = store.get(job.id)
    assert unchanged is not None and unchanged.attempts == 1


# --------------------------------------------------------------- deferral ceiling


def test_defer_ceiling_turns_a_starved_job_terminal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job = _enqueue(store, "train", "train:starved", max_deferrals=2)
    recorder = _Recorder()
    guard = _AlwaysBusy()
    orchestrator = Orchestrator(
        store,
        handlers={"train": recorder},
        guard=guard,
        guard_retry_seconds=0.0,
    )

    for expected in (1, 2):
        outcome = orchestrator.run_once(now=T0)
        assert outcome.status == "resource_refused"
        current = store.get(job.id)
        assert current is not None
        assert current.status == "queued"
        assert current.defer_count == expected
        # A refusal before the handler spends no attempt.
        assert current.attempts == 0
        assert current.error == guard.reason
        assert current.scheduled_at == T0.isoformat(timespec="microseconds")

    failed = orchestrator.run_once(now=T0)
    assert failed.status == "failed"
    current = store.get(job.id)
    assert current is not None
    assert current.status == "failed"
    assert current.defer_count == 2
    assert current.max_deferrals == 2
    assert current.attempts == 1  # the final claim is not restored
    assert current.error is not None
    assert "deferral cap reached after 2 deferrals" in current.error
    assert "max_deferrals=2" in current.error
    assert guard.reason in current.error
    assert failed.detail == current.error
    assert _event_names(store, job.id).count("defer_exhausted") == 1
    assert _event_detail(store, job.id, "defer_exhausted") == {
        "reason": guard.reason,
        "defer_count": 2,
        "max_deferrals": 2,
        "terminal": "failed",
    }
    assert recorder.calls == []
    assert guard.calls == 3
    assert store.stats().deferrals == 2
    # Deferral exhaustion is terminal; the job is not picked up again.
    assert orchestrator.run_once(now=T0 + timedelta(days=1)).status == "idle"


# --------------------------------------------------------------------- drains


def test_run_until_idle_stops_between_jobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _Recorder()
    orchestrator = Orchestrator(store, handlers={"work": recorder})
    ids = [_enqueue(store, "work", f"work:{index}").id for index in range(3)]

    assert orchestrator.run_until_idle(should_stop=lambda: True) == []
    assert recorder.calls == []
    assert len(store.list_jobs(status="queued")) == 3

    checks = {"count": 0}

    def stop_after_two_iterations() -> bool:
        checks["count"] += 1
        return checks["count"] > 2

    outcomes = orchestrator.run_until_idle(max_jobs=10, should_stop=stop_after_two_iterations)

    assert [item.status for item in outcomes] == ["succeeded", "succeeded"]
    assert recorder.calls == ids[:2]
    assert [job.id for job in store.list_jobs(status="queued")] == [ids[2]]
    # The stop was a boundary, not a partially executed job.
    untouched = store.get(ids[2])
    assert untouched is not None and untouched.attempts == 0
    assert "started" not in _event_names(store, ids[2])


def test_pause_during_a_drain_stops_it_and_resume_finishes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _Recorder()
    paused = {"once": False}
    ids = [_enqueue(store, "work", f"work:{index}").id for index in range(3)]
    orchestrator: Orchestrator

    def pause_after_first(job: Job, context: JobContext) -> Mapping[str, object]:
        result = recorder(job, context)
        if not paused["once"]:
            paused["once"] = True
            orchestrator.pause("owner paused mid-drain")
        return result

    orchestrator = Orchestrator(store, handlers={"work": pause_after_first})

    first_drain = orchestrator.run_until_idle(now=T0)
    assert [item.status for item in first_drain] == ["succeeded"]
    assert recorder.calls == ids[:1]
    assert sorted(job.id for job in store.list_jobs(status="queued")) == ids[1:]

    refused = orchestrator.run_once(now=T0)
    assert refused.status == "paused"
    assert refused.job is None
    assert "owner paused mid-drain" in refused.detail
    refused_detail = json.loads(
        [row for row in store.events() if row.event == "claim_refused"][-1].detail or "{}"
    )
    assert refused_detail["reason"] == "paused"

    orchestrator.resume()
    second_drain = orchestrator.run_until_idle(now=T0)
    assert [item.status for item in second_drain] == ["succeeded", "succeeded"]
    assert recorder.calls == ids
    controls = [row.event for row in store.events() if row.event in {"paused", "resumed"}]
    assert controls == ["paused", "resumed"]


# ----------------------------------------------------- dependencies and idempotency


def test_cancelled_dependency_skips_dependents_cascading(tmp_path: Path) -> None:
    store = _store(tmp_path)
    recorder = _Recorder()
    orchestrator = Orchestrator(store, handlers={"work": recorder})
    parent = _enqueue(store, "work", "work:parent")
    child = _enqueue(store, "work", "work:child", depends_on=(parent.id,))
    grandchild = _enqueue(store, "work", "work:grandchild", depends_on=(child.id,))

    orchestrator.cancel(parent.id, "owner stopped the parent")

    outcomes = orchestrator.run_until_idle(now=T0)
    assert [item.status for item in outcomes] == ["skipped", "skipped"]
    assert recorder.calls == []

    child_row = store.get(child.id)
    assert child_row is not None
    assert child_row.status == "skipped"
    assert child_row.error == f"skipped: dependency {parent.id} is cancelled"
    assert _event_names(store, child.id) == ["enqueued", "skipped"]

    grandchild_row = store.get(grandchild.id)
    assert grandchild_row is not None
    assert grandchild_row.status == "skipped"
    assert grandchild_row.error == f"skipped: dependency {child.id} is skipped"

    # A cancelled parent stays cancelled; the skip explains, it does not revive.
    parent_row = store.get(parent.id)
    assert parent_row is not None and parent_row.status == "cancelled"


def test_enqueue_after_failure_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    orchestrator = Orchestrator(
        store,
        handlers={"work": lambda job, context: HandlerResult.failure("boom")},
        retry=RetryPolicy(base_seconds=1, max_seconds=1),
    )
    job = _enqueue(store, "work", "work:once", payload={"version": 1}, max_retries=0)

    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "failed"
    failed = store.get(job.id)
    assert failed is not None and failed.status == "failed"

    repeated, created = store.enqueue(
        "work",
        dedupe_key="work:once",
        payload={"version": 2},
        priority=9,
        max_retries=5,
        now=T0 + timedelta(minutes=1),
    )
    assert created is False
    assert repeated.id == job.id
    assert repeated.status == "failed"
    assert repeated.error == failed.error
    assert repeated.attempts == 1
    assert dict(repeated.payload) == {"version": 1}
    assert repeated.max_retries == 0

    assert _event_names(store, job.id)[-1] == "deduplicated"
    assert _event_detail(store, job.id, "deduplicated")["existing_job"] == job.id
    assert len(store.list_jobs()) == 1
    assert orchestrator.run_once(now=T0 + timedelta(days=1)).status == "idle"
