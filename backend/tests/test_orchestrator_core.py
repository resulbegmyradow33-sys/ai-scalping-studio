"""Orchestrator loop: controls, dependencies, retries, recovery and triggers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scalping.orchestrator import (
    AllowAllGuard,
    FileTriggerSource,
    HandlerResult,
    JobContext,
    JobStore,
    Orchestrator,
    PermanentJobError,
    RetryPolicy,
    scan_once,
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
    depends_on: tuple[int, ...] = (),
    now: datetime = T0,
) -> int:
    job, _ = store.enqueue(
        kind,
        dedupe_key=key,
        priority=priority,
        max_retries=max_retries,
        depends_on=depends_on,
        now=now,
    )
    return job.id


class _Recorder:
    """Records calls and returns a small result document."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []

    def __call__(self, job: object, context: JobContext) -> dict[str, object]:
        from scalping.orchestrator.store import Job

        assert isinstance(job, Job)
        self.calls.append((job.id, job.kind, job.attempts))
        return {"kind": job.kind, "attempt": job.attempts}


class _Refuse:
    """A guard that refuses until switched off, with a reason."""

    def __init__(self, reason: str = "RAM 95% used") -> None:
        self.reason = reason
        self.allow = False
        self.checked: list[str] = []

    def can_start(self, kind: str) -> tuple[bool, str]:
        self.checked.append(kind)
        return (self.allow, "" if self.allow else self.reason)


def _orchestrator(
    store: JobStore, handlers: dict[str, object] | None = None, **kwargs: object
) -> Orchestrator:
    return Orchestrator(store, handlers=handlers, **kwargs)  # type: ignore[arg-type]


def test_run_once_succeeds_and_records_the_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "ingest", "ingest:1")
    recorder = _Recorder()
    orchestrator = _orchestrator(store, {"ingest": recorder})

    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "succeeded"
    assert outcome.job is not None and outcome.job.id == job_id
    job = store.get(job_id)
    assert job is not None
    assert job.result == {
        "ok": True,
        "permanent": False,
        "detail": "",
        "data": {"kind": "ingest", "attempt": 1},
    }
    assert [row.event for row in store.events(job_id)] == ["enqueued", "started", "succeeded"]
    assert recorder.calls == [(job_id, "ingest", 1)]


def test_dependencies_run_in_order_through_the_loop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _enqueue(store, "ingest", "ingest:p")
    child = _enqueue(store, "dataset", "dataset:c", priority=9, depends_on=(parent,))
    recorder = _Recorder()
    orchestrator = _orchestrator(store, {"ingest": recorder, "dataset": recorder})

    outcomes = orchestrator.run_until_idle(now=T0)
    assert [item.status for item in outcomes] == ["succeeded", "succeeded"]
    assert [call[0] for call in recorder.calls] == [parent, child]
    events = store.events()
    started = [row for row in events if row.event == "started"]
    assert [row.job_id for row in started] == [parent, child]


def test_dependency_failure_skips_dependents_with_a_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _enqueue(store, "ingest", "ingest:p", max_retries=3)
    child = _enqueue(store, "dataset", "dataset:c", depends_on=(parent,))
    grandchild = _enqueue(store, "train", "train:g", depends_on=(child,))
    orchestrator = _orchestrator(
        store,
        {
            "ingest": lambda job, context: HandlerResult.failure("bad csv", permanent=True),
            "dataset": _Recorder(),
            "train": _Recorder(),
        },
    )

    assert orchestrator.run_once(now=T0).status == "failed"
    assert orchestrator.run_once(now=T0).status == "skipped"
    assert orchestrator.run_once(now=T0).status == "skipped"
    assert orchestrator.run_once(now=T0).status == "idle"

    skipped = store.get(child)
    assert skipped is not None and skipped.status == "skipped"
    assert skipped.error is not None and skipped.error.startswith(f"skipped: dependency {parent}")
    downstream = store.get(grandchild)
    assert downstream is not None and downstream.status == "skipped"
    assert [row.event for row in store.events(child)] == ["enqueued", "skipped"]
    assert store.list_jobs(status="failed")[0].id == parent


def test_retry_backoff_is_bounded_then_fails(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:flaky", max_retries=2)
    orchestrator = _orchestrator(
        store,
        {"train": lambda job, context: HandlerResult.failure("flaky")},
        retry=RetryPolicy(base_seconds=10, factor=2, max_seconds=25),
    )

    first = orchestrator.run_once(now=T0)
    assert first.status == "retry"
    assert first.job is not None
    assert first.job.scheduled_at == (T0 + timedelta(seconds=10)).isoformat(timespec="microseconds")
    assert orchestrator.run_once(now=T0 + timedelta(seconds=5)).status == "idle"

    second = orchestrator.run_once(now=T0 + timedelta(seconds=10))
    assert second.status == "retry"
    assert second.job is not None
    assert second.job.attempts == 2
    assert second.job.scheduled_at == (T0 + timedelta(seconds=30)).isoformat(
        timespec="microseconds"
    )

    third = orchestrator.run_once(now=T0 + timedelta(seconds=30))
    assert third.status == "failed"
    job = store.get(job_id)
    assert job is not None
    assert job.attempts == 3
    assert job.error == "flaky"
    delays = [
        json.loads(row.detail or "{}")["delay_seconds"]
        for row in store.events(job_id)
        if row.event == "retry"
    ]
    assert delays == [10.0, 20.0]


def test_permanent_failure_is_not_retried(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def fail(job: object, context: JobContext) -> None:
        raise PermanentJobError("no usable rows")

    job_id = _enqueue(store, "train", "train:permanent", max_retries=5)
    orchestrator = _orchestrator(store, {"train": fail})
    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "failed"
    job = store.get(job_id)
    assert job is not None
    assert job.attempts == 1
    assert job.error == "no usable rows"
    assert "retry" not in [row.event for row in store.events(job_id)]


def test_exception_is_a_retryable_failure(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def explode(job: object, context: JobContext) -> None:
        raise ValueError("boom")

    _enqueue(store, "train", "train:boom", max_retries=1)
    orchestrator = _orchestrator(store, {"train": explode})
    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "retry"
    assert outcome.detail.startswith("ValueError: boom")


def test_bad_handler_return_is_a_failure_not_a_crash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:bad")
    orchestrator = _orchestrator(store, {"train": lambda job, context: 5})

    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "failed"
    job = store.get(job_id)
    assert job is not None and job.error is not None
    assert job.error.startswith("TypeError: handler must return")


def test_unserialisable_result_fails_the_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:unserialisable")
    orchestrator = _orchestrator(store, {"train": lambda job, context: {"bad": object()}})

    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "failed"
    job = store.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None and job.error.startswith("result is not storable")
    assert job.result is None


def test_cancel_while_running_discards_the_result(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:cancel")

    def cancel_self(job: object, context: JobContext) -> HandlerResult:
        context.cancel("owner asked")
        return HandlerResult.success({"computed": True})

    orchestrator = _orchestrator(store, {"train": cancel_self})
    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "cancelled"
    job = store.get(job_id)
    assert job is not None
    assert job.status == "cancelled"
    assert job.result is None
    events = [row.event for row in store.events(job_id)]
    assert events == ["enqueued", "started", "cancelled", "cancelled_while_running"]


def test_cancel_queued_job_is_never_claimed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:queued")
    orchestrator = _orchestrator(store, {"train": _Recorder()})

    assert orchestrator.cancel(job_id, "owner").status == "cancelled"
    assert orchestrator.run_once(now=T0).status == "idle"
    assert store.count_running() == 0


def test_pause_blocks_claims_and_records_the_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:paused")
    recorder = _Recorder()
    orchestrator = _orchestrator(store, {"train": recorder})

    orchestrator.pause("market close")
    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "paused"
    assert outcome.job is None
    assert orchestrator.paused is True
    assert orchestrator.pause_reason == "market close"
    assert recorder.calls == []
    job = store.get(job_id)
    assert job is not None and job.status == "queued"
    refusals = [row for row in store.events() if row.event == "claim_refused"]
    assert len(refusals) == 1
    assert json.loads(refusals[0].detail or "{}")["reason"] == "paused"

    orchestrator.resume()
    assert orchestrator.run_once(now=T0).status == "succeeded"
    assert [row.event for row in store.events() if row.event in {"paused", "resumed"}] == [
        "paused",
        "resumed",
    ]


def test_safe_mode_claims_nothing_at_all(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _enqueue(store, "ingest", "ingest:1")
    orchestrator = _orchestrator(store, {"ingest": _Recorder()})

    orchestrator.safe_mode(True, "suspicious results")
    assert orchestrator.run_once(now=T0).status == "safe_mode"
    assert orchestrator.status().safe_mode is True
    assert store.stats().counts["queued"] == 1

    orchestrator.safe_mode(False)
    assert orchestrator.run_once(now=T0).status == "succeeded"


def test_disable_research_blocks_only_research_kinds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    research = _enqueue(store, "research.hypothesis", "research:1", priority=9)
    ingest = _enqueue(store, "ingest", "ingest:1")
    recorder = _Recorder()
    orchestrator = _orchestrator(
        store,
        {"research.hypothesis": recorder, "ingest": recorder},
        research_kinds=("research.hypothesis",),
    )

    orchestrator.disable_research(True, "owner")
    assert orchestrator.run_once(now=T0).status == "succeeded"
    skipped = orchestrator.run_once(now=T0)
    assert skipped.status == "research_disabled"
    research_job = store.get(research)
    assert research_job is not None and research_job.status == "queued"
    assert [call[0] for call in recorder.calls] == [ingest]

    orchestrator.disable_research(False)
    assert orchestrator.run_once(now=T0).status == "succeeded"
    assert [call[0] for call in recorder.calls] == [ingest, research]


def test_disabled_research_is_left_queued_not_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _enqueue(store, "ingest", "ingest:p")
    research = _enqueue(store, "research.hypothesis", "research:blocked", depends_on=(parent,))
    orchestrator = _orchestrator(
        store,
        {"ingest": lambda job, context: HandlerResult.failure("bad csv", permanent=True)},
        research_kinds=("research.hypothesis",),
    )
    orchestrator.disable_research(True, "owner")
    assert orchestrator.run_once(now=T0).status == "failed"
    assert orchestrator.run_once(now=T0).status == "research_disabled"

    queued = store.get(research)
    assert queued is not None and queued.status == "queued"
    assert [row.event for row in store.events(research)] == ["enqueued"]

    orchestrator.disable_research(False)
    assert orchestrator.run_once(now=T0).status == "skipped"


def test_controls_survive_a_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _orchestrator(store)
    first.pause("overnight")
    first.safe_mode(True)
    first.disable_research(True)

    second = _orchestrator(JobStore(tmp_path / "orchestrator.sqlite3"))
    assert second.paused is True
    assert second.pause_reason == "overnight"
    assert second.safe_mode_enabled is True
    assert second.research_disabled is True


def test_start_recovers_interrupted_jobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    retryable = _enqueue(store, "train", "train:interrupted", max_retries=1)
    exhausted = _enqueue(store, "train", "train:exhausted")
    claimed = store.claim_next(now=T0)
    assert claimed is not None and claimed.id == retryable
    claimed = store.claim_next(now=T0)
    assert claimed is not None and claimed.id == exhausted

    orchestrator = _orchestrator(store, {"train": _Recorder()})
    recovered = orchestrator.start()
    assert {job.id for job in recovered} == {retryable, exhausted}
    retryable_job = store.get(retryable)
    assert retryable_job is not None and retryable_job.status == "queued"
    exhausted_job = store.get(exhausted)
    assert exhausted_job is not None and exhausted_job.status == "interrupted"
    started_events = [row for row in store.events() if row.event == "orchestrator_started"]
    assert len(started_events) == 1
    detail = json.loads(started_events[0].detail or "{}")
    assert detail == {"recovered": 2, "requeued": 1, "interrupted": 1}


def test_one_heavy_job_at_a_time_through_the_loop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    heavy_a = _enqueue(store, "train", "train:a", priority=9)
    heavy_b = _enqueue(store, "train", "train:b", priority=8)
    light = _enqueue(store, "ingest", "ingest:c")
    orchestrator = _orchestrator(
        store,
        {"train": _Recorder(), "ingest": _Recorder()},
        heavy_kinds=("train",),
    )
    claimed = store.claim_next(now=T0, heavy_kinds=("train",))
    assert claimed is not None and claimed.id == heavy_a

    light_run = orchestrator.run_once(now=T0)
    assert light_run.job is not None and light_run.job.id == light
    busy = orchestrator.run_once(now=T0)
    assert busy.status == "heavy_busy"
    heavy_b_job = store.get(heavy_b)
    assert heavy_b_job is not None and heavy_b_job.status == "queued"

    store.complete(heavy_a, now=T0)
    heavy_run = orchestrator.run_once(now=T0)
    assert heavy_run.job is not None and heavy_run.job.id == heavy_b


def test_resource_guard_refusal_requeues_with_the_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:guard", max_retries=1)
    guard = _Refuse()
    recorder = _Recorder()
    orchestrator = _orchestrator(
        store,
        {"train": recorder},
        guard=guard,
        guard_retry_seconds=60,
    )

    refused = orchestrator.run_once(now=T0)
    assert refused.status == "resource_refused"
    assert refused.detail == "RAM 95% used"
    assert recorder.calls == []
    job = store.get(job_id)
    assert job is not None
    assert job.status == "queued"
    assert job.attempts == 0  # waiting for RAM spends no attempt
    assert job.scheduled_at == (T0 + timedelta(seconds=60)).isoformat(timespec="microseconds")
    assert [row.event for row in store.events(job_id)] == [
        "enqueued",
        "started",
        "resource_refused",
        "deferred",
    ]

    guard.allow = True
    assert orchestrator.run_once(now=T0).status == "idle"  # still backing off
    assert orchestrator.run_once(now=T0 + timedelta(seconds=60)).status == "succeeded"
    assert guard.checked == ["train", "train"]
    assert recorder.calls == [(job_id, "train", 1)]


def test_a_guard_refusal_never_exhausts_the_retry_budget(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "train", "train:guard-no-retry")
    guard = _Refuse("disk full")
    orchestrator = _orchestrator(
        store,
        {"train": _Recorder()},
        guard=guard,
        guard_retry_seconds=5,
    )

    for step in range(5):
        outcome = orchestrator.run_once(now=T0 + timedelta(seconds=5 * step))
        assert outcome.status == "resource_refused"
    job = store.get(job_id)
    assert job is not None
    assert job.status == "queued"
    assert job.attempts == 0

    guard.allow = True
    assert orchestrator.run_once(now=T0 + timedelta(seconds=25)).status == "succeeded"


def test_allow_all_guard_is_the_default() -> None:
    assert AllowAllGuard().can_start("anything") == (True, "")


def test_missing_handler_fails_the_job_permanently(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "not.wired", "not.wired:1", max_retries=3)
    orchestrator = _orchestrator(store, {})
    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "no_handler"
    job = store.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None and "not.wired" in job.error
    assert [row.event for row in store.events(job_id)] == [
        "enqueued",
        "started",
        "no_handler",
        "failed",
    ]


def test_event_log_covers_every_transition(tmp_path: Path) -> None:
    store = _store(tmp_path)
    attempts: list[int] = []

    def flaky(job: object, context: JobContext) -> HandlerResult:
        from scalping.orchestrator.store import Job

        assert isinstance(job, Job)
        attempts.append(job.attempts)
        if len(attempts) == 1:
            return HandlerResult.failure("first try")
        return HandlerResult.success({"rows": 3})

    job_id = _enqueue(store, "train", "train:eventful", max_retries=1)
    orchestrator = _orchestrator(store, {"train": flaky}, retry=RetryPolicy(base_seconds=1))
    orchestrator.run_once(now=T0)
    orchestrator.pause("check")
    orchestrator.resume()
    orchestrator.run_once(now=T0 + timedelta(seconds=1))

    assert [row.event for row in store.events(job_id)] == [
        "enqueued",
        "started",
        "handler_error",
        "retry",
        "started",
        "succeeded",
    ]
    global_events = [row.event for row in store.events() if row.job_id is None]
    assert global_events == ["paused", "resumed"]
    assert attempts == [1, 2]


def test_run_until_idle_respects_its_cap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _enqueue(store, "train", "train:1")
    _enqueue(store, "train", "train:2")
    orchestrator = _orchestrator(store, {"train": _Recorder()})

    outcomes = orchestrator.run_until_idle(max_jobs=1, now=T0)
    assert [item.status for item in outcomes] == ["succeeded"]
    assert store.stats().counts["queued"] == 1


def test_status_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _enqueue(store, "train", "train:1")
    orchestrator = _orchestrator(store, {"train": _Recorder()}, heavy_kinds=("train",))
    orchestrator.pause("inspect")

    status = orchestrator.status()
    assert status.paused is True
    assert status.handlers == ("train",)
    assert status.heavy_kinds == ("train",)
    assert status.running_heavy == 0
    assert status.stats.counts["queued"] == 1
    payload = status.to_dict()
    assert payload["paused"] is True
    assert payload["jobs"]["counts"]["queued"] == 1


def test_retry_policy_caps_the_backoff() -> None:
    policy = RetryPolicy(base_seconds=10, factor=2, max_seconds=25)
    assert [policy.delay(attempt) for attempt in (1, 2, 3, 10)] == [10.0, 20.0, 25.0, 25.0]
    with pytest.raises(ValueError, match="factor"):
        RetryPolicy(factor=0.5)
    with pytest.raises(ValueError, match="base_seconds"):
        RetryPolicy(base_seconds=0)
    with pytest.raises(ValueError, match="attempt"):
        policy.delay(0)


def test_handler_result_contract() -> None:
    assert HandlerResult.success({"rows": 1}).to_payload() == {
        "ok": True,
        "permanent": False,
        "detail": "",
        "data": {"rows": 1},
    }
    with pytest.raises(ValueError, match="detail"):
        HandlerResult.failure(" ")
    with pytest.raises(ValueError, match="cannot be permanent"):
        HandlerResult(ok=True, permanent=True)


# ----------------------------------------------------------------- triggers


def _inbox(tmp_path: Path) -> Path:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    return inbox


def test_file_trigger_enqueues_a_new_file_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inbox = _inbox(tmp_path)
    (inbox / "bars.csv").write_text("ts,close\n1,2\n", encoding="utf-8")
    source = FileTriggerSource(inbox, kind="ingest.csv", store=store)
    orchestrator = _orchestrator(store, {"ingest.csv": _Recorder()})

    first = scan_once(orchestrator, [source], now=T0)
    assert len(first.queued) == 1
    assert first.sources == (source.name,)
    payload = first.queued[0].payload
    assert payload["relative_path"] == "bars.csv"
    assert payload["size"] == len("ts,close\n1,2\n")
    assert isinstance(payload["sha256"], str) and len(payload["sha256"]) == 64

    second = scan_once(orchestrator, [source], now=T0)
    assert second.queued == ()
    assert second.duplicates == ()
    assert len(store.list_jobs()) == 1
    assert [row.event for row in store.events()].count("trigger_scan") == 2


def test_file_trigger_dedupes_identical_content_under_a_new_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inbox = _inbox(tmp_path)
    (inbox / "a.csv").write_text("same bytes\n", encoding="utf-8")
    source = FileTriggerSource(inbox, kind="ingest.csv", store=store)
    orchestrator = _orchestrator(store, {})

    assert len(scan_once(orchestrator, [source], now=T0).queued) == 1
    (inbox / "b.csv").write_text("same bytes\n", encoding="utf-8")
    copied = scan_once(orchestrator, [source], now=T0)
    assert copied.queued == ()
    assert len(copied.duplicates) == 1
    assert len(store.list_jobs()) == 1


def test_file_trigger_detects_changed_content_regardless_of_mtime(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inbox = _inbox(tmp_path)
    path = inbox / "bars.csv"
    path.write_text("ts,close\n1,2\n", encoding="utf-8")
    source = FileTriggerSource(inbox, kind="ingest.csv", store=store)
    orchestrator = _orchestrator(store, {})
    assert len(scan_once(orchestrator, [source], now=T0).queued) == 1

    stamp = path.stat().st_mtime_ns
    path.write_text("ts,close\n1,3\n", encoding="utf-8")  # same size, new content
    os.utime(path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
    changed = scan_once(orchestrator, [source], now=T0)
    assert len(changed.queued) == 1
    assert changed.queued[0].payload["sha256"] != store.list_jobs()[1].payload["sha256"]
    assert len(store.list_jobs()) == 2


def test_file_trigger_rewrite_with_the_same_bytes_is_a_duplicate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inbox = _inbox(tmp_path)
    path = inbox / "bars.csv"
    path.write_text("ts,close\n1,2\n", encoding="utf-8")
    source = FileTriggerSource(inbox, kind="ingest.csv", store=store)
    orchestrator = _orchestrator(store, {})
    scan_once(orchestrator, [source], now=T0)

    stamp = path.stat().st_mtime_ns
    path.write_text("ts,close\n1,2\n", encoding="utf-8")
    os.utime(path, ns=(stamp + 5_000_000_000, stamp + 5_000_000_000))
    again = scan_once(orchestrator, [source], now=T0)
    assert again.queued == ()
    assert len(again.duplicates) == 1
    assert len(store.list_jobs()) == 1


def test_file_trigger_cursor_survives_a_store_restart(tmp_path: Path) -> None:
    inbox = _inbox(tmp_path)
    (inbox / "bars.csv").write_text("payload\n", encoding="utf-8")
    first_store = _store(tmp_path)
    first = FileTriggerSource(inbox, kind="ingest.csv", store=first_store)
    scan_once(_orchestrator(first_store, {}), [first], now=T0)
    first_store.close()

    second_store = JobStore(tmp_path / "orchestrator.sqlite3")
    second = FileTriggerSource(inbox, kind="ingest.csv", store=second_store)
    outcome = scan_once(_orchestrator(second_store, {}), [second], now=T0)
    assert outcome.queued == ()
    assert outcome.duplicates == ()
    assert len(second_store.list_jobs()) == 1


def test_file_trigger_requires_a_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        FileTriggerSource(tmp_path / "missing", kind="ingest.csv", store=_store(tmp_path))
