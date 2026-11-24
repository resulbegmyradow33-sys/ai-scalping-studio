"""Daemon: tick order, trigger dedupe, controls, guard, failure isolation, stop."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scalping.orchestrator import (
    DaemonConfig,
    HandlerResult,
    JobContext,
    JobStore,
    OrchestratorDaemon,
    TickResult,
    Trigger,
)
from scalping.orchestrator.store import Job

T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = T0 + timedelta(hours=1)


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "orchestrator.sqlite3")


def _enqueue(
    store: JobStore,
    kind: str,
    key: str,
    *,
    priority: int = 0,
    max_retries: int = 0,
    now: datetime = T0,
) -> int:
    job, _ = store.enqueue(
        kind,
        dedupe_key=key,
        priority=priority,
        max_retries=max_retries,
        now=now,
    )
    return job.id


class _Recorder:
    """A handler that records each call and succeeds."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(self, job: Job, context: JobContext) -> dict[str, object]:
        self.calls.append((job.id, job.kind))
        return {"kind": job.kind}


class _Source:
    """A trigger source that keeps returning the same triggers until acknowledged."""

    def __init__(self, name: str, triggers: list[Trigger]) -> None:
        self.name = name
        self._triggers = triggers
        self.scans = 0
        self.acks = 0

    def scan(self) -> list[Trigger]:
        self.scans += 1
        return list(self._triggers)

    def seen(self, triggers: object) -> None:
        self.acks += 1


class _RefuseGuard:
    def __init__(self, reason: str = "max_rss_mb=6000 exceeded: used_ram_mb=6100.0") -> None:
        self.reason = reason
        self.checked: list[str] = []

    def can_start(self, kind: str) -> tuple[bool, str]:
        self.checked.append(kind)
        return False, self.reason


def _daemon(
    store: JobStore,
    handlers: dict[str, object] | None = None,
    *,
    triggers: tuple[object, ...] = (),
    guard: object | None = None,
    **config_kwargs: object,
) -> OrchestratorDaemon:
    # Allow-all by default: these tests are about scheduling, and a measured
    # guard would make them depend on the host's current load and RAM.
    config_kwargs.setdefault("guard_profile", "none")
    config = DaemonConfig(**config_kwargs)  # type: ignore[arg-type]
    return OrchestratorDaemon(
        store,
        handlers or {},  # type: ignore[arg-type]
        config,
        guard=guard,  # type: ignore[arg-type]
        triggers=triggers,  # type: ignore[arg-type]
        clock=lambda: 0.0,
    )


# --------------------------------------------------------------------- config


def test_config_defaults_and_bounds() -> None:
    config = DaemonConfig()
    assert config.poll_seconds == 5.0
    assert config.guard_profile == "m2_8gb"
    assert config.scan_triggers is True
    assert config.to_dict()["heavy_kinds"] == []

    with pytest.raises(ValueError, match="poll_seconds"):
        DaemonConfig(poll_seconds=0.5)
    with pytest.raises(ValueError, match="poll_seconds"):
        DaemonConfig(poll_seconds=61)
    with pytest.raises(ValueError, match="guard_profile"):
        DaemonConfig(guard_profile="yolo")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_jobs_per_tick"):
        DaemonConfig(max_jobs_per_tick=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_jobs_per_tick"):
        DaemonConfig(max_jobs_per_tick=-1)
    with pytest.raises(ValueError, match="stop_grace_seconds"):
        DaemonConfig(stop_grace_seconds=-1.0)
    with pytest.raises(ValueError, match="stop_grace_seconds"):
        DaemonConfig(stop_grace_seconds=301)
    with pytest.raises(TypeError, match="heavy_kinds"):
        DaemonConfig(heavy_kinds="dataset.build")  # type: ignore[arg-type]


def test_measured_profile_is_the_default_guard(tmp_path: Path) -> None:
    store = _store(tmp_path)
    daemon = _daemon(store, guard_profile="m2_8gb")
    assert type(daemon.guard).__name__ == "MeasuredResourceGuard"
    status = daemon.status().to_dict()
    assert status["guard"]["profile"] == "m2_8gb"
    assert status["guard"]["type"] == "MeasuredResourceGuard"
    assert status["guard"]["unmeasured"] is False
    assert status["guard"]["snapshot"] is None
    json.dumps(status)


def test_none_profile_uses_allow_all_guard(tmp_path: Path) -> None:
    store = _store(tmp_path)
    daemon = _daemon(store, guard_profile="none")
    assert type(daemon.guard).__name__ == "AllowAllGuard"
    status = daemon.status().to_dict()
    assert status["guard"]["profile"] == "none"
    assert status["guard"]["type"] == "AllowAllGuard"
    assert status["guard"]["unmeasured"] is True


# ----------------------------------------------------------------- tick order


def test_tick_scans_then_drains_and_reports(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handler = _Recorder()
    source = _Source("files", [Trigger(kind="note", dedupe_key="note:1", payload={"n": 1})])
    daemon = _daemon(store, {"note": handler}, triggers=(source,))

    result = daemon.tick(now=T0)
    assert result.error is None
    assert result.scan is not None
    assert [job.kind for job in result.scan.queued] == ["note"]
    assert [outcome.status for outcome in result.outcomes] == ["succeeded"]
    assert handler.calls == [(1, "note")]
    assert source.scans == 1
    assert source.acks == 1

    events = [event.event for event in store.events()]
    assert events.index("trigger_scan") < events.index("started")

    status = daemon.status().to_dict()
    assert status["ticks"] == 1
    assert status["terminal_jobs"] == 1
    assert status["last_tick_at"] == T0.isoformat(timespec="microseconds")
    assert status["last_error"] is None
    assert status["queue"]["counts"]["succeeded"] == 1
    json.dumps(status)


def test_trigger_dedupe_by_source_key_across_ticks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handler = _Recorder()
    trigger = Trigger(kind="note", dedupe_key="note:same", payload={"n": 1})
    source = _Source("files", [trigger])
    daemon = _daemon(store, {"note": handler}, triggers=(source,))

    first = daemon.tick(now=T0)
    second = daemon.tick(now=T0)

    assert first.scan is not None and len(first.scan.queued) == 1
    assert second.scan is not None and len(second.scan.duplicates) == 1
    assert second.outcomes == ()
    assert handler.calls == [(1, "note")]
    assert store.stats().total == 1
    assert source.scans == 2


# ---------------------------------------------------------------- owner controls


def test_pause_scans_and_enqueues_but_claims_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handler = _Recorder()
    source = _Source("files", [Trigger(kind="note", dedupe_key="note:1")])
    daemon = _daemon(store, {"note": handler}, triggers=(source,))
    daemon.orchestrator.pause("owner is reviewing the ledger")

    result = daemon.tick(now=T0)

    assert result.scan is not None and len(result.scan.queued) == 1
    assert result.outcomes == ()
    job = store.get(1)
    assert job is not None and job.status == "queued"
    assert handler.calls == []
    events = [event.event for event in store.events()]
    assert "trigger_scan" in events
    assert "claim_refused" in events
    assert daemon.status().paused is True


def test_safe_mode_scans_and_enqueues_but_claims_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handler = _Recorder()
    source = _Source("files", [Trigger(kind="note", dedupe_key="note:1")])
    daemon = _daemon(store, {"note": handler}, triggers=(source,))
    daemon.orchestrator.safe_mode(True, "hard stop")

    result = daemon.tick(now=T0)

    assert result.scan is not None and len(result.scan.queued) == 1
    assert result.outcomes == ()
    job = store.get(1)
    assert job is not None and job.status == "queued"
    assert daemon.status().safe_mode is True


def test_disabled_research_leaves_research_queued_and_runs_ingestion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handler = _Recorder()
    research_id = _enqueue(store, "research.x", "r1", priority=5)
    ingest_id = _enqueue(store, "ingest.x", "i1", priority=0)
    daemon = _daemon(
        store,
        {"research.x": handler, "ingest.x": handler},
        research_kinds=("research.x",),
    )
    daemon.orchestrator.disable_research(True, "no OOS access today")

    result = daemon.tick(now=T0)

    assert [outcome.status for outcome in result.outcomes] == ["succeeded"]
    assert handler.calls == [(ingest_id, "ingest.x")]
    research = store.get(research_id)
    assert research is not None and research.status == "queued"
    assert research.attempts == 0
    assert daemon.status().disabled_research is True


# ---------------------------------------------------------------------- guard


def test_guard_refusal_defers_with_the_reason_in_the_event_log(tmp_path: Path) -> None:
    store = _store(tmp_path)
    guard = _RefuseGuard()
    job_id = _enqueue(store, "heavy.train", "t1")
    daemon = _daemon(store, {"heavy.train": _Recorder()}, guard=guard, guard_profile="none")

    result = daemon.tick(now=T0)

    assert [outcome.status for outcome in result.outcomes] == ["resource_refused"]
    job = store.get(job_id)
    assert job is not None and job.status == "queued"
    assert job.attempts == 0
    assert job.error == guard.reason
    assert job.scheduled_at > T0.isoformat(timespec="microseconds")
    refused = [event for event in store.events() if event.event == "resource_refused"]
    assert len(refused) == 1
    assert refused[0].detail is not None and guard.reason in refused[0].detail
    assert guard.checked == ["heavy.train"]


def test_repeated_refusals_reach_the_deferral_cap_and_fail(tmp_path: Path) -> None:
    store = _store(tmp_path)
    guard = _RefuseGuard()
    job, _ = store.enqueue(
        "heavy.train",
        dedupe_key="t1",
        max_deferrals=2,
        now=T0,
    )
    job_id = job.id
    daemon = _daemon(store, {"heavy.train": _Recorder()}, guard=guard)

    first = daemon.tick(now=T0)
    second = daemon.tick(now=T0 + timedelta(seconds=60))
    third = daemon.tick(now=T0 + timedelta(seconds=120))

    assert [item.status for item in first.outcomes] == ["resource_refused"]
    assert [item.status for item in second.outcomes] == ["resource_refused"]
    assert [item.status for item in third.outcomes] == ["failed"]
    job = store.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.defer_count == 2
    assert job.max_deferrals == 2
    assert job.error is not None and "deferral cap reached" in job.error
    events = [event.event for event in store.events(job_id)]
    assert events.count("defer_exhausted") == 1
    assert store.stats().deferrals == 2
    assert daemon.status().queue["deferrals"] == 2


# ------------------------------------------------------------ failure isolation


def test_heavy_failure_does_not_block_a_later_paper_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    calls: list[str] = []

    def train(job: Job, context: JobContext) -> HandlerResult:
        calls.append(job.kind)
        raise RuntimeError("training diverged")

    def paper(job: Job, context: JobContext) -> HandlerResult:
        calls.append(job.kind)
        return HandlerResult.success({"traded": 0})

    heavy_id = _enqueue(store, "heavy.train", "train:1", max_retries=0)
    paper_id = _enqueue(store, "paper.advance", "paper:1", max_retries=0, now=T1)
    daemon = _daemon(
        store,
        {"heavy.train": train, "paper.advance": paper},
        heavy_kinds=("heavy.train",),
    )

    first = daemon.tick(now=T0)
    assert [outcome.status for outcome in first.outcomes] == ["failed"]
    assert calls == ["heavy.train"]
    heavy = store.get(heavy_id)
    assert heavy is not None and heavy.status == "failed"
    assert heavy.error is not None and "training diverged" in heavy.error

    second = daemon.tick(now=T1)
    assert [outcome.status for outcome in second.outcomes] == ["succeeded"]
    assert calls == ["heavy.train", "paper.advance"]
    done = store.get(paper_id)
    assert done is not None and done.status == "succeeded"
    assert daemon.status().terminal_jobs == 2
    assert daemon.status().last_error is None


# ---------------------------------------------------------------- start / stop


def test_start_recovers_once_and_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "work", "w1", max_retries=1)
    running = store.claim_next(now=T0)
    assert running is not None and running.id == job_id

    sleeper_signal = threading.Event()

    def sleeper(seconds: float) -> None:
        sleeper_signal.set()

    daemon = OrchestratorDaemon(
        store,
        {"work": _Recorder()},
        DaemonConfig(poll_seconds=1.0, guard_profile="none", stop_grace_seconds=2.0),
        sleeper=sleeper,
    )

    recovered = daemon.start()
    assert [job.id for job in recovered] == [job_id]
    first_ident = daemon.worker_ident
    assert first_ident is not None
    assert daemon.start() == []
    assert daemon.worker_ident == first_ident
    assert sleeper_signal.wait(1.0)
    daemon.stop()
    daemon.stop()

    started = [event for event in store.events() if event.event == "orchestrator_started"]
    assert len(started) == 1
    assert daemon.is_running is False
    assert daemon.worker_ident is None
    job = store.get(job_id)
    assert job is not None and job.status == "succeeded"


def test_stop_during_recovery_is_not_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path)
    job_id = _enqueue(store, "work", "w1", max_retries=1)
    assert store.claim_next(now=T0) is not None

    entered = threading.Event()
    release = threading.Event()
    real_recover = JobStore.recover_interrupted

    def blocking_recover(self: JobStore, *args: object, **kwargs: object) -> object:
        entered.set()
        if not release.wait(5.0):
            raise RuntimeError("test never released recovery")
        return real_recover(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(JobStore, "recover_interrupted", blocking_recover)

    daemon = OrchestratorDaemon(
        store,
        {"work": _Recorder()},
        DaemonConfig(poll_seconds=1.0, guard_profile="none", stop_grace_seconds=2.0),
        sleeper=lambda seconds: None,
    )
    result: list[list[Job]] = []

    def run_start() -> None:
        result.append(daemon.start())

    starter = threading.Thread(target=run_start)
    starter.start()
    assert entered.wait(5.0)

    # The stop must win even though start() is still inside recovery.
    daemon.stop()
    release.set()
    starter.join(5.0)

    assert not starter.is_alive()
    assert result == [[]]
    assert daemon.is_running is False
    assert daemon.worker_ident is None
    job = store.get(job_id)
    assert job is not None and job.status == "queued"


def test_stop_joins_the_loop_thread_and_slices_the_poll(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slices: list[float] = []
    sleeping = threading.Event()
    release = threading.Event()

    def sleeper(seconds: float) -> None:
        slices.append(seconds)
        sleeping.set()
        release.wait(5.0)

    daemon = OrchestratorDaemon(
        store,
        {},
        DaemonConfig(poll_seconds=60.0, guard_profile="none", stop_grace_seconds=3.0),
        sleeper=sleeper,
    )

    daemon.start()
    assert sleeping.wait(1.0)

    stop_entered = threading.Event()

    def run_stop() -> None:
        stop_entered.set()
        daemon.stop()

    stopper = threading.Thread(target=run_stop)
    stopper.start()
    assert stop_entered.wait(1.0)
    stopper.join(0.1)
    assert stopper.is_alive()  # stop() is joining the still-sleeping loop

    release.set()
    stopper.join(3.0)

    assert not stopper.is_alive()
    assert daemon.is_running is False
    assert slices and all(0.0 < value <= 0.25 for value in slices)


def test_stop_between_jobs_ends_the_drain(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(3):
        _enqueue(store, "paper.advance", f"paper:{index}")
    calls: list[int] = []
    first_job = threading.Event()
    holder: dict[str, OrchestratorDaemon] = {}

    def handler(job: Job, context: JobContext) -> HandlerResult:
        calls.append(job.id)
        first_job.set()
        holder["daemon"].stop()  # loop thread: flag only
        return HandlerResult.success()

    daemon = OrchestratorDaemon(
        store,
        {"paper.advance": handler},
        DaemonConfig(poll_seconds=1.0, guard_profile="none", stop_grace_seconds=2.0),
        sleeper=lambda seconds: None,
    )
    holder["daemon"] = daemon

    daemon.start()
    assert first_job.wait(2.0)
    daemon.stop()

    assert calls == [1]
    done = store.get(1)
    assert done is not None and done.status == "succeeded"
    queued = [job for job in store.list_jobs() if job.status == "queued"]
    assert len(queued) == 2


def test_background_loop_rejects_an_in_memory_store(tmp_path: Path) -> None:
    store = JobStore(":memory:")
    daemon = _daemon(store, guard_profile="none")
    with pytest.raises(ValueError, match="file-backed"):
        daemon.start()
    daemon.stop()
    store.close()


def test_status_is_json_safe_after_a_tick(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _enqueue(store, "note", "n1")
    daemon = _daemon(store, {"note": _Recorder()})

    daemon.tick(now=T0)
    daemon.orchestrator.pause("owner pause")
    payload = daemon.status().to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["ticks"] == 1
    assert payload["paused"] is True
    assert payload["queue"]["total"] == 1
    assert payload["queue"]["counts"]["succeeded"] == 1
    assert payload["guard"]["profile"] == "none"
    assert set(payload) >= {
        "running",
        "paused",
        "safe_mode",
        "disabled_research",
        "queue",
        "last_tick_at",
        "last_error",
        "guard",
    }


def test_status_reads_from_another_thread(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _enqueue(store, "note", "n1")
    daemon = _daemon(store, {"note": _Recorder()})
    daemon.tick(now=T0)

    payloads: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def read_status() -> None:
        try:
            payloads.append(daemon.status().to_dict())
        except BaseException as exc:  # noqa: BLE001 - the test reports it below
            errors.append(exc)

    reader = threading.Thread(target=read_status)
    reader.start()
    reader.join(5.0)

    assert errors == []
    assert payloads[0]["ticks"] == 1
    assert payloads[0]["queue"]["total"] == 1  # type: ignore[index]


def test_status_on_an_in_memory_store_stays_on_its_thread() -> None:
    store = JobStore(":memory:")
    daemon = _daemon(store, guard_profile="none")
    errors: list[BaseException] = []

    def read_status() -> None:
        try:
            daemon.status()
        except BaseException as exc:  # noqa: BLE001 - the test asserts the failure
            errors.append(exc)

    reader = threading.Thread(target=read_status)
    reader.start()
    reader.join(5.0)
    store.close()

    assert len(errors) == 1
    assert "same thread" in str(errors[0])


# ------------------------------------------------------------------------ CLI


def test_cli_once_prints_one_json_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scalping import cli

    code = cli.main(
        [
            "orchestrator",
            "run",
            "--once",
            "--profile",
            "none",
            "--db",
            str(tmp_path / "orchestrator.sqlite3"),
            "--root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--archive-root",
            str(tmp_path / "archive"),
        ]
    )
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()

    assert code == 0
    assert captured.err == ""
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["tick"]["outcomes"] == []
    assert payload["tick"]["scan"] is None
    assert payload["status"]["queue"]["total"] == 0
    assert payload["status"]["guard"]["profile"] == "none"
    assert payload["status"]["config"]["guard_profile"] == "none"


def test_cli_requires_a_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scalping import cli

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "orchestrator",
                "run",
                "--db",
                str(tmp_path / "orchestrator.sqlite3"),
            ]
        )
    captured = capsys.readouterr()

    assert excinfo.value.code == 2
    assert captured.out == ""
    assert "one of the arguments" in json.loads(captured.err.strip())["error"]


def test_cli_once_fails_when_the_tick_reports_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scalping import cli
    from scalping.orchestrator import daemon as daemon_module

    def failing_tick(self: OrchestratorDaemon, *, now: object = None) -> TickResult:
        return TickResult(
            started_at="2026-01-01T00:00:00.000000+00:00",
            finished_at="2026-01-01T00:00:00.000000+00:00",
            elapsed_seconds=0.0,
            scan=None,
            outcomes=(),
            error="trigger scan failed: boom",
        )

    monkeypatch.setattr(daemon_module.OrchestratorDaemon, "tick", failing_tick)

    code = cli.main(
        [
            "orchestrator",
            "run",
            "--once",
            "--profile",
            "none",
            "--db",
            str(tmp_path / "orchestrator.sqlite3"),
            "--root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--archive-root",
            str(tmp_path / "archive"),
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert captured.out == ""
    assert "boom" in json.loads(captured.err.strip())["error"]


def test_forever_reports_a_dead_worker_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scalping import cli

    class _Status:
        def to_dict(self) -> dict[str, object]:
            return {"running": False}

    class _DeadDaemon:
        def __init__(self) -> None:
            self.stopped = False

        def start(self) -> None:
            return None

        @property
        def is_running(self) -> bool:
            return False

        def status(self) -> _Status:
            return _Status()

        def stop(self) -> None:
            self.stopped = True

    fake = _DeadDaemon()
    with pytest.raises(cli._ExitCode) as excinfo:
        cli._run_daemon_forever(fake)  # type: ignore[arg-type]

    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.strip().splitlines()]

    assert excinfo.value.code == 1
    assert [line["status"] for line in lines] == ["started", "failed"]
    assert lines[1]["detail"] == "worker exited without a stop request"
    assert fake.stopped is True


def test_cli_watch_dir_enqueues_a_document_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scalping import cli

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.md").write_text(
        "# Gold momentum\n\nA falsifiable note about trend persistence.\n",
        encoding="utf-8",
    )
    common = [
        "orchestrator",
        "run",
        "--once",
        "--profile",
        "none",
        "--db",
        str(tmp_path / "orchestrator.sqlite3"),
        "--root",
        str(tmp_path / "data"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--archive-root",
        str(tmp_path / "archive"),
        "--watch-dir",
        str(docs),
        "--watch-kind",
        "knowledge.ingest",
    ]

    assert cli.main(common) == 0
    first = json.loads(capsys.readouterr().out.strip())
    assert len(first["tick"]["scan"]["queued"]) == 1
    assert first["tick"]["scan"]["queued"][0]["kind"] == "knowledge.ingest"
    assert [item["job"]["kind"] for item in first["tick"]["outcomes"]] == ["knowledge.ingest"]

    assert cli.main(common) == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["tick"]["scan"]["queued"] == []
    assert second["tick"]["outcomes"] == []


def test_cli_rejects_no_scan_with_a_watch_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scalping import cli

    docs = tmp_path / "documents"
    docs.mkdir()
    code = cli.main(
        [
            "orchestrator",
            "run",
            "--once",
            "--profile",
            "none",
            "--no-scan",
            "--watch-dir",
            str(docs),
            "--db",
            str(tmp_path / "orchestrator.sqlite3"),
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert "--no-scan" in json.loads(captured.err.strip())["error"]


def test_cli_once_uses_the_registered_production_handlers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A queued pipeline kind reaches its real handler through the CLI wiring."""
    from scalping import cli

    db = tmp_path / "orchestrator.sqlite3"
    store = JobStore(db)
    store.enqueue(
        "knowledge.ingest",
        dedupe_key="doc:missing",
        payload={"document_path": str(tmp_path / "missing.md")},
    )
    store.close()

    code = cli.main(
        [
            "orchestrator",
            "run",
            "--once",
            "--profile",
            "none",
            "--db",
            str(db),
            "--root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--archive-root",
            str(tmp_path / "archive"),
        ]
    )
    payload = json.loads(capsys.readouterr().out.strip())

    assert code == 0
    assert [outcome["status"] for outcome in payload["tick"]["outcomes"]] == ["failed"]
    failed = payload["tick"]["outcomes"][0]["job"]
    assert failed["kind"] == "knowledge.ingest"
    assert "document does not exist" in failed["error"]
