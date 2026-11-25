"""End-to-end orchestrator integration across the real modules (Phase 18).

Every other orchestrator test pins one seam: the store alone, the loop alone,
the handlers alone, the daemon alone. This file runs the seams together on
temporary roots and a tiny synthetic bar store, so a wiring regression between
them fails here instead of only in production:

* one :class:`Orchestrator` drains a real dependency chain
  (``knowledge.ingest`` → ``dataset.build``) and the audit log names every
  transition;
* a heavy job that keeps failing does not block a light job, the retry budget is
  the only thing that decides how many attempts happen, and its dependent is
  ``skipped`` once the parent is terminal;
* ``PaperOutcomeTriggerSource`` + ``scan_once`` enqueue through the store,
  dedupe after a store restart and after a simulated crash between enqueue and
  acknowledgement, and the registered handler records a proposal without
  running discovery;
* ``OrchestratorDaemon.tick`` recovers a running row once, scans, drains,
  refuses claims under pause/safe mode with a recorded reason, and defers a job
  through a *measured* guard without spending an attempt;
* a store reopen keeps queued jobs, requeues an interrupted running job, and
  never executes an already-succeeded job twice;
* the flow imports no holdout or real-money module and the evaluator only ever
  sees the validation slice.

``paper.advance`` is handled by
:class:`~scalping.paper.continuous.ContinuousPaperRunner`; this file proves that
a queue-driven paper job records NO TRADE when no champion exists. The
handler-level suite (`test_orchestrator_handlers.py`) proves the continuous
cadence, restart, gap/order refusal and legacy-session resumption; this file does
not repeat those cases, and its no-champion assertions only claim what runs.

One test seam is worth naming: ``Orchestrator.start()`` has no ``now``
parameter, so crash recovery stamps a requeued row with the wall clock. The
daemon test below therefore drives the daemon with the wall clock instead of an
injected ``now``; a simulated clock would leave the recovered job scheduled in
the real future. Production calls ``start()`` once with ``now=None``.

Nothing here reads ``backend/data``, the repository's discovery archive, paper
ledger or knowledge index. Every path is under ``tmp_path``; the bar fixture is
one day of seeded synthetic M1 and the discovery run is bounded to one trial.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import scalping.discovery.run as discovery_run
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.discovery.evaluate import _holdout_id  # the production guard, asserted as-is
from scalping.discovery.run import _iso
from scalping.knowledge.index import KnowledgeIndex
from scalping.orchestrator import (
    DaemonConfig,
    Job,
    JobStore,
    Orchestrator,
    OrchestratorDaemon,
    RetryPolicy,
    scan_once,
)
from scalping.orchestrator.handlers import (
    HEAVY_HANDLER_KINDS,
    KIND_DATASET_BUILD,
    KIND_DISCOVERY_RUN,
    KIND_KNOWLEDGE_INGEST,
    KIND_PAPER_ADVANCE,
    KIND_PAPER_ANALYSE,
    HandlerContext,
    register_default_handlers,
)
from scalping.orchestrator.paper_triggers import (
    KIND_PAPER_OUTCOMES,
    PAPER_OUTCOME_FIELDS,
    PaperOutcomeTriggerSource,
    register_paper_outcome_handler,
)
from scalping.orchestrator.resources import (
    MeasuredResourceGuard,
    ResourceLimits,
    ResourceSnapshot,
)
from scalping.paper.ledger import PaperLedger

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
END = START + timedelta(days=1)
T0 = datetime(2026, 1, 5, 12, tzinfo=UTC)


# --------------------------------------------------------------------------- fixtures


def write_bars(data_root: Path) -> None:
    """One day of seeded synthetic M1; enough for a real dataset and search."""
    frame = SyntheticProvider(seed=3).fetch(SYMBOL, START, END)
    ParquetStore(data_root).write(SYMBOL, TIMEFRAME, frame)


def make_context(
    tmp_path: Path,
    *,
    discovery_runner: Any = None,
    with_bars: bool = False,
) -> HandlerContext:
    """A context whose four roots all live under ``tmp_path``."""
    data_root = tmp_path / "data"
    if with_bars:
        write_bars(data_root)
    return HandlerContext(
        data_root=data_root,
        runs_root=tmp_path / "runs",
        archive_root=tmp_path / "archive",
        use_feature_cache=False,
        discovery_runner=discovery_runner,
    )


def seed_paper_session(ledger: PaperLedger, session_id: str = "session-1") -> None:
    """One session with a single closed trade, attributable to strategy+dataset."""
    ledger.open_session(
        session_id,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        strategy="dsl",
        model_version="hash-integration",
        started_at=T0,
        config={"origin": "test", "data": {"dataset_id": "dataset:integration"}},
    )
    ledger.record_trade(
        session_id,
        side="long",
        entry_ts=T0,
        entry_price=2000.0,
        exit_ts=T0 + timedelta(minutes=5),
        exit_price=2001.0,
        lot=0.10,
        pnl=12.5,
        r_multiple=1.25,
        exit_reason="take_profit",
    )


@pytest.fixture
def paper_ledger(tmp_path: Path) -> Iterator[Path]:
    """A file-backed ledger with one matured session; closed before the test."""
    path = tmp_path / "paper.sqlite"
    with PaperLedger(path) as ledger:
        seed_paper_session(ledger)
    yield path


# --------------------------------------------------------------------------- helpers


def job_of(store: JobStore, job_id: int) -> Job:
    row = store.get(job_id)
    assert row is not None, f"job {job_id} vanished"
    return row


def data_of(store: JobStore, job: Job) -> dict[str, Any]:
    """The stored result document of a succeeded job, after a read."""
    row = job_of(store, job.id)
    assert row.result is not None and row.result["ok"] is True
    payload = row.result["data"]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    return payload


def event_names(store: JobStore, job_id: int | None = None) -> list[str]:
    return [event.event for event in store.events(job_id)]


def details(store: JobStore, event: str) -> list[dict[str, Any]]:
    return [
        json.loads(item.detail)
        for item in store.events()
        if item.event == event and item.detail is not None
    ]


def recorded(store: JobStore, job: Job) -> list[str]:
    """Handler-level events are job events too; returning them keeps the log honest."""
    return [
        item.event
        for item in store.events(job.id)
        if item.event not in {"enqueued", "started", "succeeded", "failed"}
    ]


# ------------------------------------------------------- 1. real dependency chain


def test_dependency_chain_runs_ingest_then_dataset_in_order(tmp_path: Path) -> None:
    """A single orchestrator drains ingest → dataset and the log proves the order."""
    context = make_context(tmp_path, with_bars=True)
    store = JobStore(tmp_path / "jobs.sqlite")
    orchestrator = Orchestrator(store, heavy_kinds=HEAVY_HANDLER_KINDS)
    register_default_handlers(orchestrator, context)

    document = tmp_path / "research" / "note.md"
    document.parent.mkdir()
    document.write_text(
        "# Session study\n\nWhen the price closes above the session high, the trend continues.\n",
        encoding="utf-8",
    )

    ingest, created = store.enqueue(
        KIND_KNOWLEDGE_INGEST,
        dedupe_key="ingest:session-study",
        payload={
            "document_path": str(document),
            "market_scope": SYMBOL,
            "timeframe_scope": TIMEFRAME,
        },
        now=T0,
    )
    assert created is True
    dataset, created = store.enqueue(
        KIND_DATASET_BUILD,
        dedupe_key="dataset:xauusd-m1",
        payload={"symbol": SYMBOL, "timeframe": TIMEFRAME, "horizon": 1, "purge_bars": 1},
        depends_on=(ingest.id,),
        now=T0,
    )
    assert created is True

    # The child is queued but not claimable: its dependency has not succeeded.
    assert store.blocked_jobs() == []
    first = orchestrator.run_once(now=T0)
    assert first.status == "succeeded"
    assert first.job is not None and first.job.id == ingest.id
    assert job_of(store, dataset.id).status == "queued", "a child must not leapfrog its parent"

    outcomes = orchestrator.run_until_idle(now=T0)
    assert [outcome.job.kind for outcome in outcomes if outcome.job is not None] == [
        KIND_DATASET_BUILD
    ]
    assert [outcome.status for outcome in outcomes] == ["succeeded"]

    ingest_row = job_of(store, ingest.id)
    dataset_row = job_of(store, dataset.id)
    assert ingest_row.status == "succeeded" and ingest_row.attempts == 1
    assert dataset_row.status == "succeeded" and dataset_row.attempts == 1
    assert dataset_row.depends_on == (ingest.id,)

    ingested = data_of(store, ingest)
    assert ingested["status"] == "indexed"
    assert len(ingested["doc_id"]) == 64
    assert ingested["sections"] >= 1
    assert ingested["drafts"] == 1
    assert Path(ingested["index"]) == context.knowledge_index_path
    with KnowledgeIndex(context.knowledge_index_path) as index:
        assert [row.doc_id for row in index.list_documents()] == [ingested["doc_id"]]

    built = data_of(store, dataset)
    manifest_path = Path(built["manifest_path"])
    assert manifest_path.is_file()
    assert manifest_path.parent == context.datasets_root
    assert built["dataset_id"].startswith(f"{SYMBOL}:{TIMEFRAME}:h1:")
    assert built["splits"]["validation"]["rows"] > 0
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["checksum"] == built["checksum"]
    assert written["dataset_id"] == built["dataset_id"]

    # Every transition is in the audit log, in order, per job.
    assert event_names(store, ingest.id) == [
        "enqueued",
        "started",
        "knowledge_ingested",
        "succeeded",
    ]
    assert event_names(store, dataset.id) == [
        "enqueued",
        "started",
        "dataset_built",
        "succeeded",
    ]
    assert recorded(store, ingest) == ["knowledge_ingested"]
    assert recorded(store, dataset) == ["dataset_built"]


# ------------------------------------- 2. heavy failure isolation and dependent skip


def test_failing_heavy_job_does_not_block_light_work_and_skips_dependent(
    tmp_path: Path,
) -> None:
    """A heavy retry is not a global stall; the budget and the skip are explicit."""

    def always_fails(config: object, *, root: object, should_stop: object = None) -> object:
        raise RuntimeError("simulated discovery crash")

    context = make_context(tmp_path, discovery_runner=always_fails)
    store = JobStore(tmp_path / "jobs.sqlite")
    orchestrator = Orchestrator(
        store,
        heavy_kinds=HEAVY_HANDLER_KINDS,
        retry=RetryPolicy(base_seconds=1.0, factor=1.0, max_seconds=1.0),
    )
    register_default_handlers(orchestrator, context)

    discovery, _ = store.enqueue(
        KIND_DISCOVERY_RUN,
        dedupe_key="discovery:crashes",
        payload={
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "max_trials": 1,
            "seed": 0,
            "stress": [2.0],
            "gate": {},
        },
        max_retries=1,
        now=T0,
    )
    paper, _ = store.enqueue(
        KIND_PAPER_ADVANCE,
        dedupe_key="paper:no-champion",
        payload={"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 5},
        now=T0,
    )
    dependent, _ = store.enqueue(
        KIND_PAPER_ANALYSE,
        dedupe_key="analysis:after-discovery",
        payload={"max_sessions": 5},
        depends_on=(discovery.id,),
        now=T0,
    )

    # Tick one: the heavy job spends its first attempt, then the light job runs.
    first = orchestrator.run_until_idle(now=T0)
    assert [(outcome.job.kind if outcome.job else None, outcome.status) for outcome in first] == [
        (KIND_DISCOVERY_RUN, "retry"),
        (KIND_PAPER_ADVANCE, "succeeded"),
    ]
    retried = job_of(store, discovery.id)
    assert retried.status == "queued" and retried.attempts == 1
    assert "RuntimeError: simulated discovery crash" in (retried.error or "")
    no_trade = data_of(store, paper)
    assert no_trade["action"] == "no_trade"
    assert no_trade["reason_class"] == "no_champion"
    assert job_of(store, dependent.id).status == "queued"

    # Tick two, after the retry delay: the second attempt is the last one, and
    # the dependent can now never run, so it is skipped rather than starved.
    second = orchestrator.run_until_idle(now=T0 + timedelta(seconds=5))
    assert [(outcome.job.kind if outcome.job else None, outcome.status) for outcome in second] == [
        (KIND_DISCOVERY_RUN, "failed"),
        (KIND_PAPER_ANALYSE, "skipped"),
    ]
    failed = job_of(store, discovery.id)
    assert failed.status == "failed"
    assert failed.attempts == 2, "max_retries=1 means exactly two attempts"
    assert failed.max_retries == 1
    skipped = job_of(store, dependent.id)
    assert skipped.status == "skipped"
    assert skipped.error is not None
    assert skipped.error.startswith(f"skipped: dependency {discovery.id} is failed")

    failures = details(store, "handler_error")
    assert [item["attempt"] for item in failures] == [1, 2]
    assert [item["max_retries"] for item in failures] == [1, 1]
    assert failures[0]["permanent"] is False
    retries = details(store, "retry")
    assert len(retries) == 1 and retries[0]["delay_seconds"] == 1.0

    # No third attempt exists, no matter how far the clock moves.
    assert orchestrator.run_until_idle(now=T0 + timedelta(hours=1)) == []
    assert job_of(store, discovery.id).attempts == 2


# ----------------------------------------------------------- 3. trigger → proposal


def test_paper_trigger_scan_dedupes_across_restart_and_only_proposes(
    tmp_path: Path, paper_ledger: Path
) -> None:
    """Scan enqueues once, survives a restart and a lost cursor, never runs discovery."""
    jobs_path = tmp_path / "jobs.sqlite"
    store = JobStore(jobs_path)
    orchestrator = Orchestrator(store)
    register_paper_outcome_handler(orchestrator)
    source = PaperOutcomeTriggerSource(paper_ledger, store=store)

    first = scan_once(orchestrator, [source], now=T0)
    assert len(first.queued) == 1 and first.duplicates == ()
    job = first.queued[0]
    assert job.kind == KIND_PAPER_OUTCOMES
    assert set(job.payload) == PAPER_OUTCOME_FIELDS
    assert job.payload["strategy_hash"] == "hash-integration"
    assert job.payload["dataset_id"] == "dataset:integration"
    assert job.payload["trades"] == 1
    store.close()

    # Restart: the cursor lives in the store, so the same batch is not re-queued.
    reopened = JobStore(jobs_path)
    orchestrator = Orchestrator(reopened)
    register_paper_outcome_handler(orchestrator)
    source = PaperOutcomeTriggerSource(paper_ledger, store=reopened)
    second = scan_once(orchestrator, [source], now=T0)
    assert second.queued == () and second.duplicates == ()
    assert len(reopened.list_jobs(kind=KIND_PAPER_OUTCOMES)) == 1

    # Simulate a crash between enqueue and acknowledgement: the cursor is lost,
    # the trigger is scanned again, and the store dedupe key returns the same job.
    reopened.clear_state(source.cursor_key)
    third = scan_once(orchestrator, [source], now=T0)
    assert third.queued == ()
    assert len(third.duplicates) == 1 and third.duplicates[0].id == job.id

    outcomes = orchestrator.run_until_idle(now=T0)
    assert len(outcomes) == 1 and outcomes[0].status == "succeeded"
    proposal = data_of(reopened, job)
    assert proposal["action"] == "queue_discovery"
    assert proposal["queued"] is False and proposal["executed"] is False
    assert proposal["session_id"] == "session-1"
    assert proposal["fingerprint"] == job.payload["fingerprint"]

    # A proposal is not an execution: nothing queued discovery, paper or training.
    assert reopened.list_jobs(kind=KIND_DISCOVERY_RUN) == []
    assert reopened.list_jobs(kind=KIND_PAPER_ADVANCE) == []
    assert not (tmp_path / "runs" / "discovery").exists()
    assert len(reopened.list_jobs(kind=KIND_PAPER_OUTCOMES)) == 1


# ------------------------------------------------------------------ 4. daemon tick


class _Probe:
    """A starved machine: 7.5 GiB of 8 GiB in use, so the RAM limit binds."""

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            timestamp=0.0,
            total_ram_mb=8000.0,
            available_ram_mb=500.0,
            load1=0.1,
            free_disk_mb=50_000.0,
            process_rss_mb=100.0,
        )


def test_daemon_tick_recovers_scans_drains_refuses_and_defers(
    tmp_path: Path, paper_ledger: Path
) -> None:
    """One tick recovers once, scans, drains; controls and the guard are explicit.

    The daemon is driven with the wall clock, not the injected ``now``: recovery
    happens inside ``Orchestrator.start()``, which stamps a requeued row with the
    real time, so a simulated ``now`` would leave the recovered job "in the
    future". That is a test seam, not a production path (the daemon calls
    ``start()`` once with ``now=None``); the assertions below only depend on
    ordering and status, never on a literal timestamp.
    """
    context = make_context(tmp_path)
    store = JobStore(tmp_path / "jobs.sqlite")
    registry = Orchestrator(store)
    register_default_handlers(registry, context)
    register_paper_outcome_handler(registry)

    # A job left `running` by a process that died: enqueue, then claim without
    # executing. Recovery must requeue it because an attempt remains.
    crashed, _ = store.enqueue(
        KIND_PAPER_ADVANCE,
        dedupe_key="paper:crashed",
        payload={"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 3},
        max_retries=1,
    )
    claimed = store.claim_next()
    assert claimed is not None and claimed.id == crashed.id and claimed.status == "running"

    source = PaperOutcomeTriggerSource(paper_ledger, store=store)
    daemon = OrchestratorDaemon(
        store,
        registry,
        DaemonConfig(guard_profile="none", poll_seconds=5.0),
        triggers=(source,),
        clock=lambda: 0.0,
    )

    first = daemon.tick()
    assert first.error is None
    assert first.scan is not None
    assert [job.kind for job in first.scan.queued] == [KIND_PAPER_OUTCOMES]
    assert [
        (outcome.job.kind if outcome.job else None, outcome.status) for outcome in first.outcomes
    ] == [
        (KIND_PAPER_ADVANCE, "succeeded"),
        (KIND_PAPER_OUTCOMES, "succeeded"),
    ]
    assert event_names(store, crashed.id) == [
        "enqueued",
        "started",
        "interrupted",
        "requeued",
        "started",
        "paper_no_trade",
        "succeeded",
    ]
    recovered = data_of(store, crashed)
    assert recovered["action"] == "no_trade" and recovered["reason_class"] == "no_champion"

    # Recovery is once per daemon instance, and the cursor dedupes the second scan.
    second = daemon.tick()
    assert second.error is None
    assert second.scan is not None and second.scan.queued == ()
    assert second.outcomes == ()
    started = [event for event in store.events() if event.event == "orchestrator_started"]
    assert len(started) == 1, "crash recovery must not repeat on every tick"
    status = daemon.status()
    assert status.ticks == 2
    assert status.recovered_jobs == 1
    assert status.queue["counts"]["succeeded"] == 2

    # Owner pause: scanning stays allowed, claiming does not, and the reason is logged.
    daemon.orchestrator.pause("owner is reviewing the ledger")
    paused = daemon.tick()
    assert paused.outcomes == ()
    assert paused.error is None
    assert {
        "reason": "paused",
        "detail": "orchestrator is paused: owner is reviewing the ledger",
    } in (details(store, "claim_refused"))

    # Safe mode is the hard stop: nothing is claimed even after a resume.
    daemon.orchestrator.resume()
    daemon.orchestrator.safe_mode(True, "manual stop")
    stopped = daemon.tick()
    assert stopped.outcomes == ()
    assert [item["reason"] for item in details(store, "claim_refused")][-1] == "safe_mode"

    # A measured guard defers without spending an attempt: the fake probe reports
    # a starved machine and the reason names the limit and the measured value.
    guarded_store = JobStore(tmp_path / "guarded.sqlite")
    guarded_registry = Orchestrator(guarded_store)
    register_default_handlers(guarded_registry, context)
    guard = MeasuredResourceGuard(
        ResourceLimits(max_rss_mb=6000.0),
        probe=_Probe(),
        clock=lambda: 0.0,
    )
    guarded = OrchestratorDaemon(
        guarded_store,
        guarded_registry,
        DaemonConfig(guard_profile="m2_8gb", poll_seconds=5.0),
        guard=guard,
        clock=lambda: 0.0,
    )
    deferred_job, _ = guarded_store.enqueue(
        KIND_PAPER_ADVANCE,
        dedupe_key="paper:starved",
        payload={"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 3},
        max_retries=1,
    )
    before_tick = datetime.now(UTC)
    deferred = guarded.tick()
    assert deferred.error is None
    assert [outcome.status for outcome in deferred.outcomes] == ["resource_refused"]
    row = job_of(guarded_store, deferred_job.id)
    assert row.status == "queued"
    assert row.attempts == 0, "a guard refusal before the handler costs no attempt"
    assert row.defer_count == 1
    assert datetime.fromisoformat(row.scheduled_at) > before_tick + timedelta(seconds=30)
    assert row.error is not None and "max_rss_mb=6000 exceeded" in row.error
    assert row.error.startswith("max_rss_mb=6000 exceeded: used_ram_mb=7500.0")
    refusals = details(guarded_store, "resource_refused")
    assert len(refusals) == 1 and refusals[0]["kind"] == KIND_PAPER_ADVANCE
    deferred_events = details(guarded_store, "deferred")
    assert len(deferred_events) == 1
    assert deferred_events[0]["attempts_spent"] == 0
    assert deferred_events[0]["defer_count"] == 1
    guard_status = guarded.status().to_dict()["guard"]
    assert guard_status["unmeasured"] is False
    assert guard_status["profile"] == "m2_8gb"


# --------------------------------------------------------------------- 5. restart


def test_restart_keeps_queued_jobs_requeues_running_and_never_repeats(tmp_path: Path) -> None:
    """A reopen keeps the queue; recovery requeues; a succeeded job runs once."""
    jobs_path = tmp_path / "jobs.sqlite"
    calls: Counter[str] = Counter()

    def counting(job: Job, _context: object) -> dict[str, object]:
        calls[job.kind] += 1
        return {"kind": job.kind, "seen": calls[job.kind]}

    store = JobStore(jobs_path)
    interrupted, _ = store.enqueue("test.interrupted", dedupe_key="a", max_retries=1, now=T0)
    queued, _ = store.enqueue("test.queued", dedupe_key="b", now=T0)
    claimed = store.claim_next(now=T0)
    assert claimed is not None and claimed.id == interrupted.id
    store.close()

    reopened = JobStore(jobs_path)
    assert job_of(reopened, interrupted.id).status == "running"
    assert job_of(reopened, queued.id).status == "queued"

    recovered = reopened.recover_interrupted(now=T0)
    assert len(recovered) == 1
    assert recovered[0].requeued is True
    assert job_of(reopened, interrupted.id).status == "queued"
    assert job_of(reopened, interrupted.id).attempts == 1, "recovery keeps the spent attempt"
    assert event_names(reopened, interrupted.id)[-2:] == ["interrupted", "requeued"]

    orchestrator = Orchestrator(reopened)
    orchestrator.register("test.interrupted", counting)
    orchestrator.register("test.queued", counting)

    outcomes = orchestrator.run_until_idle(now=T0)
    assert sorted(outcome.job.kind for outcome in outcomes if outcome.job is not None) == [
        "test.interrupted",
        "test.queued",
    ]
    assert dict(calls) == {"test.interrupted": 1, "test.queued": 1}
    assert job_of(reopened, interrupted.id).status == "succeeded"
    assert job_of(reopened, interrupted.id).attempts == 2
    assert job_of(reopened, queued.id).status == "succeeded"
    assert event_names(reopened, queued.id) == ["enqueued", "started", "succeeded"]

    # A drained queue stays drained, and a deduplicated enqueue is not a re-run.
    assert orchestrator.run_until_idle(now=T0) == []
    deduped, created = reopened.enqueue("test.queued", dedupe_key="b", now=T0)
    assert created is False
    assert deduped.id == queued.id and deduped.status == "succeeded"
    assert orchestrator.run_until_idle(now=T0) == []
    assert dict(calls) == {"test.interrupted": 1, "test.queued": 1}
    assert [item.event for item in reopened.events(queued.id)].count("succeeded") == 1


# --------------------------------------- 6. no holdout read and no real-money path

#: Modules that make up the queue → trigger → handler → pipeline flow. The scan
#: is static (source imports) and dynamic (what the flow adds to ``sys.modules``).
FLOW_MODULES: tuple[str, ...] = (
    "scalping.orchestrator.core",
    "scalping.orchestrator.daemon",
    "scalping.orchestrator.handlers",
    "scalping.orchestrator.paper_triggers",
    "scalping.orchestrator.triggers",
    "scalping.paper.autonomous",
    "scalping.paper.continuous",
    "scalping.paper.ledger",
    "scalping.discovery.run",
    "scalping.knowledge.ingest",
    "scalping.datasets.builder",
)

#: Fragments that would mean a holdout reader or an order-routing client.
FORBIDDEN_WORDS: tuple[str, ...] = (
    "holdout",
    "mt5",
    "metatrader",
    "broker",
    "ccxt",
    "ib_insync",
    "live_trad",
    "order_router",
)


def imported_names(source: str) -> set[str]:
    """Every module path an ``import``/``from`` statement in ``source`` names."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def forbidden(modules: set[str]) -> set[str]:
    return {name for name in modules if any(word in name.lower() for word in FORBIDDEN_WORDS)}


def test_flow_never_reads_test_split_or_reaches_a_real_money_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scored frame is the validation slice, and no order/holdout module loads."""
    # Static: no flow module even names a holdout or real-money dependency.
    for module_name in FLOW_MODULES:
        module = importlib.import_module(module_name)
        source_path = Path(module.__file__ or "")
        assert source_path.is_file(), module_name
        bad = forbidden(imported_names(source_path.read_text(encoding="utf-8")))
        assert bad == set(), f"{module_name} imports {sorted(bad)}"

    # Spy on the three decisions that would leak the test split: the dataset
    # geometry, the slice handed to the evaluator, and every frame scored.
    spans: list[tuple[int, int]] = []
    manifests: list[Any] = []
    validation_frames: list[Any] = []
    scored: list[tuple[str | None, str | None, int]] = []

    real_build = discovery_run.build_dataset
    real_slice = discovery_run._validation_slice
    real_backtest = discovery_run.run_backtest

    def build_spy(*args: Any, **kwargs: Any) -> Any:
        bundle = real_build(*args, **kwargs)
        manifests.append(bundle.manifest)
        return bundle

    def slice_spy(features: Any, span: Any) -> Any:
        spans.append((span.start, span.end))
        frame = real_slice(features, span)
        validation_frames.append(frame)
        return frame

    def backtest_spy(policy: Any, features: Any, settings: Any, **kwargs: Any) -> Any:
        scored.append(
            (_iso(features.ts, 0), _iso(features.ts, len(features.ts) - 1), len(features.ts))
        )
        return real_backtest(policy, features, settings, **kwargs)

    monkeypatch.setattr(discovery_run, "build_dataset", build_spy)
    monkeypatch.setattr(discovery_run, "_validation_slice", slice_spy)
    monkeypatch.setattr(discovery_run, "run_backtest", backtest_spy)

    context = make_context(tmp_path, with_bars=True)
    store = JobStore(tmp_path / "jobs.sqlite")
    orchestrator = Orchestrator(store, heavy_kinds=HEAVY_HANDLER_KINDS)
    register_default_handlers(orchestrator, context)

    before = {name for name in sys.modules if forbidden({name})}
    job, _ = store.enqueue(
        KIND_DISCOVERY_RUN,
        dedupe_key="discovery:holdout-check",
        payload={
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "max_trials": 1,
            "population": 2,
            "generations": 1,
            "patience": 1,
            "seed": 0,
            "stress": [2.0],
            "gate": {},
        },
        now=T0,
    )
    outcome = orchestrator.run_once(now=T0)
    assert outcome.status == "succeeded", outcome.detail
    after = {name for name in sys.modules if forbidden({name})}
    assert after == before, f"the flow imported {sorted(after - before)}"

    # Geometry: the test split exists and starts after the validation split.
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.test.rows > 0
    assert manifest.test.start >= manifest.validation.end

    # Slicing: the only frame built is exactly the validation range.
    assert spans == [(manifest.validation.start, manifest.validation.end)]
    assert len(validation_frames) == 1
    validation = validation_frames[0]

    # Scoring: every frame lies inside the validation frame, and one covers it.
    first_ts = _iso(validation.ts, 0)
    last_ts = _iso(validation.ts, len(validation.ts) - 1)
    assert scored, "the evaluator and baselines must have been scored"
    assert all(
        first is not None
        and last is not None
        and first_ts is not None
        and last_ts is not None
        and first >= first_ts
        and last <= last_ts
        for first, last, _rows in scored
    )
    assert (first_ts, last_ts, len(validation.ts)) in scored

    # Provenance: the recorded dataset id names validation and trips no holdout guard.
    result = data_of(store, job)
    assert result["dataset_id"].startswith("validation:")
    assert _holdout_id(result["dataset_id"]) is False
    assert result["promoted"] is False
    assert result["decision"] in {"no_trade", "no_change", "rejected"}
    assert Path(result["archive"]).is_file()

    # The queue never reaches a paper advance in this flow, so no ledger exists.
    assert store.list_jobs(kind=KIND_PAPER_ADVANCE) == []
    assert not context.paper_ledger_path.exists()
    assert KIND_PAPER_ANALYSE not in [job.kind for job in store.list_jobs()]
