"""The orchestrator loop: claim one job, run its handler, record the outcome.

The store knows *what* is queued; this module decides *whether* a job may run
and what its failure means. The seam is deliberately narrow: a handler is any
callable ``(job, context) -> HandlerResult`` and the registry maps a job kind to
one. No ML, DSL or paper-trading code is imported here, so the queue can be
tested with toy handlers and wired to the real pipelines after they merge.

Bounded autonomy, expressed in code:

* one heavy job at a time — the store refuses to claim a heavy kind while
  another heavy job is running, and a light job is claimed in the meantime;
* an explicit :class:`ResourceGuard` hook decides whether *this* kind may start
  now; a refusal is a delayed retry with its reason in the audit log, not a
  silent drop;
* ``pause``, ``safe mode`` and ``disable research`` are persisted in the store,
  so a restart does not silently resume work the owner stopped;
* a dependency that failed makes its dependents ``skipped`` with a reason, so a
  broken parent cannot hold a child in the queue forever;
* retries are bounded by ``max_retries`` and spaced by exponential backoff;
  permanent failures are not retried at all.

The loop is synchronous on purpose. There is no scheduler thread in this slice:
``run_once`` does at most one unit of work and ``run_until_idle`` drains the
queue for a CLI or a test, so its behaviour is reproducible without sleeping.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from scalping.orchestrator.store import (
    BlockedJob,
    Event,
    Job,
    JobStateError,
    JobStats,
    JobStore,
)

__all__ = [
    "CONTROL_EVENTS",
    "AllowAllGuard",
    "Handler",
    "HandlerResult",
    "JobContext",
    "JobStoreReader",
    "Orchestrator",
    "OrchestratorStatus",
    "PermanentJobError",
    "ResourceGuard",
    "RetryPolicy",
    "RunOutcome",
]

#: Persisted control flags. Stored as JSON so the reason a control was switched
#: is visible next to the flag itself.
STATE_PAUSED: str = "orchestrator.paused"
STATE_SAFE_MODE: str = "orchestrator.safe_mode"
STATE_RESEARCH_DISABLED: str = "orchestrator.research_disabled"

#: Event names written for control changes, for the audit page to filter on.
CONTROL_EVENTS: frozenset[str] = frozenset(
    {
        "orchestrator_started",
        "paused",
        "resumed",
        "safe_mode_on",
        "safe_mode_off",
        "research_disabled",
        "research_enabled",
        "claim_refused",
        "resource_refused",
        "cancelled_while_running",
        "no_handler",
        "handler_error",
    }
)

#: Why ``run_once`` did what it did. ``idle`` means "nothing due"; the rest name
#: the refusal or the terminal state of the job that was processed.
OutcomeStatus = Literal[
    "idle",
    "paused",
    "safe_mode",
    "research_disabled",
    "heavy_busy",
    "resource_refused",
    "no_handler",
    "skipped",
    "succeeded",
    "failed",
    "retry",
    "cancelled",
]


class PermanentJobError(Exception):
    """Raised by a handler for a failure a retry cannot fix."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff between attempts.

    ``delay(attempt)`` is ``base_seconds * factor ** (attempt - 1)`` capped at
    ``max_seconds``. Attempt 1 is the first start, so the first retry waits
    ``base_seconds``. The store enforces that a retry only happens while one is
    left; this class only decides how long it waits.
    """

    base_seconds: float = 30.0
    factor: float = 2.0
    max_seconds: float = 900.0

    def __post_init__(self) -> None:
        for name in ("base_seconds", "factor", "max_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not (float(value) > 0.0):
                raise ValueError(f"{name} must be > 0")
        if float(self.factor) < 1.0:
            raise ValueError("factor must be >= 1 so backoff does not shrink")

    def delay(self, attempt: int) -> float:
        """Seconds to wait after ``attempt`` failed; never above the cap."""
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            raise TypeError("attempt must be an int")
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        seconds = float(self.base_seconds) * float(self.factor) ** (attempt - 1)
        return min(seconds, float(self.max_seconds))


@runtime_checkable
class ResourceGuard(Protocol):
    """A veto over starting a job, supplied by whatever measures resources.

    ``can_start`` runs after the job is claimed and before its handler, so a
    refusal is tied to a real job and its reason reaches the audit log. The
    guard is a hook, not a resource monitor: measuring RAM/disk is the caller's
    job, and this slice ships :class:`AllowAllGuard` as the default.
    """

    def can_start(self, kind: str) -> tuple[bool, str]: ...


class AllowAllGuard:
    """The default guard: no measurement, no veto, reason left empty."""

    def can_start(self, kind: str) -> tuple[bool, str]:
        return True, ""


@dataclass(frozen=True, slots=True)
class HandlerResult:
    """What a handler reports back.

    Success results are stored as ``{"ok": true, "data": {...}}``; a failure
    records its ``detail`` in ``jobs.error`` and in the event log. ``permanent``
    skips the remaining retries even when the job has attempts left.
    """

    ok: bool = True
    permanent: bool = False
    detail: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be a bool")
        if not isinstance(self.permanent, bool):
            raise TypeError("permanent must be a bool")
        if not self.ok and not self.detail.strip():
            raise ValueError("a failed HandlerResult needs a detail")
        if self.ok and self.permanent:
            raise ValueError("a successful HandlerResult cannot be permanent")
        if self.ok and not isinstance(self.data, Mapping):
            raise TypeError("data must be a mapping")

    @classmethod
    def success(cls, data: Mapping[str, Any] | None = None, *, detail: str = "") -> HandlerResult:
        return cls(ok=True, detail=detail, data={} if data is None else data)

    @classmethod
    def failure(cls, detail: str, *, permanent: bool = False) -> HandlerResult:
        return cls(ok=False, permanent=permanent, detail=detail)

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "permanent": self.permanent,
            "detail": self.detail,
            "data": dict(self.data),
        }


#: A handler returning ``None`` is a success with an empty result; a plain
#: mapping is a success whose content is the result document.
HandlerReturn = HandlerResult | Mapping[str, Any] | None
Handler = Callable[[Job, "JobContext"], HandlerReturn]


class JobStoreReader:
    """The read-only slice of :class:`JobStore` a handler is allowed to see.

    The orchestrator owns the queue: a handler reports what it found and the
    loop decides whether another job exists. Handing the raw store to a handler
    contradicted that (``context.store.enqueue`` was reachable), so the context
    carries this facade instead. Reads and audit lines pass through; enqueue,
    claim, retry, defer, cancel and every state write are absent and raise
    ``AttributeError`` if a handler tries them.

    This is accidental-misuse protection, not a security boundary. Python cannot
    hide the delegate completely: the store is held in a private slot and a
    determined handler can still reach it through introspection. It stops normal
    code from enqueuing or cancelling by mistake, which is what the context has
    always promised; it is not a sandbox for hostile in-process code.

    ``append_event`` is deliberately exposed: a handler must be able to explain
    a decision in the audit log, and that write cannot create work.
    """

    __slots__ = ("_store",)

    def __init__(self, store: JobStore) -> None:
        if not isinstance(store, JobStore):
            raise TypeError(f"store must be a JobStore, got {type(store).__name__}")
        self._store = store

    @property
    def path(self) -> str | Path:
        """The store's SQLite path, for an audit line that names its source."""
        return self._store.path

    def get(self, job_id: int) -> Job | None:
        return self._store.get(job_id)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Job]:
        return self._store.list_jobs(status=status, kind=kind, limit=limit)

    def due(
        self,
        *,
        kinds: Sequence[str] = (),
        now: str | datetime | None = None,
    ) -> list[Job]:
        return self._store.due(kinds=kinds, now=now)

    def has_due(
        self,
        *,
        kinds: Sequence[str] = (),
        now: str | datetime | None = None,
    ) -> bool:
        return self._store.has_due(kinds=kinds, now=now)

    def count_running(self, *, kinds: Sequence[str] = ()) -> int:
        return self._store.count_running(kinds=kinds)

    def blocked_jobs(self) -> list[BlockedJob]:
        return self._store.blocked_jobs()

    def events(
        self,
        job_id: int | None = None,
        *,
        since_id: int | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        return self._store.events(job_id, since_id=since_id, limit=limit)

    def get_state(self, key: str) -> str | None:
        return self._store.get_state(key)

    def stats(self, *, now: str | datetime | None = None) -> JobStats:
        return self._store.stats(now=now)

    def append_event(
        self,
        event: str,
        *,
        job_id: int | None = None,
        detail: str | None = None,
        now: str | datetime | None = None,
    ) -> Event:
        return self._store.append_event(event, job_id=job_id, detail=detail, now=now)


@dataclass(frozen=True, slots=True)
class JobContext:
    """Everything a handler may look at, and the only side effects it needs.

    ``store`` is a :class:`JobStoreReader`: a handler can inspect jobs, events
    and state, and append an audit line, but it cannot enqueue, claim or cancel
    another job. The orchestrator records the result. ``cancel`` is cooperative
    and scoped to this job: it flips the row, and the orchestrator refuses to
    bank a result for a cancelled job.
    """

    store: JobStoreReader
    job: Job
    safe_mode: bool
    research_enabled: bool
    #: The orchestrator's own cancel hook, limited to this handler's job id.
    #: Kept off the reader so a handler cannot cancel a different job.
    _cancel_job: Callable[[int, str | None], Job] = field(repr=False, compare=False)

    def log(self, event: str, detail: str | None = None) -> None:
        """Append an audit line for this job; use it to explain a real decision."""
        self.store.append_event(event, job_id=self.job.id, detail=detail)

    def cancel(self, reason: str | None = None) -> Job:
        """Cancel the current job from inside its handler."""
        return self._cancel_job(self.job.id, reason)

    def cancelled(self) -> bool:
        """True when the row was cancelled after the handler started."""
        current = self.store.get(self.job.id)
        return current is not None and current.status == "cancelled"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """One iteration of the loop: what ran (or why nothing did)."""

    status: OutcomeStatus
    job: Job | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "job": None if self.job is None else self.job.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OrchestratorStatus:
    """A snapshot for a CLI or a status page; no behaviour hides behind it."""

    paused: bool
    safe_mode: bool
    research_disabled: bool
    handlers: tuple[str, ...]
    heavy_kinds: tuple[str, ...]
    running_heavy: int
    stats: JobStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "paused": self.paused,
            "safe_mode": self.safe_mode,
            "research_disabled": self.research_disabled,
            "handlers": list(self.handlers),
            "heavy_kinds": list(self.heavy_kinds),
            "running_heavy": self.running_heavy,
            "jobs": self.stats.to_dict(),
        }


class Orchestrator:
    """Claims jobs, runs handlers and records every transition.

    Controls are read from the store on every iteration rather than cached, so
    two orchestrators over one database agree and a restart honours the last
    owner decision.
    """

    def __init__(
        self,
        store: JobStore,
        *,
        handlers: Mapping[str, Handler] | None = None,
        heavy_kinds: Iterable[str] = (),
        research_kinds: Iterable[str] = (),
        retry: RetryPolicy | None = None,
        guard: ResourceGuard | None = None,
        guard_retry_seconds: float = 60.0,
    ) -> None:
        self.store = store
        self._handlers: dict[str, Handler] = dict(handlers or {})
        self.heavy_kinds: frozenset[str] = frozenset(heavy_kinds)
        self.research_kinds: frozenset[str] = frozenset(research_kinds)
        self.retry_policy = RetryPolicy() if retry is None else retry
        self.guard: ResourceGuard = AllowAllGuard() if guard is None else guard
        if guard_retry_seconds < 0:
            raise ValueError("guard_retry_seconds must be >= 0")
        self.guard_retry_seconds = float(guard_retry_seconds)

    # ---------------------------------------------------------------- handlers

    def register(self, kind: str, handler: Handler) -> None:
        """Bind a job kind to its handler; the last registration wins."""
        self._handlers[str(kind)] = handler

    @property
    def handlers(self) -> Mapping[str, Handler]:
        return dict(self._handlers)

    def has_handler(self, kind: str) -> bool:
        return kind in self._handlers

    # ---------------------------------------------------------------- controls

    def pause(self, reason: str | None = None, *, actor: str | None = None) -> None:
        """Stop claiming work until :meth:`resume`; the queue keeps growing."""
        self._set_flag(STATE_PAUSED, True, reason, "paused", actor=actor)

    def resume(self, *, actor: str | None = None) -> None:
        self._set_flag(STATE_PAUSED, False, None, "resumed", actor=actor)

    def safe_mode(
        self, enabled: bool, reason: str | None = None, *, actor: str | None = None
    ) -> None:
        """Safe mode claims nothing at all; it is the owner's hard stop."""
        self._set_flag(
            STATE_SAFE_MODE,
            enabled,
            reason,
            "safe_mode_on" if enabled else "safe_mode_off",
            actor=actor,
        )

    def disable_research(
        self, enabled: bool, reason: str | None = None, *, actor: str | None = None
    ) -> None:
        """Block research kinds while still allowing ingestion-style jobs."""
        self._set_flag(
            STATE_RESEARCH_DISABLED,
            enabled,
            reason,
            "research_disabled" if enabled else "research_enabled",
            actor=actor,
        )

    @property
    def paused(self) -> bool:
        return self._flag(STATE_PAUSED)

    @property
    def safe_mode_enabled(self) -> bool:
        return self._flag(STATE_SAFE_MODE)

    @property
    def research_disabled(self) -> bool:
        return self._flag(STATE_RESEARCH_DISABLED)

    def cancel(self, job_id: int, reason: str | None = None) -> Job:
        """Cancel a queued job, or ask a running one to discard its result."""
        return self.store.cancel(job_id, reason)

    @property
    def pause_reason(self) -> str | None:
        return self._flag_reason(STATE_PAUSED)

    def _flag(self, key: str) -> bool:
        raw = self.store.get_state(key)
        if raw is None:
            return False
        return bool(self._read_flag(key, raw)[0])

    def _flag_reason(self, key: str) -> str | None:
        raw = self.store.get_state(key)
        if raw is None:
            return None
        return self._read_flag(key, raw)[1]

    def _read_flag(self, key: str, raw: str) -> tuple[bool, str | None]:
        try:
            parsed = json.loads(raw)
        except ValueError:
            return False, None
        if not isinstance(parsed, dict):
            return False, None
        on = bool(parsed.get("on", False))
        reason = parsed.get("reason")
        return on, None if reason is None else str(reason)

    def _set_flag(
        self,
        key: str,
        on: bool,
        reason: str | None,
        event: str,
        *,
        actor: str | None = None,
    ) -> None:
        stamp = _utc_now()
        payload: dict[str, Any] = {"on": on, "reason": reason, "at": stamp}
        detail: dict[str, Any] = {"on": on, "reason": reason}
        if actor is not None:
            # The audit line names who switched the control. CLI/daemon callers
            # omit the actor and keep the historical event payload unchanged.
            payload["actor"] = actor
            detail["actor"] = actor
        self.store.set_state(key, _json(payload), now=stamp)
        self.store.append_event(
            event,
            detail=_json(detail),
            now=stamp,
        )

    # -------------------------------------------------------------------- loop

    def start(self) -> list[Job]:
        """Recover jobs left running by a previous process, then report.

        Call once at process start. The returned rows are post-recovery: a
        requeued job is ``queued`` again, an exhausted one is ``interrupted``.
        """
        recovered = self.store.recover_interrupted()
        requeued = sum(1 for item in recovered if item.requeued)
        if recovered:
            self.store.append_event(
                "orchestrator_started",
                detail=_json(
                    {
                        "recovered": len(recovered),
                        "requeued": requeued,
                        "interrupted": len(recovered) - requeued,
                    }
                ),
            )
        return [item.job for item in recovered]

    def run_once(self, *, now: str | datetime | None = None) -> RunOutcome:
        """Process at most one unit of work; deterministic for tests.

        Order: control checks, then one dependency-blocked job may be skipped,
        then one due job may be claimed and run. A skip and a claim never happen
        in the same call, so a test can follow the queue transition by
        transition.

        With research disabled, research jobs are left exactly as they are: not
        skipped, not claimed, not retried. The owner asked for a pause on that
        class of work, not for it to be resolved.
        """
        if self.safe_mode_enabled:
            return self._refuse("safe_mode", "safe mode is on; nothing is claimed")
        if self.paused:
            reason = self.pause_reason
            detail = "orchestrator is paused" if not reason else f"orchestrator is paused: {reason}"
            return self._refuse("paused", detail)

        research_disabled = self.research_disabled
        blocked = [
            item
            for item in self.store.blocked_jobs()
            if not (research_disabled and item.job.kind in self.research_kinds)
        ]
        if blocked:
            return self._skip_blocked(blocked[0], now=now)

        excluded = tuple(self.research_kinds) if research_disabled else ()
        job = self.store.claim_next(
            now=now,
            heavy_kinds=tuple(self.heavy_kinds),
            exclude_kinds=excluded,
        )
        if job is None:
            return self._explain_idle(research_disabled, now=now)
        return self._execute(job, research_disabled, now=now)

    def run_until_idle(
        self,
        max_jobs: int = 100,
        *,
        now: str | datetime | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[RunOutcome]:
        """Drain due work, stopping at the first call that finds nothing to do.

        Only iterations that transitioned a job are returned, so the list is the
        batch's history rather than a log of every empty poll.

        ``should_stop`` is checked before every iteration, including the first. A
        daemon passes its shutdown flag here so a requested stop does not start
        up to ``max_jobs`` further jobs.
        """
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int):
            raise TypeError("max_jobs must be an int")
        if max_jobs < 0:
            raise ValueError("max_jobs must be >= 0")
        if should_stop is not None and not callable(should_stop):
            raise TypeError("should_stop must be callable or None")
        outcomes: list[RunOutcome] = []
        while len(outcomes) < max_jobs:
            if should_stop is not None and should_stop():
                break
            outcome = self.run_once(now=now)
            if outcome.job is None:
                break
            outcomes.append(outcome)
        return outcomes

    def status(self) -> OrchestratorStatus:
        return OrchestratorStatus(
            paused=self.paused,
            safe_mode=self.safe_mode_enabled,
            research_disabled=self.research_disabled,
            handlers=tuple(sorted(self._handlers)),
            heavy_kinds=tuple(sorted(self.heavy_kinds)),
            running_heavy=self.store.count_running(kinds=tuple(self.heavy_kinds)),
            stats=self.store.stats(),
        )

    # ---------------------------------------------------------------- internal

    def _refuse(self, status: OutcomeStatus, detail: str) -> RunOutcome:
        self.store.append_event(
            "claim_refused",
            detail=_json({"reason": status, "detail": detail}),
        )
        return RunOutcome(status=status, job=None, detail=detail)

    def _explain_idle(self, research_disabled: bool, *, now: str | datetime | None) -> RunOutcome:
        if research_disabled and self.store.has_due(kinds=tuple(self.research_kinds), now=now):
            return self._refuse(
                "research_disabled",
                "research is disabled; due research jobs stay queued",
            )
        if self.store.count_running(kinds=tuple(self.heavy_kinds)) > 0 and self.store.has_due(
            kinds=tuple(self.heavy_kinds), now=now
        ):
            self.store.append_event(
                "claim_refused",
                detail=_json({"reason": "heavy_busy", "detail": "a heavy job is already running"}),
            )
            return RunOutcome(
                status="heavy_busy",
                job=None,
                detail="a heavy job is already running; light jobs had nothing due",
            )
        return RunOutcome(status="idle", job=None, detail="no due job")

    def _skip_blocked(self, blocked: BlockedJob, *, now: str | datetime | None) -> RunOutcome:
        culprit = ", ".join(f"{job_id} is {status}" for job_id, status in blocked.blocking)
        reason = f"dependency {culprit}"
        try:
            job = self.store.skip(blocked.job.id, reason, now=now)
        except JobStateError:
            current = self.store.get(blocked.job.id)
            if current is not None and current.status == "cancelled":
                return RunOutcome(
                    status="cancelled",
                    job=current,
                    detail="cancelled before the dependency could skip it",
                )
            raise
        return RunOutcome(status="skipped", job=job, detail=reason)

    def _execute(
        self, job: Job, research_disabled: bool, *, now: str | datetime | None
    ) -> RunOutcome:
        allowed, why = self.guard.can_start(job.kind)
        if not allowed:
            reason = why.strip() or "resource guard refused the job"
            self.store.append_event(
                "resource_refused",
                job_id=job.id,
                detail=_json({"kind": job.kind, "reason": reason}),
            )
            try:
                job = self.store.defer(
                    job.id,
                    reason,
                    delay_seconds=self.guard_retry_seconds,
                    now=now,
                )
            except JobStateError:
                current = self.store.get(job.id)
                if current is not None and current.status == "cancelled":
                    return RunOutcome(
                        status="cancelled",
                        job=current,
                        detail="cancelled before the guard could defer it",
                    )
                raise
            if job.status == "failed":
                # The deferral cap turned a repeated refusal terminal.
                return RunOutcome(
                    status="failed",
                    job=job,
                    detail=job.error or reason,
                )
            return RunOutcome(status="resource_refused", job=job, detail=reason)

        handler = self._handlers.get(job.kind)
        if handler is None:
            error = f"no handler registered for kind {job.kind!r}"
            self.store.append_event(
                "no_handler",
                job_id=job.id,
                detail=_json({"kind": job.kind, "error": error}),
            )
            job = self.store.fail(job.id, error, now=now)
            return RunOutcome(status="no_handler", job=job, detail=error)

        context = JobContext(
            store=JobStoreReader(self.store),
            job=job,
            safe_mode=self.safe_mode_enabled,
            research_enabled=not research_disabled,
            _cancel_job=self.store.cancel,
        )
        permanent = False
        try:
            result = _coerce(handler(job, context))
        except PermanentJobError as exc:
            result = HandlerResult.failure(str(exc) or type(exc).__name__, permanent=True)
            permanent = True
        except Exception as exc:  # noqa: BLE001 - a handler bug is a job failure, not a crash
            result = HandlerResult.failure(f"{type(exc).__name__}: {exc}")

        current = self.store.get(job.id)
        if current is None:
            raise JobStateError(f"job {job.id} disappeared while running")
        if current.status != "running":
            self.store.append_event(
                "cancelled_while_running",
                job_id=job.id,
                detail=_json({"status": current.status, "result_discarded": True}),
            )
            return RunOutcome(
                status="cancelled",
                job=current,
                detail=f"job became {current.status} while running; its result was discarded",
            )

        try:
            if result.ok:
                try:
                    finished = self.store.complete(job.id, result.to_payload(), now=now)
                except (TypeError, ValueError) as exc:
                    error = f"result is not storable as JSON: {exc}"
                    failed = self.store.fail(job.id, error, now=now)
                    return RunOutcome(status="failed", job=failed, detail=error)
                return RunOutcome(status="succeeded", job=finished, detail=result.detail)
            return self._fail(job, result, permanent=permanent or result.permanent, now=now)
        except JobStateError as exc:
            current = self.store.get(job.id)
            if current is not None and current.status == "cancelled":
                return RunOutcome(status="cancelled", job=current, detail=str(exc))
            raise

    def _fail(
        self,
        job: Job,
        result: HandlerResult,
        *,
        permanent: bool,
        now: str | datetime | None,
    ) -> RunOutcome:
        detail = result.detail
        self.store.append_event(
            "handler_error",
            job_id=job.id,
            detail=_json(
                {
                    "attempt": job.attempts,
                    "max_retries": job.max_retries,
                    "permanent": permanent,
                    "error": detail,
                }
            ),
        )
        if permanent or job.attempts > job.max_retries:
            failed = self.store.fail(job.id, detail, now=now)
            return RunOutcome(status="failed", job=failed, detail=detail)
        delay = self.retry_policy.delay(job.attempts)
        retried = self.store.retry(job.id, detail, delay_seconds=delay, now=now)
        return RunOutcome(
            status="retry",
            job=retried,
            detail=f"{detail} (retry in {delay:g}s)",
        )


def _coerce(returned: HandlerReturn) -> HandlerResult:
    """Accept the documented handler returns and nothing else."""
    if returned is None:
        return HandlerResult()
    if isinstance(returned, HandlerResult):
        return returned
    if isinstance(returned, Mapping):
        return HandlerResult.success(returned)
    raise TypeError(
        f"handler must return HandlerResult, a mapping or None, got {type(returned).__name__}"
    )


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
