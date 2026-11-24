"""Long-running scheduler that turns the queue into one supervised process.

P3 slice 2 (audit item 2 and the daemon half of item 3) joins three pieces
that each worked in isolation: the persistent queue and its controls in
:mod:`scalping.orchestrator.core`, the production handler registry in
:mod:`scalping.orchestrator.handlers`, and the measured guard in
:mod:`scalping.orchestrator.resources`. Nothing here re-implements them; the
daemon decides *when* to scan and drain, and delegates *what* may run.

Tick order is fixed and documented in ``docs/orchestrator.md``:

1. crash recovery, exactly once per daemon instance;
2. scan every trigger source and enqueue what it found;
3. drain due work with ``run_until_idle(max_jobs_per_tick)``.

Scanning and enqueueing stay allowed while the queue is paused, in safe mode or
with research disabled. The owner stopped *claims*, and the core already refuses
them with a recorded reason; a hidden scanner would instead lose the event that
caused the work. Every refusal is therefore visible in the audit log.

Failure isolation is the queue's, not a new mechanism: a handler exception is a
failed job, so the next due paper job still runs in the same drain. The daemon
adds only the outer boundary — a crash in scanning or in the store is recorded
in ``last_error`` and the loop continues.

The background loop opens its own :class:`~scalping.orchestrator.store.JobStore`
from the same file, because SQLite connections belong to the thread that made
them. ``tick()`` stays synchronous for tests and ``--once``; ``start()`` and
``stop()`` wrap it in a daemon thread and never leave a partly running process
behind on purpose. This is still one process: there is no operating-system
service, no multi-worker scheduler and no timer persistence beyond a job's own
``scheduled_at``.

Stdlib only. No ML, DSL or paper module is imported here.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

from scalping.orchestrator.core import (
    AllowAllGuard,
    Handler,
    Orchestrator,
    ResourceGuard,
    RunOutcome,
)
from scalping.orchestrator.resources import (
    EXPLORATORY_PROFILE,
    M2_8GB_PROFILE,
    MeasuredResourceGuard,
)
from scalping.orchestrator.store import Job, JobStore
from scalping.orchestrator.triggers import ScanOutcome, TriggerSource, scan_once

__all__ = [
    "GUARD_PROFILES",
    "DaemonConfig",
    "DaemonStatus",
    "OrchestratorDaemon",
    "TickResult",
]

#: Guard profiles accepted by :class:`DaemonConfig`. ``none`` is for tests and
#: throwaway development only; it does not measure anything.
GuardProfile = Literal["m2_8gb", "exploratory", "none"]

GUARD_PROFILES: tuple[str, ...] = ("m2_8gb", "exploratory", "none")

#: Longest single sleep slice. ``stop()`` waits at most this long for the loop to
#: notice the flag, independent of ``poll_seconds``.
_SLEEP_SLICE_SECONDS: float = 0.25


def _require_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_kinds(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an iterable of job kinds, not a string")
    kinds: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{name} entries must be str, got {type(item).__name__}")
        text = item.strip()
        if not text:
            raise ValueError(f"{name} entries must not be empty")
        kinds.add(text)
    return tuple(sorted(kinds))


@dataclass(frozen=True, slots=True)
class DaemonConfig:
    """Everything the scheduler needs that is not a handler or a guard.

    ``poll_seconds`` is bounded to 1–60 so a typo cannot busy-spin the queue or
    hide a stop for an hour. ``heavy_kinds`` is passed straight to the core, which
    already refuses a second heavy claim; ``research_kinds`` makes the persisted
    ``disable research`` control meaningful for this process.

    ``guard_profile`` selects the guard only when no explicit ``guard`` was
    injected. ``m2_8gb`` is the default because the target is the Mac mini
    8 GB; ``none`` installs :class:`~scalping.orchestrator.core.AllowAllGuard`
    and must not be used for an unattended run.
    """

    poll_seconds: float = 5.0
    guard_profile: GuardProfile = "m2_8gb"
    scan_triggers: bool = True
    max_jobs_per_tick: int = 10
    stop_grace_seconds: float = 5.0
    heavy_kinds: tuple[str, ...] = ()
    research_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        poll = _require_number("poll_seconds", self.poll_seconds)
        if not 1.0 <= poll <= 60.0:
            raise ValueError("poll_seconds must be between 1 and 60")
        if self.guard_profile not in GUARD_PROFILES:
            raise ValueError(
                f"guard_profile must be one of {list(GUARD_PROFILES)}, got {self.guard_profile!r}"
            )
        if not isinstance(self.scan_triggers, bool):
            raise TypeError("scan_triggers must be a bool")
        if isinstance(self.max_jobs_per_tick, bool) or not isinstance(self.max_jobs_per_tick, int):
            raise TypeError("max_jobs_per_tick must be an int")
        if self.max_jobs_per_tick < 0:
            raise ValueError("max_jobs_per_tick must be >= 0")
        grace = _require_number("stop_grace_seconds", self.stop_grace_seconds)
        if grace < 0.0:
            raise ValueError("stop_grace_seconds must be >= 0")
        object.__setattr__(self, "poll_seconds", poll)
        object.__setattr__(self, "stop_grace_seconds", grace)
        object.__setattr__(self, "heavy_kinds", _require_kinds(self.heavy_kinds, "heavy_kinds"))
        object.__setattr__(
            self, "research_kinds", _require_kinds(self.research_kinds, "research_kinds")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "poll_seconds": self.poll_seconds,
            "guard_profile": self.guard_profile,
            "scan_triggers": self.scan_triggers,
            "max_jobs_per_tick": self.max_jobs_per_tick,
            "stop_grace_seconds": self.stop_grace_seconds,
            "heavy_kinds": list(self.heavy_kinds),
            "research_kinds": list(self.research_kinds),
        }


@dataclass(frozen=True, slots=True)
class TickResult:
    """One scan-and-drain pass, returned by ``tick()`` for tests and ``--once``."""

    started_at: str
    finished_at: str
    elapsed_seconds: float
    scan: ScanOutcome | None
    outcomes: tuple[RunOutcome, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(float(self.elapsed_seconds), 6),
            "scan": None if self.scan is None else self.scan.to_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    """A JSON-safe snapshot; no behaviour hides behind it."""

    running: bool
    paused: bool
    safe_mode: bool
    disabled_research: bool
    queue: Mapping[str, Any]
    ticks: int
    jobs_processed: int
    last_tick_at: str | None
    last_tick_seconds: float | None
    last_error: str | None
    recovered_jobs: int
    guard: Mapping[str, Any]
    config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "paused": self.paused,
            "safe_mode": self.safe_mode,
            "disabled_research": self.disabled_research,
            "queue": dict(self.queue),
            "ticks": self.ticks,
            "jobs_processed": self.jobs_processed,
            "last_tick_at": self.last_tick_at,
            "last_tick_seconds": self.last_tick_seconds,
            "last_error": self.last_error,
            "recovered_jobs": self.recovered_jobs,
            "guard": dict(self.guard),
            "config": dict(self.config),
        }


class HandlerRegistryView(Protocol):
    """Anything that exposes its handlers mapping; ``Orchestrator`` qualifies."""

    @property
    def handlers(self) -> Mapping[str, Handler]: ...


def _handler_mapping(registry: object) -> dict[str, Handler]:
    if isinstance(registry, Mapping):
        return {str(kind): cast(Handler, handler) for kind, handler in registry.items()}
    view = getattr(registry, "handlers", None)
    if callable(view):
        view = view()
    if isinstance(view, Mapping):
        return {str(kind): cast(Handler, handler) for kind, handler in view.items()}
    raise TypeError("registry must be a mapping of kind to handler or expose a 'handlers' mapping")


def _guard_for(config: DaemonConfig, guard: ResourceGuard | None) -> ResourceGuard:
    if guard is not None:
        if not callable(getattr(guard, "can_start", None)):
            raise TypeError("guard must implement can_start(kind) -> (bool, str)")
        return guard
    if config.guard_profile == "m2_8gb":
        return MeasuredResourceGuard(M2_8GB_PROFILE)
    if config.guard_profile == "exploratory":
        return MeasuredResourceGuard(EXPLORATORY_PROFILE)
    return AllowAllGuard()


def _trigger_sources(triggers: Iterable[TriggerSource]) -> tuple[TriggerSource, ...]:
    sources: list[TriggerSource] = []
    for source in triggers:
        if not callable(getattr(source, "scan", None)) or not callable(
            getattr(source, "seen", None)
        ):
            raise TypeError("a trigger source must implement scan() and seen(triggers)")
        name = getattr(source, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise TypeError("a trigger source must have a non-empty 'name'")
        sources.append(source)
    return tuple(sources)


def _stamp(now: str | datetime | None) -> str:
    """Spell a tick time exactly the way the store spells timestamps."""
    if now is None:
        return datetime.now(UTC).isoformat(timespec="microseconds")
    if isinstance(now, datetime):
        value = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    return str(now)


class OrchestratorDaemon:
    """Scan triggers, drain the queue and report a snapshot.

    ``registry`` is either ``{kind: handler}`` or anything exposing that mapping
    (an :class:`Orchestrator` used as a registration target). The daemon builds
    its own orchestrator so the guard, heavy kinds and research kinds come from
    one :class:`DaemonConfig`; controls still live in the store, so an owner
    pause and a daemon restart agree.

    Call :meth:`tick` directly from the thread that created ``store`` (tests,
    ``--once``), or :meth:`start` for a background thread. The background loop
    opens its own connection to the same database file.
    """

    def __init__(
        self,
        store: JobStore,
        registry: Mapping[str, Handler] | HandlerRegistryView,
        config: DaemonConfig,
        *,
        guard: ResourceGuard | None = None,
        triggers: Sequence[TriggerSource] = (),
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(store, JobStore):
            raise TypeError(f"store must be a JobStore, got {type(store).__name__}")
        if not isinstance(config, DaemonConfig):
            raise TypeError(f"config must be a DaemonConfig, got {type(config).__name__}")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")

        self.store = store
        self.config = config
        self._handlers = _handler_mapping(registry)
        self._guard = _guard_for(config, guard)
        self._triggers = _trigger_sources(triggers)
        self.clock = clock
        self._sleeper = sleeper
        self._orchestrator = Orchestrator(
            store,
            handlers=self._handlers,
            heavy_kinds=config.heavy_kinds,
            research_kinds=config.research_kinds,
            guard=self._guard,
        )
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._recover_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._recovered = False
        self._recovered_jobs = 0
        self._ticks = 0
        self._jobs_processed = 0
        self._last_tick_at: str | None = None
        self._last_tick_seconds: float | None = None
        self._last_error: str | None = None

    # ------------------------------------------------------------------ views

    @property
    def orchestrator(self) -> Orchestrator:
        """The caller-thread orchestrator (controls and ``tick``)."""
        return self._orchestrator

    @property
    def guard(self) -> ResourceGuard:
        return self._guard

    @property
    def triggers(self) -> tuple[TriggerSource, ...]:
        return self._triggers

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------- lifecycle

    def start(self) -> list[Job]:
        """Recover interrupted work once, then run ticks on a daemon thread.

        Idempotent: a second call while the loop is alive does nothing and
        returns an empty list. A stopped daemon may be started again, but crash
        recovery is never repeated for the same instance.
        """
        with self._lifecycle_lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return []
            if self.store.path == ":memory:":
                raise ValueError(
                    "the background loop needs a file-backed store; use tick() with :memory:"
                )
            recovered = self._recover_once()
            self._stop.clear()
            thread = threading.Thread(
                target=self._run_loop,
                name="orchestrator-daemon",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return recovered

    def stop(self) -> None:
        """Ask the loop to stop and wait up to ``stop_grace_seconds``.

        Never raises. A handler already inside ``tick()`` cannot be interrupted
        from here; the loop stops after it returns. ``stop()`` called from the
        loop thread itself (for example a test sleeper) only sets the flag.
        """
        try:
            self._stop.set()
            with self._lifecycle_lock:
                thread = self._thread
            if thread is None or thread is threading.current_thread():
                return
            thread.join(timeout=float(self.config.stop_grace_seconds))
            if not thread.is_alive():
                with self._lifecycle_lock:
                    if self._thread is thread:
                        self._thread = None
        except Exception:  # noqa: BLE001, S110 - shutdown must not raise into the owner
            pass

    # -------------------------------------------------------------------- tick

    def tick(self, *, now: str | datetime | None = None) -> TickResult:
        """Recover once, scan, then drain; safe to call without ``start()``.

        ``now`` exists for deterministic tests and advances both the trigger
        scan and every claim/schedule comparison the store makes. Use either
        ``tick()`` or a running loop for an instance, not both: two live drivers
        would race for claims (the store keeps them unique, but the tick
        counters would be split).
        """
        return self._tick_with(self._orchestrator, now=now)

    def _tick_with(self, orchestrator: Orchestrator, *, now: str | datetime | None) -> TickResult:
        started_stamp = _stamp(now)
        started_clock = float(self.clock())
        error: str | None = None
        scan: ScanOutcome | None = None
        outcomes: list[RunOutcome] = []

        try:
            self._recover_once()
        except Exception as exc:  # noqa: BLE001 - a broken store is reported, not hidden
            error = f"crash recovery failed: {type(exc).__name__}: {exc}"

        if error is None:
            if self.config.scan_triggers and self._triggers:
                try:
                    scan = scan_once(orchestrator, self._triggers, now=now)
                except Exception as exc:  # noqa: BLE001 - scan failure is recorded
                    error = f"trigger scan failed: {type(exc).__name__}: {exc}"
            try:
                outcomes = orchestrator.run_until_idle(self.config.max_jobs_per_tick, now=now)
            except Exception as exc:  # noqa: BLE001 - the loop reports and retries next tick
                drain_error = f"queue drain failed: {type(exc).__name__}: {exc}"
                error = drain_error if error is None else f"{error}; {drain_error}"

        elapsed = max(float(self.clock()) - started_clock, 0.0)
        result = TickResult(
            started_at=started_stamp,
            finished_at=_stamp(now),
            elapsed_seconds=elapsed,
            scan=scan,
            outcomes=tuple(outcomes),
            error=error,
        )
        with self._state_lock:
            self._ticks += 1
            self._jobs_processed += len(outcomes)
            self._last_tick_at = started_stamp
            self._last_tick_seconds = elapsed
            self._last_error = error
        return result

    def status(self) -> DaemonStatus:
        """Read the queue and this daemon's counters; no probing, no claims."""
        stats = self.store.stats()
        with self._state_lock:
            ticks = self._ticks
            jobs_processed = self._jobs_processed
            last_tick_at = self._last_tick_at
            last_tick_seconds = self._last_tick_seconds
            last_error = self._last_error
            recovered = self._recovered_jobs
        return DaemonStatus(
            running=self.is_running,
            paused=self._orchestrator.paused,
            safe_mode=self._orchestrator.safe_mode_enabled,
            disabled_research=self._orchestrator.research_disabled,
            queue=stats.to_dict(),
            ticks=ticks,
            jobs_processed=jobs_processed,
            last_tick_at=last_tick_at,
            last_tick_seconds=None
            if last_tick_seconds is None
            else round(float(last_tick_seconds), 6),
            last_error=last_error,
            recovered_jobs=recovered,
            guard=self._guard_status(),
            config=self.config.to_dict(),
        )

    # --------------------------------------------------------------- internals

    def _recover_once(self) -> list[Job]:
        with self._recover_lock:
            if self._recovered:
                return []
            recovered = self._orchestrator.start()
            self._recovered = True
            self._recovered_jobs = len(recovered)
            return recovered

    def _run_loop(self) -> None:
        worker_store: JobStore | None = None
        try:
            # SQLite connections are thread-bound. The loop owns a second
            # connection to the same file; WAL lets status() keep reading on the
            # caller's connection while jobs are written here.
            worker_store = JobStore(self.store.path)
            worker = Orchestrator(
                worker_store,
                handlers=self._handlers,
                heavy_kinds=self.config.heavy_kinds,
                research_kinds=self.config.research_kinds,
                guard=self._guard,
            )
            while not self._stop.is_set():
                self._tick_with(worker, now=None)
                if self._stop.is_set():
                    break
                self._sleep(self.config.poll_seconds)
        except Exception as exc:  # noqa: BLE001 - the thread must not die silently
            self._record_error(f"daemon loop stopped: {type(exc).__name__}: {exc}")
        finally:
            if worker_store is not None:
                try:
                    worker_store.close()
                except Exception:  # noqa: BLE001, S110 - closing a broken store is best effort
                    pass

    def _sleep(self, seconds: float) -> None:
        remaining = float(seconds)
        while remaining > 0.0 and not self._stop.is_set():
            step = min(remaining, _SLEEP_SLICE_SECONDS)
            self._sleeper(step)
            remaining -= step

    def _record_error(self, message: str) -> None:
        with self._state_lock:
            self._last_error = message

    def _guard_status(self) -> dict[str, Any]:
        snapshot = getattr(self._guard, "last_snapshot", None)
        payload: dict[str, Any] | None = None
        if snapshot is not None and hasattr(snapshot, "to_dict"):
            payload = snapshot.to_dict()
        error = getattr(self._guard, "last_error", None)
        return {
            "profile": self.config.guard_profile,
            "snapshot": payload,
            "last_error": None if error is None else str(error),
        }
