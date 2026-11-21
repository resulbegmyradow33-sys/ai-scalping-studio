"""Background PPO training behind the HTTP API.

A request never waits for learning. :meth:`TrainRunner.start` reads the stored
bars, builds the feature matrix, derives the run id and reserves it in a
process-local map; the actual :func:`~scalping.rl.train.train` call then runs on
a daemon worker thread. That split is deliberate: a store below
:data:`~scalping.rl.train.MIN_BARS` must be answered with the trainer's message
*before* a thread exists, so the API returns 409 instead of a run that fails in
the background.

The map keeps the last known status of every run started by this process,
including terminal ones, so ``pause``/``resume``/``stop`` can answer 409 for a
run this process owns but cannot change. A run that is only on disk (started by
another process or by the CLI) is not in the map and is a 404. All map access
happens under one lock; joining threads happens after the lock is released so a
worker finishing cannot deadlock against a shutdown that waits for it.

The worker thread only writes files through the registry's atomic writers, and
``train`` checkpoints after every rollout: a stop therefore leaves
``status="interrupted"`` next to a loadable ``model.zip``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from scalping.api.runner import MIN_BARS as SIM_MIN_BARS
from scalping.api.runner import TIMEFRAME
from scalping.backtest.engine import run_backtest
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.features.build import FeatureFrame, build_features
from scalping.rl.policy import load_policy
from scalping.rl.registry import (
    CONFIG_NAME,
    METRICS_NAME,
    load_run,
    save_manifest,
    write_json,
)
from scalping.rl.train import (
    MIN_BARS,
    TrainConfig,
    data_hash_for,
    env_starts_for,
    run_identity,
    train,
)
from scalping.serialize import metrics_payload

__all__ = [
    "LIVE_STATUSES",
    "RunControlError",
    "RunState",
    "TrainError",
    "TrainRunner",
    "UnknownRunError",
]

logger = logging.getLogger(__name__)

#: Statuses of a run whose worker thread is still alive. ``stopping`` is the
#: window between a stop request and the terminal ``interrupted`` manifest.
LIVE_STATUSES: Final[frozenset[str]] = frozenset({"running", "paused", "stopping"})

#: Seconds a lifespan shutdown waits for threads after asking them to stop.
SHUTDOWN_TIMEOUT: Final[float] = 10.0

_SAFE_NAME: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


class TrainError(RuntimeError):
    """A training or backtest request cannot be served as asked (HTTP 409)."""


class RunControlError(RuntimeError):
    """A control action does not apply to the run's current status (HTTP 409)."""


class UnknownRunError(LookupError):
    """No run with this id exists in this process or under ``runs_root`` (404)."""


@dataclass
class RunState:
    """Process-local view of one run: its status, control flags and thread."""

    run_id: str
    status: str
    created_at: str
    run_dir: Path
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, or ``{}`` when it is absent, unreadable or not one."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _steps(value: Any) -> int:
    """Manifest step count as an int; a hand-edited value counts as zero."""
    if isinstance(value, bool):
        return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class TrainRunner:
    """Own the training threads and the ``runs/`` registry for one app."""

    def __init__(
        self,
        store: ParquetStore,
        settings: Settings,
        runs_root: str | Path,
    ) -> None:
        self.store = store
        self.settings = settings
        self.runs_root = Path(runs_root)
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}

    # -- start -----------------------------------------------------------

    def start(
        self,
        *,
        total_steps: int,
        n_envs: int,
        seed: int,
        symbol: str,
    ) -> str:
        """Prepare a run, reserve its id and start the worker thread.

        Raises :class:`TrainError` (409) for a store below the bar floor or a
        run id that is already training. Data problems from the feature layer
        stay ``ValueError`` and map to 422 in the route.
        """
        frame = self.store.read(symbol, TIMEFRAME)
        if frame.height < MIN_BARS:
            raise TrainError(f"not enough bars: need {MIN_BARS}")
        features = build_features(frame, self.settings.costs)
        if features.X.shape[0] < MIN_BARS:
            raise TrainError(f"not enough bars: need {MIN_BARS}")

        # Validate the env count here, with the trainer's own check: doing it in
        # the worker would return 202 and then leave a half-created run behind.
        try:
            env_starts_for(features, n_envs)
        except ValueError as exc:
            raise TrainError(str(exc)) from exc

        config = TrainConfig(
            run_name=symbol,
            total_steps=total_steps,
            n_envs=n_envs,
            seed=seed,
        )
        run_id, identity = run_identity(config, features, self.settings)
        run_dir = self.runs_root / run_id
        created_at = _now()

        with self._lock:
            existing = self._runs.get(run_id)
            if existing is not None and existing.status in LIVE_STATUSES:
                raise TrainError(f"run {run_id} is already training")
            state = RunState(
                run_id=run_id,
                status="running",
                created_at=created_at,
                run_dir=run_dir,
            )
            self._runs[run_id] = state

        try:
            # Publish the run before the thread builds PPO so a listing right
            # after the 202 already contains it. ``train`` replaces this
            # manifest with the full payload on its first checkpoint.
            save_manifest(
                run_dir,
                {
                    "run_id": run_id,
                    "run_name": config.run_name,
                    "created_at": created_at,
                    "seed": seed,
                    "data_hash": identity["data_hash"],
                    "status": "running",
                    "steps": 0,
                    "reward_mean": None,
                    "updated_at": created_at,
                },
            )
            thread = threading.Thread(
                target=self._train,
                args=(state, config, features),
                name=f"train-{run_id}",
                daemon=True,
            )
            with self._lock:
                state.thread = thread
            thread.start()
        except BaseException:
            with self._lock:
                self._runs.pop(run_id, None)
            raise
        return run_id

    def _train(self, state: RunState, config: TrainConfig, features: FeatureFrame) -> None:
        """Run ``train`` in the worker thread and publish its final status."""
        status = "failed"
        try:
            result = train(
                config,
                features,
                self.settings,
                out_dir=self.runs_root,
                stop_event=state.stop_event,
                pause_event=state.pause_event,
            )
            manifest = load_run(result.model_path.parent) or {}
            status = str(manifest.get("status") or "completed")
        except Exception as exc:
            logger.exception("training run %s failed", state.run_id)
            state.error = str(exc)
            self._mark_failed(state, str(exc))
        finally:
            with self._lock:
                state.status = status
                state.thread = None

    def _mark_failed(self, state: RunState, message: str) -> None:
        """Leave an honest manifest behind when the worker raised."""
        try:
            payload = load_run(state.run_dir) or {
                "run_id": state.run_id,
                "created_at": state.created_at,
            }
            payload.update(
                {
                    "status": "failed",
                    "error": message,
                    "updated_at": _now(),
                }
            )
            save_manifest(state.run_dir, payload)
        except (OSError, ValueError, TypeError):
            logger.exception("cannot record the failure of run %s", state.run_id)

    # -- control ---------------------------------------------------------

    def control(self, run_id: str, action: str) -> bool:
        """Apply ``pause``/``resume``/``stop`` to a run this process started."""
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                raise UnknownRunError(run_id)
            if action == "pause":
                if state.status != "running":
                    raise RunControlError(
                        f"cannot pause run {run_id} while it is {state.status}"
                    )
                state.status = "paused"
                state.pause_event.set()
            elif action == "resume":
                if state.status != "paused":
                    raise RunControlError(
                        f"cannot resume run {run_id} while it is {state.status}"
                    )
                state.status = "running"
                state.pause_event.clear()
            elif action == "stop":
                if state.status not in LIVE_STATUSES:
                    raise RunControlError(
                        f"cannot stop run {run_id} while it is {state.status}"
                    )
                # Make the stop observable before the worker's terminal
                # manifest lands; pause/resume no longer fit ``stopping``.
                state.status = "stopping"
                state.stop_event.set()
            else:  # pragma: no cover - the request schema only admits the three
                raise RunControlError(f"unknown control action {action!r}")
        return True

    # -- registry --------------------------------------------------------

    def list_runs(self) -> list[dict[str, Any]]:
        """Manifests under ``runs_root``, newest first, with live statuses overlaid."""
        with self._lock:
            live = {run_id: state.status for run_id, state in self._runs.items()}
        root = self.runs_root
        if not root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            try:
                manifest = load_run(entry)
            except (OSError, ValueError, TypeError) as exc:
                # One unreadable manifest must not hide every other run.
                logger.warning("skipping run directory %s: %s", entry, exc)
                continue
            if manifest is None:
                continue
            run_id = str(manifest.get("run_id") or entry.name)
            created_at = manifest.get("created_at")
            items.append(
                {
                    "run_id": run_id,
                    "status": live.get(run_id, str(manifest.get("status") or "unknown")),
                    "created_at": str(created_at) if created_at is not None else None,
                    "steps": _steps(manifest.get("steps")),
                    "reward_mean": manifest.get("reward_mean"),
                }
            )
        items.sort(key=lambda item: (item["created_at"] or "", item["run_id"]), reverse=True)
        return items

    def detail(self, run_id: str) -> dict[str, Any]:
        """Config, latest metrics and status for one run."""
        run_dir = self._run_dir(run_id)
        manifest = load_run(run_dir)
        with self._lock:
            state = self._runs.get(run_id)
        if manifest is None and state is None:
            raise UnknownRunError(run_id)
        status = state.status if state is not None else str((manifest or {}).get("status") or "unknown")
        return {
            "config": _read_json(run_dir / CONFIG_NAME),
            "metrics": _read_json(run_dir / METRICS_NAME),
            "status": status,
        }

    def _run_dir(self, run_id: str) -> Path:
        """Resolve one run id, refusing anything that is not a single component."""
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise UnknownRunError(run_id)
        return self.runs_root / run_id

    # -- backtest --------------------------------------------------------

    def backtest(self, run_id: str, symbol: str) -> dict[str, Any]:
        """Replay one stored policy over the stored bars and write a report.

        Only a finished run with a policy can be replayed. A live status
        (``running``/``paused``/``stopping``) is a 409 because the worker is
        still rewriting the run; ``failed`` is a 409 because there is no policy
        to load. The stored bars must hash to the run's own ``data_hash`` — a
        policy replayed on different prices would produce a report that looks
        valid and means nothing.
        """
        run_dir = self._run_dir(run_id)
        manifest = load_run(run_dir)
        if manifest is None:
            raise UnknownRunError(run_id)
        with self._lock:
            state = self._runs.get(run_id)
        status = state.status if state is not None else str(manifest.get("status") or "unknown")
        if status in LIVE_STATUSES:
            raise TrainError(f"run {run_id} is not finished (status {status})")
        if status == "failed":
            raise TrainError(f"run {run_id} is failed; there is no policy to replay")

        frame = self.store.read(symbol, TIMEFRAME)
        if frame.height < SIM_MIN_BARS:
            raise TrainError(f"not enough bars: need {SIM_MIN_BARS}")
        features = build_features(frame, self.settings.costs)
        try:
            policy = load_policy(run_dir)
        except (OSError, TypeError, ValueError) as exc:
            raise TrainError(f"run {run_id} cannot be loaded: {exc}") from exc

        width = int(features.X.shape[1])
        declared = policy.config.get("n_features")
        if declared is not None and int(declared) != width:
            raise TrainError(
                f"run {run_id} was trained on {declared} features, "
                f"the stored data produces {width}"
            )
        data_hash = data_hash_for(features)
        run_data_hash = policy.config.get("data_hash")
        if not isinstance(run_data_hash, str) or run_data_hash != data_hash:
            raise TrainError(
                f"run {run_id} was trained on different data: "
                f"run data_hash {run_data_hash}, stored data_hash {data_hash}"
            )

        try:
            result = run_backtest(policy, features, self.settings, policy_name=run_id)
        except (TypeError, ValueError) as exc:
            # A policy trained with the bandit context cannot act on the plain
            # simulator observation; that is a mismatch, not a server fault.
            raise TrainError(f"run {run_id} cannot be replayed: {exc}") from exc
        metrics = metrics_payload(result.metrics)
        report_path = run_dir / f"backtest-{_safe_name(symbol)}.json"
        write_json(
            report_path,
            {
                "run_id": run_id,
                "backtest_run_id": result.run_id,
                "symbol": symbol,
                "config_hash": result.config_hash,
                "data_hash": data_hash,
                "run_data_hash": run_data_hash,
                "created_at": _now(),
                "metrics": metrics,
            },
        )
        return {"report_path": str(report_path), "metrics": metrics}

    # -- lifecycle -------------------------------------------------------

    def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """Ask live runs to stop, then join their threads within ``timeout``."""
        with self._lock:
            states = list(self._runs.values())
        for state in states:
            if state.status in LIVE_STATUSES:
                state.stop_event.set()
        self.join_all(timeout)

    def join_all(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """Join every worker thread, waiting at most ``timeout`` seconds total."""
        with self._lock:
            threads = [
                state.thread
                for state in self._runs.values()
                # A thread whose ``start`` never ran has no ident and would make
                # ``join`` raise; the start path removes that state anyway.
                if state.thread is not None and state.thread.ident is not None
            ]
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)

    def active_threads(self) -> list[threading.Thread]:
        """Worker threads that are still alive (empty once shutdown finished)."""
        with self._lock:
            threads = [state.thread for state in self._runs.values()]
        return [thread for thread in threads if thread is not None and thread.is_alive()]


def _safe_name(symbol: str) -> str:
    """A file-name-safe symbol: no separators, never empty or dot-only."""
    cleaned = _SAFE_NAME.sub("_", symbol).strip("._")
    return cleaned or "symbol"
