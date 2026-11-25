"""Classical ML training behind the HTTP API.

The shape follows :mod:`scalping.api.train_runner`: a request never waits for a
fit. :meth:`MLRunner.start` validates the request, reads the stored bars and
builds the feature matrix on the request thread, so a store the dataset builder
rejects is answered with a 409/422 *before* a run directory exists. The fit, the
trading evaluation and the artefact writes then happen on one daemon worker.

**One heavy job at a time.** :class:`MLJobSlot` is a process-wide token held by
every ML entry point: a second ``start`` or ``evaluate`` returns 409 instead of
putting two fits in one process. It is released in the worker's ``finally``, so
success, failure and cancellation all free it.

**Cancellation is cooperative.** The slot owns a flag the worker checks before
the fit, immediately after it, and between evaluation stages. The estimator fit
itself is a single blocking call: it cannot be interrupted mid-fit, so a cancel
during that window stays ``cancelling`` until the fit returns. The terminal
``cancelled`` is written by the worker, never by the control request.

There are no dataset caches in this slice. A run holds one feature frame
(bounded by :data:`MAX_BARS`) and leaves it to the garbage collector when the
worker ends. The **feature matrix** is cached on disk under
``<runs_root>/.feature-cache`` (see :mod:`scalping.features.cache`): repeated
training and evaluation of the same store reuses the 59-column matrix instead of
rebuilding it. The cache is on by default, scoped to this app's runs root, and a
constructor flag or ``MLRunner(..., use_cache=False)`` disables it; a cache hit
is bit-identical to a build, and any corrupt entry degrades to a rebuild.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.api.errors import public_error_text
from scalping.api.schemas import is_ml_run_id
from scalping.config import Settings
from scalping.data.registry import TIMEFRAMES
from scalping.data.store import ParquetStore
from scalping.datasets.builder import build_dataset
from scalping.features.build import FeatureFrame
from scalping.features.cache import FEATURE_CACHE_DIRNAME, FeatureCache, cached_build_features
from scalping.features.registry import columns_for
from scalping.ml.evaluate import EvaluationResult, evaluate_model
from scalping.ml.models import MODEL_NAMES
from scalping.ml.train import (
    MLResult,
    TrainSpec,
    load_result,
    save_result,
    train_classifier,
)
from scalping.rl.registry import write_json

__all__ = [
    "LIVE_STATUSES",
    "MAX_BARS",
    "MAX_FEATURES",
    "MLControlError",
    "MLError",
    "MLJobSlot",
    "MLRequest",
    "MLRunner",
    "UnknownMLRunError",
]

logger = logging.getLogger(__name__)

#: Directory under ``runs_root`` that holds every ML run.
RUNS_DIRNAME: Final[str] = "ml"
META_NAME: Final[str] = "meta.json"
REPORT_NAME: Final[str] = "report.json"
MODEL_NAME: Final[str] = "model.joblib"

#: Statuses of a run whose worker thread can still observe a cancel flag.
LIVE_STATUSES: Final[frozenset[str]] = frozenset({"running", "cancelling"})

#: Terminal statuses; a run in one of these no longer holds the slot.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed", "cancelled"})

#: Seconds a lifespan shutdown waits for worker threads.
SHUTDOWN_TIMEOUT: Final[float] = 10.0

#: Bars one API run may read. The real 17-year M1 store is far above this;
#: the API refuses it with a 409 instead of materialising a multi-gigabyte
#: float32 matrix. H1 history fits comfortably.
MAX_BARS: Final[int] = 400_000

#: Upper bound on selected feature columns (the registry currently has 59).
MAX_FEATURES: Final[int] = 128

#: Cost-stress multipliers per evaluation.
MAX_STRESS: Final[int] = 5
MAX_STRESS_MULTIPLIER: Final[float] = 10.0

_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation", "test"})

#: Split name to the dataset builder's integer label.
_SPLITS_INDEX: Final[dict[str, int]] = {"train": 0, "validation": 1, "test": 2}


class MLError(RuntimeError):
    """A request cannot be served as asked (HTTP 409)."""


class MLControlError(RuntimeError):
    """A control action does not apply to the run's current status (HTTP 409)."""


class UnknownMLRunError(LookupError):
    """No ML run with this id exists, or it has no usable artefact (404)."""


class MLJobSlot:
    """At most one heavy ML job in this process.

    The owner is an opaque string (the run id for training, a request token for
    evaluation). ``acquire`` is non-blocking: the caller turns a ``False`` into
    a 409 rather than queueing a second fit in RAM.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: str | None = None

    def acquire(self, owner: str) -> bool:
        """Reserve the slot for ``owner``; ``False`` when another job holds it."""
        with self._lock:
            if self._owner is not None:
                return False
            self._owner = owner
            return True

    def release(self, owner: str) -> None:
        """Free the slot, but only if ``owner`` still holds it."""
        with self._lock:
            if self._owner == owner:
                self._owner = None

    def current(self) -> str | None:
        with self._lock:
            return self._owner


@dataclass(frozen=True, slots=True)
class MLRequest:
    """One training request, already narrowed to a safe spec."""

    symbol: str
    timeframe: str
    spec: TrainSpec
    split: str = "validation"
    threshold: float = 0.55
    exit_threshold: float = 0.55
    window: int = 60
    stress: tuple[float, ...] = ()
    regime: bool = True

    def config(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "split": self.split,
            "threshold": self.threshold,
            "exit_threshold": self.exit_threshold,
            "window": self.window,
            "stress": list(self.stress),
            "regime": self.regime,
            "spec": asdict(self.spec),
        }


@dataclass
class MLRunState:
    """Process-local view of one run: the slot token, flags and thread."""

    run_id: str
    status: str
    created_at: str
    run_dir: Path
    request: MLRequest
    dataset_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
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


def _finite_float(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` for missing/non-finite input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _calibration(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Brier/log loss and the guarded ECE/MCE of one split, flattened."""
    if not isinstance(metrics, dict):
        return None
    calibration = metrics.get("calibration")
    table = calibration if isinstance(calibration, dict) else {}
    return {
        "brier": _finite_float(metrics.get("brier")),
        "log_loss": _finite_float(metrics.get("log_loss")),
        "ece": _finite_float(table.get("ece")),
        "mce": _finite_float(table.get("mce")),
    }


def _evaluation_payload(evaluation: EvaluationResult) -> dict[str, Any]:
    return {
        "split": evaluation.split,
        "threshold": evaluation.threshold,
        "exit_threshold": evaluation.exit_threshold,
        "metrics": evaluation.metrics,
        "hold": evaluation.hold,
        "stress": evaluation.stress,
        "regime": evaluation.regime,
    }


def _validate_thresholds(threshold: float, exit_threshold: float, window: int) -> None:
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must be in [0.5, 1)")
    if not 0.5 <= exit_threshold <= 1.0:
        raise ValueError("exit_threshold must be in [0.5, 1]")
    if window < 1:
        raise ValueError("window must be positive")


def _validate_stress(stress: tuple[float, ...]) -> None:
    if len(stress) > MAX_STRESS:
        raise ValueError(f"at most {MAX_STRESS} stress multipliers")
    for value in stress:
        if not math.isfinite(value) or value <= 0.0 or value > MAX_STRESS_MULTIPLIER:
            raise ValueError(
                f"stress multipliers must be finite and in (0, {MAX_STRESS_MULTIPLIER:g}]"
            )


def _validate_request(request: MLRequest) -> TrainSpec:
    """Cheap field validation; returns the normalised spec.

    Every bad field raises ``ValueError`` so the route answers 422 before the
    slot is touched. Limits that depend on the stored data are ``MLError``.
    """
    if request.timeframe not in TIMEFRAMES:
        raise ValueError(
            f"unknown timeframe {request.timeframe!r}; available: {', '.join(TIMEFRAMES)}"
        )
    spec = request.spec
    model = spec.model.strip().lower()
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {spec.model!r}; available: {', '.join(MODEL_NAMES)}")
    if spec.horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 < spec.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= spec.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if spec.train_fraction + spec.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    if spec.purge_bars < 0 or spec.embargo_bars < 0:
        raise ValueError("purge/embargo must be non-negative")
    columns = columns_for(spec.groups)
    if not columns:
        raise ValueError("no feature columns selected")
    if len(columns) > MAX_FEATURES:
        raise ValueError(f"{len(columns)} feature columns exceeds the API limit of {MAX_FEATURES}")
    if request.split not in _SPLITS:
        raise ValueError(f"unknown split {request.split!r}; choose train/validation/test")
    _validate_thresholds(request.threshold, request.exit_threshold, request.window)
    _validate_stress(request.stress)
    return replace(spec, model=model)


def _new_run_id(root: Path) -> str:
    """An opaque, separator-free token that is not already on disk."""
    for _ in range(8):
        run_id = "ml" + uuid.uuid4().hex
        if not (root / run_id).exists():
            return run_id
    raise MLError("cannot allocate a run id")  # pragma: no cover - 64-bit collision


class MLRunner:
    """Own the ML worker thread, the slot and ``runs/ml`` for one app."""

    def __init__(
        self,
        store: ParquetStore,
        settings: Settings,
        runs_root: str | Path,
        *,
        cache: FeatureCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self.store = store
        self.settings = settings
        self.runs_root = Path(runs_root)
        self.slot = MLJobSlot()
        self.cache: FeatureCache | None = None
        if use_cache:
            if cache is not None:
                self.cache = cache
            else:
                self.cache = FeatureCache(self.runs_root / FEATURE_CACHE_DIRNAME)
        self._lock = threading.Lock()
        self._runs: dict[str, MLRunState] = {}

    def _features(
        self,
        frame: pl.DataFrame,
        symbol: str,
        timeframe: str,
        *,
        groups: tuple[str, ...] | None = None,
    ) -> FeatureFrame:
        """The shared feature-matrix path: cache hit, else build and store."""
        return cached_build_features(
            self.cache,
            symbol=symbol,
            timeframe=timeframe,
            frame=frame,
            settings=self.settings,
            groups=groups,
        )

    # -- start -----------------------------------------------------------

    def start(self, request: MLRequest) -> str:
        """Validate, prepare the dataset, reserve the run and start the worker.

        The slot is acquired before the store read. Preparation failures release
        it and leave no run directory behind; only a reserved run reaches
        ``runs/ml/<run_id>``.
        """
        spec = _validate_request(request)
        request = replace(request, spec=spec)
        root = self.runs_root / RUNS_DIRNAME
        run_id = _new_run_id(root)
        if not self.slot.acquire(run_id):
            raise MLError(
                f"another ML job is running ({self.slot.current()}); wait for it to finish"
            )

        try:
            frame = self.store.read(request.symbol, request.timeframe)
            if frame.is_empty():
                raise MLError(f"no {request.symbol} {request.timeframe} bars in the store")
            if frame.height > MAX_BARS:
                raise MLError(
                    f"refusing {frame.height} bars: the API limit is {MAX_BARS}; "
                    "narrow the stored history"
                )
            features = self._features(
                frame, request.symbol, request.timeframe, groups=request.spec.groups
            )
            if features.X.shape[1] > MAX_FEATURES:  # pragma: no cover - registry guard
                raise MLError(
                    f"{features.X.shape[1]} feature columns exceeds the limit of {MAX_FEATURES}"
                )
            bundle = self._build_bundle(features, request)
            self._preflight(bundle, request)
            dataset_id = bundle.manifest.dataset_id
        except BaseException:
            self.slot.release(run_id)
            raise

        created_at = _now()
        run_dir = root / run_id
        state = MLRunState(
            run_id=run_id,
            status="running",
            created_at=created_at,
            run_dir=run_dir,
            request=request,
            dataset_id=dataset_id,
        )
        with self._lock:
            self._runs[run_id] = state
        try:
            self._write_meta(state, "running")
            thread = threading.Thread(
                target=self._work,
                args=(state, features),
                name=f"ml-{run_id}",
                daemon=True,
            )
            with self._lock:
                state.thread = thread
            thread.start()
        except BaseException as exc:
            with self._lock:
                self._runs.pop(run_id, None)
            self.slot.release(run_id)
            try:
                self._write_meta(state, "failed", str(exc))
            except OSError:  # pragma: no cover - best effort on a broken disk
                logger.exception("cannot record the failed start of run %s", run_id)
            raise
        return run_id

    def _build_bundle(self, features: FeatureFrame, request: MLRequest) -> Any:
        """Build the dataset the trainer will build again; the split geometry
        and dataset id must be known before the 202."""
        spec = request.spec
        try:
            return build_dataset(
                features,
                symbol=request.symbol,
                timeframe=request.timeframe,
                groups=spec.groups,
                horizon=spec.horizon,
                train_fraction=spec.train_fraction,
                validation_fraction=spec.validation_fraction,
                purge_bars=max(spec.purge_bars, spec.horizon),
                embargo_bars=spec.embargo_bars,
            )
        except ValueError as exc:
            raise MLError(str(exc)) from exc

    @staticmethod
    def _preflight(bundle: Any, request: MLRequest) -> None:
        """Mirror the trainer's row floors so the 202 does not hide a dead run."""
        spec = request.spec
        flat = bundle.y_direction == 0
        keep = np.ones(bundle.split.shape[0], dtype=bool) if spec.include_flat else ~flat
        train = (bundle.split == 0) & keep
        validation = (bundle.split == 1) & keep
        if int(train.sum()) < 20:
            raise MLError("not enough training rows; adjust fractions or horizon")
        if int(validation.sum()) < 5:
            raise MLError("not enough validation rows; adjust fractions or embargo")
        if np.unique(bundle.y_direction[train]).size < 2:
            raise MLError("the training split has a single class; cannot fit a direction model")
        split_mask = (bundle.split == _SPLITS_INDEX[request.split]) & keep
        if int(split_mask.sum()) == 0:
            raise MLError(f"the {request.split} split has no rows to evaluate")

    # -- worker ----------------------------------------------------------

    def _work(self, state: MLRunState, features: FeatureFrame) -> None:
        """Fit, evaluate, persist and publish a terminal status."""
        try:
            self._check_cancel(state)
            result = train_classifier(
                features,
                symbol=state.request.symbol,
                timeframe=state.request.timeframe,
                spec=state.request.spec,
            )
            # The fit is one blocking call. A cancel that arrived during it is
            # observable here, before anything is written.
            self._check_cancel(state)
            save_result(result, state.run_dir)
            self._check_cancel(state)
            evaluation = evaluate_model(
                result,
                features,
                self.settings,
                split=state.request.split,
                threshold=state.request.threshold,
                exit_threshold=state.request.exit_threshold,
                window=state.request.window,
                stress=state.request.stress,
                compute_regime=state.request.regime,
                cancel_check=state.cancel_event.is_set,
            )
            self._check_cancel(state)
            self._finish(state, "completed", report=self._report(state, result, evaluation))
        except InterruptedError:
            self._finish(state, "cancelled", error="cancelled by request")
        except Exception as exc:
            logger.exception("ML run %s failed", state.run_id)
            self._finish(state, "failed", error=str(exc))
        finally:
            with self._lock:
                state.thread = None
            self.slot.release(state.run_id)

    @staticmethod
    def _check_cancel(state: MLRunState) -> None:
        if state.cancel_event.is_set():
            raise InterruptedError("cancelled")

    def _report(
        self,
        state: MLRunState,
        result: MLResult,
        evaluation: EvaluationResult,
    ) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "status": "completed",
            "created_at": state.created_at,
            "finished_at": _now(),
            "model": result.spec.model,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "dataset_id": result.dataset_id,
            "config": state.request.config(),
            "spec": asdict(result.spec),
            "columns": list(result.columns),
            "validation": result.validation,
            "test": result.test,
            "baseline": result.baseline,
            "split_counts": result.split_counts,
            "calibration": _calibration(result.validation),
            "evaluation": _evaluation_payload(evaluation),
        }

    def _finish(
        self,
        state: MLRunState,
        status: str,
        *,
        error: str | None = None,
        report: dict[str, Any] | None = None,
    ) -> None:
        """Write the terminal report/meta and publish the status atomically.

        The runner lock is held across both writes so a concurrent ``cancel``
        either wins before the terminal state (the worker then records
        ``cancelled``) or loses to it (409). It also means a reader never sees a
        ``completed`` run whose report is still missing.
        """
        payload = report or {
            "run_id": state.run_id,
            "status": status,
            "created_at": state.created_at,
            "finished_at": _now(),
            "model": state.request.spec.model,
            "symbol": state.request.symbol,
            "timeframe": state.request.timeframe,
            "dataset_id": state.dataset_id,
            "config": state.request.config(),
            "spec": asdict(state.request.spec),
            "columns": [],
            "validation": None,
            "test": None,
            "baseline": None,
            "split_counts": {},
            "calibration": None,
            "evaluation": None,
            "error": error,
        }
        with self._lock:
            if state.status in TERMINAL_STATUSES:  # pragma: no cover - defensive
                return
            # A cancel that landed after the worker's last check still wins: the
            # visible status never goes cancelling -> completed.
            if status == "completed" and state.cancel_event.is_set():
                status = "cancelled"
                error = "cancelled by request"
                payload = {
                    **payload,
                    "status": "cancelled",
                    "finished_at": _now(),
                    "validation": None,
                    "test": None,
                    "baseline": None,
                    "split_counts": {},
                    "calibration": None,
                    "evaluation": None,
                    "error": error,
                }
            try:
                write_json(state.run_dir / REPORT_NAME, payload)
            except (OSError, ValueError, TypeError):
                logger.exception("cannot write the report of ML run %s", state.run_id)
            try:
                self._write_meta(state, status, error)
            except (OSError, ValueError, TypeError):
                logger.exception("cannot write the manifest of ML run %s", state.run_id)
            state.status = status
            state.error = error

    def _write_meta(self, state: MLRunState, status: str, error: str | None = None) -> None:
        now = _now()
        write_json(
            state.run_dir / META_NAME,
            {
                "run_id": state.run_id,
                "status": status,
                "created_at": state.created_at,
                "updated_at": now,
                "finished_at": now if status in TERMINAL_STATUSES else None,
                "symbol": state.request.symbol,
                "timeframe": state.request.timeframe,
                "model": state.request.spec.model,
                "dataset_id": state.dataset_id,
                "config": state.request.config(),
                "error": error,
            },
        )

    # -- control ---------------------------------------------------------

    def cancel(self, run_id: str) -> bool:
        """Mark a live run as cancelling; the worker writes the terminal state.

        The meta write happens under the runner lock, so a worker finishing at
        the same moment either records its terminal status first (this call then
        raises 409) or sees the cancel flag and records ``cancelled``.
        """
        self._run_dir(run_id)
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                raise UnknownMLRunError(run_id)
            if state.status != "running":
                raise MLControlError(f"cannot cancel run {run_id} while it is {state.status}")
            state.status = "cancelling"
            state.cancel_event.set()
            try:
                self._write_meta(state, "cancelling")
            except (OSError, ValueError, TypeError):
                # The in-memory flag is authoritative for this process; a failed
                # meta write must not turn the control request into a 500.
                logger.exception("cannot record the cancellation of ML run %s", run_id)
        return True

    # -- registry --------------------------------------------------------

    def status(self, run_id: str) -> str:
        """The live status when this process owns the run, else the meta file."""
        run_dir = self._run_dir(run_id)
        with self._lock:
            state = self._runs.get(run_id)
            if state is not None:
                return state.status
            meta = _read_json(run_dir / META_NAME)
        if not meta:
            raise UnknownMLRunError(run_id)
        return str(meta.get("status") or "unknown")

    def list_runs(self) -> list[dict[str, Any]]:
        """``meta.json`` rows under ``runs/ml``, newest first, live status overlaid."""
        with self._lock:
            live = {run_id: state.status for run_id, state in self._runs.items()}
        root = self.runs_root / RUNS_DIRNAME
        if not root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            meta = _read_json(entry / META_NAME)
            if not meta:
                logger.warning("skipping ML directory without meta.json: %s", entry)
                continue
            run_id = str(meta.get("run_id") or entry.name)
            created_at = meta.get("created_at")
            items.append(
                {
                    "run_id": run_id,
                    "status": live.get(run_id, str(meta.get("status") or "unknown")),
                    "created_at": str(created_at) if created_at is not None else None,
                    "model": meta.get("model"),
                    "symbol": meta.get("symbol"),
                    "timeframe": meta.get("timeframe"),
                    "dataset_id": meta.get("dataset_id"),
                }
            )
        items.sort(key=lambda item: (item["created_at"] or "", item["run_id"]), reverse=True)
        return items

    def detail(self, run_id: str) -> dict[str, Any]:
        """Status, config, prediction metrics and the evaluation of one run.

        The lock is held across the state lookup and both file reads, so a
        detail request cannot pair a terminal in-memory status with a report
        that is still being written.
        """
        run_dir = self._run_dir(run_id)
        with self._lock:
            state = self._runs.get(run_id)
            meta = _read_json(run_dir / META_NAME)
            report = _read_json(run_dir / REPORT_NAME)
            if not meta and not report and state is None:
                raise UnknownMLRunError(run_id)

            status = (
                state.status
                if state is not None
                else str(meta.get("status") or report.get("status") or "unknown")
            )
            config = report.get("config") or meta.get("config")
            config = config if isinstance(config, dict) else {}
            spec = report.get("spec") or config.get("spec")
            spec = spec if isinstance(spec, dict) else {}
            columns = report.get("columns")
            validation = report.get("validation")
            validation = validation if isinstance(validation, dict) else None
            calibration = report.get("calibration")
            if not isinstance(calibration, dict):
                calibration = _calibration(validation)
            split_counts = report.get("split_counts")
            split_counts = split_counts if isinstance(split_counts, dict) else {}

            return {
                "run_id": run_id,
                "status": status,
                "created_at": str(meta.get("created_at") or report.get("created_at") or "") or None,
                "finished_at": meta.get("finished_at") or report.get("finished_at"),
                "model": report.get("model") or meta.get("model"),
                "symbol": report.get("symbol") or meta.get("symbol"),
                "timeframe": report.get("timeframe") or meta.get("timeframe"),
                "dataset_id": report.get("dataset_id") or meta.get("dataset_id"),
                "config": config,
                "spec": spec,
                "columns": list(columns) if isinstance(columns, list) else [],
                "validation": validation,
                "test": report.get("test") if isinstance(report.get("test"), dict) else None,
                "baseline": report.get("baseline")
                if isinstance(report.get("baseline"), dict)
                else None,
                "split_counts": split_counts,
                "calibration": calibration,
                "error": meta.get("error") or report.get("error"),
            }

    def _run_dir(self, run_id: str) -> Path:
        """Resolve one run id, refusing anything that is not a bare token."""
        if not is_ml_run_id(run_id):
            raise UnknownMLRunError(run_id)
        return self.runs_root / RUNS_DIRNAME / run_id

    # -- trading evaluation ----------------------------------------------

    def evaluate(
        self,
        run_id: str,
        *,
        split: str = "validation",
        threshold: float = 0.55,
        exit_threshold: float = 0.55,
        window: int = 60,
        stress: tuple[float, ...] = (),
        regime: bool = True,
    ) -> dict[str, Any]:
        """Trade a finished run's artefact on one split.

        The store must reproduce the run's ``dataset_id``; otherwise the metrics
        would describe a model and prices that never met. Evaluation shares the
        one-job slot, because cost stress is another heavy simulator pass.
        """
        if split not in _SPLITS:
            raise ValueError(f"unknown split {split!r}; choose train/validation/test")
        _validate_thresholds(threshold, exit_threshold, window)
        _validate_stress(stress)

        run_dir = self._run_dir(run_id)
        meta = _read_json(run_dir / META_NAME)
        with self._lock:
            state = self._runs.get(run_id)
        if not meta and state is None:
            raise UnknownMLRunError(run_id)
        status = state.status if state is not None else str(meta.get("status") or "unknown")
        if status in LIVE_STATUSES:
            raise MLError(f"run {run_id} is not finished (status {status})")
        if not (run_dir / MODEL_NAME).is_file():
            raise UnknownMLRunError(f"run {run_id} has no model artefact")

        owner = f"evaluate-{run_id}-{uuid.uuid4().hex}"
        if not self.slot.acquire(owner):
            raise MLError(
                f"another ML job is running ({self.slot.current()}); wait for it to finish"
            )
        try:
            return self._evaluate_locked(
                run_id,
                run_dir,
                split=split,
                threshold=threshold,
                exit_threshold=exit_threshold,
                window=window,
                stress=stress,
                regime=regime,
            )
        finally:
            self.slot.release(owner)

    def _evaluate_locked(
        self,
        run_id: str,
        run_dir: Path,
        *,
        split: str,
        threshold: float,
        exit_threshold: float,
        window: int,
        stress: tuple[float, ...],
        regime: bool,
    ) -> dict[str, Any]:
        try:
            result = load_result(run_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise MLError(
                f"run {run_id} cannot be loaded: {public_error_text(exc, root=self.runs_root)}"
            ) from exc

        frame = self.store.read(result.symbol, result.timeframe)
        if frame.is_empty():
            raise MLError(f"no {result.symbol} {result.timeframe} bars in the store")
        if frame.height > MAX_BARS:
            raise MLError(f"refusing {frame.height} bars: the API limit is {MAX_BARS}")
        features = self._features(frame, result.symbol, result.timeframe)

        spec = result.spec
        try:
            bundle = build_dataset(
                features,
                symbol=result.symbol,
                timeframe=result.timeframe,
                groups=spec.groups,
                horizon=spec.horizon,
                train_fraction=spec.train_fraction,
                validation_fraction=spec.validation_fraction,
                purge_bars=max(spec.purge_bars, spec.horizon),
                embargo_bars=spec.embargo_bars,
            )
        except ValueError as exc:
            raise MLError(str(exc)) from exc
        if bundle.manifest.dataset_id != result.dataset_id:
            raise MLError(
                f"run {run_id} was trained on different data: "
                f"run dataset_id {result.dataset_id}, stored dataset_id "
                f"{bundle.manifest.dataset_id}"
            )

        try:
            evaluation = evaluate_model(
                result,
                features,
                self.settings,
                split=split,
                threshold=threshold,
                exit_threshold=exit_threshold,
                window=window,
                stress=stress,
                compute_regime=regime,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise MLError(
                f"run {run_id} cannot be evaluated: {public_error_text(exc, root=self.runs_root)}"
            ) from exc

        return {
            "run_id": run_id,
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "dataset_id": result.dataset_id,
            "split": evaluation.split,
            "threshold": evaluation.threshold,
            "exit_threshold": evaluation.exit_threshold,
            "metrics": evaluation.metrics,
            "hold": evaluation.hold,
            "stress": evaluation.stress,
            "regime": evaluation.regime,
        }

    # -- lifecycle -------------------------------------------------------

    def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """Ask live runs to cancel, then join their threads within ``timeout``."""
        with self._lock:
            states = list(self._runs.values())
        for state in states:
            if state.status in LIVE_STATUSES:
                state.cancel_event.set()
        self.join_all(timeout)

    def join_all(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        with self._lock:
            threads = [
                state.thread
                for state in self._runs.values()
                if state.thread is not None and state.thread.ident is not None
            ]
        end = time.monotonic() + timeout
        for thread in threads:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)

    def active_threads(self) -> list[threading.Thread]:
        with self._lock:
            threads = [state.thread for state in self._runs.values()]
        return [thread for thread in threads if thread is not None and thread.is_alive()]
