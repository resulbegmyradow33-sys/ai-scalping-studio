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
import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.api.runner import MIN_BARS as SIM_MIN_BARS
from scalping.api.runner import TIMEFRAME
from scalping.backtest.engine import run_backtest
from scalping.backtest.evaluate import FoldResult, cost_stress, evaluate_walkforward
from scalping.config import ContractSpec, CostConfig, RiskConfig, Settings
from scalping.data.store import ParquetStore
from scalping.features.build import FeatureFrame, build_features
from scalping.rl.policy import load_policy
from scalping.rl.registry import (
    BANDIT_NAME,
    CONFIG_NAME,
    METRICS_HISTORY_NAME,
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
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM, bandit_training_data, label_events
from scalping.strategy.signals import ALL_STRATEGIES

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


def _utc(value: datetime) -> datetime:
    """Naive timestamps are read as UTC, matching the data layer."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime | str | None) -> str | None:
    """Normalise a split boundary to an ISO string (naive means UTC)."""
    if value is None:
        return None
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return _utc(parsed).isoformat()
    return _utc(value).isoformat()


def _parsed_ts(value: str) -> datetime:
    return _utc(datetime.fromisoformat(value))


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


def _finite_float(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` for missing/non-finite input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    """``value`` as an int built from a finite number, or ``None``."""
    number = _finite_float(value)
    return None if number is None else int(number)


def _series_from_history(run_dir: Path) -> dict[str, list[dict[str, Any]]] | None:
    """Build the four training curves from ``metrics.jsonl`` in file order.

    ``reward`` prefers the per-interval mean and falls back to the cumulative
    one. A row whose value is missing or non-finite is skipped for that curve
    only, so a null entropy does not drop its reward point. ``None`` means the
    file is absent or empty and the caller should omit the whole block.
    """
    path = run_dir / METRICS_HISTORY_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None

    series: dict[str, list[dict[str, Any]]] = {
        "reward": [],
        "episode_length": [],
        "entropy": [],
        "loss": [],
    }
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        step = _finite_int(row.get("step"))
        if step is None:
            continue
        reward_key = "reward_mean_interval" if "reward_mean_interval" in row else "reward_mean"
        for name, key in (
            ("reward", reward_key),
            ("episode_length", "ep_len_mean"),
            ("entropy", "entropy_loss"),
            ("loss", "policy_loss"),
        ):
            value = _finite_float(row.get(key))
            if value is not None:
                series[name].append({"step": step, "value": value})
    return series


def _range_pair(value: tuple[datetime, datetime]) -> list[str]:
    return [_utc(value[0]).isoformat(), _utc(value[1]).isoformat()]


def _fold_payload(fold: FoldResult) -> dict[str, Any]:
    """One walk-forward fold as a JSON-safe response object."""
    return {
        "train_range": _range_pair(fold.train_range),
        "test_range": _range_pair(fold.test_range),
        "evaluated_range": _range_pair(fold.evaluated_range),
        "metrics": metrics_payload(fold.metrics),
        "train_expectancy": _finite_float(fold.train_expectancy),
    }


def _bandit_weights(run_dir: Path) -> list[dict[str, Any]] | None:
    """Per-arm coefficient norms from ``bandit.npz``, in strategy order.

    ``weight`` is ``||theta||`` of the arm's context coefficients, not a reward:
    ``LinUCB.weights()`` drops the intercept, so ``theta @ x`` is neither the
    predicted reward nor the UCB score. It is shown as a relative magnitude.
    ``None`` means the file is absent (or unreadable) and the block is omitted.
    """
    path = run_dir / BANDIT_NAME
    if not path.is_file():
        return None
    try:
        weights = LinUCB.load(path).weights()
    except (OSError, TypeError, ValueError, KeyError) as exc:
        logger.warning("ignoring unreadable bandit file %s: %s", path, exc)
        return None
    rows: list[dict[str, Any]] = []
    # Arm index is a frozen position in ALL_STRATEGIES; a shorter checkpoint is
    # paired up to its own length instead of inventing arms.
    for strategy, theta in zip(ALL_STRATEGIES, weights, strict=False):
        rows.append(
            {
                "strategy": strategy.value,
                "weight": _finite_float(float(np.linalg.norm(theta))),
            }
        )
    return rows


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
        use_strategy: bool = True,
        entry_penalty: float = 0.05,
        gate_signals: bool = True,
        window: int = 60,
        timeframe: str = "M1",
        sl_atr_mult: float = 1.0,
        tp_atr_mult: float = 2.0,
        max_bars_in_trade: int = 120,
        train_end: datetime | None = None,
    ) -> str:
        """Prepare a run, reserve its id and start the worker thread.

        ``use_strategy`` fits a LinUCB on the label batch, freezes it and saves
        ``bandit.npz`` *before* the worker starts PPO, so the strategy context is
        real from the first step. ``train_end`` slices the raw bars: the run only
        sees bars at or before it, and ``eval_start`` records the first bar after
        it for the evaluation endpoint.

        Raises :class:`TrainError` (409) for a store below the bar floor or a
        run id that is already training. Data problems from the feature layer
        stay ``ValueError`` and map to 422 in the route.
        """
        if timeframe not in ("M1", "M5", "M15"):
            raise TrainError(f"unsupported timeframe: {timeframe!r}")
        run_settings = replace(
            self.settings,
            risk=replace(
                self.settings.risk,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                max_bars_in_trade=max_bars_in_trade,
            ),
        )
        full = self.store.read(symbol, timeframe)
        if full.height < MIN_BARS:
            raise TrainError(f"not enough bars: need {MIN_BARS}")

        train_end_iso = _iso(train_end)
        eval_start: str | None = None
        frame = full
        if train_end is not None:
            end = _utc(train_end)
            frame = full.filter(pl.col("ts") <= end)
            after = full.filter(pl.col("ts") > end)
            if after.height:
                first_after = after["ts"].min()
                if isinstance(first_after, datetime):
                    eval_start = _utc(first_after).isoformat()
        if frame.height < MIN_BARS:
            raise TrainError(f"not enough bars: need {MIN_BARS}")

        features = build_features(frame, run_settings.costs)
        if features.X.shape[0] < MIN_BARS:
            raise TrainError(f"not enough bars: need {MIN_BARS}")

        # Validate the env count here, with the trainer's own check: doing it in
        # the worker would return 202 and then leave a half-created run behind.
        try:
            env_starts_for(features, n_envs, window)
        except ValueError as exc:
            raise TrainError(str(exc)) from exc

        labels: pl.DataFrame | None = None
        selector: LinUCB | None = None
        if use_strategy:
            labels = label_events(frame, run_settings)
            try:
                X, arms, rewards = bandit_training_data(features, labels)
                if X.shape[0] == 0:
                    raise TrainError("no strategy labels: cannot fit the bandit")
                selector = LinUCB(dim=CONTEXT_DIM).fit_offline(X, arms, rewards).freeze()
            except TrainError:
                raise
            except (TypeError, ValueError) as exc:
                raise TrainError(f"cannot fit the strategy bandit: {exc}") from exc

        config = TrainConfig(
            run_name=symbol,
            total_steps=total_steps,
            n_envs=n_envs,
            seed=seed,
            entry_penalty=entry_penalty,
            gate_signals=gate_signals,
            window=window,
            timeframe=timeframe,
        )
        split = (
            {"train_end": train_end_iso, "eval_start": eval_start}
            if train_end_iso is not None or eval_start is not None
            else None
        )
        run_id, identity = run_identity(config, features, run_settings, selector, split)
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
                    "use_strategy": selector is not None,
                    "random_start": config.random_start,
                    "train_end": train_end_iso,
                    "eval_start": eval_start,
                    "updated_at": created_at,
                },
            )
            if selector is not None:
                # The frozen bandit exists before the first PPO step: the panel
                # can inspect the weights while the policy is still untrained.
                selector.save(run_dir / BANDIT_NAME)
            thread = threading.Thread(
                target=self._train,
                args=(
                    state,
                    config,
                    features,
                    selector,
                    labels,
                    train_end_iso,
                    eval_start,
                    run_settings,
                ),
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

    def _train(
        self,
        state: RunState,
        config: TrainConfig,
        features: FeatureFrame,
        selector: LinUCB | None,
        labels: pl.DataFrame | None,
        train_end: str | None,
        eval_start: str | None,
        settings: Settings,
    ) -> None:
        """Run ``train`` in the worker thread and publish its final status."""
        status = "failed"
        try:
            result = train(
                config,
                features,
                settings,
                selector=selector,
                out_dir=self.runs_root,
                labels=labels,
                train_end=train_end,
                eval_start=eval_start,
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
                    raise RunControlError(f"cannot pause run {run_id} while it is {state.status}")
                state.status = "paused"
                state.pause_event.set()
            elif action == "resume":
                if state.status != "paused":
                    raise RunControlError(f"cannot resume run {run_id} while it is {state.status}")
                state.status = "running"
                state.pause_event.clear()
            elif action == "stop":
                if state.status not in LIVE_STATUSES:
                    raise RunControlError(f"cannot stop run {run_id} while it is {state.status}")
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
                    "reward_mean": _finite_float(manifest.get("reward_mean")),
                    "reward_mean_normalized": _finite_float(manifest.get("reward_mean_normalized")),
                    "degenerate": (
                        manifest.get("degenerate")
                        if isinstance(manifest.get("degenerate"), bool)
                        else None
                    ),
                    "baseline_hold_net_profit": _finite_float(
                        manifest.get("baseline_hold_net_profit")
                    ),
                }
            )
        items.sort(key=lambda item: (item["created_at"] or "", item["run_id"]), reverse=True)
        return items

    def detail(self, run_id: str) -> dict[str, Any]:
        """Config, latest metrics, status and the chart data for one run.

        ``step`` and ``total_steps`` drive the progress bar; ``series`` and
        ``bandit`` are only present when the run has the corresponding file, and
        the route omits unset fields rather than serialising nulls.
        """
        run_dir = self._run_dir(run_id)
        manifest = load_run(run_dir)
        with self._lock:
            state = self._runs.get(run_id)
        if manifest is None and state is None:
            raise UnknownRunError(run_id)
        status = (
            state.status if state is not None else str((manifest or {}).get("status") or "unknown")
        )

        config = _read_json(run_dir / CONFIG_NAME)
        metrics = _read_json(run_dir / METRICS_NAME)
        detail: dict[str, Any] = {
            "config": config,
            "metrics": metrics,
            "status": status,
            "step": _finite_int(metrics.get("step")) or 0,
        }
        params = config.get("params")
        if isinstance(params, dict):
            total_steps = _finite_int(params.get("total_steps"))
            if total_steps is not None:
                detail["total_steps"] = total_steps
        series = _series_from_history(run_dir)
        if series is not None:
            detail["series"] = series
        bandit = _bandit_weights(run_dir)
        if bandit is not None:
            detail["bandit"] = bandit
        return detail

    def _run_dir(self, run_id: str) -> Path:
        """Resolve one run id, refusing anything that is not a single component."""
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise UnknownRunError(run_id)
        return self.runs_root / run_id

    # -- backtest --------------------------------------------------------

    def _load_selector(self, run_dir: Path, config: dict[str, Any]) -> LinUCB | None:
        """Load the frozen bandit when the run was trained with strategy context."""
        selector = config.get("selector")
        if not isinstance(selector, dict) or not selector.get("enabled"):
            return None
        path = run_dir / BANDIT_NAME
        if not path.is_file():
            raise TrainError(f"run directory {run_dir.name} has no {BANDIT_NAME}")
        try:
            return LinUCB.load(path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise TrainError(f"cannot load {BANDIT_NAME}: {exc}") from exc

    def _settings_from_config(self, config: dict[str, Any]) -> Settings:
        """Rebuild the run's settings, falling back to the app's when absent."""
        raw = config.get("settings")
        if not isinstance(raw, dict):
            return self.settings
        try:
            return Settings(
                contract=ContractSpec(**raw["contract"]),
                risk=RiskConfig(**raw["risk"]),
                costs=CostConfig(**raw["costs"]),
            )
        except (KeyError, TypeError):
            return self.settings

    def _expected_data_hash(
        self, frame: pl.DataFrame, config: dict[str, Any]
    ) -> str:
        """The hash of the bars this run was trained on (the slice when split)."""
        train_end = config.get("train_end")
        if isinstance(train_end, str):
            end = _parsed_ts(train_end)
            sliced = frame.filter(pl.col("ts") <= end)
            return data_hash_for(build_features(sliced, self.settings.costs))
        return data_hash_for(build_features(frame, self.settings.costs))

    def backtest(self, run_id: str, symbol: str) -> dict[str, Any]:
        """Replay one stored policy over the stored bars and write a report.

        Only a finished run with a policy can be replayed. A live status
        (``running``/``paused``/``stopping``) is a 409 because the worker is
        still rewriting the run; ``failed`` is a 409 because there is no policy
        to load. The stored bars must hash to the run's own ``data_hash`` — a
        policy replayed on different prices would produce a report that looks
        valid and means nothing.

        ``in_sample`` compares the last evaluated bar with ``train_end``: a run
        trained on the full stored range is in-sample everywhere, a run with a
        split is out of sample as soon as the replay extends past it.
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
        config = _read_json(run_dir / CONFIG_NAME)
        run_settings = self._settings_from_config(config)
        try:
            policy = load_policy(run_dir)
        except (OSError, TypeError, ValueError) as exc:
            raise TrainError(f"run {run_id} cannot be loaded: {exc}") from exc

        width = int(features.X.shape[1])
        declared = policy.config.get("n_features")
        if declared is not None and int(declared) != width:
            raise TrainError(
                f"run {run_id} was trained on {declared} features, the stored data produces {width}"
            )
        expected_hash = self._expected_data_hash(frame, config)
        run_data_hash = policy.config.get("data_hash")
        if not isinstance(run_data_hash, str) or run_data_hash != expected_hash:
            raise TrainError(
                f"run {run_id} was trained on different data: "
                f"run data_hash {run_data_hash}, stored data_hash {expected_hash}"
            )

        selector = self._load_selector(run_dir, config)
        params = config.get("params") if isinstance(config.get("params"), dict) else {}
        window = int(params.get("window", 60)) if isinstance(params, dict) else 60
        labels = label_events(frame, self.settings) if selector is not None else None
        try:
            result = run_backtest(
                policy,
                features,
                self.settings,
                policy_name=run_id,
                max_steps=max(len(features.ts) + 1, 1),
                strategy_selector=selector,
                labels=labels,
                window=window,
            )
        except (TypeError, ValueError) as exc:
            raise TrainError(f"run {run_id} cannot be replayed: {exc}") from exc
        metrics = metrics_payload(result.metrics)
        train_end = config.get("train_end")
        evaluated_end = features.ts[-1] if len(features.ts) else None
        if train_end is None:
            in_sample = True
        elif evaluated_end is None:
            in_sample = False
        else:
            last = np.datetime64(evaluated_end, "us").astype(datetime).replace(tzinfo=UTC)
            in_sample = last <= _parsed_ts(str(train_end))
        report_path = run_dir / f"backtest-{_safe_name(symbol)}.json"
        write_json(
            report_path,
            {
                "run_id": run_id,
                "backtest_run_id": result.run_id,
                "symbol": symbol,
                "config_hash": result.config_hash,
                "data_hash": data_hash_for(features),
                "run_data_hash": run_data_hash,
                "train_end": train_end,
                "eval_start": config.get("eval_start"),
                "in_sample": in_sample,
                "created_at": _now(),
                "metrics": metrics,
            },
        )
        return {"report_path": str(report_path), "metrics": metrics}

    # -- evaluation ------------------------------------------------------

    def evaluate(self, run_id: str, symbol: str, folds: int = 3) -> dict[str, Any]:
        """Walk-forward and cost-stress a finished run on out-of-sample bars.

        The evaluation window starts at the run's ``eval_start`` when the run was
        trained with a split; otherwise the whole stored history is used. The
        slice must match the run's data identity: a no-split run is evaluated on
        exactly its training bars, and a split run's training slice is re-hashed
        against ``data_hash`` before its newer bars are evaluated. Features are
        rebuilt inside every fold, so no fold sees another's bars, and a
        strategy-aware policy gets its frozen bandit and per-window labels so the
        observation width matches training. The report is written to
        ``evaluation.json`` in the run directory.
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
            raise TrainError(f"run {run_id} is failed; there is no policy to evaluate")

        config = _read_json(run_dir / CONFIG_NAME)
        try:
            policy = load_policy(run_dir)
        except (OSError, TypeError, ValueError) as exc:
            raise TrainError(f"run {run_id} cannot be loaded: {exc}") from exc
        selector = self._load_selector(run_dir, config)
        raw_params = config.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        window = int(params.get("window", 60))
        run_settings = self._settings_from_config(config)
        run_timeframe = str(params.get("timeframe", "M1"))
        full = self.store.read(symbol, run_timeframe)
        train_end = config.get("train_end")
        eval_start = config.get("eval_start")
        frame = full
        if isinstance(eval_start, str):
            frame = frame.filter(pl.col("ts") >= _parsed_ts(eval_start))
        if frame.height < SIM_MIN_BARS:
            raise TrainError(f"not enough bars: need {SIM_MIN_BARS}")

        features = build_features(frame, run_settings.costs)
        eval_data_hash = data_hash_for(features)
        run_data_hash = config.get("data_hash")
        if not isinstance(run_data_hash, str):
            raise TrainError(f"run {run_id} has no data hash")
        # A no-split run must be evaluated on the exact bars it trained on. A
        # split run evaluates newer bars by design, so its guard is the training
        # slice, exactly as the backtest guard is.
        guard_hash = (
            self._expected_data_hash(full, config)
            if isinstance(train_end, str)
            else eval_data_hash
        )
        if guard_hash != run_data_hash:
            raise TrainError(
                f"run {run_id} was trained on different data: "
                f"run data_hash {run_data_hash}, evaluation data_hash {eval_data_hash}"
            )

        method = "walk_forward"
        report = None
        holdout = None
        try:
            try:
                report = evaluate_walkforward(
                    policy,
                    frame,
                    run_settings,
                    folds=folds,
                    strategy_selector=selector,
                    window=window,
                )
            except ValueError as exc:
                if "not enough data" not in str(exc):
                    raise
                method = "holdout"
                labels_holdout = (
                    label_events(frame, run_settings) if selector is not None else None
                )
                holdout = run_backtest(
                    policy,
                    features,
                    run_settings,
                    max_steps=len(features.ts) + 1,
                    strategy_selector=selector,
                    labels=labels_holdout,
                    window=window,
                )
            labels = label_events(frame, run_settings) if selector is not None else None
            stress = cost_stress(
                policy,
                features,
                run_settings,
                strategy_selector=selector,
                labels=labels,
                window=window,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TrainError(f"run {run_id} cannot be evaluated: {exc}") from exc

        if report is None:
            if holdout is None:  # pragma: no cover - impossible: set with method
                raise TrainError(f"run {run_id} produced no evaluation")
            eval_first = str(features.ts[0])
            eval_last = str(features.ts[-1])
            fold_payloads = [
                {
                    "train_range": [],
                    "test_range": [eval_first, eval_last],
                    "evaluated_range": [eval_first, eval_last],
                    "metrics": metrics_payload(holdout.metrics),
                    "train_expectancy": None,
                }
            ]
            overall_payload = metrics_payload(holdout.metrics)
            overfit = False
            folds_requested, folds_produced = folds, 1
        else:
            fold_payloads = [_fold_payload(fold) for fold in report.folds]
            overall_payload = metrics_payload(report.overall)
            overfit = bool(report.overfit_warning)
            folds_requested, folds_produced = report.folds_requested, report.folds_produced

        payload: dict[str, Any] = {
            "run_id": run_id,
            "symbol": symbol,
            "train_end": train_end,
            "eval_start": eval_start,
            "data_hash": run_data_hash,
            "eval_data_hash": eval_data_hash,
            "in_sample": False,
            "created_at": _now(),
            "method": method,
            "folds": fold_payloads,
            "overall": overall_payload,
            "overfit_warning": overfit,
            "folds_requested": folds_requested,
            "folds_produced": folds_produced,
            "stress": [
                {
                    "multiplier": float(result.multiplier),
                    "metrics": metrics_payload(result.metrics),
                }
                for result in stress
            ],
        }
        report_path = run_dir / "evaluation.json"
        write_json(report_path, payload)
        return {
            "report_path": str(report_path),
            "train_end": train_end,
            "eval_start": eval_start,
            "data_hash": run_data_hash,
            "eval_data_hash": eval_data_hash,
            "in_sample": False,
            "folds": payload["folds"],
            "overall": payload["overall"],
            "overfit_warning": payload["overfit_warning"],
            "stress": payload["stress"],
            "method": method,
        }

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
