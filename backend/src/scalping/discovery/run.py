"""One end-to-end discovery run: stored bars to a recorded champion decision.

This module is the glue the earlier slices deliberately left out. It reads a
stored symbol, builds the causal feature matrix through the same cache every
other command uses, cuts a chronological dataset, keeps **only the validation
split**, scores two simple baselines on that exact slice and then lets the
bounded search generate and simulate strategies. When the budget is spent it
writes a decision to the archive.

What this module owns
---------------------
* **Split discipline.** It calls :func:`scalping.datasets.builder.build_dataset`
  for the purge/embargo geometry and slices ``[validation.start, validation.end)``.
  The test rows are never copied, never scored and never named to an evaluator;
  the dataset id carries the ``validation:`` prefix the evaluator requires. A
  build whose validation split is empty is an error, not an empty search.
* **Cost parity.** Candidates and baselines run on the same frame, with the same
  ``Settings``, the same observation ``window`` and the same weekly session gate.
  A strategy that beats ``hold`` only because the baseline paid different costs
  is not beating anything.
* **An honest decision.** The archive accepts ``promoted``, but this module never
  writes it: discovery produces hypotheses and the registry gate owns promotion.
  The decision vocabulary here is ``no_trade``, ``no_change`` and ``rejected``.

Decision semantics
------------------
``no_trade``
    The gate rejected every trial, or the best accepted candidate is flat by
    construction (zero closed trades). NO TRADE is a result, not a failure.
``no_change``
    No accepted candidate improved on the best baseline. This includes a rerun
    that added no trials because the archive already held every candidate.
``rejected``
    Some accepted candidate beat the best baseline on validation, so the run
    hands it to the champion gate. The word is the archive's, not a verdict:
    discovery cannot promote, and a validation winner after N trials is a
    multiple-testing warning, not evidence of a durable edge.

Reproducibility
---------------
The dataset id comes from the manifest digest, the trial seeds from the search
RNG and the code version from a digest of the modules that change arithmetic.
Same store slice, same seed, same code and an empty archive therefore reproduce
the same candidate hashes and metrics; a second run on the same archive is
deduplicated by candidate hash instead of paying for the same trials again.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from scalping.backtest.engine import run_backtest
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.builder import DatasetBundle, SplitRange, build_dataset
from scalping.discovery.archive import DECISIONS, DiscoveryArchive
from scalping.discovery.evaluate import DslEvaluator
from scalping.discovery.evidence import settings_digest
from scalping.discovery.protocols import Factory
from scalping.discovery.search import SearchConfig, SearchResult, evolve
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.features.cache import FEATURE_CACHE_DIRNAME, FeatureCache, cached_build_features
from scalping.registry.gate import GateCriteria
from scalping.serialize import metrics_payload
from scalping.strategy.dsl.mutate import StrategyFactory

__all__ = [
    "BASELINES",
    "BASELINE_LABELS",
    "DiscoveryRunConfig",
    "DiscoveryRunError",
    "DiscoveryRunResult",
    "code_version",
    "decide",
    "run_discovery",
]

#: Baselines a run can score. Both are rule-free: ``hold`` never trades and
#: ``ret_5`` buys while the last bar's 5-bar momentum is positive.
BASELINES: Final[tuple[str, ...]] = ("hold", "ret_5")

#: Recorded policy labels. ``ret_5`` keeps the CLI's label so a trade report
#: from a discovery baseline cannot be mistaken for the learning agent's.
BASELINE_LABELS: Final[dict[str, str]] = {
    "hold": "hold",
    "ret_5": "rule:ret_5-baseline",
}

#: Modules whose source defines the arithmetic of a run. Their digest is the
#: default code version, so a fixed evaluator bug is a new experiment and an
#: unchanged tree replays the same trials.
_VERSION_MODULES: Final[tuple[str, ...]] = (
    "scalping.backtest.metrics",
    "scalping.datasets.builder",
    "scalping.discovery.archive",
    "scalping.discovery.evaluate",
    "scalping.discovery.run",
    "scalping.discovery.search",
    "scalping.env.simulator",
    "scalping.features.build",
    "scalping.strategy.dsl.canonical",
    "scalping.strategy.dsl.compile",
    "scalping.strategy.dsl.mutate",
    "scalping.strategy.dsl.nodes",
    "scalping.strategy.dsl.validate",
)

_HOLD_ACTION: Final[int] = 0
_BUY_ACTION: Final[int] = 1


class DiscoveryRunError(ValueError):
    """A run cannot be served from the store/config as asked; nothing is recorded."""


def _computed_code_version() -> str:
    """Digest the source of the modules a run's arithmetic depends on."""
    digest = hashlib.sha256()
    for name in _VERSION_MODULES:
        digest.update(name.encode("utf-8"))
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - every module is in this package
            digest.update(b"<missing>")
            continue
        source = getattr(module, "__file__", None)
        if source is None:  # pragma: no cover - namespace packages only
            digest.update(b"<no-file>")
            continue
        try:
            digest.update(Path(source).read_bytes())
        except OSError:  # pragma: no cover - unreadable install
            digest.update(b"<unreadable>")
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def code_version() -> str:
    """Stable 16-hex digest of the discovery arithmetic, computed once per process."""
    return _computed_code_version()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number!r}")
    return number


@dataclass(frozen=True, slots=True)
class DiscoveryRunConfig:
    """Everything one run needs: the slice, the search budget and the decision policy.

    ``archive`` is the SQLite evidence file; it is created and migrated on open.
    ``stress`` are simulator cost multipliers applied to every candidate (and
    required by the gate if ``gate.require_positive_cost_stress`` names them).
    ``baselines`` is the ordered list of rule-free comparisons; order also breaks
    a tie between equal baseline scores.
    """

    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    max_trials: int = 100
    population: int = 16
    generations: int = 5
    seed: int = 0
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    complexity_cap: int = 40
    archive: str | Path = "runs/discovery/discovery.sqlite"
    use_cache: bool = True
    stress: tuple[float, ...] = (1.5, 2.0)
    gate: GateCriteria = field(default_factory=GateCriteria)
    baselines: tuple[str, ...] = BASELINES
    horizon: int = 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    purge_bars: int = 0
    embargo_bars: int = 0
    window: int = 60
    patience: int | None = 5
    selection_metric: str = "net_profit"
    max_lookback: int = 20
    max_depth: int = 3
    code_version: str | None = None

    def __post_init__(self) -> None:
        _text(self.symbol, "symbol")
        _text(self.timeframe, "timeframe")
        _text(self.selection_metric, "selection_metric")
        object.__setattr__(self, "archive", str(Path(self.archive)))

        for name, minimum in (
            ("max_trials", 1),
            ("population", 2),
            ("generations", 1),
            ("complexity_cap", 1),
            ("window", 1),
            ("horizon", 1),
            ("max_lookback", 1),
            ("max_depth", 1),
        ):
            count = _count(getattr(self, name), name)
            if count < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {count}")
        if _count(self.seed, "seed") < 0:
            raise ValueError("seed must be >= 0")
        for name in ("purge_bars", "embargo_bars"):
            if _count(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.patience is not None:
            patience = _count(self.patience, "patience")
            if patience < 1:
                raise ValueError("patience must be >= 1 when set")

        for name in ("mutation_rate", "crossover_rate"):
            rate = _finite(getattr(self, name), name)
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {rate!r}")
        if float(self.mutation_rate) + float(self.crossover_rate) > 1.0:
            raise ValueError("mutation_rate + crossover_rate must be <= 1")
        if not 0.0 < float(self.train_fraction) < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if float(self.train_fraction) + float(self.validation_fraction) >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")

        if not isinstance(self.use_cache, bool):
            raise TypeError("use_cache must be a bool")
        if not isinstance(self.gate, GateCriteria):
            raise TypeError(f"gate must be a GateCriteria, got {type(self.gate).__name__}")
        if not isinstance(self.baselines, tuple) or not self.baselines:
            raise ValueError("baselines must be a non-empty tuple")
        if len(set(self.baselines)) != len(self.baselines):
            raise ValueError("baselines must be distinct")
        for name in self.baselines:
            if name not in BASELINES:
                raise ValueError(f"baseline must be one of {list(BASELINES)}, got {name!r}")

        multipliers: list[float] = []
        for raw in self.stress:
            number = _finite(raw, "stress")
            if number <= 0.0:
                raise ValueError(f"stress multipliers must be > 0, got {raw!r}")
            multipliers.append(number)
        if len({f"{value:g}" for value in multipliers}) != len(multipliers):
            raise ValueError("stress multipliers must be distinct")
        object.__setattr__(self, "stress", tuple(multipliers))

        if self.code_version is not None:
            _text(self.code_version, "code_version")


@dataclass(frozen=True, slots=True)
class DiscoveryRunResult:
    """The run's evidence and its decision, in a JSON-safe shape.

    ``best_canonical`` is the accepted candidate's own representation, so a
    reader can replay it without the archive. ``baselines`` is the ordered table
    that was scored on the same slice. ``comparison`` repeats the numbers the
    decision used, so a decision cannot be read out of context.
    """

    dataset_id: str
    symbol: str
    timeframe: str
    validation: dict[str, Any]
    seed: int
    code_version: str
    trials: int
    skipped: int
    deduped: int
    accepted: int
    stop_reason: str
    rejections: dict[str, int]
    best_id: str | None
    best_canonical: dict[str, Any] | None
    best_metrics: dict[str, float] | None
    baselines: tuple[dict[str, Any], ...]
    comparison: dict[str, Any]
    decision: str
    reason: str
    decision_id: int | None
    decision_candidate: str
    archive: str

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}, got {self.decision!r}")
        if self.decision == "promoted":
            raise ValueError(
                "a discovery run cannot record 'promoted'; the registry gate owns promotion"
            )
        if self.best_id is None and self.best_metrics is not None:
            raise ValueError("best_metrics require a best candidate")
        if self.best_id is not None and self.best_canonical is None:
            raise ValueError("a best candidate requires its canonical payload")

    def to_dict(self) -> dict[str, Any]:
        """One JSON-safe payload; the CLI prints it and ``--out`` writes it."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_id": self.dataset_id,
            "validation": dict(self.validation),
            "seed": self.seed,
            "code_version": self.code_version,
            "trials": self.trials,
            "skipped": self.skipped,
            "deduped": self.deduped,
            "accepted": self.accepted,
            "stop_reason": self.stop_reason,
            "rejections": dict(sorted(self.rejections.items())),
            "best": (
                None
                if self.best_id is None
                else {
                    "candidate_id": self.best_id,
                    "canonical": self.best_canonical,
                    "metrics": self.best_metrics,
                }
            ),
            "baseline": [dict(row) for row in self.baselines],
            "comparison": dict(self.comparison),
            "decision": self.decision,
            "reason": self.reason,
            "decision_id": self.decision_id,
            "decision_candidate": self.decision_candidate,
            "archive": self.archive,
        }


# --- validation slice ---------------------------------------------------------


def _iso(values: np.ndarray, index: int) -> str | None:
    """One timestamp as ISO-8601 UTC, or ``None`` for an empty array."""
    if values.size == 0 or index >= values.size or index < 0:
        return None
    stamp = np.datetime64(values[index], "us").astype(datetime)
    return stamp.replace(tzinfo=UTC).isoformat()


def _validation_slice(features: FeatureFrame, span: SplitRange) -> FeatureFrame:
    """Copy only ``[span.start, span.end)`` of a built frame.

    The slice keeps every feature column (the DSL addresses columns by name) and
    resets ``valid_from`` to 0: the builder starts the training split at the
    original warm-up boundary, so the validation rows are already complete. Its
    ``ts``/``ohlc``/``spread``/``atr`` are copied so the simulator sees the same
    quotes and stops the full frame would have produced.
    """
    rows = int(features.X.shape[0])
    if span.start < int(features.valid_from) or span.end > rows or span.end <= span.start:
        raise DiscoveryRunError(
            f"validation range [{span.start}, {span.end}) is not usable inside a "
            f"{rows}-row frame with valid_from={features.valid_from}"
        )
    return FeatureFrame(
        X=np.ascontiguousarray(features.X[span.start : span.end]),
        ts=np.asarray(features.ts[span.start : span.end]),
        ohlc=np.asarray(features.ohlc[span.start : span.end]),
        spread=np.asarray(features.spread[span.start : span.end]),
        atr=np.asarray(features.atr[span.start : span.end]),
        valid_from=0,
        columns=list(features.columns),
    )


# --- baselines ----------------------------------------------------------------


def _hold_policy(_obs: np.ndarray) -> int:
    """Control baseline: never trade."""
    return _HOLD_ACTION


def _ret5_policy(window: int) -> Callable[[np.ndarray], int]:
    """The CLI's ``ret_5`` rule for an arbitrary observation window.

    ``scalping.cli._rule_baseline`` hard-codes ``WINDOW = 60``; the formula is
    the same one its ``_RET5_INDEX`` uses. Reading the index from ``window``
    keeps the baseline identical even when a run shortens the window for a small
    validation slice.
    """
    index = (window - 1) * len(FEATURE_COLUMNS) + FEATURE_COLUMNS.index("ret_5")

    def policy(obs: np.ndarray) -> int:
        if obs.size <= index:
            return _HOLD_ACTION
        return _BUY_ACTION if float(obs[index]) > 0.0 else _HOLD_ACTION

    return policy


def _baseline_policy(name: str, window: int) -> Callable[[np.ndarray], int]:
    if name == "hold":
        return _hold_policy
    if name == "ret_5":
        return _ret5_policy(window)
    raise DiscoveryRunError(f"unknown baseline {name!r}")  # pragma: no cover - config guards


def _baselines(
    frame: FeatureFrame,
    settings: Settings,
    config: DiscoveryRunConfig,
) -> tuple[dict[str, Any], ...]:
    """Score every configured baseline on the exact validation slice."""
    rows: list[dict[str, Any]] = []
    for name in config.baselines:
        policy = _baseline_policy(name, config.window)
        label = BASELINE_LABELS[name]
        result = run_backtest(
            policy,
            frame,
            settings,
            policy_name=label,
            max_steps=len(frame.ts),
            window=config.window,
            symbol=config.symbol,
        )
        metrics = metrics_payload(result.metrics)
        selection = metrics.get(config.selection_metric)
        rows.append(
            {
                "name": name,
                "policy": label,
                "run_id": result.run_id,
                "metrics": metrics,
                "selection": (float(selection) if isinstance(selection, (int, float)) else None),
            }
        )
    return tuple(rows)


def _best_baseline(baselines: Sequence[Mapping[str, Any]]) -> tuple[str | None, float | None]:
    """Highest-scoring baseline; equal values keep the configured order."""
    best_name: str | None = None
    best_value: float | None = None
    for row in baselines:
        value = row.get("selection")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        if best_value is None or number > best_value:
            best_name, best_value = str(row["name"]), number
    return best_name, best_value


# --- decision -----------------------------------------------------------------


def decide(
    *,
    metric: str,
    trials: int,
    accepted: int,
    best_id: str | None,
    best_metrics: Mapping[str, float] | None,
    baselines: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """The decision policy, as a pure function of this run's numbers.

    Order matters and is part of the contract: a gate that rejected everything
    is ``no_trade`` even though the baseline comparison would be undefined; a
    flat accepted candidate is ``no_trade`` even when every baseline lost money;
    an improvement is ``rejected`` because discovery cannot promote. ``promoted``
    is never returned.
    """
    if trials > 0 and accepted == 0:
        return (
            "no_trade",
            (
                f"the gate rejected all {trials} trial(s); no candidate is eligible, "
                "so NO TRADE is the recorded result"
            ),
        )
    if best_id is None or best_metrics is None:
        return (
            "no_change",
            f"no accepted candidate was recorded in this run ({trials} trial(s)); nothing changes",
        )
    candidate = f"candidate {best_id[:12]}"
    trades = best_metrics.get("trades")
    value = best_metrics.get(metric)
    if trades is not None and float(trades) == 0.0:
        return (
            "no_trade",
            (
                f"{candidate} is flat by construction (0 closed trades); "
                "NO TRADE is the recorded result"
            ),
        )
    if value is None:
        return (
            "no_change",
            f"{candidate} did not publish the selection metric {metric!r}; nothing can be compared",
        )
    score = float(value)
    base_name, base_value = _best_baseline(baselines)
    if base_name is None or base_value is None:
        return (
            "no_change",
            (
                f"{candidate} has {metric} {score:g} but no baseline published that "
                "metric; nothing can be compared"
            ),
        )
    if score > base_value:
        return (
            "rejected",
            (
                f"{candidate} beats the best baseline ({base_name} {metric} {base_value:g}) "
                f"with {metric} {score:g} after {trials} trial(s); discovery never promotes, "
                "so the registry gate must decide"
            ),
        )
    return (
        "no_change",
        (
            f"{candidate} {metric} {score:g} does not beat the best baseline "
            f"({base_name} {metric} {base_value:g}) after {trials} trial(s)"
        ),
    )


# --- the run ------------------------------------------------------------------


def _search_config(
    config: DiscoveryRunConfig,
    dataset_id: str,
    version: str,
    *,
    dataset_checksum: str,
    settings_hash: str,
) -> SearchConfig:
    return SearchConfig(
        dataset_id=dataset_id,
        code_version=version,
        dataset_checksum=dataset_checksum,
        settings_digest=settings_hash,
        population=config.population,
        generations=config.generations,
        mutation_rate=config.mutation_rate,
        crossover_rate=config.crossover_rate,
        max_trials=config.max_trials,
        seed=config.seed,
        patience=config.patience,
        complexity_cap=config.complexity_cap,
        selection_metric=config.selection_metric,
    )


def _validation_payload(span: SplitRange, frame: FeatureFrame) -> dict[str, Any]:
    return {
        **span.to_dict(),
        "first_ts": _iso(frame.ts, 0),
        "last_ts": _iso(frame.ts, len(frame.ts) - 1),
    }


def _decision_target(search: SearchResult, archive: DiscoveryArchive, metric: str) -> str:
    """The candidate a decision is recorded against; every row needs one.

    The best accepted candidate of this run is the natural target. A run that
    recorded nothing new (all duplicates) still has to attach its ``no_change``
    to a row, so it falls back to the best candidate the archive already holds
    on any dataset. An empty archive and no trial is an error: there is nothing
    to decide about.
    """
    if search.best_id is not None:
        return search.best_id
    stored = archive.best(metric, minimum_trials=1, accepted_only=False)
    if stored is None:
        raise DiscoveryRunError(
            "no candidate was evaluated and the archive has none; there is nothing to decide"
        )
    return stored.candidate_id


def run_discovery(
    config: DiscoveryRunConfig | None = None,
    *,
    root: str | Path = "data",
    should_stop: Callable[[], bool] | None = None,
) -> DiscoveryRunResult:
    """Run one discovery experiment from stored bars and record its decision.

    ``root`` is the Parquet store root (the CLI's ``--root``); the feature cache
    lives under ``<root>/.feature-cache`` and is bypassed with ``use_cache=False``.
    ``should_stop`` is the cancellation hook passed to :func:`evolve`; a run that
    stops early is recorded with ``stop_reason="stopped"`` and still decides.
    """
    cfg = DiscoveryRunConfig() if config is None else config
    if not isinstance(cfg, DiscoveryRunConfig):
        raise TypeError(f"config must be a DiscoveryRunConfig, got {type(cfg).__name__}")

    store = ParquetStore(root)
    bars = store.read(cfg.symbol, cfg.timeframe)
    if bars.is_empty():
        raise DiscoveryRunError(
            f"no {cfg.symbol} {cfg.timeframe} bars under {store.root}; ingest bars first"
        )

    settings = Settings()
    cache = FeatureCache(Path(root) / FEATURE_CACHE_DIRNAME) if cfg.use_cache else None
    features = cached_build_features(
        cache,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        frame=bars,
        settings=settings,
    )

    try:
        bundle = _dataset(features, cfg)
    except ValueError as exc:
        raise DiscoveryRunError(f"cannot build a chronological dataset: {exc}") from exc
    span = bundle.manifest.validation
    if span.rows <= 0:
        raise DiscoveryRunError(
            "the built dataset has no validation rows; discovery may not read the test "
            "split, so there is nothing to search on"
        )
    if span.rows <= cfg.window:
        raise DiscoveryRunError(
            f"the validation split has {span.rows} rows; a {cfg.window}-bar observation "
            f"window needs at least {cfg.window + 1} to make one decision"
        )

    frame = _validation_slice(features, span)
    dataset_id = f"validation:{bundle.manifest.dataset_id}"
    version = cfg.code_version or code_version()

    evaluator = DslEvaluator(
        frame,
        settings,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        dataset_id=dataset_id,
        gate=cfg.gate,
        window=cfg.window,
        stress=cfg.stress,
        max_steps=len(frame.ts),
    )
    search_config = _search_config(
        cfg,
        dataset_id,
        version,
        dataset_checksum=bundle.manifest.checksum,
        settings_hash=settings_digest(settings),
    )
    # Structural protocols take ``Candidate``/``Factory``; this run only ever
    # produces ``StrategySpec``, so the narrower factory satisfies the loop at
    # runtime and the cast records that invariant for the type checker.
    factory = cast(Factory, StrategyFactory(max_lookback=cfg.max_lookback, max_depth=cfg.max_depth))

    baselines = _baselines(frame, settings, cfg)

    with DiscoveryArchive(cfg.archive) as archive:
        search = evolve(factory, evaluator, archive, search_config, should_stop=should_stop)

        decision, reason = decide(
            metric=cfg.selection_metric,
            trials=search.trials,
            accepted=search.accepted,
            best_id=search.best_id,
            best_metrics=search.best_metrics,
            baselines=baselines,
        )
        target = _decision_target(search, archive, cfg.selection_metric)
        record = archive.record_decision(target, decision, reason)

    best_canonical = None if search.best is None else search.best.canonical_dict()
    best_value = (
        None if search.best_metrics is None else search.best_metrics.get(cfg.selection_metric)
    )
    base_name, base_value = _best_baseline(baselines)
    comparison: dict[str, Any] = {
        "metric": cfg.selection_metric,
        "best_candidate": None if best_value is None else float(best_value),
        "best_baseline": base_name,
        "best_baseline_value": base_value,
        "beats_baseline": (
            None if best_value is None or base_value is None else float(best_value) > base_value
        ),
        "trials": search.trials,
        "accepted": search.accepted,
    }

    return DiscoveryRunResult(
        dataset_id=dataset_id,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        validation=_validation_payload(span, frame),
        seed=cfg.seed,
        code_version=version,
        trials=search.trials,
        skipped=search.skipped,
        deduped=search.skipped_reasons.get("duplicate", 0),
        accepted=search.accepted,
        stop_reason=search.stop_reason,
        rejections=dict(search.rejections),
        best_id=search.best_id,
        best_canonical=best_canonical,
        best_metrics=None if search.best_metrics is None else dict(search.best_metrics),
        baselines=baselines,
        comparison=comparison,
        decision=decision,
        reason=reason,
        decision_id=record.id,
        decision_candidate=target,
        archive=str(cfg.archive),
    )


def _dataset(features: FeatureFrame, cfg: DiscoveryRunConfig) -> DatasetBundle:
    """Build the chronological split; kept separate so errors name the geometry."""
    return build_dataset(
        features,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        horizon=cfg.horizon,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
        purge_bars=cfg.purge_bars,
        embargo_bars=cfg.embargo_bars,
    )
