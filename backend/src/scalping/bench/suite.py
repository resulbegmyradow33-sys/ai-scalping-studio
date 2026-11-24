"""The M2 measurement suite: named stages, one JSON report, explicit claims.

``run_suite`` is the only writer. It measures each :class:`Stage`, builds the
report and writes it atomically to ``out_path``. The report has four parts:

``environment``
    Python version and implementation, platform string, machine, CPU count and
    total RAM in MiB, all read through the standard library.
``stages``
    One :meth:`StageMeasurement.to_dict` per stage, in the order given. A
    failed stage is present with ``ok=false``; it does not erase the others.
``claims``
    A fixed block that says these are local measurements on this machine at
    this time, not validation, not profitability and not M2 acceptance.
``summary``
    Counts plus the sum of per-stage median wall time and the largest RSS
    delta, for a quick read. The per-stage rows are the evidence; the summary
    is only a convenience.

Built-in stage factories
------------------------
Every factory accepts a ``store_root`` and a ``real_store`` flag. The flag is
``False`` by default and then **no read of ``store_root`` happens at all**:
bars come from :class:`~scalping.data.synthetic.SyntheticProvider`, or from a
deterministic flat tape for the evidence stage. Tests therefore never need the
real 17-year store. ``real_store=True`` reads (or copies) real bars and is an
explicit opt-in.

Real-store reads are bounded by ``rows``. ``None`` means a conservative sample
(20 000 bars for most stages, 200 000 for walk-forward); ``0`` means the whole
stored series. The full store can exceed the 8 GB target, so read
``docs/mac-mini-optimization.md`` before using ``rows=0``.

None of the factories is an acceptance claim. ``features_build``,
``dataset_build``, ``evidence_build`` and ``walk_forward`` compute in memory;
``discovery_smoke`` and ``paper_advance`` need a writable store, which the
factory creates under a temporary directory and removes when the suite ends.
The real store itself is never written to.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from scalping.backtest.evaluate import evaluate_walkforward
from scalping.bench.pipeline import (
    StageError,
    StageMeasurement,
    measure_stage,
)
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.builder import build_dataset
from scalping.discovery.evidence import EvidenceConfig, build_evidence
from scalping.discovery.run import DiscoveryRunConfig, run_discovery
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame, build_features
from scalping.orchestrator.resources import SystemProbe
from scalping.paper.continuous import ContinuousPaperRunner
from scalping.paper.ledger import PaperLedger
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

__all__ = [
    "CLAIMS_SCHEMA",
    "REPORT_SCHEMA",
    "Stage",
    "StageError",
    "dataset_build_stage",
    "discovery_smoke_stage",
    "evidence_build_stage",
    "features_build_stage",
    "paper_advance_stage",
    "run_suite",
    "walk_forward_stage",
]

REPORT_SCHEMA: Final[str] = "scalping.bench.suite/1"
CLAIMS_SCHEMA: Final[str] = "scalping.bench.claims/1"

#: Synthetic timestamps start on a Monday so session-gated stages have a full week.
BENCH_START: Final[datetime] = datetime(2026, 1, 5, tzinfo=UTC)

#: Conservative real-store sample sizes. ``rows=0`` opts into the whole store.
_DEFAULT_REAL_ROWS: Final[int] = 20_000
_DEFAULT_REAL_WALK_FORWARD_ROWS: Final[int] = 200_000

_ALWAYS = Compare(Const(0.0), "<", Const(1.0))
_NEVER = Compare(Const(0.0), ">", Const(1.0))


@dataclass(frozen=True, slots=True)
class Stage:
    """One named measurement plus the cleanup its factory owns.

    ``data_source`` is ``"synthetic"``, ``"real_store"`` or ``"custom"`` and is
    copied into the report so a reader knows what produced the numbers.
    ``cleanup`` is called exactly once when :func:`run_suite` finishes, even if
    another stage failed. The stage function is called once per warmup and once
    per measured repeat.
    """

    name: str
    fn: Callable[[], object]
    repeats: int = 1
    warmup: int = 0
    disk_path: Path | None = None
    data_source: str = "custom"
    cleanup: Callable[[], None] | None = None


def _environment() -> dict[str, object]:
    """Machine and interpreter identity, all from the standard library."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "total_ram_mb": round(float(SystemProbe.total_ram_mb()), 1),
    }


def _claims(data_sources: Sequence[str]) -> dict[str, object]:
    return {
        "schema": CLAIMS_SCHEMA,
        "kind": "local_measurement",
        "data_sources": sorted(set(data_sources)),
        "validation": False,
        "profitability": False,
        "m2_acceptance": False,
        "text": (
            "These are local wall/CPU/RSS/disk measurements on one machine at one "
            "time. They do not validate a strategy, prove profitability or satisfy "
            "M2 acceptance. Load, thermal state, cache warmth and other processes "
            "change the numbers; compare runs on the same idle machine instead."
        ),
    }


def _summary(measurements: Sequence[StageMeasurement]) -> dict[str, object]:
    deltas = [m.rss_delta_mb for m in measurements if m.rss_delta_mb is not None]
    return {
        "stages": len(measurements),
        "ok": sum(1 for m in measurements if m.ok),
        "failed": sum(1 for m in measurements if not m.ok),
        "wall_seconds_median_total": round(sum(m.wall_seconds_median for m in measurements), 6),
        "max_rss_delta_mb": None if not deltas else round(max(deltas), 3),
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def run_suite(
    stages: Sequence[Stage],
    out_path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, object]:
    """Measure every stage, write one JSON report and return it.

    A stage failure is recorded with ``ok=false`` and later stages still run.
    With ``strict=True`` the first failed stage raises :class:`StageError`; the
    report is not written in that case. Cleanup callbacks always run.
    """
    items = list(stages)
    if not items:
        raise ValueError("stages must not be empty")
    names = [stage.name for stage in items]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"stage names must be unique, duplicated: {duplicates}")
    for stage in items:
        if not isinstance(stage, Stage):
            raise TypeError(f"stages must contain Stage, got {type(stage).__name__}")

    measurements: list[StageMeasurement] = []
    try:
        for stage in items:
            measurement = measure_stage(
                stage.name,
                stage.fn,
                repeats=stage.repeats,
                warmup=stage.warmup,
                disk_path=stage.disk_path,
                strict=strict,
            )
            measurements.append(measurement)
    finally:
        for stage in items:
            if stage.cleanup is not None:
                with contextlib.suppress(Exception):
                    stage.cleanup()

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(),
        "claims": _claims([stage.data_source for stage in items]),
        "stages": [measurement.to_dict() for measurement in measurements],
        "summary": _summary(measurements),
    }
    _write_report(Path(out_path), report)
    return report


# --------------------------------------------------------------------- helpers


def _resolve_rows(
    rows: int | None,
    *,
    real_store: bool,
    synthetic_default: int,
    real_default: int = _DEFAULT_REAL_ROWS,
) -> int:
    """Resolve ``rows`` for one mode; ``0`` means all real bars, never zero synthetic."""
    if rows is not None:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"rows must be an int >= 0 or None, got {rows!r}")
        if rows == 0 and not real_store:
            return synthetic_default
        return rows
    return real_default if real_store else synthetic_default


def _synthetic_frame(symbol: str, timeframe: str, rows: int, seed: int) -> pl.DataFrame:
    if timeframe != "M1":
        raise ValueError(
            f"synthetic bars are M1 only, got timeframe={timeframe!r}; pass real_store=True"
        )
    if rows <= 0:
        raise ValueError("synthetic rows must be > 0")
    provider = SyntheticProvider(seed=seed)
    return provider.fetch(symbol, BENCH_START, BENCH_START + timedelta(minutes=rows))


def _real_frame(
    store_root: str | Path,
    symbol: str,
    timeframe: str,
    rows: int,
) -> pl.DataFrame:
    frame = ParquetStore(store_root).read(symbol, timeframe)
    if frame.is_empty():
        raise FileNotFoundError(
            f"no {symbol} {timeframe} bars under {store_root}; ingest data or omit --real-store"
        )
    return frame if rows == 0 else frame.head(rows)


def _source_frame(
    store_root: str | Path,
    symbol: str,
    timeframe: str,
    *,
    real_store: bool,
    rows: int | None,
    synthetic_rows: int,
    seed: int,
) -> pl.DataFrame:
    resolved = _resolve_rows(rows, real_store=real_store, synthetic_default=synthetic_rows)
    if real_store:
        return _real_frame(store_root, symbol, timeframe, resolved)
    return _synthetic_frame(symbol, timeframe, resolved, seed)


def _temp_store(
    frame: pl.DataFrame, symbol: str, timeframe: str
) -> tuple[Path, Callable[[], None]]:
    """Write ``frame`` into a fresh temp store; return its root and cleanup."""
    holder = tempfile.TemporaryDirectory(prefix="scalping-bench-")
    root = Path(holder.name)
    ParquetStore(root).write(symbol, timeframe, frame)
    return root, holder.cleanup


def _flat_bars(rows: int, *, price: float = 1000.0) -> pl.DataFrame:
    """Deterministic flat M1 tape with a 0.20 spread; every round turn pays costs."""
    stamps = [BENCH_START + timedelta(minutes=index) for index in range(rows)]
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": [price] * rows,
            "high": [price + 0.001] * rows,
            "low": [price - 0.001] * rows,
            "close": [price] * rows,
            "bid_close": [price] * rows,
            "ask_close": [price + 0.2] * rows,
            "spread": [0.2] * rows,
            "volume": [10] * rows,
        }
    )


def _long_spec(*, holding: int = 10) -> StrategySpec:
    """Always enter long when flat; the holding limit closes the position."""
    return StrategySpec(
        entries=(EntryRule("long", _ALWAYS),),
        exits=(ExitRule(_NEVER),),
        holding=HoldingLimit(holding),
        max_lookback=2,
    )


# --------------------------------------------------------------------- stages


def features_build_stage(
    store_root: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    real_store: bool = False,
    rows: int | None = None,
    seed: int = 7,
    repeats: int = 1,
    warmup: int = 0,
) -> Stage:
    """Time ``build_features`` on a synthetic slice (default) or real bars."""
    frame = _source_frame(
        store_root,
        symbol,
        timeframe,
        real_store=real_store,
        rows=rows,
        synthetic_rows=480,
        seed=seed,
    )

    def run() -> dict[str, int]:
        features: FeatureFrame = build_features(frame)
        return {"rows": int(features.X.shape[0])}

    return Stage(
        "features_build",
        run,
        repeats=repeats,
        warmup=warmup,
        data_source="real_store" if real_store else "synthetic",
    )


def dataset_build_stage(
    store_root: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    real_store: bool = False,
    rows: int | None = None,
    seed: int = 7,
    repeats: int = 1,
    warmup: int = 0,
) -> Stage:
    """Time feature build plus the chronological dataset cut.

    The dataset needs ``FeatureFrame``, so feature extraction is part of this
    stage's cost. It is intentionally not deduplicated against
    ``features_build``: each stage answers "what does this pipeline segment
    cost", and a shared cache would make the two numbers incomparable.
    """
    frame = _source_frame(
        store_root,
        symbol,
        timeframe,
        real_store=real_store,
        rows=rows,
        synthetic_rows=480,
        seed=seed,
    )

    def run() -> dict[str, int]:
        features = build_features(frame)
        bundle = build_dataset(features, symbol=symbol, timeframe=timeframe)
        return {"rows": int(bundle.manifest.rows)}

    return Stage(
        "dataset_build",
        run,
        repeats=repeats,
        warmup=warmup,
        data_source="real_store" if real_store else "synthetic",
    )


def discovery_smoke_stage(
    store_root: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    real_store: bool = False,
    rows: int | None = None,
    seed: int = 0,
    max_trials: int = 2,
    repeats: int = 1,
    warmup: int = 0,
) -> Stage:
    """Time a tiny bounded discovery run: feature build, dataset, two trials.

    ``max_trials`` defaults to 2 and the population to 2, so this is a smoke
    measurement of the pipeline, not a search experiment. The archive is a
    fresh temporary file per repeat; otherwise later repeats would deduplicate
    every trial and measure an empty run. A real-store read uses a temporary
    copy of the sampled bars and never writes to the source store.
    """
    frame = _source_frame(
        store_root,
        symbol,
        timeframe,
        real_store=real_store,
        rows=rows,
        synthetic_rows=1200,
        seed=seed,
    )
    root, cleanup = _temp_store(frame, symbol, timeframe)
    archive_dir = root / "archive"

    def run() -> dict[str, int]:
        shutil.rmtree(archive_dir, ignore_errors=True)
        archive_dir.mkdir(parents=True, exist_ok=True)
        config = DiscoveryRunConfig(
            symbol=symbol,
            timeframe=timeframe,
            max_trials=max_trials,
            population=2,
            generations=1,
            seed=seed,
            archive=archive_dir / "discovery.sqlite",
            use_cache=False,
            patience=None,
        )
        result = run_discovery(config, root=root)
        return {"rows": int(result.validation["rows"])}

    return Stage(
        "discovery_smoke",
        run,
        repeats=repeats,
        warmup=warmup,
        data_source="real_store" if real_store else "synthetic",
        cleanup=cleanup,
    )


def evidence_build_stage(
    store_root: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    real_store: bool = False,
    rows: int | None = None,
    seed: int = 7,
    folds: int = 3,
    repeats: int = 1,
    warmup: int = 0,
) -> Stage:
    """Time a tiny evidence bundle: features, dataset, ``folds`` fold evaluations.

    The default is a flat tape and ``folds=3`` (the evidence minimum), one
    bootstrap sample per fold, one required trade per fold and no regime
    coverage requirement. It exercises the fold geometry and cost arithmetic,
    not a statistically meaningful bundle.
    """
    if folds < 3:
        raise ValueError("folds must be >= 3; the evidence builder refuses fewer")
    if real_store:
        frame = _source_frame(
            store_root,
            symbol,
            timeframe,
            real_store=True,
            rows=rows,
            synthetic_rows=1200,
            seed=seed,
        )
    else:
        if timeframe != "M1":
            raise ValueError(f"the flat evidence tape is M1 only, got timeframe={timeframe!r}")
        frame = _flat_bars(_resolve_rows(rows, real_store=False, synthetic_default=1200))

    def run() -> dict[str, int]:
        features = build_features(frame)
        bundle = build_dataset(
            features,
            symbol=symbol,
            timeframe=timeframe,
            horizon=1,
            train_fraction=0.6,
            validation_fraction=0.2,
            purge_bars=3,
            embargo_bars=2,
        )
        config = EvidenceConfig(
            code_version="bench-evidence-v1",
            symbol=symbol,
            timeframe=timeframe,
            folds=folds,
            min_trades_per_fold=1,
            cost_stress=(1.5, 2.0),
            regime_coverage_min=0.0,
            bootstrap_samples=100,
            seed=seed,
            search_budget=50,
            trial_count=4,
        )
        evidence = build_evidence(bundle, _long_spec(holding=1), config, features=features)
        return {"rows": int(bundle.manifest.rows), "folds": len(evidence.folds)}

    return Stage(
        "evidence_build",
        run,
        repeats=repeats,
        warmup=warmup,
        data_source="real_store" if real_store else "synthetic",
    )


def paper_advance_stage(
    store_root: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    real_store: bool = False,
    rows: int | None = None,
    seed: int = 7,
    repeats: int = 1,
    warmup: int = 0,
) -> Stage:
    """Time one ``ContinuousPaperRunner.advance`` over a small stored slice.

    Each repeat uses a fresh in-memory ledger, so it processes the whole slice
    instead of resuming at the previous cursor. The store is a temporary copy
    of the sample; the source store is never modified. This measures one paper
    advance, not a long-running paper session.
    """
    frame = _source_frame(
        store_root,
        symbol,
        timeframe,
        real_store=real_store,
        rows=rows,
        synthetic_rows=400,
        seed=seed,
    )
    root, cleanup = _temp_store(frame, symbol, timeframe)
    settings = Settings()

    def run() -> dict[str, int]:
        with PaperLedger(":memory:") as ledger:
            runner = ContinuousPaperRunner(
                ledger,
                _long_spec(holding=10),
                symbol,
                timeframe,
                settings,
                store_root=root,
                cache=False,
            )
            summary = runner.advance()
            return {"rows": int(summary.bars_processed)}

    return Stage(
        "paper_advance",
        run,
        repeats=repeats,
        warmup=warmup,
        data_source="real_store" if real_store else "synthetic",
        cleanup=cleanup,
    )


def walk_forward_stage(
    store_root: str | Path,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    real_store: bool = False,
    rows: int | None = None,
    seed: int = 7,
    folds: int = 1,
    repeats: int = 1,
    warmup: int = 0,
) -> Stage:
    """Time ``evaluate_walkforward`` with a ``ret_5`` policy; optional and slower.

    Synthetic history is 100 days, enough for one one-month fold. Real mode
    defaults to 200 000 bars; ``rows=0`` uses the whole store and can be slow
    and memory-hungry. This is a calendar-split walk-forward evaluation, not
    promotion evidence and not a locked OOS test.
    """
    resolved = _resolve_rows(
        rows,
        real_store=real_store,
        synthetic_default=100 * 1440,
        real_default=_DEFAULT_REAL_WALK_FORWARD_ROWS,
    )
    if real_store:
        frame = _real_frame(store_root, symbol, timeframe, resolved)
    else:
        frame = _synthetic_frame(symbol, timeframe, resolved, seed)

    index = (60 - 1) * len(FEATURE_COLUMNS) + FEATURE_COLUMNS.index("ret_5")

    def policy(observation: np.ndarray) -> int:
        if observation.size <= index:
            return 0
        return 1 if float(observation[index]) > 0.0 else 0

    def run() -> dict[str, int]:
        report = evaluate_walkforward(
            policy,
            frame,
            Settings(),
            train_months=1,
            folds=folds,
        )
        return {"rows": len(frame), "folds": len(report.folds)}

    return Stage(
        "walk_forward",
        run,
        repeats=repeats,
        warmup=warmup,
        data_source="real_store" if real_store else "synthetic",
        cleanup=None,
    )
