"""Command line interface for data ingestion, backtests and the API.

Every command prints exactly one JSON line to stdout on success. Errors print
one JSON line to stderr: exit code 2 for data problems (missing file, bad CSV,
no stored bars) and 1 for unexpected failures. That makes the CLI scriptable:
stdout is always parseable JSON and never mixed with progress or log output.

The ``rule`` backtest policy is a **baseline**, not the learning agent: it buys
while the latest bar's 5-bar momentum (the ``ret_5`` feature) is positive and
holds otherwise. Its run label, ``rule:ret_5-baseline``, says so in the report
and on every trade it opens.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import socket
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

import numpy as np
import polars as pl

from scalping.backtest.engine import BacktestResult, run_backtest
from scalping.backtest.montecarlo import MonteCarloConfig, run_montecarlo
from scalping.backtest.sensitivity import (
    COMMISSION_AXIS,
    SLIPPAGE_AXIS,
    SPREAD_AXIS,
    SensitivityRow,
    parameter_sensitivity,
)
from scalping.config import Settings
from scalping.data.csv_provider import CsvProvider, DataError
from scalping.data.json_provider import JsonProvider
from scalping.data.provenance import build_provenance, save_provenance
from scalping.data.registry import TIMEFRAMES, timeframe_rule
from scalping.data.resample import resample
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.data.validate import validate_bars
from scalping.datasets.builder import build_dataset, save_manifest
from scalping.discovery.evidence import CandidateEvidence, EvidenceConfig, EvidenceError
from scalping.discovery.persist import (
    PersistError,
    load_strategy_spec,
    persist_evidence_run,
)
from scalping.discovery.reproduce import (
    ReproduceError,
    ReproduceMismatch,
    reproduce_bundle,
)
from scalping.discovery.run import (
    DiscoveryRunConfig,
    DiscoveryRunError,
    run_discovery,
)
from scalping.discovery.run import (
    code_version as discovery_code_version,
)
from scalping.env.simulator import WINDOW
from scalping.features.build import FEATURE_COLUMNS, WARMUP, FeatureFrame
from scalping.features.cache import FEATURE_CACHE_DIRNAME, FeatureCache, cached_build_features
from scalping.features.registry import DEFAULT_GROUPS
from scalping.ml.cv import CVConfig, cross_validate
from scalping.ml.evaluate import evaluate_model
from scalping.ml.train import TrainSpec, load_result, save_result, train_classifier
from scalping.orchestrator.triggers import FileTriggerSource, Trigger, TriggerSource
from scalping.registry.gate import GateCriteria
from scalping.serialize import metrics_payload

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    from scalping.orchestrator.daemon import OrchestratorDaemon

__all__ = ["main"]

#: Timeframe every command reads and writes.
TIMEFRAME: Final[str] = "M1"

#: Default store root, matching the API factory and the Makefile.
DEFAULT_ROOT: Final[Path] = Path("data")

#: Symbol used when the caller does not name one.
DEFAULT_SYMBOL: Final[str] = "XAUUSD"

#: Help text of the repeated ``--no-cache`` flag (dataset, ML and backtest paths).
_NO_CACHE_HELP: Final[str] = "rebuild the feature matrix instead of reusing <root>/.feature-cache"

_POLICY_HOLD: Final[str] = "hold"
_POLICY_RULE: Final[str] = "rule"

_HOLD: Final[int] = 0
_BUY: Final[int] = 1

#: Explicit label so a baseline trade can never be mistaken for the agent's.
RULE_LABEL: Final[str] = "rule:ret_5-baseline"

#: Model names accepted by ``nn train``. Kept here (not imported from
#: ``scalping.nn``) so data commands do not import torch; a test asserts it
#: matches ``scalping.nn.MODEL_NAMES``.
_NN_MODELS: Final[tuple[str, ...]] = (
    "mlp",
    "deep_mlp",
    "residual_mlp",
    "rnn",
    "lstm",
    "gru",
    "cnn",
    "dilated_cnn",
    "cnn_lstm",
    "tcn",
    "transformer",
    "causal_transformer",
    "patchtst",
    "itransformer",
)

#: Bars needed before the simulator has its first decision bar: the feature
#: warm-up plus one full observation window. Mirrors ``api.runner.MIN_BARS``.
_MIN_BARS: Final[int] = WARMUP + WINDOW

#: Where the last closed bar's ``ret_5`` sits in a flattened observation.
_RET5_INDEX: Final[int] = (WINDOW - 1) * len(FEATURE_COLUMNS) + FEATURE_COLUMNS.index("ret_5")


class CliError(ValueError):
    """A request that cannot be served because of the input data or range."""


class _ExitCode(Exception):
    """The handler already printed its JSON line and only sets the exit code.

    A failed validation report is a normal, inspectable result: the report goes
    to stdout like every other command, but the process exits non-zero so a
    script can gate on it. ``main`` maps this to the code without adding an
    error line.
    """

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


# --- JSON output -------------------------------------------------------------
def _json_default(value: object) -> Any:
    """Serialise the non-JSON-native values the report contains."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _print_json(payload: dict[str, Any]) -> None:
    """Write one compact JSON line to stdout and flush it."""
    print(json.dumps(payload, default=_json_default), file=sys.stdout, flush=True)


def _print_error(message: str, kind: str | None = None) -> None:
    """Write one JSON line to stderr; the caller owns the exit code."""
    payload: dict[str, Any] = {"error": message}
    if kind is not None:
        payload["type"] = kind
    print(json.dumps(payload, default=_json_default), file=sys.stderr, flush=True)


class _Parser(argparse.ArgumentParser):
    """Argument parser whose usage errors obey the one-JSON-line contract."""

    def error(self, message: str) -> NoReturn:
        _print_error(message)
        raise SystemExit(2)


# --- argument types ----------------------------------------------------------
def _symbol(text: str) -> str:
    symbol = text.strip()
    if not symbol:
        raise argparse.ArgumentTypeError("symbol must be non-empty")
    return symbol


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a YYYY-MM-DD date") from None


# --- data commands -----------------------------------------------------------
def _summary(symbol: str, store: ParquetStore) -> dict[str, Any]:
    """Store status after an ingest, in the API's data-status shape."""
    coverage = store.coverage(symbol, TIMEFRAME)
    return {
        "rows": coverage.rows,
        "symbol": symbol,
        "first_ts": None if coverage.first_ts is None else coverage.first_ts.isoformat(),
        "last_ts": None if coverage.last_ts is None else coverage.last_ts.isoformat(),
    }


def _ingest(
    frame: pl.DataFrame,
    symbol: str,
    store: ParquetStore,
    *,
    source: str,
    synthetic: bool = False,
    timeframe: str = TIMEFRAME,
) -> dict[str, Any]:
    if frame.is_empty():
        raise CliError(f"no bars for {symbol} in the requested range")
    store.write(symbol, timeframe, frame)
    provenance = build_provenance(
        frame,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        synthetic=synthetic,
    )
    save_provenance(store.root, provenance)
    summary = _summary(symbol, store)
    summary["provenance"] = provenance.to_dict()
    return summary


def _cmd_data_synth(args: argparse.Namespace) -> None:
    """Generate deterministic synthetic bars and merge them into the store."""
    start = datetime.combine(args.start, time.min, tzinfo=UTC)
    # A date-only ``--end`` covers the whole UTC day, so ``--start X --end X``
    # is one complete day rather than a single midnight bar.
    end = datetime.combine(args.end, time(23, 59), tzinfo=UTC)
    if start > end:
        raise CliError(f"--start {args.start} is after --end {args.end}")
    provider = SyntheticProvider(seed=args.seed, start_price=args.start_price)
    frame = provider.fetch(args.symbol, start, end)
    store = ParquetStore(args.root)
    _print_json(_ingest(frame, args.symbol, store, source="synthetic", synthetic=True))


def _cmd_data_csv(args: argparse.Namespace) -> None:
    """Read a broker CSV export and merge it into the store."""
    path = Path(args.path)
    provider = CsvProvider(path, tz_offset_hours=args.tz_offset)
    try:
        frame = provider.fetch(args.symbol)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise CliError(f"cannot read {path}: {reason}") from exc
    except UnicodeError as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise CliError(f"cannot decode {path} as UTF-8: {reason}") from exc
    store = ParquetStore(args.root)
    _print_json(_ingest(frame, args.symbol, store, source=f"csv:{path.name}"))


def _cmd_data_json(args: argparse.Namespace) -> None:
    """Read a JSON bar array and merge it into the store."""
    path = Path(args.path)
    provider = JsonProvider(path)
    try:
        frame = provider.fetch(args.symbol)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise CliError(f"cannot read {path}: {reason}") from exc
    except (ValueError, UnicodeError) as exc:
        raise CliError(str(exc)) from exc
    store = ParquetStore(args.root)
    _print_json(_ingest(frame, args.symbol, store, source=f"json:{path.name}"))


def _cmd_data_derive(args: argparse.Namespace) -> None:
    """Derive M5–D1 partitions from a stored source timeframe."""
    store = ParquetStore(args.root)
    source = store.read(args.symbol, args.source)
    if source.is_empty():
        raise CliError(f"no {args.symbol} {args.source} bars under {args.root}")
    written: dict[str, int] = {}
    for timeframe in args.timeframes.split(","):
        key = timeframe.strip().upper()
        if key not in TIMEFRAMES or key == args.source:
            continue
        frame = resample(source, timeframe_rule(key)).rename({"spread_mean": "spread"})
        rows = store.write(args.symbol, key, frame)
        provenance = build_provenance(
            frame,
            symbol=args.symbol,
            timeframe=key,
            source=f"derived:{args.source}",
        )
        save_provenance(store.root, provenance)
        written[key] = rows
    _print_json({"symbol": args.symbol, "source": args.source, "written": written})


def _cmd_data_validate(args: argparse.Namespace) -> None:
    """Print the structural validation report for one stored slice.

    The report is the command's normal output, so it goes to stdout even when
    it fails. A structural error only changes the exit code to 2, which lets a
    pipeline stop on bad data without parsing the message.
    """
    store = ParquetStore(args.root)
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise CliError(f"no {args.symbol} {args.timeframe} bars under {args.root}")
    report = validate_bars(frame, symbol=args.symbol, timeframe=args.timeframe)
    _print_json(report.to_dict())
    if not report.ok:
        raise _ExitCode(2)


# --- neural networks ---------------------------------------------------------
def _nn_data(args: argparse.Namespace, *, window: int, clip: float = 5.0):
    """Build windowed tensors for one stored symbol/timeframe."""
    from scalping.datasets.builder import build_dataset
    from scalping.features.build import build_features
    from scalping.nn import WindowConfig, build_windowed_dataset

    store = ParquetStore(args.root)
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise CliError(f"no {args.symbol} {args.timeframe} bars under {args.root}")
    features = build_features(frame, Settings().costs)
    bundle = build_dataset(
        features,
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_bars=args.purge,
    )
    data = build_windowed_dataset(bundle, WindowConfig(window=window, clip=clip))
    return bundle, data


def _cmd_nn_train(args: argparse.Namespace) -> None:
    """Fit one sequence model and write a resumable checkpoint."""
    from scalping.nn import TrainConfig, select_device, train_classifier

    _, data = _nn_data(args, window=args.window, clip=args.clip)
    checkpoint = Path(args.checkpoint)
    config = TrainConfig(
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden=args.hidden,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    result = train_classifier(data, config, checkpoint=checkpoint)
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "dataset_id": data.dataset_id,
        "counts": data.counts(),
        "window": data.window,
        "device": select_device(args.device).type,
        **result.to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    _print_json(
        {
            "checkpoint": str(checkpoint),
            "out": str(out),
            "dataset_id": data.dataset_id,
            "best_epoch": result.best_epoch,
            "val_accuracy": result.val_accuracy,
            "stopped": result.stopped,
            "device": payload["device"],
            "counts": data.counts(),
        }
    )


def _cmd_nn_evaluate(args: argparse.Namespace) -> None:
    """Score a checkpoint on one split of a freshly rebuilt dataset."""
    from scalping.nn import evaluate_checkpoint, load_checkpoint

    payload = load_checkpoint(args.checkpoint)
    _, data = _nn_data(args, window=int(payload["window"]), clip=args.clip)
    report = evaluate_checkpoint(args.checkpoint, data, split=args.split, device=args.device)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    metrics = report["metrics"]
    _print_json(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "rows": metrics["rows"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "log_loss": metrics["log_loss"],
            "brier": metrics["brier"],
            "dataset_id": report["dataset_id"],
            "trading_claim": report["trading_claim"],
        }
    )


def _feature_cache(args: argparse.Namespace) -> FeatureCache | None:
    """The disk feature cache for this command, or ``None`` with ``--no-cache``."""
    if getattr(args, "no_cache", False):
        return None
    return FeatureCache(Path(args.root) / FEATURE_CACHE_DIRNAME)


def _features_for(
    args: argparse.Namespace, frame: pl.DataFrame, symbol: str, timeframe: str
) -> FeatureFrame:
    """Build the matrix through the cache; a hit is bit-identical to a build."""
    return cached_build_features(
        _feature_cache(args),
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        settings=Settings(),
    )


def _cmd_dataset_build(args: argparse.Namespace) -> None:
    """Build a chronological, purged dataset manifest from a stored symbol."""
    store = ParquetStore(args.root)
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise CliError(f"no {args.symbol} {args.timeframe} bars under {args.root}")
    features = _features_for(args, frame, args.symbol, args.timeframe)
    groups = (
        None
        if args.groups == "all"
        else tuple(part.strip() for part in args.groups.split(",") if part.strip())
    )
    bundle = build_dataset(
        features,
        symbol=args.symbol,
        timeframe=args.timeframe,
        groups=groups,
        horizon=args.horizon,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_bars=args.purge,
        embargo_bars=args.embargo,
    )
    path = save_manifest(args.out, bundle)
    _print_json(
        {
            "manifest": str(path),
            "dataset_id": bundle.manifest.dataset_id,
            "rows": bundle.manifest.rows,
            "columns": len(bundle.manifest.columns),
            "train": bundle.rows("train"),
            "validation": bundle.rows("validation"),
            "test": bundle.rows("test"),
            "purged": int((bundle.split == -1).sum()),
        }
    )


def _cmd_ml_train(args: argparse.Namespace) -> None:
    """Fit a classical direction classifier on a chronological dataset."""
    store = ParquetStore(args.root)
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise CliError(f"no {args.symbol} {args.timeframe} bars under {args.root}")
    features = _features_for(args, frame, args.symbol, args.timeframe)
    groups = (
        DEFAULT_GROUPS
        if args.groups == "all"
        else tuple(part.strip() for part in args.groups.split(",") if part.strip())
    )
    spec = TrainSpec(
        model=args.model,
        horizon=args.horizon,
        groups=groups,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_bars=args.purge,
        embargo_bars=args.embargo,
        seed=args.seed,
        evaluate_test=args.evaluate_test,
        include_flat=args.include_flat,
    )
    result = train_classifier(
        features,
        symbol=args.symbol,
        timeframe=args.timeframe,
        spec=spec,
    )
    directory = Path(args.out) / f"ml-{args.model}-{result.dataset_id.rsplit(':', 1)[-1]}"
    model_path, metrics_path = save_result(result, directory)
    _print_json(
        {
            "model": args.model,
            "dataset_id": result.dataset_id,
            "train": result.split_counts["train"],
            "validation": result.split_counts["validation"],
            "test": result.split_counts["test"],
            "validation_metrics": result.validation,
            "baseline": result.baseline,
            "test_metrics": result.test,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        }
    )


def _cmd_ml_cv(args: argparse.Namespace) -> None:
    """Run purged expanding-window CV without reading validation or test rows."""
    store = ParquetStore(args.root)
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise CliError(f"no {args.symbol} {args.timeframe} bars under {args.root}")
    features = _features_for(args, frame, args.symbol, args.timeframe)
    groups = (
        DEFAULT_GROUPS
        if args.groups == "all"
        else tuple(part.strip() for part in args.groups.split(",") if part.strip())
    )
    spec = TrainSpec(
        model=args.model,
        horizon=args.horizon,
        groups=groups,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        purge_bars=args.purge,
        embargo_bars=args.embargo,
        seed=args.seed,
        include_flat=args.include_flat,
    )
    config = CVConfig(
        n_splits=args.folds,
        purge_bars=args.purge,
        embargo_bars=args.embargo,
        min_train_rows=args.min_train_rows,
    )
    result = cross_validate(
        features,
        symbol=args.symbol,
        timeframe=args.timeframe,
        spec=spec,
        config=config,
    )
    payload = result.to_dict()
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
        payload["out"] = str(out)
    _print_json(payload)


def _cmd_ml_evaluate(args: argparse.Namespace) -> None:
    """Trade a saved classifier on one chronological split."""
    result = load_result(Path(args.run))
    store = ParquetStore(args.root)
    frame = store.read(result.symbol, result.timeframe)
    if frame.is_empty():
        raise CliError(f"no {result.symbol} {result.timeframe} bars under {args.root}")
    features = _features_for(args, frame, result.symbol, result.timeframe)
    multipliers = tuple(float(value) for value in args.stress.split(",") if value)
    regime_labels = None
    if args.regime_labels:
        path = Path(args.regime_labels)
        if not path.is_file():
            raise CliError(f"regime labels not found: {path}")
        regime_labels = pl.read_parquet(path)
    evaluation = evaluate_model(
        result,
        features,
        Settings(),
        split=args.split,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        window=args.window,
        stress=multipliers,
        regime_labels=regime_labels,
        compute_regime=args.regime,
    )
    _print_json(
        {
            "run": str(Path(args.run)),
            "dataset_id": result.dataset_id,
            "split": evaluation.split,
            "threshold": evaluation.threshold,
            "exit_threshold": evaluation.exit_threshold,
            "metrics": evaluation.metrics,
            "hold": evaluation.hold,
            "stress": evaluation.stress,
            "regime": evaluation.regime,
        }
    )


# --- discovery -----------------------------------------------------------------
def _stress_values(text: str | None) -> tuple[float, ...] | None:
    """Parse ``--stress``: ``None`` keeps the config default, ``""`` disables it."""
    if text is None:
        return None
    if not text.strip():
        return ()
    return _floats(text, flag="--stress")


def _cmd_discovery_run(args: argparse.Namespace) -> None:
    """Generate, evaluate and archive strategies, then record the decision."""
    required = (
        ()
        if not args.require_stress.strip()
        else _floats(args.require_stress, flag="--require-stress")
    )
    gate = GateCriteria(
        min_trades=args.min_trades,
        min_expectancy_r=args.min_expectancy_r,
        min_profit_factor=args.min_profit_factor,
        max_drawdown=args.max_drawdown,
        require_positive_cost_stress=required,
    )
    baselines = tuple(part.strip() for part in args.baseline.split(",") if part.strip())
    stress = _stress_values(args.stress)
    try:
        config = DiscoveryRunConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            max_trials=args.max_trials,
            population=args.population,
            generations=args.generations,
            seed=args.seed,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            complexity_cap=args.complexity_cap,
            archive=args.archive,
            use_cache=not args.no_cache,
            stress=(1.5, 2.0) if stress is None else stress,
            gate=gate,
            baselines=baselines,
            horizon=args.horizon,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            purge_bars=args.purge,
            embargo_bars=args.embargo,
            window=args.window,
            patience=args.patience,
            max_lookback=args.max_lookback,
            max_depth=args.max_depth,
        )
        result = run_discovery(config, root=args.root)
    except (DiscoveryRunError, ValueError) as exc:
        raise CliError(str(exc)) from exc

    payload = result.to_dict()
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
        payload["out"] = str(out)
    _print_json(payload)


def _reproduce_payload(outcome: CandidateEvidence | ReproduceMismatch) -> dict[str, Any]:
    """The one-line contract: reproduced flag, both checksums and mismatches."""
    if isinstance(outcome, ReproduceMismatch):
        return {
            "reproduced": False,
            "checksum": outcome.reproduced,
            "expected": outcome.expected,
            "mismatches": list(outcome.sections),
        }
    return {
        "reproduced": True,
        "checksum": outcome.checksum,
        "expected": outcome.checksum,
        "mismatches": [],
    }


def _write_reproduce_report(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _cmd_discovery_reproduce(args: argparse.Namespace) -> None:
    """Re-run one stored evidence bundle and report whether its checksum matches.

    Exit codes: 0 when the checksum is reproduced, 1 when the stored identity
    does not match this environment (settings, code version, data, or the
    rebuilt evidence), and 2 when the input cannot be read or reproduced at all.
    The original bundle file is never rewritten; ``--out`` is a separate report.
    """
    settings = None
    if args.settings is not None:
        try:
            settings = Settings.load(args.settings)
        except (OSError, TypeError, ValueError) as exc:
            raise CliError(f"cannot read --settings {args.settings}: {exc}") from exc

    try:
        outcome = reproduce_bundle(
            Path(args.bundle),
            store_root=args.store_root,
            spec=args.spec,
            settings=settings,
        )
    except ReproduceError as exc:
        if exc.invalid:
            raise CliError(str(exc)) from exc
        payload: dict[str, Any] = {
            "reproduced": False,
            "checksum": None,
            "expected": exc.expected,
            "mismatches": [exc.code],
            "reason": str(exc),
        }
        _write_reproduce_report(args.out, payload)
        _print_json(payload)
        raise _ExitCode(1) from exc
    except EvidenceError as exc:
        raise CliError(str(exc)) from exc

    payload = _reproduce_payload(outcome)
    if isinstance(outcome, ReproduceMismatch):
        report = outcome.to_dict()
    else:
        report = {**payload, "evidence": outcome.to_dict()}
    _write_reproduce_report(args.out, report)
    _print_json(payload)
    if isinstance(outcome, ReproduceMismatch):
        raise _ExitCode(1)


def _cmd_discovery_evidence(args: argparse.Namespace) -> None:
    """Build one experiment from stored bars and persist its artifact and bundle.

    This is the production writer ``discovery reproduce`` depends on: the
    dataset artifact under ``<root>/datasets/<checksum>/`` and the evidence JSON
    plus strategy sidecar at ``--out``. One compact JSON line on success; input
    or refusal errors go to stderr with exit code 2 like every other command.
    """
    settings = Settings()
    if args.settings is not None:
        try:
            settings = Settings.load(args.settings)
        except (OSError, TypeError, ValueError) as exc:
            raise CliError(f"cannot read --settings {args.settings}: {exc}") from exc
    stress = _stress_values(args.stress)
    try:
        spec = load_strategy_spec(args.spec)
        config = EvidenceConfig(
            code_version=discovery_code_version(),
            symbol=args.symbol,
            timeframe=args.timeframe,
            folds=args.folds,
            purge_bars=args.fold_purge,
            embargo_bars=args.fold_embargo,
            min_trades_per_fold=args.min_trades_per_fold,
            cost_stress=(1.5, 2.0) if stress is None else stress,
            regime_coverage_min=args.regime_coverage_min,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_confidence=args.bootstrap_confidence,
            seed=args.seed,
            search_budget=args.search_budget,
            trial_count=args.trials,
            complexity=int(spec.complexity()),
            window=args.window,
        )
        result = persist_evidence_run(
            Path(args.root),
            Path(args.out),
            spec=spec,
            symbol=args.symbol,
            timeframe=args.timeframe,
            config=config,
            settings=settings,
            horizon=args.horizon,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            dataset_purge_bars=args.purge,
            dataset_embargo_bars=args.embargo,
            use_cache=not args.no_cache,
        )
    except (PersistError, EvidenceError, ValueError) as exc:
        raise CliError(str(exc)) from exc

    _print_json(result.to_dict())


# --- backtest ------------------------------------------------------------------
def _hold(obs: np.ndarray) -> int:
    """Control policy: never trade."""
    return _HOLD


def _rule_baseline(obs: np.ndarray) -> int:
    """Baseline only: buy while the last bar's 5-bar momentum is positive.

    ``obs`` is the simulator's flattened 60-bar window followed by five
    position features, so the last bar's ``ret_5`` is at ``_RET5_INDEX``.
    A baseline, not the learning agent: it ignores position state, so an
    already-open position simply blocks the repeated buy.
    """
    return _BUY if float(obs[_RET5_INDEX]) > 0.0 else _HOLD


def _report(result: BacktestResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "config_hash": result.config_hash,
        "metrics": metrics_payload(result.metrics),
        "trades": [asdict(trade) for trade in result.trades],
    }


def _select_policy(name: str) -> tuple[Callable[[np.ndarray], int], str]:
    """Map the CLI policy name to its callable and explicit run label."""
    if name == _POLICY_RULE:
        return _rule_baseline, RULE_LABEL
    return _hold, _POLICY_HOLD


def _backtest_inputs(args: argparse.Namespace) -> tuple[Settings, FeatureFrame]:
    """Load the stored M1 frame and build the features every backtest shares."""
    store = ParquetStore(args.root)
    frame = store.read(args.symbol, TIMEFRAME)
    if frame.is_empty():
        raise CliError(
            f"no data for {args.symbol} ({TIMEFRAME}) under {store.root}; ingest bars first"
        )
    if frame.height < _MIN_BARS:
        raise CliError(
            f"not enough data for {args.symbol} ({TIMEFRAME}): "
            f"need {_MIN_BARS} bars, have {frame.height}"
        )
    settings = Settings()
    return settings, cached_build_features(
        _feature_cache(args),
        symbol=args.symbol,
        timeframe=TIMEFRAME,
        frame=frame,
        settings=settings,
    )


def _cmd_backtest(args: argparse.Namespace) -> None:
    """Run a stored symbol through the simulator and write a JSON report."""
    if args.out is None:
        raise CliError("backtest requires --out (or use 'backtest montecarlo'/'sensitivity')")
    settings, features = _backtest_inputs(args)
    policy, label = _select_policy(args.policy)
    result = run_backtest(
        policy,
        features,
        settings,
        policy_name=label,
        symbol=args.symbol,
        mask_actions=args.mask_actions,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_report(result), indent=2, default=_json_default) + "\n")
    _print_json({"run_id": result.run_id, "config_hash": result.config_hash, "out": str(out)})


def _floats(text: str, *, flag: str) -> tuple[float, ...]:
    """Parse a comma-separated number list, failing with a CLI error."""
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise CliError(f"{flag} expects comma-separated numbers, got {text!r}") from None
    if not values:
        raise CliError(f"{flag} expects at least one number")
    return values


def _finite(value: float) -> float | None:
    """JSON has no infinity; a non-finite metric becomes ``null``."""
    return float(value) if math.isfinite(value) else None


def _sensitivity_row(row: SensitivityRow) -> dict[str, Any]:
    payload = asdict(row)
    payload["profit_factor"] = _finite(row.profit_factor)
    return payload


def _cmd_backtest_montecarlo(args: argparse.Namespace) -> None:
    """Resample one backtest's trades and report the path distribution."""
    settings, features = _backtest_inputs(args)
    policy, label = _select_policy(args.policy)
    result = run_backtest(policy, features, settings, policy_name=label, symbol=args.symbol)
    levels = _floats(args.confidence, flag="--confidence")
    try:
        config = MonteCarloConfig(
            n_paths=args.n_paths,
            seed=args.seed,
            mode=args.mode,
            block_size=args.block_size,
            confidence_levels=levels,
            starting_balance=float(settings.contract.initial_balance),
            ruin_fraction=args.ruin_fraction,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    montecarlo = run_montecarlo(result, config)
    summary: dict[str, Any] = {
        "run_id": result.run_id,
        "trades": montecarlo.trades,
        "paths": montecarlo.paths,
        "mode": montecarlo.mode,
        "seed": montecarlo.seed,
        "block_size": montecarlo.block_size,
        "starting_balance": montecarlo.starting_balance,
        "ruin_threshold": montecarlo.ruin_threshold,
        "degenerate": montecarlo.degenerate,
        "probability_of_ruin": montecarlo.probability_of_ruin,
        "final_pnl": montecarlo.final_pnl_percentiles,
        "max_drawdown": montecarlo.max_drawdown_percentiles,
        "ending_balance": montecarlo.ending_balance_percentiles,
        "worst_path": (None if montecarlo.worst_path is None else asdict(montecarlo.worst_path)),
    }
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        full = {
            **summary,
            "confidence_levels": list(montecarlo.confidence_levels),
            "outcomes": [asdict(outcome) for outcome in montecarlo.outcomes],
        }
        out.write_text(json.dumps(full, indent=2, default=_json_default) + "\n")
        summary["out"] = str(out)
    _print_json(summary)


def _cmd_backtest_sensitivity(args: argparse.Namespace) -> None:
    """Rerun one backtest across execution-cost grids."""
    settings, features = _backtest_inputs(args)
    policy, _ = _select_policy(args.policy)
    grids: dict[str, tuple[float, ...]] = {}
    if args.spread:
        grids[SPREAD_AXIS] = _floats(args.spread, flag="--spread")
    if args.slippage:
        grids[SLIPPAGE_AXIS] = _floats(args.slippage, flag="--slippage")
    if args.commission:
        grids[COMMISSION_AXIS] = _floats(args.commission, flag="--commission")
    try:
        report = parameter_sensitivity(
            policy,
            features,
            settings,
            axes=grids or None,
            symbol=args.symbol,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    payload: dict[str, Any] = {
        "base": _sensitivity_row(report.base),
        "axes": [
            {
                "parameter": axis.parameter,
                "breakeven_multiplier": axis.breakeven_multiplier,
                "rows": [_sensitivity_row(row) for row in axis.rows],
            }
            for axis in report.axes
        ],
    }
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
        payload["out"] = str(out)
    _print_json(payload)


# --- serve ---------------------------------------------------------------------
def _bind(host: str, port: int) -> socket.socket:
    """Bind and listen before uvicorn starts, so a failure stays ours to report."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(2048)
    except OSError as exc:
        listener.close()
        reason = exc.strerror or str(exc)
        raise RuntimeError(f"cannot bind {host}:{port}: {reason}") from exc
    return listener


def _stderr_log_config(base: dict[str, Any]) -> dict[str, Any]:
    """Copy a uvicorn logging config with every handler pointed at stderr.

    uvicorn's default sends access logs to stdout, which would break the CLI's
    one-JSON-line contract. ``access_log=False`` already disables them; routing
    the handlers too keeps any future logger off stdout as well.
    """
    config = copy.deepcopy(base)
    for handler in config["handlers"].values():
        handler["stream"] = "ext://sys.stderr"
    return config


def _cmd_serve(args: argparse.Namespace) -> None:
    """Bind the port, then serve the API with one worker.

    Binding ourselves reports the real port for ``--port 0`` and turns
    ``EADDRINUSE`` into a normal CLI error instead of uvicorn's ``SystemExit``.
    All state is process-local, so one worker is the contract.
    """
    import uvicorn

    from scalping.api.app import create_app

    listener = _bind(args.host, args.port)
    port = int(listener.getsockname()[1])
    try:
        config = uvicorn.Config(
            create_app(
                data_root=Path(args.root),
                flow_root=Path(args.flow_root) if args.flow_root else None,
            ),
            host=args.host,
            port=port,
            log_config=_stderr_log_config(uvicorn.config.LOGGING_CONFIG),
            access_log=False,
            workers=1,
        )
        _print_json({"status": "serving", "host": args.host, "port": port})
        uvicorn.Server(config).run(sockets=[listener])
    finally:
        listener.close()


# --- orchestrator daemon ----------------------------------------------------
class _WatchSource:
    """Adapt ``FileTriggerSource`` payloads for the pipeline handlers.

    ``FileTriggerSource`` describes a file (``path``, size, hash); a pipeline
    handler expects its own payload. For ``knowledge.ingest`` the only required
    field is ``document_path``, so the adapter maps one to the other and keeps
    the file's content hash as the dedupe key. Any other kind receives the raw
    file payload, and its handler must accept that shape.
    """

    def __init__(self, source: FileTriggerSource, kind: str) -> None:
        self._source = source
        self.kind = kind
        self.name = source.name

    def scan(self) -> list[Trigger]:
        triggers = self._source.scan()
        adapted: list[Trigger] = []
        for trigger in triggers:
            payload = (
                {"document_path": str(Path(str(trigger.payload.get("path", ""))).resolve())}
                if self.kind == "knowledge.ingest"
                else dict(trigger.payload)
            )
            adapted.append(
                Trigger(
                    kind=trigger.kind,
                    dedupe_key=trigger.dedupe_key,
                    payload=payload,
                    priority=trigger.priority,
                    max_retries=trigger.max_retries,
                    depends_on=trigger.depends_on,
                )
            )
        return adapted

    def seen(self, triggers: Sequence[Trigger]) -> None:
        self._source.seen(triggers)


def _cmd_orchestrator_run(args: argparse.Namespace) -> None:
    """Run one tick (``--once``) or a supervised loop (``--forever``)."""
    from scalping.orchestrator.core import Orchestrator
    from scalping.orchestrator.daemon import DaemonConfig, OrchestratorDaemon
    from scalping.orchestrator.handlers import (
        HEAVY_HANDLER_KINDS,
        KIND_DATASET_BUILD,
        KIND_DISCOVERY_RUN,
        KIND_PAPER_ADVANCE,
        KIND_PAPER_ANALYSE,
        HandlerContext,
        register_default_handlers,
    )
    from scalping.orchestrator.store import JobStore

    if args.no_scan and args.watch_dir:
        raise CliError("--no-scan and --watch-dir cannot be combined")
    research_kinds = tuple(
        sorted(
            {
                KIND_DATASET_BUILD,
                KIND_DISCOVERY_RUN,
                KIND_PAPER_ADVANCE,
                KIND_PAPER_ANALYSE,
            }
        )
    )
    config = DaemonConfig(
        poll_seconds=args.poll_seconds,
        guard_profile=args.profile,
        scan_triggers=not args.no_scan,
        max_jobs_per_tick=args.max_jobs_per_tick,
        stop_grace_seconds=args.stop_grace_seconds,
        heavy_kinds=tuple(sorted(HEAVY_HANDLER_KINDS)),
        research_kinds=research_kinds,
    )
    store = JobStore(Path(args.db))
    try:
        triggers: tuple[TriggerSource, ...] = ()
        if args.watch_dir:
            try:
                source = FileTriggerSource(
                    Path(args.watch_dir),
                    kind=args.watch_kind,
                    store=store,
                )
            except NotADirectoryError as exc:
                raise CliError(str(exc)) from exc
            triggers = (_WatchSource(source, args.watch_kind),)
        registry = Orchestrator(store)
        register_default_handlers(
            registry,
            HandlerContext(
                data_root=Path(args.root),
                runs_root=Path(args.runs_root),
                archive_root=Path(args.archive_root),
            ),
        )
        daemon = OrchestratorDaemon(store, registry, config, triggers=triggers)
        if args.once:
            result = daemon.tick()
            if result.error is not None:
                raise CliError(result.error)
            _print_json({"status": daemon.status().to_dict(), "tick": result.to_dict()})
            return
        _run_daemon_forever(daemon)
    finally:
        store.close()


def _run_daemon_forever(daemon: OrchestratorDaemon) -> None:
    """Start the daemon and block until SIGINT/SIGTERM or a dead worker."""
    stop = threading.Event()
    failed = False

    def _request_stop(signum: int, frame: object) -> None:
        stop.set()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, _request_stop)
        daemon.start()
        _print_json({"status": "started", "daemon": daemon.status().to_dict()})
        try:
            while not stop.is_set():
                if not daemon.is_running:
                    stop.set()
                    failed = True
                    break
                stop.wait(1.0)
        except KeyboardInterrupt:  # pragma: no cover - signal handler normally wins
            stop.set()
    finally:
        try:
            daemon.stop()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if failed:
        _print_json(
            {
                "status": "failed",
                "detail": "worker exited without a stop request",
                "daemon": daemon.status().to_dict(),
            }
        )
        raise _ExitCode(1)
    _print_json({"status": "stopped", "daemon": daemon.status().to_dict()})


# --- parser --------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="scalping.cli", description="Data ingestion, backtests and the API.")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="ingest bars into the Parquet store")
    sources = data.add_subparsers(dest="source", required=True)

    synth = sources.add_parser("synth", help="generate deterministic synthetic M1 bars")
    synth.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    synth.add_argument("--start", type=_day, required=True, help="first UTC day (inclusive)")
    synth.add_argument("--end", type=_day, required=True, help="last UTC day (inclusive)")
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--start-price", type=float, default=2350.0)
    synth.add_argument("--root", default=str(DEFAULT_ROOT))
    synth.set_defaults(handler=_cmd_data_synth)

    csv = sources.add_parser("csv", help="read a broker CSV export")
    csv.add_argument("--path", required=True, help="CSV file to read")
    csv.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    csv.add_argument(
        "--tz-offset",
        type=int,
        default=None,
        help="broker UTC offset in hours (assumed +3 when omitted)",
    )
    csv.add_argument("--root", default=str(DEFAULT_ROOT))
    csv.set_defaults(handler=_cmd_data_csv)

    json_source = sources.add_parser("json", help="read a JSON array of bars")
    json_source.add_argument("--path", required=True, help="JSON file to read")
    json_source.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    json_source.add_argument("--root", default=str(DEFAULT_ROOT))
    json_source.set_defaults(handler=_cmd_data_json)

    derive = sources.add_parser("derive", help="derive M5-M30-H1-H4-D1 from a timeframe")
    derive.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    derive.add_argument("--source", default="M1")
    derive.add_argument("--timeframes", default="M5,M15,M30,H1,H4,D1")
    derive.add_argument("--root", default=str(DEFAULT_ROOT))
    derive.set_defaults(handler=_cmd_data_derive)

    validate = sources.add_parser("validate", help="print the structural report for a stored slice")
    validate.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    validate.add_argument("--timeframe", default="M1")
    validate.add_argument("--root", default=str(DEFAULT_ROOT))
    validate.set_defaults(handler=_cmd_data_validate)

    ml = commands.add_parser("ml", help="classical machine-learning experiments")
    ml_commands = ml.add_subparsers(dest="ml_command", required=True)
    ml_train = ml_commands.add_parser("train", help="fit a direction classifier")
    ml_train.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    ml_train.add_argument("--timeframe", default="H1")
    ml_train.add_argument("--model", default="logistic")
    ml_train.add_argument("--groups", default="all")
    ml_train.add_argument("--horizon", type=int, default=5)
    ml_train.add_argument("--train-fraction", type=float, default=0.70)
    ml_train.add_argument("--validation-fraction", type=float, default=0.15)
    ml_train.add_argument("--purge", type=int, default=5)
    ml_train.add_argument("--embargo", type=int, default=5)
    ml_train.add_argument("--seed", type=int, default=0)
    ml_train.add_argument("--evaluate-test", action="store_true")
    ml_train.add_argument(
        "--include-flat",
        action="store_true",
        help="keep exact-zero returns as a third class (usually harmful)",
    )
    ml_train.add_argument("--root", default=str(DEFAULT_ROOT))
    ml_train.add_argument("--out", default="runs/ml")
    ml_train.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    ml_train.set_defaults(handler=_cmd_ml_train)

    ml_cv = ml_commands.add_parser(
        "cv", help="purged time-series cross-validation on the training split"
    )
    ml_cv.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    ml_cv.add_argument("--timeframe", default="H1")
    ml_cv.add_argument("--model", default="logistic")
    ml_cv.add_argument("--groups", default="all")
    ml_cv.add_argument("--horizon", type=int, default=5)
    ml_cv.add_argument("--folds", type=int, default=5)
    ml_cv.add_argument("--purge", type=int, default=5)
    ml_cv.add_argument(
        "--embargo",
        type=int,
        default=5,
        help="extra bars before each test block (matches ml train)",
    )
    ml_cv.add_argument("--min-train-rows", type=int, default=50)
    ml_cv.add_argument("--train-fraction", type=float, default=0.70)
    ml_cv.add_argument("--validation-fraction", type=float, default=0.15)
    ml_cv.add_argument("--seed", type=int, default=0)
    ml_cv.add_argument(
        "--include-flat",
        action="store_true",
        help="keep exact-zero returns as a third class (usually harmful)",
    )
    ml_cv.add_argument("--root", default=str(DEFAULT_ROOT))
    ml_cv.add_argument("--out", default=None, help="optional full report path")
    ml_cv.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    ml_cv.set_defaults(handler=_cmd_ml_cv)

    ml_evaluate = ml_commands.add_parser("evaluate", help="trade a saved classifier")
    ml_evaluate.add_argument("--run", required=True, help="runs/ml/<run> directory")
    ml_evaluate.add_argument("--root", default=str(DEFAULT_ROOT))
    ml_evaluate.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    ml_evaluate.add_argument("--threshold", type=float, default=0.55)
    ml_evaluate.add_argument("--exit-threshold", type=float, default=0.55)
    ml_evaluate.add_argument("--window", type=int, default=WINDOW)
    ml_evaluate.add_argument("--stress", default="", help="comma-separated cost multipliers")
    ml_evaluate.add_argument(
        "--regime",
        action="store_true",
        help="attach rule-based regime labels to the base run's trades",
    )
    ml_evaluate.add_argument(
        "--regime-labels",
        default="",
        help="parquet with ts + regime columns; overrides --regime labels",
    )
    ml_evaluate.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    ml_evaluate.set_defaults(handler=_cmd_ml_evaluate)

    dataset = commands.add_parser("dataset", help="build leakage-safe datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    build = dataset_commands.add_parser("build", help="build a chronological split manifest")
    build.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    build.add_argument("--timeframe", default="H1")
    build.add_argument("--groups", default="all", help="comma-separated feature groups, or all")
    build.add_argument("--horizon", type=int, default=1, help="future bars for the label")
    build.add_argument("--train-fraction", type=float, default=0.70)
    build.add_argument("--validation-fraction", type=float, default=0.15)
    build.add_argument("--purge", type=int, default=0, help="bars dropped at each boundary")
    build.add_argument("--embargo", type=int, default=0, help="extra bars after the purge")
    build.add_argument("--root", default=str(DEFAULT_ROOT))
    build.add_argument("--out", required=True, help="manifest JSON path")
    build.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    build.set_defaults(handler=_cmd_dataset_build)

    nn_parser = commands.add_parser("nn", help="train and evaluate sequence models")
    nn_commands = nn_parser.add_subparsers(dest="nn_command", required=True)
    nn_train = nn_commands.add_parser("train", help="fit one sequence model")
    nn_train.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    nn_train.add_argument("--timeframe", default="H1")
    nn_train.add_argument("--model", choices=_NN_MODELS, default="gru")
    nn_train.add_argument("--window", type=int, default=60)
    nn_train.add_argument("--clip", type=float, default=5.0)
    nn_train.add_argument("--horizon", type=int, default=1)
    nn_train.add_argument("--epochs", type=int, default=10)
    nn_train.add_argument("--batch-size", type=int, default=64)
    nn_train.add_argument("--hidden", type=int, default=64)
    nn_train.add_argument("--dropout", type=float, default=0.1)
    nn_train.add_argument("--learning-rate", type=float, default=1e-3)
    nn_train.add_argument("--patience", type=int, default=5)
    nn_train.add_argument("--checkpoint-every", type=int, default=1)
    nn_train.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    nn_train.add_argument("--seed", type=int, default=0)
    nn_train.add_argument("--train-fraction", type=float, default=0.70)
    nn_train.add_argument("--validation-fraction", type=float, default=0.15)
    nn_train.add_argument("--purge", type=int, default=0)
    nn_train.add_argument("--root", default=str(DEFAULT_ROOT))
    nn_train.add_argument("--checkpoint", required=True, help="checkpoint path to write")
    nn_train.add_argument("--out", required=True, help="training report JSON path")
    nn_train.set_defaults(handler=_cmd_nn_train)

    nn_eval = nn_commands.add_parser("evaluate", help="score a checkpoint on one split")
    nn_eval.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    nn_eval.add_argument("--timeframe", default="H1")
    nn_eval.add_argument("--horizon", type=int, default=1)
    nn_eval.add_argument("--train-fraction", type=float, default=0.70)
    nn_eval.add_argument("--validation-fraction", type=float, default=0.15)
    nn_eval.add_argument("--purge", type=int, default=0)
    nn_eval.add_argument("--split", choices=("train", "validation", "test"), default="test")
    nn_eval.add_argument("--clip", type=float, default=5.0)
    nn_eval.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    nn_eval.add_argument("--root", default=str(DEFAULT_ROOT))
    nn_eval.add_argument("--checkpoint", required=True)
    nn_eval.add_argument("--out", default=None, help="optional full report path")
    nn_eval.set_defaults(handler=_cmd_nn_evaluate)

    backtest = commands.add_parser("backtest", help="replay a stored symbol")
    backtest.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    backtest.add_argument(
        "--policy", choices=sorted((_POLICY_HOLD, _POLICY_RULE)), default=_POLICY_HOLD
    )
    backtest.add_argument("--root", default=str(DEFAULT_ROOT))
    backtest.add_argument("--out", default=None, help="path of the JSON report")
    backtest.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    backtest.add_argument(
        "--mask-actions",
        action="store_true",
        help="map actions the simulator mask forbids to HOLD",
    )
    backtest.set_defaults(handler=_cmd_backtest)

    backtest_commands = backtest.add_subparsers(dest="backtest_command", required=False)

    mc = backtest_commands.add_parser(
        "montecarlo", help="resample the run's trades and report path outcomes"
    )
    mc.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    mc.add_argument("--policy", choices=sorted((_POLICY_HOLD, _POLICY_RULE)), default=_POLICY_RULE)
    mc.add_argument("--n-paths", type=int, default=1_000)
    mc.add_argument("--seed", type=int, default=0)
    mc.add_argument(
        "--mode",
        choices=("shuffle", "bootstrap"),
        default="bootstrap",
        help="resampling mode (library default is shuffle)",
    )
    mc.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="block length for bootstrap mode (1 = IID bootstrap)",
    )
    mc.add_argument(
        "--confidence",
        default="0.05,0.5,0.95",
        help="comma-separated quantiles in (0, 1)",
    )
    mc.add_argument(
        "--ruin-fraction",
        type=float,
        default=0.0,
        help="ruin floor as a fraction of the starting balance, in [0, 1)",
    )
    mc.add_argument("--root", default=str(DEFAULT_ROOT))
    mc.add_argument("--out", default=None, help="optional full report path")
    mc.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    mc.set_defaults(handler=_cmd_backtest_montecarlo)

    sensitivity = backtest_commands.add_parser(
        "sensitivity", help="rerun the strategy across execution-cost grids"
    )
    sensitivity.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    sensitivity.add_argument(
        "--policy", choices=sorted((_POLICY_HOLD, _POLICY_RULE)), default=_POLICY_RULE
    )
    sensitivity.add_argument(
        "--spread",
        default="",
        help="comma-separated spread multipliers (default: 0.5,1,2,3)",
    )
    sensitivity.add_argument(
        "--slippage",
        default="",
        help="comma-separated absolute slippage_atr_frac values (default: from settings)",
    )
    sensitivity.add_argument(
        "--commission",
        default="",
        help="comma-separated absolute commission_per_lot values (default: from settings)",
    )
    sensitivity.add_argument("--root", default=str(DEFAULT_ROOT))
    sensitivity.add_argument("--out", default=None, help="optional full report path")
    sensitivity.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    sensitivity.set_defaults(handler=_cmd_backtest_sensitivity)

    discovery = commands.add_parser("discovery", help="generate, evaluate and archive strategies")
    discovery_commands = discovery.add_subparsers(dest="discovery_command", required=True)
    d_run = discovery_commands.add_parser(
        "run",
        help="run one evolutionary search on the validation split and record a decision",
    )
    d_run.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    d_run.add_argument("--timeframe", default=TIMEFRAME)
    d_run.add_argument(
        "--max-trials",
        type=int,
        default=24,
        help="attempt cap, duplicates and invalid trees included "
        "(a real run needs thousands; 24 only proves the wiring)",
    )
    d_run.add_argument("--population", type=int, default=8)
    d_run.add_argument("--generations", type=int, default=4)
    d_run.add_argument("--seed", type=int, default=0)
    d_run.add_argument("--mutation-rate", type=float, default=0.3)
    d_run.add_argument("--crossover-rate", type=float, default=0.2)
    d_run.add_argument("--complexity-cap", type=int, default=40)
    d_run.add_argument("--patience", type=int, default=3)
    d_run.add_argument("--max-lookback", type=int, default=20)
    d_run.add_argument("--max-depth", type=int, default=3)
    d_run.add_argument("--window", type=int, default=WINDOW)
    d_run.add_argument("--horizon", type=int, default=1)
    d_run.add_argument("--train-fraction", type=float, default=0.70)
    d_run.add_argument("--validation-fraction", type=float, default=0.15)
    d_run.add_argument("--purge", type=int, default=0)
    d_run.add_argument("--embargo", type=int, default=0)
    d_run.add_argument(
        "--archive",
        default="runs/discovery/discovery.sqlite",
        help="SQLite evidence archive (created when missing)",
    )
    d_run.add_argument(
        "--baseline",
        default="hold,ret_5",
        help="comma-separated rule-free baselines: hold,ret_5",
    )
    d_run.add_argument(
        "--stress",
        default=None,
        help="comma-separated cost multipliers (default 1.5,2.0; '' disables)",
    )
    d_run.add_argument("--min-trades", type=int, default=0)
    d_run.add_argument("--min-expectancy-r", type=float, default=0.0)
    d_run.add_argument("--min-profit-factor", type=float, default=0.0)
    d_run.add_argument("--max-drawdown", type=float, default=0.0)
    d_run.add_argument(
        "--require-stress",
        default="",
        help="comma-separated multipliers the gate requires to be strictly profitable",
    )
    d_run.add_argument("--root", default=str(DEFAULT_ROOT))
    d_run.add_argument("--out", default=None, help="optional full JSON report path")
    d_run.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    d_run.set_defaults(handler=_cmd_discovery_run)

    d_reproduce = discovery_commands.add_parser(
        "reproduce",
        help="re-run one stored evidence bundle and compare its checksum",
    )
    d_reproduce.add_argument("--bundle", required=True, help="evidence JSON written earlier")
    d_reproduce.add_argument(
        "--store-root",
        required=True,
        help="bar store plus <store-root>/datasets/<checksum>/ artifacts",
    )
    d_reproduce.add_argument(
        "--spec",
        default=None,
        help=(
            "strategy spec JSON; defaults to the sidecar <bundle stem>.spec.json next to the bundle"
        ),
    )
    d_reproduce.add_argument(
        "--settings",
        default=None,
        help="YAML settings; defaults to built-in Settings()",
    )
    d_reproduce.add_argument("--out", default=None, help="optional report path")
    d_reproduce.set_defaults(handler=_cmd_discovery_reproduce)

    d_evidence = discovery_commands.add_parser(
        "evidence",
        help="build evidence from stored bars and persist the dataset artifact and bundle",
    )
    d_evidence.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    d_evidence.add_argument("--timeframe", default=TIMEFRAME)
    d_evidence.add_argument(
        "--spec",
        required=True,
        help="strategy spec JSON path or inline JSON (written to the sidecar)",
    )
    d_evidence.add_argument(
        "--out",
        required=True,
        help="evidence bundle path; the sidecar is written next to it",
    )
    d_evidence.add_argument("--root", default=str(DEFAULT_ROOT))
    d_evidence.add_argument(
        "--settings", default=None, help="YAML settings; defaults to Settings()"
    )
    d_evidence.add_argument("--folds", type=int, default=3, help="chronological folds (>= 3)")
    d_evidence.add_argument("--window", type=int, default=WINDOW)
    d_evidence.add_argument("--horizon", type=int, default=1)
    d_evidence.add_argument("--train-fraction", type=float, default=0.70)
    d_evidence.add_argument("--validation-fraction", type=float, default=0.15)
    d_evidence.add_argument("--purge", type=int, default=0, help="dataset purge bars")
    d_evidence.add_argument("--embargo", type=int, default=0, help="dataset embargo bars")
    d_evidence.add_argument("--fold-purge", type=int, default=0, help="fold purge bars")
    d_evidence.add_argument("--fold-embargo", type=int, default=0, help="fold embargo bars")
    d_evidence.add_argument("--min-trades-per-fold", type=int, default=5)
    d_evidence.add_argument(
        "--stress",
        default=None,
        help="comma-separated cost multipliers (default 1.5,2.0; '' is refused)",
    )
    d_evidence.add_argument("--regime-coverage-min", type=float, default=1.0)
    d_evidence.add_argument("--bootstrap-samples", type=int, default=1000)
    d_evidence.add_argument("--bootstrap-confidence", type=float, default=0.95)
    d_evidence.add_argument("--seed", type=int, default=0)
    d_evidence.add_argument(
        "--search-budget",
        type=int,
        default=100,
        help="trials the candidate search spent; recorded in the bundle identity",
    )
    d_evidence.add_argument(
        "--trials", type=int, default=1, help="trial_count (<= --search-budget)"
    )
    d_evidence.add_argument("--no-cache", action="store_true", help=_NO_CACHE_HELP)
    d_evidence.set_defaults(handler=_cmd_discovery_evidence)

    serve = commands.add_parser("serve", help="serve the API (single worker)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--root", default=str(DEFAULT_ROOT))
    serve.add_argument(
        "--flow-root",
        default=None,
        help="tick-flow side store; enables the promoted flow-momentum card",
    )
    serve.set_defaults(handler=_cmd_serve)

    orchestrator = commands.add_parser(
        "orchestrator", help="run the persistent autonomous job queue"
    )
    orchestrator_commands = orchestrator.add_subparsers(dest="orchestrator_command", required=True)
    run = orchestrator_commands.add_parser(
        "run", help="scan triggers and drain due jobs once or until a signal"
    )
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run one tick, print it and exit")
    mode.add_argument(
        "--forever", action="store_true", help="loop until SIGINT/SIGTERM, then stop cleanly"
    )
    run.add_argument(
        "--profile",
        choices=("m2_8gb", "exploratory", "none"),
        default="m2_8gb",
        help="measured resource guard profile; 'none' is tests/dev only",
    )
    run.add_argument("--poll-seconds", type=float, default=5.0)
    run.add_argument("--max-jobs-per-tick", type=int, default=10)
    run.add_argument("--stop-grace-seconds", type=float, default=5.0)
    run.add_argument(
        "--no-scan", action="store_true", help="do not call trigger sources on this run"
    )
    run.add_argument(
        "--watch-dir",
        default=None,
        help="watch this directory with FileTriggerSource (no watcher unless given)",
    )
    run.add_argument(
        "--watch-kind",
        default="knowledge.ingest",
        help="job kind a watched file enqueues (default knowledge.ingest)",
    )
    run.add_argument(
        "--db", default="runs/orchestrator.sqlite3", help="SQLite job queue (created when missing)"
    )
    run.add_argument("--root", default=str(DEFAULT_ROOT), help="Parquet bar store")
    run.add_argument("--runs-root", default="runs", help="dataset and discovery run products")
    run.add_argument(
        "--archive-root",
        default=str(DEFAULT_ROOT),
        help="durable evidence root (paper ledger and knowledge index)",
    )
    run.set_defaults(handler=_cmd_orchestrator_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        args.handler(args)
    except _ExitCode as exc:
        # The handler's JSON line is already on stdout; only the code is left.
        return exc.code
    except CliError as exc:
        _print_error(str(exc))
        return 2
    except DataError as exc:
        _print_error(str(exc))
        return 2
    except SystemExit as exc:
        # uvicorn can sys.exit() on a startup failure; SystemExit is a
        # BaseException and would otherwise bypass the JSON error contract.
        code = exc.code if isinstance(exc.code, int) else 0
        message = "server exited during startup" if code <= 0 else f"server exited with code {code}"
        _print_error(message, kind="SystemExit")
        return 1
    except Exception as exc:  # noqa: BLE001 - the CLI owns the top-level boundary
        _print_error(str(exc) or type(exc).__name__, kind=type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
