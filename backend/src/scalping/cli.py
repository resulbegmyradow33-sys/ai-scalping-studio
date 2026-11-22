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
import socket
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Final, NoReturn

import numpy as np
import polars as pl

from scalping.backtest.engine import BacktestResult, run_backtest
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
from scalping.env.simulator import WINDOW
from scalping.features.build import FEATURE_COLUMNS, WARMUP, build_features
from scalping.features.registry import DEFAULT_GROUPS
from scalping.ml.evaluate import evaluate_model
from scalping.ml.train import TrainSpec, load_result, save_result, train_classifier
from scalping.serialize import metrics_payload

__all__ = ["main"]

#: Timeframe every command reads and writes.
TIMEFRAME: Final[str] = "M1"

#: Default store root, matching the API factory and the Makefile.
DEFAULT_ROOT: Final[Path] = Path("data")

#: Symbol used when the caller does not name one.
DEFAULT_SYMBOL: Final[str] = "XAUUSD"

_POLICY_HOLD: Final[str] = "hold"
_POLICY_RULE: Final[str] = "rule"

_HOLD: Final[int] = 0
_BUY: Final[int] = 1

#: Explicit label so a baseline trade can never be mistaken for the agent's.
RULE_LABEL: Final[str] = "rule:ret_5-baseline"

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
    _print_json(
        _ingest(frame, args.symbol, store, source="synthetic", synthetic=True)
    )


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


def _cmd_dataset_build(args: argparse.Namespace) -> None:
    """Build a chronological, purged dataset manifest from a stored symbol."""
    store = ParquetStore(args.root)
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise CliError(f"no {args.symbol} {args.timeframe} bars under {args.root}")
    features = build_features(frame, Settings().costs)
    groups = None if args.groups == "all" else tuple(
        part.strip() for part in args.groups.split(",") if part.strip()
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
    features = build_features(frame, Settings().costs)
    groups = DEFAULT_GROUPS if args.groups == "all" else tuple(
        part.strip() for part in args.groups.split(",") if part.strip()
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


def _cmd_ml_evaluate(args: argparse.Namespace) -> None:
    """Trade a saved classifier on one chronological split."""
    result = load_result(Path(args.run))
    store = ParquetStore(args.root)
    frame = store.read(result.symbol, result.timeframe)
    if frame.is_empty():
        raise CliError(
            f"no {result.symbol} {result.timeframe} bars under {args.root}"
        )
    features = build_features(frame, Settings().costs)
    multipliers = tuple(float(value) for value in args.stress.split(",") if value)
    evaluation = evaluate_model(
        result,
        features,
        Settings(),
        split=args.split,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        window=args.window,
        stress=multipliers,
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
        }
    )


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


def _cmd_backtest(args: argparse.Namespace) -> None:
    """Run a stored symbol through the simulator and write a JSON report."""
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
    features = build_features(frame, settings.costs)
    if args.policy == _POLICY_RULE:
        result = run_backtest(_rule_baseline, features, settings, policy_name=RULE_LABEL)
    else:
        result = run_backtest(_hold, features, settings, policy_name=_POLICY_HOLD)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_report(result), indent=2, default=_json_default) + "\n")
    _print_json({"run_id": result.run_id, "config_hash": result.config_hash, "out": str(out)})


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
    ml_train.set_defaults(handler=_cmd_ml_train)

    ml_evaluate = ml_commands.add_parser("evaluate", help="trade a saved classifier")
    ml_evaluate.add_argument("--run", required=True, help="runs/ml/<run> directory")
    ml_evaluate.add_argument("--root", default=str(DEFAULT_ROOT))
    ml_evaluate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    ml_evaluate.add_argument("--threshold", type=float, default=0.55)
    ml_evaluate.add_argument("--exit-threshold", type=float, default=0.55)
    ml_evaluate.add_argument("--window", type=int, default=WINDOW)
    ml_evaluate.add_argument("--stress", default="", help="comma-separated cost multipliers")
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
    build.set_defaults(handler=_cmd_dataset_build)

    backtest = commands.add_parser("backtest", help="replay a stored symbol")
    backtest.add_argument("--symbol", type=_symbol, default=DEFAULT_SYMBOL)
    backtest.add_argument(
        "--policy", choices=sorted((_POLICY_HOLD, _POLICY_RULE)), default=_POLICY_HOLD
    )
    backtest.add_argument("--root", default=str(DEFAULT_ROOT))
    backtest.add_argument("--out", required=True, help="path of the JSON report")
    backtest.set_defaults(handler=_cmd_backtest)

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
