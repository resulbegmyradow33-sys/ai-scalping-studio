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
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, fields
from datetime import UTC, date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Final, NoReturn

import numpy as np
import polars as pl

from scalping.backtest.engine import BacktestResult, run_backtest
from scalping.backtest.metrics import Metrics
from scalping.config import Settings
from scalping.data.csv_provider import CsvProvider, DataError
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.env.simulator import WINDOW
from scalping.features.build import FEATURE_COLUMNS, WARMUP, build_features

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


def _ingest(frame: pl.DataFrame, symbol: str, store: ParquetStore) -> dict[str, Any]:
    if frame.is_empty():
        raise CliError(f"no bars for {symbol} in the requested range")
    store.write(symbol, TIMEFRAME, frame)
    return _summary(symbol, store)


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
    _print_json(_ingest(frame, args.symbol, store))


def _cmd_data_csv(args: argparse.Namespace) -> None:
    """Read a broker CSV export and merge it into the store."""
    path = Path(args.path)
    provider = CsvProvider(path, tz_offset_hours=args.tz_offset)
    try:
        frame = provider.fetch(args.symbol)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise CliError(f"cannot read {path}: {reason}") from exc
    store = ParquetStore(args.root)
    _print_json(_ingest(frame, args.symbol, store))


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


def _metric_values(metrics: Metrics) -> dict[str, Any]:
    """Metrics by dataclass field name; non-finite values become JSON ``null``."""
    values: dict[str, Any] = {}
    for field_ in fields(metrics):
        value = getattr(metrics, field_.name)
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        values[field_.name] = value
    return values


def _report(result: BacktestResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "config_hash": result.config_hash,
        "metrics": _metric_values(result.metrics),
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
    features = build_features(frame)
    settings = Settings()
    if args.policy == _POLICY_RULE:
        result = run_backtest(_rule_baseline, features, settings, policy_name=RULE_LABEL)
    else:
        result = run_backtest(_hold, features, settings, policy_name=_POLICY_HOLD)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_report(result), indent=2, default=_json_default) + "\n")
    _print_json({"run_id": result.run_id, "config_hash": result.config_hash, "out": str(out)})


# --- serve ---------------------------------------------------------------------
def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the API with one worker (all state is process-local)."""
    import uvicorn

    from scalping.api.app import create_app

    _print_json({"status": "serving", "host": args.host, "port": args.port})
    uvicorn.run(
        create_app(data_root=Path(args.root)),
        host=args.host,
        port=args.port,
        workers=1,
    )


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
        help="broker UTC offset in hours (auto-detect when omitted)",
    )
    csv.add_argument("--root", default=str(DEFAULT_ROOT))
    csv.set_defaults(handler=_cmd_data_csv)

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
    serve.set_defaults(handler=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        args.handler(args)
    except CliError as exc:
        _print_error(str(exc))
        return 2
    except DataError as exc:
        _print_error(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - the CLI owns the top-level boundary
        _print_error(str(exc) or type(exc).__name__, kind=type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
