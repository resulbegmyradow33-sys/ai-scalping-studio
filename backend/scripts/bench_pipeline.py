"""Run the local M2 pipeline measurement suite and write one JSON report.

This script measures; it does not validate a strategy, prove profitability or
accept the M2 target. By default every stage uses synthetic data, so the real
17-year store is not needed and is never opened. Real-bar measurement requires
the explicit ``--real-store`` flag and is bounded by ``--rows``.

Examples
--------
    uv run python scripts/bench_pipeline.py
    uv run python scripts/bench_pipeline.py --suite full --repeats 3
    uv run python scripts/bench_pipeline.py --real-store --store-root data --rows 50000
    uv run python scripts/bench_pipeline.py --real-store --store-root data --rows 0

The full report is written to ``--out`` (default ``runs/bench/pipeline.json``)
and the same JSON object is printed on stdout as one line. Errors print one JSON
line to stderr: exit code 2 for an unusable input or a failed stage, 1 for an
unexpected failure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scalping.bench import (
    Stage,
    StageError,
    dataset_build_stage,
    discovery_smoke_stage,
    evidence_build_stage,
    features_build_stage,
    paper_advance_stage,
    run_suite,
    walk_forward_stage,
)

#: Suites the script can build. ``full`` adds the optional walk-forward stage.
SUITES: tuple[str, ...] = ("smoke", "full")

#: Match the CLI and Makefile store root.
DEFAULT_STORE_ROOT: Path = Path("data")

#: Default report path; ``runs/`` is already git-ignored for local artefacts.
DEFAULT_OUT: Path = Path("runs/bench/pipeline.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="smoke")
    parser.add_argument("--store-root", default=str(DEFAULT_STORE_ROOT))
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--repeats", type=int, default=1, help="measured repeats per stage (>= 1)")
    parser.add_argument(
        "--warmup", type=int, default=0, help="discarded warmup calls per stage (>= 0)"
    )
    parser.add_argument(
        "--real-store",
        action="store_true",
        help="read real bars from --store-root instead of synthetic data",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help=(
            "bars per stage; default is a bounded real sample or the synthetic size, "
            "0 means the whole real store"
        ),
    )
    parser.add_argument("--strict", action="store_true", help="abort on the first failed stage")
    return parser


def _stages(args: argparse.Namespace) -> list[Stage]:
    common = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "real_store": args.real_store,
        "rows": args.rows,
        "repeats": args.repeats,
        "warmup": args.warmup,
    }
    stages: list[Stage] = []
    try:
        stages.append(features_build_stage(args.store_root, **common))
        stages.append(dataset_build_stage(args.store_root, **common))
        stages.append(discovery_smoke_stage(args.store_root, **common))
        stages.append(evidence_build_stage(args.store_root, **common))
        stages.append(paper_advance_stage(args.store_root, **common))
        if args.suite == "full":
            stages.append(walk_forward_stage(args.store_root, **common))
    except BaseException:
        # A later factory can fail after an earlier one created a temp store.
        for stage in stages:
            if stage.cleanup is not None:
                with contextlib.suppress(Exception):
                    stage.cleanup()
        raise
    return stages


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_suite(_stages(args), Path(args.out), strict=args.strict)
    except (FileNotFoundError, StageError, ValueError) as exc:
        print(
            json.dumps({"error": str(exc), "type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - one JSON line beats a traceback
        print(
            json.dumps({"error": str(exc), "type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
