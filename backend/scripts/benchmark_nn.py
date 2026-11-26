"""Benchmark the sequence-model zoo at a fixed input shape.

Prints one JSON line with parameters, median forward latency, throughput and
the process high-water RSS for every requested model. The RSS figure is
process-wide, not isolated per model; run one model per invocation for that.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scalping.nn.benchmark import benchmark_models
from scalping.nn.models import MODEL_NAMES, ModelName


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the sequence-model zoo.")
    parser.add_argument("--models", default="all", help="comma-separated names, or all")
    parser.add_argument("--n-features", type=int, default=59)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument(
        "--compare-devices",
        action="store_true",
        help="also run on MPS when the self-test passes",
    )
    parser.add_argument("--out", default=None, help="optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    names: tuple[ModelName, ...] | None = None
    if args.models.strip().lower() != "all":
        requested = tuple(part.strip() for part in args.models.split(",") if part.strip())
        unknown = [name for name in requested if name not in MODEL_NAMES]
        if unknown:
            raise SystemExit(f"unknown models: {unknown}; choose from {MODEL_NAMES}")
        names = requested  # type: ignore[assignment]
    devices: list[str] = [args.device]
    if args.compare_devices:
        from scalping.nn.device import mps_self_test

        if mps_self_test()[0]:
            devices = ["cpu", "mps"]
    rows = []
    for device in devices:
        rows.extend(
            benchmark_models(
                names,
                n_features=args.n_features,
                window=args.window,
                batch=args.batch,
                repeats=args.repeats,
                warmup=args.warmup,
                hidden=args.hidden,
                device=device,
            )
        )
    payload = {"rows": [row.to_dict() for row in rows]}
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
