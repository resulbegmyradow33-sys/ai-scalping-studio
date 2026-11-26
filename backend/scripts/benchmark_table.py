"""Merge benchmark reports into one Markdown table.

This script does not run models; it reads the JSON written by
``benchmark_ml``, ``benchmark_nn`` and ``benchmark_ppo`` and renders a single
wide table. A missing report is skipped, not invented. Throughput units differ
by source (rows/s for ML is not measured; windows/s for NN; steps/s for PPO), so
the column header says so.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

COLUMNS = (
    "model",
    "source",
    "device",
    "fit/train s",
    "predict ms",
    "throughput",
    "accuracy",
    "parameters",
)


def _load(path: str | None) -> list[dict[str, object]]:
    if path is None:
        return []
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path}: report has no rows list")
    return [row for row in rows if isinstance(row, dict)]


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _row(row: dict[str, object], source: str) -> tuple[str, ...]:
    if row.get("ok") is False:
        return (str(row.get("model", "?")), source, "-", "-", "-", "-", "-", f"error: {row.get('error', '?')}")
    if source == "ml":
        return (
            str(row.get("model")),
            "ml",
            "-",
            _fmt(row.get("fit_seconds")),
            _fmt(row.get("predict_ms")),
            "-",
            _fmt(row.get("accuracy")),
            "-",
        )
    if source == "nn":
        return (
            str(row.get("model")),
            "nn",
            str(row.get("device", "-")),
            "-",
            _fmt(row.get("latency_ms_median")),
            f"{_fmt(row.get('throughput_windows_per_s'), 1)} win/s",
            "-",
            _fmt(row.get("parameters"), 0),
        )
    return (
        str(row.get("model")),
        "ppo",
        str(row.get("device", "-")),
        _fmt(row.get("train_seconds")),
        _fmt(row.get("predict_ms_median")),
        f"{_fmt(row.get('steps_per_second'), 1)} steps/s",
        "-",
        "-",
    )


def render(rows: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge benchmark JSON reports.")
    parser.add_argument("--ml-report", default=None)
    parser.add_argument("--nn-report", default=None)
    parser.add_argument("--ppo-report", default=None)
    parser.add_argument("--out", default=None, help="optional Markdown path")
    args = parser.parse_args(argv)

    rows: list[tuple[str, ...]] = []
    for source, path in (
        ("ml", args.ml_report),
        ("nn", args.nn_report),
        ("ppo", args.ppo_report),
    ):
        rows.extend(_row(row, source) for row in _load(path))
    if not rows:
        raise SystemExit("no benchmark reports supplied")
    table = render(rows)
    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "markdown": table}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
