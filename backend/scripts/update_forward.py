"""Weekly forward validation: fetch new weeks, rebuild, test, record.

The GitHub mirror publishes one gzip CSV per week. This script downloads any
missing week from the current forward window, re-runs the M1/flow ingest, then
runs the promoted card unchanged and appends the result to
``runs/forward_history.jsonl``.

It is deliberately separate from the nightly rule calibrator: forward data must
never feed parameter selection, or the forward test stops being forward.

Usage:
    python -m scripts.update_forward
    python -m scripts.update_forward --weeks 22-50
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["main"]

_BASE = "https://raw.githubusercontent.com/Touhid-HZ/xauusd-tick-data-2026/main/data/tick/"


def _parse_weeks(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in value.split(",") if part]


def _download(directory: Path, week: int) -> str:
    name = f"XAUUSD_2026_w{week:02d}.csv.gz"
    target = directory / name
    if target.exists() and target.stat().st_size > 1000:
        return "skip"
    try:
        urllib.request.urlretrieve(_BASE + name, target)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - a missing future week is normal
        return f"miss:{exc}"


def _json_tail(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _run(module: str, args: list[str]) -> str:
    completed = subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{module} failed: {completed.stderr.strip()[-800:]}")
    return completed.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly forward validation")
    parser.add_argument("--dir", default="../data-2026-ticks")
    parser.add_argument("--cache", default="../data-2026-cache")
    parser.add_argument("--store", default="../data-forward-store")
    parser.add_argument("--flow-root", default="../data-forward-flow")
    parser.add_argument("--history", default="runs/forward_history.jsonl")
    parser.add_argument("--weeks", default="22-50")
    parser.add_argument("--force", action="store_true", help="rerun even without new weeks")
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-11-21")
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    weeks = _parse_weeks(args.weeks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(lambda week: (week, _download(directory, week)), weeks))
    print(
        json.dumps(
            {
                "event": "downloaded",
                "ok": sum(1 for _, status in downloads if status == "ok"),
                "skip": sum(1 for _, status in downloads if status == "skip"),
                "missing": [week for week, status in downloads if status.startswith("miss:")],
            }
        ),
        flush=True,
    )

    new_files = sum(1 for _, status in downloads if status == "ok")
    if new_files == 0 and not args.force and Path(args.history).exists():
        print(
            json.dumps(
                {"event": "skipped", "reason": "no new weekly file", "history": args.history}
            ),
            flush=True,
        )
        return 0

    _run(
        "scripts.ingest_2026_ticks",
        [
            "--dir",
            args.dir,
            "--root",
            args.store,
            "--flow-root",
            args.flow_root,
            "--cache",
            args.cache,
        ],
    )
    stdout = _run(
        "scripts.forward_test",
        [
            "--start",
            args.start,
            "--end",
            args.end,
            "--history-store",
            "../data-ticks5-store",
            "--history-flow",
            "../data-flow-store",
            "--forward-store",
            args.store,
            "--forward-flow",
            args.flow_root,
        ],
    )
    result = _json_tail(stdout)
    if result is None:
        raise RuntimeError("forward_test produced no JSON result")
    record = {
        "event": "forward_update",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "weeks": weeks,
        **result,
    }
    history = Path(args.history)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"event": "recorded", "history": str(history)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
