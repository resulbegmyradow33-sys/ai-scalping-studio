"""Tests for the feature-cache measurement script.

The script reads a real store and writes an evidence JSON; CI has neither, so
the test builds a small synthetic parquet store and checks the contract: the
report is printed and written identically, the measured case carries the cache
evidence, and the store is not touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider

START = datetime(2026, 1, 5, tzinfo=UTC)


def _store(root: Path, rows: int = 600) -> tuple[Path, int]:
    bars = SyntheticProvider(seed=7).fetch("XAUUSD", START, START + timedelta(minutes=rows))
    ParquetStore(root).write("XAUUSD", "M1", bars)
    return root, bars.height


def _run(backend: Path, case: str, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/bench_feature_cache.py",
            "--case",
            case,
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--out",
            str(out),
        ],
        cwd=backend,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def test_cli_measures_one_bounded_case_without_touching_the_store(tmp_path: Path) -> None:
    backend = Path(__file__).resolve().parents[1]
    store, available = _store(tmp_path / "store")
    before = sorted(path.relative_to(store) for path in store.rglob("*"))

    out = tmp_path / "cache-report.json"
    proc = _run(backend, f"tiny={store}:M1:200", out)

    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert printed == written
    assert printed["schema"] == "scalping.bench.cache/1"
    assert printed["claims"]["m2_acceptance"] is False
    assert printed["claims"]["validation"] is False

    case = printed["cases"][0]
    assert case["label"] == "tiny"
    assert case["rows"] == 200
    assert case["rows_available"] == available
    assert set(case["stages"]) == {"frame_meta", "build_features", "key_for", "save", "load"}
    assert all(stage["ok"] for stage in case["stages"].values())
    assert case["entry"]["entry_bytes"] > case["entry"]["array_bytes"] > 0
    assert case["call_loop"]["hit_ratio"] == 0.9
    assert case["corruption"]["load_returns_none"] is True
    assert case["corruption"]["quarantined_files"] == 2
    assert case["invalidation"]["settings_key_differs"] is True
    assert case["invalidation"]["data_key_differs"] is True
    assert case["version"]["old_entry_rejected"] is True

    after = sorted(path.relative_to(store) for path in store.rglob("*"))
    assert after == before, "the benchmark must not write to the store"


def test_cli_rejects_a_malformed_case(tmp_path: Path) -> None:
    backend = Path(__file__).resolve().parents[1]
    proc = _run(backend, "broken", tmp_path / "unused.json")
    assert proc.returncode == 2
    assert "LABEL=ROOT:TIMEFRAME:ROWS" in json.loads(proc.stderr)["error"]
