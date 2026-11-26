"""Unified benchmark table: merge, formatting and failure rows."""

from __future__ import annotations

import json
import subprocess
import sys


def report(path, rows):
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return str(path)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_table", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_merges_three_sources_into_one_table(tmp_path) -> None:
    ml = report(
        tmp_path / "ml.json",
        [{"model": "logistic", "ok": True, "fit_seconds": 0.12, "predict_ms": 1.5, "accuracy": 0.62}],
    )
    nn = report(
        tmp_path / "nn.json",
        [
            {
                "model": "gru",
                "ok": True,
                "device": "cpu",
                "parameters": 12345,
                "latency_ms_median": 2.0,
                "throughput_windows_per_s": 400.0,
            }
        ],
    )
    ppo = report(
        tmp_path / "ppo.json",
        [
            {
                "model": "ppo",
                "ok": True,
                "device": "cpu",
                "train_seconds": 3.0,
                "steps_per_second": 85.0,
                "predict_ms_median": 0.4,
            }
        ],
    )
    out = tmp_path / "table.md"

    result = run(
        "--ml-report", ml, "--nn-report", nn, "--ppo-report", ppo, "--out", str(out)
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["rows"] == 3
    table = out.read_text()
    assert "| model | source | device |" in table
    assert "| logistic | ml |" in table
    assert "| gru | nn | cpu |" in table and "400.0 win/s" in table
    assert "| ppo | ppo | cpu |" in table and "85.0 steps/s" in table


def test_failed_rows_and_missing_reports(tmp_path) -> None:
    ml = report(tmp_path / "ml.json", [{"model": "xgboost", "ok": False, "error": "no libomp"}])
    result = run("--ml-report", ml)

    assert result.returncode == 0, result.stderr
    assert "error: no libomp" in json.loads(result.stdout)["markdown"]

    empty = run()
    assert empty.returncode != 0
