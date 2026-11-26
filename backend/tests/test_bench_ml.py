"""Classical ML benchmark script: JSON contract and guard rails."""

from __future__ import annotations

import json
import subprocess
import sys


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_ml", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_logistic_row_is_measured() -> None:
    result = run(
        "--models",
        "logistic",
        "--rows",
        "300",
        "--features",
        "8",
        "--classes",
        "3",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    row = payload["rows"][0]
    assert row["model"] == "logistic"
    assert row["ok"] is True
    assert row["fit_seconds"] > 0
    assert row["predict_ms"] > 0
    assert 0.0 <= row["accuracy"] <= 1.0
    assert row["train_rows"] == 210 and row["test_rows"] == 90


def test_unknown_model_and_small_input_are_rejected() -> None:
    unknown = run("--models", "magic", "--rows", "100")
    assert unknown.returncode != 0
    assert "unknown models" in (unknown.stderr + unknown.stdout)

    small = run("--rows", "10")
    assert small.returncode != 0


def test_report_can_be_written(tmp_path) -> None:
    out = tmp_path / "bench.json"
    result = run(
        "--models",
        "logistic",
        "--rows",
        "120",
        "--features",
        "5",
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["rows"][0]["model"] == "logistic"
