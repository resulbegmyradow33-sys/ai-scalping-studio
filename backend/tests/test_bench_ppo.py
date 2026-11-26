"""PPO benchmark script: JSON contract and guard rails."""

from __future__ import annotations

import json
import subprocess
import sys


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.benchmark_ppo", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def test_tiny_ppo_row_is_measured() -> None:
    result = run(
        "--bars",
        "1200",
        "--timesteps",
        "64",
        "--n-steps",
        "32",
        "--batch-size",
        "16",
        "--predictions",
        "10",
        "--device",
        "cpu",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n") == 1
    row = json.loads(result.stdout)["rows"][0]
    assert row["model"] == "ppo"
    assert row["ok"] is True
    assert row["device"] == "cpu"
    assert row["train_seconds"] > 0
    assert row["steps_per_second"] > 0
    assert row["predict_ms_median"] > 0


def test_bad_arguments_are_rejected() -> None:
    small = run("--bars", "100")
    assert small.returncode != 0

    short = run("--timesteps", "8", "--n-steps", "32")
    assert short.returncode != 0
