"""Sequence-model profiler: JSON contract and argument guards."""

from __future__ import annotations

import json
import subprocess
import sys


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.profile_nn", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def test_tiny_profile_has_ops_and_memory_rows() -> None:
    result = run(
        "--model",
        "mlp",
        "--n-features",
        "4",
        "--window",
        "8",
        "--batch",
        "2",
        "--hidden",
        "8",
        "--steps",
        "2",
        "--device",
        "cpu",
    )

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["model"] == "mlp"
    assert payload["device"] == "cpu"
    assert payload["parameters"] > 0
    assert payload["top_cpu"] and payload["top_memory"]
    assert payload["top_cpu"][0]["calls"] >= 1
    assert "Self CPU" in payload["table"]


def test_bad_arguments_are_rejected() -> None:
    with_zero = run("--batch", "0")
    assert with_zero.returncode != 0

    unknown = run("--model", "magic")
    assert unknown.returncode != 0
