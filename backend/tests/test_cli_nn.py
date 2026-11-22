"""End-to-end ``nn train`` / ``nn evaluate`` on a stored H1 slice.

The paths run the real module as a subprocess, so argparse, the one-JSON-line
contract, exit codes and the checkpoint round trip are all exercised. The model
is deliberately tiny: this test is about wiring, not about learning an edge.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scalping.nn import MODEL_NAMES

FIXTURES = Path(__file__).parent / "fixtures"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def seed_h1(root: Path) -> None:
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-01",
        "--end",
        "2026-01-31",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr
    derive = run(
        "data",
        "derive",
        "--root",
        str(root),
        "--symbol",
        "XAUUSD",
        "--source",
        "M1",
        "--timeframes",
        "H1",
    )
    assert derive.returncode == 0, derive.stderr


def test_cli_model_choices_match_the_zoo() -> None:
    from scalping import cli

    assert tuple(cli._NN_MODELS) == tuple(MODEL_NAMES)


def test_nn_train_and_evaluate_roundtrip(tmp_path) -> None:
    seed_h1(tmp_path)
    checkpoint = tmp_path / "runs" / "gru.pt"
    report_path = tmp_path / "runs" / "train.json"

    trained = run(
        "nn",
        "train",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "H1",
        "--model",
        "mlp",
        "--window",
        "10",
        "--hidden",
        "8",
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--device",
        "cpu",
        "--root",
        str(tmp_path),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(report_path),
    )

    assert trained.returncode == 0, trained.stderr
    assert trained.stderr == ""
    assert trained.stdout.count("\n") == 1
    summary = json.loads(trained.stdout)
    assert summary["checkpoint"] == str(checkpoint)
    assert summary["stopped"] in {"epochs", "early_stop"}
    assert summary["device"] == "cpu"
    assert summary["counts"]["train"] > 0

    assert checkpoint.exists()
    report = json.loads(report_path.read_text())
    assert report["window"] == 10
    assert len(report["history"]) == 1
    assert report["dataset_id"].startswith("XAUUSD:H1:")

    evaluated = run(
        "nn",
        "evaluate",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "H1",
        "--root",
        str(tmp_path),
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
    )

    assert evaluated.returncode == 0, evaluated.stderr
    metrics = json.loads(evaluated.stdout)
    assert metrics["rows"] > 0
    assert metrics["split"] == "test"
    assert metrics["trading_claim"] is False
    assert metrics["dataset_id"].startswith("XAUUSD:H1:")
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_nn_train_without_data_exits_2(tmp_path) -> None:
    result = run(
        "nn",
        "train",
        "--root",
        str(tmp_path),
        "--checkpoint",
        str(tmp_path / "never.pt"),
        "--out",
        str(tmp_path / "never.json"),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert "no XAUUSD H1 bars" in error["error"]
