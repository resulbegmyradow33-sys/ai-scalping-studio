"""Benchmark helper: shape of the table and its guard rails."""

from __future__ import annotations

import pytest

from scalping.nn.benchmark import benchmark_models, process_rss_mb


def test_benchmark_returns_one_row_per_model_in_order() -> None:
    rows = benchmark_models(
        ["mlp", "gru", "tcn"],
        n_features=6,
        window=12,
        batch=2,
        repeats=2,
        warmup=1,
        hidden=8,
    )

    assert [row.model for row in rows] == ["mlp", "gru", "tcn"]
    for row in rows:
        assert row.parameters > 0
        assert row.parameter_mb == pytest.approx(row.parameters * 4 / (1024 * 1024))
        assert row.latency_ms_median > 0
        assert row.throughput_windows_per_s > 0
        assert row.device == "cpu"
        assert 0 < row.rss_high_water_mb < 100_000
        assert set(row.to_dict()) == {
            "model",
            "parameters",
            "parameter_mb",
            "latency_ms_median",
            "throughput_windows_per_s",
            "rss_high_water_mb",
            "device",
        }
    assert rows[0].parameters != rows[1].parameters


def test_benchmark_rejects_bad_requests() -> None:
    with pytest.raises(ValueError, match="unknown models"):
        benchmark_models(["spectral"], n_features=2, window=2)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="batch"):
        benchmark_models(["mlp"], n_features=2, window=2, batch=0)
    with pytest.raises(ValueError, match="repeats"):
        benchmark_models(["mlp"], n_features=2, window=2, repeats=0)


def test_process_rss_is_plausible() -> None:
    assert process_rss_mb() > 0


def test_benchmark_script_prints_one_json_line(tmp_path) -> None:
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_nn",
            "--models",
            "mlp,gru",
            "--n-features",
            "4",
            "--window",
            "8",
            "--batch",
            "2",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--hidden",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert [row["model"] for row in payload["rows"]] == ["mlp", "gru"]
