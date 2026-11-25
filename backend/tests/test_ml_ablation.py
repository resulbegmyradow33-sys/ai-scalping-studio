"""Feature-group ablation: fixed protocol, deltas, refusals and CLI shape."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.features.registry import DEFAULT_GROUPS, FEATURE_GROUPS, columns_for
from scalping.ml.ablation import ABLATION_METRICS, run_ablation
from scalping.ml.train import TrainSpec


def _learnable_frame(n: int = 700, valid_from: int = 10) -> FeatureFrame:
    """Feature 0 (``ret_1``) alternates +/-1 and drives the next bar's return."""
    signal = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for index in range(1, n):
        close[index] = close[index - 1] * (1.0 + 0.002 * signal[index - 1])
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    X[:, 0] = signal
    ts = np.array(
        [np.datetime64("2026-01-05T00:00", "us") + np.timedelta64(i, "m") for i in range(n)]
    )
    ohlc = np.column_stack((close, close + 0.01, close - 0.01, close))
    return FeatureFrame(
        X=X,
        ts=ts,
        ohlc=ohlc,
        spread=np.full(n, 0.2),
        atr=np.full(n, 0.5),
        valid_from=valid_from,
        columns=list(FEATURE_COLUMNS),
    )


def _spec(**overrides: object) -> TrainSpec:
    values: dict[str, object] = {
        "model": "logistic",
        "horizon": 1,
        "groups": DEFAULT_GROUPS,
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "purge_bars": 1,
        "embargo_bars": 1,
        "seed": 0,
    }
    values.update(overrides)
    return TrainSpec(**values)  # type: ignore[arg-type]


def _ablate(frame: FeatureFrame | None = None, **overrides: object):
    return run_ablation(
        _learnable_frame() if frame is None else frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=_spec(**overrides),
    )


def test_group_registry_partitions_every_feature_column() -> None:
    """The ablation assumes each column belongs to exactly one group."""
    assert len(FEATURE_GROUPS) == 7
    assert len(FEATURE_COLUMNS) == 59
    owner: dict[str, str] = {}
    for name, columns in FEATURE_GROUPS.items():
        assert columns, f"group {name!r} is empty"
        for column in columns:
            assert column in FEATURE_COLUMNS, f"{column!r} is not a feature column"
            assert column not in owner, f"{column!r} is in {name!r} and {owner[column]!r}"
            owner[column] = name
    assert set(owner) == set(FEATURE_COLUMNS)
    assert columns_for() == FEATURE_COLUMNS


def test_protocol_is_deterministic() -> None:
    first = _ablate().to_dict()
    second = _ablate().to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_runs_are_all_leave_one_out_and_single_group() -> None:
    result = _ablate()
    expected = (
        ["all"]
        + [f"without:{group}" for group in DEFAULT_GROUPS]
        + [f"only:{group}" for group in DEFAULT_GROUPS]
    )
    assert [run.name for run in result.runs] == expected
    assert [run.kind for run in result.runs] == ["all"] + ["leave_one_out"] * 7 + [
        "single_group"
    ] * 7
    assert len(result.runs) == 1 + 2 * len(DEFAULT_GROUPS)
    assert result.protocol["protocol"] == "feature-group-ablation-v1"
    assert result.protocol["runs"] == len(result.runs) == 15
    assert result.protocol["hyperparameter_search"] is False
    assert result.columns == FEATURE_COLUMNS
    for run in result.runs:
        assert run.columns == columns_for(run.groups)
        assert set(run.metrics) == set(ABLATION_METRICS) | {"rows"}


def test_deltas_are_run_minus_all() -> None:
    result = _ablate()
    reference = result.runs[0]
    assert reference.name == "all"
    assert reference.delta is None
    for run in result.runs[1:]:
        assert run.delta is not None
        assert set(run.delta) == set(ABLATION_METRICS)
        for metric in ABLATION_METRICS:
            assert run.delta[metric] == pytest.approx(
                run.metrics[metric] - reference.metrics[metric], abs=1e-12
            )

    by_name = {run.name: run for run in result.runs}
    # The only real signal is ret_1, which lives in the price group.
    assert by_name["only:price"].metrics["accuracy"] > 0.9
    assert by_name["only:price"].delta["accuracy"] >= 0.0
    assert by_name["without:price"].delta["accuracy"] < 0.0
    # A group of all-zero columns cannot carry the signal.
    assert by_name["only:volume"].delta["accuracy"] < 0.0


def test_test_split_is_reserved_and_never_scored() -> None:
    """NaN test features would break any prediction, so success proves no read."""
    frame = _learnable_frame()
    X = frame.X.copy()
    X[int(X.shape[0] * 0.9) :] = np.nan
    corrupted = FeatureFrame(
        X=X,
        ts=frame.ts,
        ohlc=frame.ohlc,
        spread=frame.spread,
        atr=frame.atr,
        valid_from=frame.valid_from,
        columns=frame.columns,
    )
    result = _ablate(corrupted)
    assert result.protocol["scored_split"] == "validation"
    assert result.protocol["test_scored"] is False
    assert result.protocol["test_rows_reserved"] == result.split_counts["test"] > 0
    for run in result.runs:
        assert run.metrics["rows"] == result.split_counts["validation"]
    assert "test" not in result.to_dict()


def test_ablation_ignores_evaluate_test() -> None:
    result = _ablate(evaluate_test=True)
    assert result.protocol["test_scored"] is False
    assert result.protocol["evaluate_test_ignored"] is True
    assert result.to_dict()["runs"] == _ablate().to_dict()["runs"]


def test_refuses_empty_group_selection() -> None:
    with pytest.raises(ValueError, match="at least one feature group"):
        _ablate(groups=())


def test_refuses_single_group_selection() -> None:
    with pytest.raises(ValueError, match="at least two feature groups"):
        _ablate(groups=("price",))


def test_refuses_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown feature groups: nope"):
        _ablate(groups=("price", "nope"))


def test_refuses_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        _ablate(model="magic")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_cli_ablation_prints_one_json_line_and_writes_report(tmp_path: Path) -> None:
    root = tmp_path / "store"
    synth = _run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr

    out = tmp_path / "ablation.json"
    completed = _run(
        "ml",
        "ablation",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--model",
        "logistic",
        "--groups",
        "price,momentum",
        "--horizon",
        "5",
        "--purge",
        "5",
        "--embargo",
        "2",
        "--root",
        str(root),
        "--out",
        str(out),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["model"] == "logistic"
    assert payload["spec"]["groups"] == ["price", "momentum"]
    assert [run["name"] for run in payload["runs"]] == [
        "all",
        "without:price",
        "without:momentum",
        "only:price",
        "only:momentum",
    ]
    assert payload["protocol"]["test_scored"] is False
    assert set(payload["runs"][0]["metrics"]) == set(ABLATION_METRICS) | {"rows"}
    assert payload["runs"][0]["delta_vs_all"] is None
    assert "out" in payload
    report = json.loads(out.read_text())
    assert "out" not in report
    assert report["dataset_id"] == payload["dataset_id"]
    assert report["runs"] == payload["runs"]


def test_cli_ablation_without_data_exits_2(tmp_path: Path) -> None:
    completed = _run(
        "ml",
        "ablation",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--root",
        str(tmp_path / "empty"),
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    error = json.loads(completed.stderr)
    assert "XAUUSD" in error["error"] and "M1" in error["error"]
