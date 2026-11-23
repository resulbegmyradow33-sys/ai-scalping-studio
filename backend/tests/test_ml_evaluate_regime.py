"""Regime breakdown of the classifier's trading evaluation.

The breakdown is a join, not a second backtest: it must partition exactly the
closed trades the headline metrics were computed from, attach only labels that
were current at each entry, and admit missing coverage instead of dropping it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.evaluate import _regime_breakdown, evaluate_model
from scalping.ml.train import TrainSpec, train_classifier
from scalping.types import Side, Trade

T0 = datetime(2026, 1, 5, tzinfo=UTC)
ROW_KEYS = {"regime", "trades", "win_rate", "expectancy_r", "total_r", "pnl", "profit_factor"}


def _learnable_frame(n: int = 700, valid_from: int = 10) -> FeatureFrame:
    """Alternating +/-1 feature that drives the next bar's return."""
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


@pytest.fixture(scope="module")
def trained() -> tuple[FeatureFrame, object]:
    frame = _learnable_frame()
    result = train_classifier(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        spec=TrainSpec(model="logistic", horizon=1, purge_bars=1, embargo_bars=1),
    )
    return frame, result


def _labels(boundaries: dict[datetime, str]) -> pl.DataFrame:
    stamps = sorted(boundaries)
    return pl.DataFrame({"ts": stamps, "regime": [boundaries[stamp] for stamp in stamps]}).cast(
        {"ts": pl.Datetime("us", "UTC")}
    )


def _trade(entry: datetime, r_multiple: float, *, pnl: float | None = None) -> Trade:
    return Trade(
        entry_ts=entry,
        exit_ts=entry + timedelta(minutes=5),
        side=Side.LONG,
        lot=0.1,
        entry=100.0,
        exit=101.0,
        sl=99.0,
        tp=102.0,
        pnl=r_multiple * 100.0 if pnl is None else pnl,
        costs=1.0,
        r_multiple=r_multiple,
        reason="tp",
        strategy="ml-logistic-validation",
    )


def test_rule_breakdown_partitions_every_trade(trained) -> None:
    frame, result = trained
    evaluation = evaluate_model(result, frame, Settings(), split="validation")

    rows = evaluation.regime["rows"]
    assert rows, "the rule engine should label the evaluated window"
    assert all(set(row) == ROW_KEYS for row in rows)
    assert evaluation.regime["unmatched_trades"] == 0
    assert sum(row["trades"] for row in rows) == evaluation.metrics["trades"] > 0


def test_per_regime_total_r_sums_to_the_headline_total(trained) -> None:
    frame, result = trained
    evaluation = evaluate_model(result, frame, Settings(), split="validation")

    headline_total = evaluation.metrics["expectancy_r"] * evaluation.metrics["trades"]
    bucket_total = sum(row["total_r"] for row in evaluation.regime["rows"])
    assert bucket_total == pytest.approx(headline_total, abs=1e-9)
    assert evaluation.regime["unmatched_trades"] == 0


def test_null_and_nan_labels_become_unknown(trained) -> None:
    frame, result = trained
    stamps = np.asarray(frame.ts, dtype="datetime64[us]")

    null_labels = pl.DataFrame(
        {"ts": stamps, "regime": pl.Series([None] * len(stamps), dtype=pl.String)}
    )
    nan_labels = pl.DataFrame(
        {"ts": stamps, "regime": pl.Series([float("nan")] * len(stamps), dtype=pl.Float64)}
    )

    for labels in (null_labels, nan_labels):
        evaluation = evaluate_model(
            result, frame, Settings(), split="validation", regime_labels=labels
        )
        assert evaluation.regime["rows"] == []
        assert evaluation.regime["unmatched_trades"] == evaluation.metrics["trades"]
        assert evaluation.metrics["trades"] > 0


def test_labels_after_the_entry_never_attach() -> None:
    labels = _labels({T0: "before", T0 + timedelta(minutes=10): "after"})
    trades = [
        _trade(T0 + timedelta(minutes=5), 1.0),
        _trade(T0 + timedelta(minutes=10), -0.5),
        _trade(T0 - timedelta(minutes=1), 0.25),
    ]

    breakdown = _regime_breakdown(trades, labels)
    rows = {row["regime"]: row for row in breakdown["rows"]}

    assert set(rows) == {"before", "after"}
    assert rows["before"]["trades"] == 1
    assert rows["before"]["total_r"] == pytest.approx(1.0)
    assert rows["after"]["trades"] == 1
    assert breakdown["unmatched_trades"] == 1
    assert sum(row["trades"] for row in breakdown["rows"]) + 1 == len(trades)


def test_future_only_labels_leave_every_trade_unmatched(trained) -> None:
    frame, result = trained
    last = np.asarray([frame.ts.max()], dtype="datetime64[us]")
    labels = pl.DataFrame({"ts": last, "regime": ["late"]})

    evaluation = evaluate_model(result, frame, Settings(), split="validation", regime_labels=labels)

    assert evaluation.regime["rows"] == []
    assert evaluation.regime["unmatched_trades"] == evaluation.metrics["trades"]


def test_external_labels_override_rule_labels(trained) -> None:
    frame, result = trained
    stamps = np.asarray(frame.ts, dtype="datetime64[us]")
    labels = pl.DataFrame({"ts": stamps, "regime": ["custom"] * len(stamps)})

    evaluation = evaluate_model(result, frame, Settings(), split="validation", regime_labels=labels)

    assert [row["regime"] for row in evaluation.regime["rows"]] == ["custom"]
    assert evaluation.regime["rows"][0]["trades"] == evaluation.metrics["trades"]


def test_no_labels_and_no_rules_leave_regime_empty(trained) -> None:
    frame, result = trained
    evaluation = evaluate_model(result, frame, Settings(), split="validation", compute_regime=False)
    assert evaluation.regime == {}


def test_stress_runs_stay_headline_only(trained) -> None:
    frame, result = trained
    evaluation = evaluate_model(result, frame, Settings(), split="validation", stress=(2.0,))
    assert set(evaluation.stress["x2"]) == set(evaluation.metrics)
    assert "regime" not in evaluation.stress["x2"]
    assert evaluation.regime["rows"]


# --- CLI ---------------------------------------------------------------------
def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_cli_ml_evaluate_reports_regime(tmp_path) -> None:
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
    trained = _run(
        "ml",
        "train",
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
        str(tmp_path / "ml"),
    )
    assert trained.returncode == 0, trained.stderr
    run_dir = Path(json.loads(trained.stdout)["model_path"]).parent

    plain = _run("ml", "evaluate", "--run", str(run_dir), "--root", str(root))
    assert plain.returncode == 0, plain.stderr
    plain_payload = json.loads(plain.stdout)
    assert plain_payload["regime"] == {}
    assert {"metrics", "hold", "stress", "split"} <= set(plain_payload)

    labelled = _run("ml", "evaluate", "--run", str(run_dir), "--root", str(root), "--regime")
    assert labelled.returncode == 0, labelled.stderr
    payload = json.loads(labelled.stdout)
    assert set(plain_payload) == set(payload)
    assert payload["metrics"]["trades"] > 0
    assert isinstance(payload["regime"]["rows"], list)
    assert payload["regime"]["rows"]
    assert isinstance(payload["regime"]["unmatched_trades"], int)
    for row in payload["regime"]["rows"]:
        assert set(row) == ROW_KEYS
    total = sum(row["trades"] for row in payload["regime"]["rows"])
    assert total + payload["regime"]["unmatched_trades"] == payload["metrics"]["trades"]

    labels_path = tmp_path / "labels.parquet"
    pl.DataFrame({"ts": [datetime(2020, 1, 1, tzinfo=UTC)], "regime": ["all"]}).write_parquet(
        labels_path
    )
    external = _run(
        "ml",
        "evaluate",
        "--run",
        str(run_dir),
        "--root",
        str(root),
        "--regime-labels",
        str(labels_path),
    )
    assert external.returncode == 0, external.stderr
    external_payload = json.loads(external.stdout)
    assert [row["regime"] for row in external_payload["regime"]["rows"]] == ["all"]
    assert external_payload["regime"]["unmatched_trades"] == 0
    assert external_payload["regime"]["rows"][0]["trades"] == payload["metrics"]["trades"]
