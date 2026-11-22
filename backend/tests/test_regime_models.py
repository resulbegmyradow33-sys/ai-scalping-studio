"""Statistical regime models and their evaluation.

The fixtures are the same known-shape paths used by the rule tests: a trend, a
quiet range and a decline. Model assertions care about the contract
(deterministic, aligned, train-only fitting), not about winning a clustering
beauty contest.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scalping.regime import (
    MODEL_NAMES,
    REGIME_FEATURES,
    ModelConfig,
    RegimeModelError,
    build_regime_features,
    classify,
    dwell_stats,
    fit_predict,
    transition_matrix,
)

START = datetime(2026, 1, 5, tzinfo=UTC)


def bars(closes: object, *, wick: float = 0.05) -> pl.DataFrame:
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=i) for i in range(close.size)],
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "bid_close": close,
            "ask_close": close + 0.2,
            "spread": np.full(close.size, 0.2),
            "volume": np.full(close.size, 10, dtype=np.int64),
        }
    )


def three_regimes(n: int = 250) -> np.ndarray:
    up = 100.0 * np.exp(np.linspace(0.0, 0.08, n))
    side = up[-1] + 0.3 * np.sin(np.arange(n) / 7.0)
    down = side[-1] * np.exp(np.linspace(0.0, -0.08, n))
    return np.concatenate([up, side, down])


def test_features_are_finite_aligned_and_causal() -> None:
    path = three_regimes()
    features = build_regime_features(bars(path))

    assert features.columns == ["ts", *REGIME_FEATURES]
    assert features.height == path.size
    assert features.select(REGIME_FEATURES).to_numpy().shape == (path.size, len(REGIME_FEATURES))
    assert np.isfinite(features.select(REGIME_FEATURES).to_numpy()).all()

    mutated = path.copy()
    mutated[500:] = mutated[500:] * 2.0
    assert_frame_equal(features.head(500), build_regime_features(bars(mutated)).head(500))


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_models_are_deterministic_and_label_every_row(model: str) -> None:
    frame = bars(three_regimes())

    first = fit_predict(frame, model)  # type: ignore[arg-type]
    second = fit_predict(frame, model)  # type: ignore[arg-type]

    assert first.rows == frame.height
    assert first.labels.height == frame.height
    assert set(first.labels["state"].to_list()).issubset(set(first.states))
    assert first.labels["state"].n_unique() >= 2
    assert sum(first.counts().values()) == first.rows
    assert_frame_equal(first.labels, second.labels)


@pytest.mark.parametrize("model", ["kmeans", "gmm", "hmm", "supervised"])
def test_fitting_never_reads_past_the_training_slice(model: str) -> None:
    path = three_regimes()
    base = fit_predict(bars(path), model)  # type: ignore[arg-type]

    mutated = path.copy()
    mutated[600:] = mutated[600:] * 1.5
    after_future_change = fit_predict(bars(mutated), model)  # type: ignore[arg-type]

    # The training slice ends at 525 rows (70% of 750); everything the model
    # learned, including the scaler, must be untouched by later bars.
    assert base.train_rows == 525
    assert_frame_equal(base.labels.head(600), after_future_change.labels.head(600))


def test_supervised_baseline_learns_the_rule_labels() -> None:
    frame = bars(three_regimes())
    rule = classify(frame)["regime"]
    fit = fit_predict(frame, "supervised")

    predicted = fit.labels["state"].to_list()
    actual = rule.to_list()
    correct = sum(left == right for left, right in zip(predicted, actual))
    assert correct / fit.train_rows >= 0.6
    assert set(predicted).issubset(set(fit.states))


def test_too_short_a_frame_is_rejected() -> None:
    with pytest.raises(RegimeModelError, match="at least"):
        fit_predict(bars(np.linspace(100.0, 101.0, 40)))


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown regime model"):
        fit_predict(bars(three_regimes()), "spectral")  # type: ignore[arg-type]


def test_config_rejects_impossible_values() -> None:
    with pytest.raises(ValueError, match="n_states"):
        ModelConfig(n_states=1)
    with pytest.raises(ValueError, match="train_fraction"):
        ModelConfig(train_fraction=0.4)
    with pytest.raises(ValueError, match="seed"):
        ModelConfig(seed=-1)


def test_transition_matrix_counts_and_normalises() -> None:
    labels = pl.DataFrame(
        {
            "ts": list(range(6)),
            "state": ["a", "a", "b", "b", "b", "a"],
        }
    )
    matrix = transition_matrix(labels)
    lookup = {(row["from"], row["to"]): row for row in matrix.to_dicts()}

    assert lookup[("a", "a")]["count"] == 1
    assert lookup[("a", "b")]["count"] == 1
    assert lookup[("b", "b")]["count"] == 2
    assert lookup[("b", "a")]["count"] == 1
    assert lookup[("a", "b")]["probability"] == pytest.approx(0.5)

    totals = matrix.group_by("from").agg(pl.col("probability").sum())["probability"].to_list()
    assert all(abs(total - 1.0) < 1e-9 for total in totals)


def test_dwell_stats_describe_run_lengths() -> None:
    labels = pl.DataFrame(
        {
            "ts": list(range(6)),
            "state": ["a", "a", "b", "b", "b", "a"],
        }
    )
    stats = {row["state"]: row for row in dwell_stats(labels).to_dicts()}

    assert stats["a"]["runs"] == 2
    assert stats["a"]["bars"] == 3
    assert stats["a"]["max_run"] == 2
    assert stats["a"]["share"] == pytest.approx(0.5)
    assert stats["b"]["runs"] == 1
    assert stats["b"]["max_run"] == 3
    assert stats["b"]["share"] == pytest.approx(0.5)


def test_evaluation_handles_empty_labels() -> None:
    empty = pl.DataFrame(
        schema={"ts": pl.Datetime("us", "UTC"), "state": pl.String}
    )
    assert transition_matrix(empty).height == 0
    assert dwell_stats(empty).height == 0
