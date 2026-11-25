"""Regime-dependent member selection: causality, fallback and honest reporting.

The fixtures separate three concerns: the rule-label join (real bars, real
:func:`scalping.regime.rules.classify`), the weight arithmetic (hand-built OOF
matrices with known losses) and the report (a case where regime weighting is
*worse* than the best member, so the report cannot pretend otherwise).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.ensemble import (
    EnsembleConfig,
    OOFResult,
    RegimeSelectionConfig,
    RegimeWeightedEnsemble,
    RegimeWeights,
    RegimeWeightTable,
    align_regime_labels,
    fit_oof,
    fit_regime_weights,
    regime_selection_report,
)
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.cv import CVConfig
from scalping.regime import classify
from scalping.regime.strategy import UNKNOWN_REGIME

# ---------------------------------------------------------------------------
# Toy data shared with the ensemble tests, kept local to this module.
# ---------------------------------------------------------------------------


def _toy_frame(n: int = 400, valid_from: int = 10) -> FeatureFrame:
    signal = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    close = np.empty(n, dtype=np.float64)
    close[0] = 100.0
    for index in range(1, n):
        close[index] = close[index - 1] * (1.0 + 0.002 * signal[index - 1])
    X = np.zeros((n, len(FEATURE_COLUMNS)), dtype=np.float32)
    X[:, FEATURE_COLUMNS.index("ret_1")] = np.arange(n)
    X[:, FEATURE_COLUMNS.index("ret_5")] = signal
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


def _toy_bundle(frame: FeatureFrame) -> DatasetBundle:
    return build_dataset(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        groups=("price",),
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=2,
        embargo_bars=1,
    )


def _bars(frame: FeatureFrame) -> pl.DataFrame:
    """Minimal schema-correct bar frame for the rule engine."""
    ohlc = np.asarray(frame.ohlc, dtype=np.float64)
    close = ohlc[:, 3]
    return pl.DataFrame(
        {
            "ts": np.asarray(frame.ts, dtype="datetime64[us]"),
            "open": ohlc[:, 0],
            "high": ohlc[:, 1],
            "low": ohlc[:, 2],
            "close": close,
            "bid_close": close,
            "ask_close": close + 0.2,
            "spread": np.full(close.size, 0.2),
            "volume": np.zeros(close.size, dtype=np.int64),
        }
    )


class _ToyMember:
    """Predicts from ``ret_5``; ``classes_`` comes from the fold it was fitted on."""

    def __init__(self, *, strength: float = 0.3) -> None:
        self.strength = strength
        self.classes_ = np.array([], dtype=np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ToyMember:
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = np.asarray(X[:, 1], dtype=np.float64)
        up = 0.5 + self.strength * signal
        return np.column_stack((1.0 - up, up))


def _oof(
    probabilities: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    classes: tuple[int, ...] = (-1, 1),
) -> OOFResult:
    return OOFResult(
        members=tuple(probabilities),
        classes=classes,
        row_index=np.arange(y_true.size),
        y_true=y_true,
        probabilities=probabilities,
        folds=(),
    )


def _two_member_case() -> tuple[OOFResult, np.ndarray, np.ndarray]:
    """``a`` is strong in ``good`` and wrong in ``bad``; ``b`` is the reverse.

    All labels are ``+1`` and the probabilities are in ``(-1, 1)`` order, so the
    per-member log loss and the combined argmax are straightforward to check by
    hand instead of re-deriving them through a model.
    """
    good, bad = 60, 40
    y = np.ones(good + bad, dtype=np.int64)
    regimes = np.array(["good"] * good + ["bad"] * bad, dtype=object)

    a = np.empty((y.size, 2), dtype=np.float64)
    a[:good] = [0.2, 0.8]
    a[good:] = [0.65, 0.35]

    b = np.empty((y.size, 2), dtype=np.float64)
    b[:good] = [0.2, 0.8]
    b[good:] = [0.45, 0.55]

    return _oof({"a": a, "b": b}, y), regimes, y


# ---------------------------------------------------------------------------
# Regime labels: backward as-of and never future.
# ---------------------------------------------------------------------------


def test_align_regime_labels_is_backward_asof() -> None:
    bundle = _toy_bundle(_toy_frame())
    ts = bundle.ts
    labels = pl.DataFrame(
        {
            "ts": np.array([ts[10], ts[100]], dtype="datetime64[us]"),
            "regime": ["old", "new"],
        }
    )

    aligned = align_regime_labels(bundle, labels)

    assert aligned[0] == UNKNOWN_REGIME  # before the first label
    assert aligned[10] == "old"
    assert aligned[99] == "old"  # the newer label starts at row 100; not used early
    assert aligned[100] == "new"
    assert aligned[-1] == "new"


def test_align_regime_labels_does_not_use_future_bars() -> None:
    frame = _toy_frame()
    bundle = _toy_bundle(frame)
    cut = 200

    mutated = frame.X.copy(), frame.ohlc.copy()
    ohlc = mutated[1]
    ohlc[cut:] *= 3.0
    altered = FeatureFrame(
        X=frame.X,
        ts=frame.ts,
        ohlc=ohlc,
        spread=frame.spread,
        atr=frame.atr,
        valid_from=frame.valid_from,
        columns=frame.columns,
    )

    full = align_regime_labels(bundle, classify(_bars(frame)))
    changed = align_regime_labels(bundle, classify(_bars(altered)))

    np.testing.assert_array_equal(full[: cut - frame.valid_from], changed[: cut - frame.valid_from])
    # And the rule labels themselves are prefix-stable.
    assert_frame_equal(
        classify(_bars(frame)).head(cut),
        classify(_bars(altered)).head(cut),
    )


def test_align_regime_labels_rejects_bad_label_frames() -> None:
    bundle = _toy_bundle(_toy_frame())

    with pytest.raises(ValueError, match="missing columns"):
        align_regime_labels(bundle, pl.DataFrame({"ts": bundle.ts}))
    duplicated = np.array([bundle.ts[0], bundle.ts[0]], dtype="datetime64[us]")
    with pytest.raises(ValueError, match="duplicate"):
        align_regime_labels(
            bundle,
            pl.DataFrame({"ts": duplicated, "regime": ["a", "b"]}),
        )
    with pytest.raises(TypeError, match="datetime"):
        align_regime_labels(bundle, pl.DataFrame({"ts": [1, 2], "regime": ["a", "b"]}))

    absent = align_regime_labels(
        bundle,
        pl.DataFrame({"ts": np.array([bundle.ts[0]], dtype="datetime64[us]"), "regime": [None]}),
    )
    assert absent[0] == UNKNOWN_REGIME


# ---------------------------------------------------------------------------
# Weights: deterministic, thresholded and computed from OOF rows only.
# ---------------------------------------------------------------------------


def test_regime_weights_are_deterministic_and_order_independent() -> None:
    oof, regimes, _ = _two_member_case()
    config = RegimeSelectionConfig(min_rows_per_regime=30)

    first = fit_regime_weights(oof, regimes, config)
    second = fit_regime_weights(oof, regimes, config)
    assert first.to_dict() == second.to_dict()

    # A row permutation that moves labels and probabilities together must not
    # move the weights: they are means of the same per-row losses.
    order = np.random.default_rng(7).permutation(regimes.size)
    shuffled = OOFResult(
        members=oof.members,
        classes=oof.classes,
        row_index=np.arange(order.size),
        y_true=np.asarray(oof.y_true)[order],
        probabilities={name: np.asarray(oof.probabilities[name])[order] for name in oof.members},
        folds=(),
    )
    reordered = fit_regime_weights(shuffled, regimes[order], config)

    for label in first.regimes:
        left = first.per_regime[label].weights
        right = reordered.per_regime[label].weights
        assert set(left) == set(right)
        for name in left:
            assert left[name] == pytest.approx(right[name], rel=1e-12)
    for name, value in first.global_weights.items():
        assert value == pytest.approx(reordered.global_weights[name], rel=1e-12)


def test_regime_weights_follow_the_bucket_losses() -> None:
    oof, regimes, _ = _two_member_case()
    table = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=30))

    good = table.per_regime["good"]
    bad = table.per_regime["bad"]
    assert good.source == bad.source == "regime"
    assert good.weights["a"] == pytest.approx(0.5)
    assert good.weights["b"] == pytest.approx(0.5)
    assert bad.weights["b"] > bad.weights["a"]

    loss_a = -np.log(0.35)
    loss_b = -np.log(0.55)
    assert bad.weights["a"] == pytest.approx((1.0 / loss_a) / (1.0 / loss_a + 1.0 / loss_b))
    assert sum(bad.weights.values()) == pytest.approx(1.0)


def test_small_regime_falls_back_to_global_weights() -> None:
    oof, regimes, _ = _two_member_case()
    table = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=50))

    assert table.per_regime["bad"].source == "global"
    assert table.per_regime["bad"].rows == 40
    assert table.per_regime["bad"].weights == table.global_weights
    assert table.per_regime["good"].source == "regime"

    # Exactly at the threshold the bucket keeps its own weights.
    at_threshold = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=40))
    assert at_threshold.per_regime["bad"].source == "regime"

    unseen = table.for_regime("never_seen")
    assert (unseen.rows, unseen.source) == (0, "global")
    assert unseen.weights == table.global_weights


def test_regime_weight_inputs_are_validated() -> None:
    oof, regimes, _ = _two_member_case()

    with pytest.raises(ValueError, match="do not match"):
        fit_regime_weights(oof, regimes[:-1])
    with pytest.raises(ValueError, match="not a non-empty string"):
        bad = regimes.copy()
        bad[0] = ""
        fit_regime_weights(oof, bad)
    with pytest.raises(ValueError, match="not a non-empty string"):
        bad = regimes.copy()
        bad[0] = None
        fit_regime_weights(oof, bad)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="min_rows_per_regime"):
        RegimeSelectionConfig(min_rows_per_regime=0)
    with pytest.raises(ValueError, match="1-D"):
        fit_regime_weights(oof, np.array([["good"] * oof.y_true.size]))


def test_regime_weight_objects_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        RegimeWeights(
            regime="r",
            rows=10,
            source="regime",
            weights={"a": 0.6, "b": 0.6},
            loss={"a": 0.1, "b": 0.1},
        )
    with pytest.raises(ValueError, match="same members"):
        RegimeWeights(
            regime="r",
            rows=10,
            source="regime",
            weights={"a": 0.5, "b": 0.5},
            loss={"a": 0.1},
        )

    with pytest.raises(ValueError, match="exactly the members"):
        RegimeWeightTable(
            members=("a", "b", "c"),
            classes=(-1, 1),
            rows=10,
            min_rows_per_regime=5,
            global_weights={"a": 0.5, "b": 0.5},
            global_loss={"a": 0.1, "b": 0.2, "c": 0.3},
            per_regime={},
        )


# ---------------------------------------------------------------------------
# Row-wise combination and its shape/class guards.
# ---------------------------------------------------------------------------


def test_regime_ensemble_uses_row_weights_and_global_fallback() -> None:
    oof, regimes, _ = _two_member_case()
    table = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=30))
    ensemble = RegimeWeightedEnsemble(table)

    probabilities = {
        "a": np.stack(
            [oof.probabilities["a"][0], oof.probabilities["a"][60], oof.probabilities["a"][60]]
        ),
        "b": np.stack(
            [oof.probabilities["b"][0], oof.probabilities["b"][60], oof.probabilities["b"][60]]
        ),
    }
    combined = ensemble.predict_proba(probabilities, np.array(["good", "bad", "never"]))

    good_weights = table.per_regime["good"].weights
    bad_weights = table.per_regime["bad"].weights
    expected_good = np.asarray(
        [
            probabilities["a"][0][i] * good_weights["a"]
            + probabilities["b"][0][i] * good_weights["b"]
            for i in range(2)
        ]
    )
    expected_bad = np.asarray(
        [
            probabilities["a"][1][i] * bad_weights["a"]
            + probabilities["b"][1][i] * bad_weights["b"]
            for i in range(2)
        ]
    )
    expected_global = np.asarray(
        [
            probabilities["a"][2][i] * table.global_weights["a"]
            + probabilities["b"][2][i] * table.global_weights["b"]
            for i in range(2)
        ]
    )

    np.testing.assert_allclose(combined[0], expected_good)
    np.testing.assert_allclose(combined[1], expected_bad)
    np.testing.assert_allclose(combined[2], expected_global)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)


def test_regime_ensemble_rejects_shape_and_class_errors() -> None:
    oof, regimes, _ = _two_member_case()
    table = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=30))
    ensemble = RegimeWeightedEnsemble(table)
    probabilities = oof.probabilities

    with pytest.raises(ValueError, match="do not match"):
        ensemble.predict_proba(probabilities, regimes[:-1])
    with pytest.raises(ValueError, match="missing member"):
        ensemble.predict_proba({"a": probabilities["a"]}, regimes)
    three_classes = {
        "a": np.tile([0.5, 0.3, 0.2], (regimes.size, 1)),
        "b": np.tile([0.4, 0.4, 0.2], (regimes.size, 1)),
    }
    with pytest.raises(ValueError, match="class columns"):
        ensemble.predict_proba(three_classes, regimes)
    with pytest.raises(ValueError, match="non-finite"):
        ensemble.predict_proba(
            {
                "a": np.full_like(probabilities["a"], np.nan),
                "b": probabilities["b"],
            },
            regimes,
        )
    with pytest.raises(ValueError, match="probability_floor"):
        RegimeWeightedEnsemble(table, probability_floor=0.1, renormalise=False)
    with pytest.raises(ValueError, match="probability_floor"):
        RegimeWeightedEnsemble(table, probability_floor=1.0)


# ---------------------------------------------------------------------------
# Report: descriptive only, with coverage and fallback accounting.
# ---------------------------------------------------------------------------


def test_report_reports_weights_coverage_and_fallback() -> None:
    oof, regimes, _ = _two_member_case()
    table = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=50))
    report = regime_selection_report(oof, regimes, table)

    assert report["rows"] == 100
    assert report["descriptive_only"] is True
    assert report["best_member"] == "b"
    assert report["coverage"]["fallback_rows"] == 40
    assert report["coverage"]["fallback_share"] == pytest.approx(0.4)
    assert report["coverage"]["regime_weighted_rows"] == 60
    assert report["coverage"]["fallback_regimes"] == ["bad"]
    assert report["fallback"] == {
        "rows": 40,
        "share": pytest.approx(0.4),
        "regimes": ["bad"],
    }

    by_regime = {row["regime"]: row for row in report["regimes"]}
    assert by_regime["bad"]["source"] == "global"
    assert by_regime["good"]["source"] == "regime"
    assert by_regime["good"]["weights"] == pytest.approx({"a": 0.5, "b": 0.5}, rel=1e-12)
    assert by_regime["good"]["rows"] + by_regime["bad"]["rows"] == report["rows"]
    for row in report["regimes"]:
        assert {member["name"] for member in row["members"]} == {"a", "b"}
        assert row["members"][0]["rows"] == row["rows"]
        assert row["ensemble"]["rows"] == row["rows"]
        assert row["global_ensemble"]["rows"] == row["rows"]


def test_report_does_not_assume_improvement() -> None:
    oof, regimes, _ = _two_member_case()
    table = fit_regime_weights(oof, regimes, RegimeSelectionConfig(min_rows_per_regime=30))
    report = regime_selection_report(oof, regimes, table)

    comparison = {row["name"]: row for row in report["comparison"]}
    assert set(comparison) == {"regime_ensemble", "global_ensemble", "a", "b"}

    # In this fixture regime weighting loses to the best single member, and the
    # report must say so instead of ranking the ensemble first.
    assert comparison["b"]["accuracy"] == pytest.approx(1.0)
    assert comparison["regime_ensemble"]["accuracy"] == pytest.approx(0.6)
    assert report["comparison"][0]["name"] == "b"

    # Descriptive only: no promotion/improvement verdict anywhere in the payload.
    assert "improved" not in report
    assert "promoted" not in report
    assert "edge" not in report
    assert all("improved" not in row for row in report["regimes"])


# ---------------------------------------------------------------------------
# End-to-end: OOF rows only, no validation/test split in the evidence.
# ---------------------------------------------------------------------------


def test_regime_selection_uses_only_out_of_fold_training_rows() -> None:
    frame = _toy_frame()
    bundle = _toy_bundle(frame)
    config = CVConfig(n_splits=4, purge_bars=2, embargo_bars=1, min_train_rows=20)

    oof = fit_oof(
        {"a": lambda: _ToyMember(strength=0.3), "b": lambda: _ToyMember(strength=0.1)},
        bundle,
        config,
    )
    labels = align_regime_labels(bundle, classify(_bars(frame)))

    # The OOF result only describes training rows; regime selection inherits it.
    assert np.all(bundle.split[oof.row_index] == 0)
    table = fit_regime_weights(oof, labels[oof.row_index])
    assert table.rows == oof.row_index.size

    # Mutating the test split cannot move the labels the selection reads.
    ohlc = np.asarray(frame.ohlc).copy()
    test_rows = np.flatnonzero(bundle.split == 2) + frame.valid_from
    ohlc[test_rows] *= 5.0
    altered = FeatureFrame(
        X=frame.X,
        ts=frame.ts,
        ohlc=ohlc,
        spread=frame.spread,
        atr=frame.atr,
        valid_from=frame.valid_from,
        columns=frame.columns,
    )
    changed = align_regime_labels(bundle, classify(_bars(altered)))
    np.testing.assert_array_equal(labels[oof.row_index], changed[oof.row_index])

    again = fit_regime_weights(oof, changed[oof.row_index])
    assert table.to_dict() == again.to_dict()


def test_ensemble_config_does_not_accept_regime_mode_by_accident() -> None:
    # The regime rule is a separate object; it must not be smuggled in as a
    # WeightingMode and silently lose the per-row weights.
    with pytest.raises(ValueError, match="unknown weighting mode"):
        EnsembleConfig(members=("a", "b"), mode="regime")  # type: ignore[arg-type]
