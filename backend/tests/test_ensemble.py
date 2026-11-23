"""Ensemble core, OOF geometry and decision-engine behaviour."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.ensemble import (
    Decision,
    DecisionEngine,
    EnsembleConfig,
    HardVotingEnsemble,
    OOFResult,
    SoftVotingEnsemble,
    StackedEnsemble,
    WeightedVotingEnsemble,
    compare_to_best_member,
    disagreement,
    fit_oof,
)
from scalping.ensemble.decision import (
    REASON_BELOW_ENTRY,
    REASON_DISAGREEMENT,
    REASON_ENTRY_LONG,
    REASON_ENTRY_SHORT,
    REASON_EXIT_LONG,
    REASON_EXIT_SHORT,
    REASON_FLAT_CLASS,
    REASON_HOLD_LONG,
    REASON_HOLD_SHORT,
    REASON_LOW_EDGE,
)
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.ml.cv import CVConfig, cross_validate, fold_geometry
from scalping.ml.train import TrainSpec

# ---------------------------------------------------------------------------
# Fixtures: a deterministic feature frame and toy members.
# ---------------------------------------------------------------------------


def _toy_frame(n: int = 400, valid_from: int = 10) -> FeatureFrame:
    """Alternating +/- returns; ``ret_1`` carries the source row index."""
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


def _toy_bundle(
    frame: FeatureFrame,
    *,
    horizon: int = 1,
    purge_bars: int = 2,
    embargo_bars: int = 1,
) -> DatasetBundle:
    return build_dataset(
        frame,
        symbol="XAUUSD",
        timeframe="M1",
        groups=("price",),
        horizon=horizon,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )


def _toy_spec(**overrides: object) -> TrainSpec:
    values: dict[str, object] = {
        "model": "logistic",
        "horizon": 1,
        "groups": ("price",),
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "purge_bars": 2,
        "embargo_bars": 1,
        "include_flat": False,
    }
    values.update(overrides)
    return TrainSpec(**values)  # type: ignore[arg-type]


class _ToyMember:
    """Predicts from ``ret_5`` and records which source rows it was fitted on."""

    def __init__(self, record: list[_ToyMember], *, strength: float = 0.3) -> None:
        self.record = record
        self.strength = strength
        self.fit_rows = np.array([], dtype=np.int64)
        self.classes_ = np.array([], dtype=np.int64)
        record.append(self)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ToyMember:
        self.fit_rows = np.rint(X[:, 0]).astype(np.int64)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = np.asarray(X[:, 1], dtype=np.float64)
        up = 0.5 + self.strength * signal
        return np.column_stack((1.0 - up, up))


class _OneClassMember:
    classes_ = np.array([-1], dtype=np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _OneClassMember:
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((X.shape[0], 1))


class _NaNToyMember(_ToyMember):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0], 2), np.nan)


# ---------------------------------------------------------------------------
# Fold geometry: shared with cross_validate, no copied purge math.
# ---------------------------------------------------------------------------


def test_fold_geometry_matches_cross_validate() -> None:
    frame = _toy_frame()
    spec = _toy_spec()
    config = CVConfig(n_splits=4, purge_bars=2, embargo_bars=1, min_train_rows=20)

    plan = fold_geometry(_toy_bundle(frame), spec=spec, config=config)
    result = cross_validate(frame, symbol="XAUUSD", timeframe="M1", spec=spec, config=config)

    assert [fold.to_dict() for fold in plan.folds] == [fold.to_dict() for fold in result.folds]
    assert plan.split_counts == result.split_counts
    assert plan.purge_bars == 2
    assert plan.embargo_bars == 1


def test_oof_uses_only_training_rows_and_respects_purge() -> None:
    frame = _toy_frame()
    bundle = _toy_bundle(frame)
    config = CVConfig(n_splits=4, purge_bars=2, embargo_bars=1, min_train_rows=20)
    record: list[_ToyMember] = []

    oof = fit_oof({"toy": lambda: _ToyMember(record)}, bundle, config)

    assert oof.classes == (-1, 1)
    assert np.all(bundle.split[oof.row_index] == 0)
    usable = np.flatnonzero((bundle.split == 0) & (bundle.y_direction != 0))
    scored = usable[usable >= oof.folds[0].test_start]
    np.testing.assert_array_equal(oof.row_index, scored)
    np.testing.assert_array_equal(oof.y_true, bundle.y_direction[scored])
    assert oof.row_index[0] == oof.folds[0].test_start

    gap = max(config.purge_bars, bundle.manifest.purge_bars, bundle.manifest.horizon)
    gap += config.embargo_bars
    assert len(record) == config.n_splits
    for member, fold in zip(record, oof.folds, strict=True):
        assert member.fit_rows.min() == frame.valid_from + fold.train_start
        assert member.fit_rows.max() == frame.valid_from + fold.train_end - 1
        assert fold.test_start - fold.train_end >= gap
        test_rows = np.arange(
            frame.valid_from + fold.test_start,
            frame.valid_from + fold.test_end,
        )
        assert not np.isin(member.fit_rows, test_rows).any()
        assert np.all(bundle.split[member.fit_rows - frame.valid_from] == 0)

    for name in oof.members:
        matrix = oof.probabilities[name]
        assert matrix.shape == (scored.size, 2)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        predicted = np.asarray(oof.classes)[np.argmax(matrix, axis=1)]
        assert float(np.mean(predicted == oof.y_true)) > 0.9


def test_fit_oof_rejects_bad_members() -> None:
    bundle = _toy_bundle(_toy_frame())
    config = CVConfig(n_splits=3, purge_bars=2, embargo_bars=1, min_train_rows=20)

    with pytest.raises(ValueError, match="at least one member"):
        fit_oof({}, bundle, config)
    with pytest.raises(ValueError, match="do not match"):
        fit_oof({"bad": _OneClassMember}, bundle, config)
    with pytest.raises(ValueError, match="non-finite"):
        fit_oof({"bad": lambda: _NaNToyMember([])}, bundle, config)


# ---------------------------------------------------------------------------
# Combination rules on hand-built probabilities.
# ---------------------------------------------------------------------------


def _config(**overrides: object) -> EnsembleConfig:
    values: dict[str, object] = {"members": ("a", "b")}
    values.update(overrides)
    return EnsembleConfig(**values)  # type: ignore[arg-type]


def test_soft_voting_averages_probabilities() -> None:
    ensemble = SoftVotingEnsemble(_config())
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.5, 0.3, 0.2]])

    combined = ensemble.predict_proba({"a": a, "b": b})

    np.testing.assert_allclose(combined, [[0.6, 0.25, 0.15]])


def test_weighted_voting_uses_normalised_weights() -> None:
    ensemble = WeightedVotingEnsemble(_config(mode="weighted", weights={"a": 3.0, "b": 1.0}))
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.5, 0.3, 0.2]])

    combined = ensemble.predict_proba({"a": a, "b": b})

    np.testing.assert_allclose(combined, [[0.65, 0.225, 0.125]])
    np.testing.assert_allclose(ensemble.weights_, [0.75, 0.25])


def test_probability_floor_prevents_a_zero_column() -> None:
    ensemble = SoftVotingEnsemble(_config(probability_floor=0.1))
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])

    combined = ensemble.predict_proba({"a": a, "b": b})

    assert (combined > 0.0).all()
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)


def test_hard_voting_majority_and_tie() -> None:
    ensemble = HardVotingEnsemble(_config())
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.5, 0.3, 0.2]])

    majority, ties = ensemble.vote({"a": a, "b": b})
    np.testing.assert_allclose(majority, [[1.0, 0.0, 0.0]])
    assert not ties[0]

    tied, ties = ensemble.vote({"a": np.array([[0.6, 0.3, 0.1]]), "b": np.array([[0.2, 0.7, 0.1]])})
    np.testing.assert_allclose(tied, [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    assert ties[0]


def test_stacked_meta_learner_fits_oof_and_keeps_the_class_mapping() -> None:
    rng = np.random.default_rng(3)
    classes = (-1, 0, 1)
    y = np.repeat(np.asarray(classes), 20)
    centres = {
        -1: np.array([0.75, 0.15, 0.10]),
        0: np.array([0.15, 0.70, 0.15]),
        1: np.array([0.10, 0.15, 0.75]),
    }

    def member() -> np.ndarray:
        matrix = np.stack([centres[int(label)] for label in y])
        matrix = matrix + rng.uniform(0.0, 0.03, size=matrix.shape)
        return matrix / matrix.sum(axis=1, keepdims=True)

    oof = OOFResult(
        members=("a", "b"),
        classes=classes,
        row_index=np.arange(y.size),
        y_true=y,
        probabilities={"a": member(), "b": member()},
        folds=(),
    )
    ensemble = StackedEnsemble(_config(mode="stacked")).fit(oof)
    combined = ensemble.predict_proba(oof.probabilities)
    again = StackedEnsemble(_config(mode="stacked")).fit(oof).predict_proba(oof.probabilities)

    assert combined.shape == (y.size, 3)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)
    np.testing.assert_allclose(combined, again)
    predicted = np.asarray(classes)[np.argmax(combined, axis=1)]
    assert float(np.mean(predicted == y)) >= 0.95


def test_stacked_requires_every_class_and_a_fit() -> None:
    classes = (-1, 0, 1)
    y = np.array([-1, 1, -1, 1])
    oof = OOFResult(
        members=("a", "b"),
        classes=classes,
        row_index=np.arange(y.size),
        y_true=y,
        probabilities={
            "a": np.tile([0.45, 0.10, 0.45], (y.size, 1)),
            "b": np.tile([0.45, 0.10, 0.45], (y.size, 1)),
        },
        folds=(),
    )

    with pytest.raises(ValueError, match="every class"):
        StackedEnsemble(_config(mode="stacked")).fit(oof)
    with pytest.raises(ValueError, match="not fitted"):
        StackedEnsemble(_config(mode="stacked")).predict_proba(oof.probabilities)


def test_ensemble_config_rejections() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleConfig(members=())
    with pytest.raises(ValueError, match="at least 2 members"):
        EnsembleConfig(members=("a",))
    with pytest.raises(ValueError, match="unique"):
        EnsembleConfig(members=("a", "a"))
    with pytest.raises(ValueError, match="requires weights"):
        EnsembleConfig(members=("a", "b"), mode="weighted")
    with pytest.raises(ValueError, match="exactly the members"):
        EnsembleConfig(members=("a", "b"), mode="weighted", weights={"a": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        EnsembleConfig(members=("a", "b"), mode="weighted", weights={"a": 1.0, "b": -1.0})
    with pytest.raises(ValueError, match="positive value"):
        EnsembleConfig(members=("a", "b"), mode="weighted", weights={"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="positive weight"):
        EnsembleConfig(
            members=("a", "b", "c"),
            mode="weighted",
            weights={"a": 1.0, "b": 0.0, "c": 0.0},
        )
    with pytest.raises(ValueError, match="does not use weights"):
        EnsembleConfig(members=("a", "b"), weights={"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError, match="needs mode"):
        SoftVotingEnsemble(_config(mode="weighted", weights={"a": 1.0, "b": 1.0}))


def test_ensemble_rejects_bad_probability_matrices() -> None:
    ensemble = SoftVotingEnsemble(_config())
    good = np.array([[0.6, 0.3, 0.1]])

    with pytest.raises(ValueError, match="missing member"):
        ensemble.predict_proba({"a": good})
    with pytest.raises(ValueError, match="unknown member"):
        ensemble.predict_proba({"a": good, "b": good, "c": good})
    with pytest.raises(ValueError, match="do not match"):
        ensemble.predict_proba({"a": good, "b": np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]])})
    with pytest.raises(ValueError, match="class columns"):
        ensemble.predict_proba({"a": good, "b": np.array([[0.5, 0.5]])})
    with pytest.raises(ValueError, match="non-finite"):
        ensemble.predict_proba({"a": good, "b": np.array([[np.nan, 0.0, 0.0]])})
    with pytest.raises(ValueError, match="negative"):
        ensemble.predict_proba({"a": good, "b": np.array([[-0.1, 0.6, 0.5]])})
    with pytest.raises(ValueError, match="positive value"):
        ensemble.predict_proba({"a": good, "b": np.zeros((1, 3))})
    with pytest.raises(ValueError, match="2-D"):
        ensemble.predict_proba({"a": good, "b": np.array([0.5, 0.5])})


def test_oof_result_rejects_nan_labels_and_duplicate_rows() -> None:
    good = np.array([[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError):
        OOFResult(
            members=("a",),
            classes=(-1, 1),
            row_index=np.array([0, 1]),
            y_true=np.array([-1.0, np.nan]),
            probabilities={"a": good},
            folds=(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        OOFResult(
            members=("a",),
            classes=(-1, 1),
            row_index=np.array([0, 0]),
            y_true=np.array([-1, 1]),
            probabilities={"a": good},
            folds=(),
        )


# ---------------------------------------------------------------------------
# Disagreement.
# ---------------------------------------------------------------------------


def test_disagreement_is_zero_when_identical_and_one_at_opposite_corners() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])

    scores = disagreement({"a": a, "b": b})

    assert scores[0] == 0.0
    assert np.isclose(scores[1], 0.5)

    opposite = disagreement(
        {
            "a": np.array([[1.0, 0.0, 0.0]]),
            "b": np.array([[0.0, 0.0, 1.0]]),
        }
    )
    assert np.isclose(opposite[0], 1.0)

    single = disagreement({"a": a})
    np.testing.assert_allclose(single, np.zeros(2))


def test_disagreement_three_way_is_the_mean_pairwise_tv() -> None:
    corners = {
        "a": np.array([[1.0, 0.0, 0.0]]),
        "b": np.array([[0.0, 1.0, 0.0]]),
        "c": np.array([[0.0, 0.0, 1.0]]),
    }
    assert np.isclose(disagreement(corners)[0], 1.0)

    half = {
        "a": np.array([[0.5, 0.5, 0.0]]),
        "b": np.array([[0.5, 0.0, 0.5]]),
    }
    assert np.isclose(disagreement(half)[0], 0.5)


# ---------------------------------------------------------------------------
# Decision engine.
# ---------------------------------------------------------------------------


def _engine(**overrides: object) -> DecisionEngine:
    values: dict[str, object] = {
        "entry_threshold": 0.55,
        "exit_threshold": 0.45,
        "max_disagreement": 0.35,
        "min_edge": 0.05,
        "classes": (-1, 0, 1),
        "model_version": "ensemble-test",
    }
    values.update(overrides)
    return DecisionEngine(**values)  # type: ignore[arg-type]


def _decide(engine: DecisionEngine, probabilities: object, **kwargs: object) -> Decision:
    kwargs.setdefault("disagreement", 0.0)
    return engine.decide(probabilities, **kwargs)  # type: ignore[arg-type]


def test_decision_thresholds_and_reasons() -> None:
    engine = _engine()
    assert _decide(engine, [0.1, 0.2, 0.7], ts="2026-01-05T10:00:00+00:00").action == "BUY"
    assert _decide(engine, [0.7, 0.2, 0.1]).action == "SELL"
    assert _decide(engine, [0.1, 0.2, 0.7]).reason == REASON_ENTRY_LONG
    assert _decide(engine, [0.7, 0.2, 0.1]).reason == REASON_ENTRY_SHORT

    below = _decide(engine, [0.4, 0.35, 0.25])
    assert (below.action, below.reason) == ("NO_TRADE", REASON_BELOW_ENTRY)

    noisy = _decide(engine, [0.1, 0.1, 0.8], disagreement=0.5)
    assert (noisy.action, noisy.reason) == ("NO_TRADE", REASON_DISAGREEMENT)

    flat = _decide(engine, [0.1, 0.7, 0.2])
    assert (flat.action, flat.reason) == ("NO_TRADE", REASON_FLAT_CLASS)

    thin_engine = _engine(entry_threshold=0.45, exit_threshold=0.35, min_edge=0.10)
    thin = _decide(thin_engine, [0.5, 0.45, 0.05])
    assert (thin.action, thin.reason) == ("NO_TRADE", REASON_LOW_EDGE)


def test_decision_position_handling() -> None:
    engine = _engine()

    hold_long = _decide(engine, [0.2, 0.1, 0.7], position="long")
    assert (hold_long.action, hold_long.reason) == ("HOLD", REASON_HOLD_LONG)

    hold_short = _decide(engine, [0.7, 0.1, 0.2], position="short")
    assert (hold_short.action, hold_short.reason) == ("HOLD", REASON_HOLD_SHORT)

    exit_long = _decide(engine, [0.6, 0.2, 0.2], position="long")
    assert (exit_long.action, exit_long.reason) == ("NO_TRADE", REASON_EXIT_LONG)

    exit_short = _decide(engine, [0.2, 0.2, 0.6], position="short")
    assert (exit_short.action, exit_short.reason) == ("NO_TRADE", REASON_EXIT_SHORT)

    # Disagreement gates entries; with a position it is logged but the held
    # class still decides between HOLD and an exit.
    held = _decide(engine, [0.2, 0.1, 0.7], position="long", disagreement=0.9)
    assert held.action == "HOLD"
    assert held.disagreement == 0.9


def test_hard_vote_tie_becomes_no_trade() -> None:
    ensemble = HardVotingEnsemble(_config())
    combined = ensemble.predict_proba(
        {"a": np.array([[0.6, 0.3, 0.1]]), "b": np.array([[0.2, 0.7, 0.1]])}
    )
    decision = _decide(_engine(), combined[0])
    assert decision.action == "NO_TRADE"


def test_decision_probabilities_are_normalised_and_validated() -> None:
    engine = _engine()
    decision = _decide(engine, [2.0, 2.0, 6.0])
    assert decision.action == "BUY"
    np.testing.assert_allclose(decision.probabilities, [0.2, 0.2, 0.6])

    for bad in ([0.5, 0.5], [np.nan, 0.5, 0.5], [-0.1, 0.6, 0.5], [0.0, 0.0, 0.0]):
        with pytest.raises(ValueError):
            _decide(engine, bad)
    with pytest.raises(ValueError, match="disagreement"):
        _decide(engine, [0.1, 0.2, 0.7], disagreement=1.5)
    with pytest.raises(ValueError, match="position"):
        _decide(engine, [0.1, 0.2, 0.7], position="sideways")


def test_engine_config_is_validated() -> None:
    with pytest.raises(ValueError, match="entry_threshold"):
        _engine(entry_threshold=0.0)
    with pytest.raises(ValueError, match="exit_threshold"):
        _engine(exit_threshold=0.6)
    with pytest.raises(ValueError, match="max_disagreement"):
        _engine(max_disagreement=1.5)
    with pytest.raises(ValueError, match="min_edge"):
        _engine(min_edge=-0.1)
    with pytest.raises(ValueError, match="classes"):
        _engine(classes=(-1,))


def test_decision_log_round_trip() -> None:
    engine = _engine()
    decision = engine.decide(
        [0.1, 0.2, 0.7],
        disagreement=0.125,
        regime="trend",
        ts="2026-01-05T10:00:00+00:00",
        model_version="ens-2026-01",
    )

    payload = decision.to_dict()
    assert payload["action"] == "BUY"
    assert payload["reason"] == REASON_ENTRY_LONG
    assert payload["probabilities"] == {"-1": 0.1, "0": 0.2, "1": 0.7}
    assert payload["regime"] == "trend"
    assert payload["model_version"] == "ens-2026-01"
    assert payload["ts"] == "2026-01-05T10:00:00+00:00"

    restored = json.loads(json.dumps(payload))
    assert restored == payload
    assert json.loads(json.dumps(restored)) == payload

    # A decision object cannot exist without a reason.
    with pytest.raises(ValueError, match="reason"):
        Decision(
            action="NO_TRADE",
            probabilities=(0.5, 0.3, 0.2),
            classes=(-1, 0, 1),
            disagreement=0.1,
            regime=None,
            reason="",
            model_version="v1",
            ts="2026-01-05T10:00:00+00:00",
        )


def test_decision_is_deterministic_for_fixed_inputs() -> None:
    engine = _engine()
    kwargs = {
        "disagreement": 0.1,
        "regime": "range",
        "ts": "2026-01-05T10:00:00+00:00",
    }
    first = engine.decide([0.1, 0.2, 0.7], **kwargs)  # type: ignore[arg-type]
    second = engine.decide([0.1, 0.2, 0.7], **kwargs)  # type: ignore[arg-type]
    assert first == second


# ---------------------------------------------------------------------------
# Comparison table.
# ---------------------------------------------------------------------------


def test_compare_to_best_member_sorts_like_compare_predictors() -> None:
    y = np.array([-1, -1, 1, 1, -1, 1])
    member_a = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.3, 0.7],
        ]
    )
    member_b = np.array(
        [
            [0.4, 0.6],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.55, 0.45],
            [0.5, 0.5],
            [0.2, 0.8],
        ]
    )
    combined = (member_a + member_b) / 2.0

    table = compare_to_best_member(
        y,
        {"member_a": member_a, "member_b": member_b},
        combined,
        classes=(-1, 1),
    )

    assert [row["name"] for row in table] == ["ensemble", "member_a", "member_b"]
    assert [row["rows"] for row in table] == [6, 6, 6]
    assert table[0]["accuracy"] == pytest.approx(1.0)
    assert table[1]["accuracy"] == pytest.approx(1.0)
    assert table[2]["accuracy"] == pytest.approx(4.0 / 6.0)
    assert table[0]["log_loss"] is not None
    assert table[0]["brier"] is not None
