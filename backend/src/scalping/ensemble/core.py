"""Combine member models into one probability distribution.

This is Phase 9 slice 1: soft voting, weighted averaging, hard voting and a
stacked meta-learner, plus the out-of-fold (OOF) plumbing they need. It does
**not** pick members by regime yet and it does not train on the real store.

Leakage discipline is inherited, not re-implemented: :func:`fit_oof` asks
:func:`scalping.ml.cv.fold_geometry` for the purged expanding-window boundaries,
so a member's OOF probability for a row comes from a fit that never saw that row
or the purge/embargo gap before it. The stacked meta-learner is fitted on those
OOF probabilities, never on in-sample member predictions.

Nothing here is a trading claim. An ensemble that scores better than its members
still has to pass the simulator and the cost gate before it is an edge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, Self

import numpy as np
from sklearn.linear_model import LogisticRegression

from scalping.datasets.builder import DatasetBundle
from scalping.ml.cv import CVConfig, CVFold, fold_geometry
from scalping.ml.train import TrainSpec
from scalping.nn.evaluate import ClassificationMetrics, classification_metrics

__all__ = [
    "Ensemble",
    "EnsembleConfig",
    "HardVotingEnsemble",
    "MemberFactory",
    "OOFResult",
    "SoftVotingEnsemble",
    "StackedEnsemble",
    "WeightedVotingEnsemble",
    "WeightingMode",
    "compare_to_best_member",
    "disagreement",
    "fit_oof",
]

#: A zero-argument callable returning a **fresh, unfitted** estimator.
MemberFactory = Callable[[], Any]

WeightingMode = Literal["equal", "weighted", "stacked"]

_MODES: Final[tuple[str, ...]] = ("equal", "weighted", "stacked")


def _probability_matrix(
    name: str,
    values: Any,
    *,
    rows: int | None = None,
    columns: int | None = None,
    renormalise: bool = True,
) -> np.ndarray:
    """Validate one probability matrix and optionally normalise its rows."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"member {name!r}: probabilities must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError(f"member {name!r}: probabilities have no rows")
    if matrix.shape[1] < 2:
        raise ValueError(
            f"member {name!r}: probabilities need at least two classes, got {matrix.shape[1]}"
        )
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(
            f"member {name!r}: {matrix.shape[0]} rows do not match the {rows} rows "
            "of the other members"
        )
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(
            f"member {name!r}: {matrix.shape[1]} class columns do not match the {columns} "
            "of the other members"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"member {name!r}: probabilities contain non-finite values")
    if (matrix < 0.0).any():
        raise ValueError(f"member {name!r}: probabilities contain negative values")
    totals = matrix.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise ValueError(f"member {name!r}: probability rows do not sum to a positive value")
    if renormalise:
        matrix = matrix / totals
    return matrix


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    """Members, weighting mode and the guards around a combination.

    ``weights`` is required for ``mode="weighted"`` and rejected otherwise. A
    weight of zero is allowed (the member is muted) but at least ``min_members``
    members must keep a positive weight. ``probability_floor`` clamps every
    member probability before averaging so one confident member cannot zero a
    class; it requires ``renormalise=True`` because the floor breaks the row sum.
    """

    members: tuple[str, ...]
    mode: WeightingMode = "equal"
    weights: Mapping[str, float] | None = None
    min_members: int = 2
    probability_floor: float = 0.0
    renormalise: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.members)
        if not names:
            raise ValueError("an ensemble needs at least one member")
        if len(set(names)) != len(names):
            raise ValueError(f"member names must be unique, got {names}")
        if self.min_members < 2:
            raise ValueError("min_members must be >= 2; a single member is not an ensemble")
        if len(names) < self.min_members:
            raise ValueError(
                f"an ensemble needs at least {self.min_members} members, got {len(names)}"
            )
        if self.mode not in _MODES:
            raise ValueError(f"unknown weighting mode {self.mode!r}; choose from {_MODES}")
        if not 0.0 <= self.probability_floor < 1.0:
            raise ValueError("probability_floor must be in [0, 1)")
        if self.probability_floor > 0.0 and not self.renormalise:
            raise ValueError("probability_floor requires renormalise=True")
        if self.mode == "weighted":
            if self.weights is None:
                raise ValueError("weighted mode requires weights")
            if set(self.weights) != set(names):
                raise ValueError(
                    f"weights must name exactly the members {names}, got {tuple(self.weights)}"
                )
            values = np.asarray([float(self.weights[name]) for name in names], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("weights must be finite")
            if (values < 0.0).any():
                raise ValueError("weights must be non-negative")
            if float(values.sum()) <= 0.0:
                raise ValueError("weights must sum to a positive value")
            if int((values > 0.0).sum()) < self.min_members:
                raise ValueError(
                    f"at least {self.min_members} members need a positive weight, "
                    f"got {int((values > 0.0).sum())}"
                )
        elif self.weights is not None:
            raise ValueError(f"{self.mode} mode does not use weights")


@dataclass(frozen=True, slots=True)
class OOFResult:
    """Out-of-fold probabilities for the scored training rows.

    ``row_index`` are ``DatasetBundle`` row offsets, ascending. The fold test
    blocks tile the training split **after** the first block, so the first block
    has no OOF prediction: it is only ever a training window. Every scored row
    comes from a fit that did not see it or the purge/embargo gap before it.
    """

    members: tuple[str, ...]
    classes: tuple[int, ...]
    row_index: np.ndarray
    y_true: np.ndarray
    probabilities: Mapping[str, np.ndarray]
    folds: tuple[CVFold, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("an OOF result needs at least one member")
        if len(set(self.members)) != len(self.members):
            raise ValueError("OOF member names must be unique")
        if len(self.classes) < 2:
            raise ValueError("OOF probabilities need at least two classes")
        raw_index = np.asarray(self.row_index, dtype=np.float64)
        if not np.isfinite(raw_index).all():
            raise ValueError("OOF row_index contains non-finite values")
        index = raw_index.astype(np.int64)
        if not np.equal(index, raw_index).all():
            raise ValueError("OOF row_index must be integer bundle offsets")
        raw_truth = np.asarray(self.y_true, dtype=np.float64)
        if not np.isfinite(raw_truth).all():
            raise ValueError("OOF labels contain non-finite values")
        truth = raw_truth.astype(np.int64)
        if not np.equal(truth, raw_truth).all():
            raise ValueError("OOF labels must be integers")
        if index.ndim != 1 or truth.shape != index.shape:
            raise ValueError("row_index and y_true must be 1-D and the same length")
        if index.size == 0:
            raise ValueError("an OOF result needs at least one row")
        if np.unique(index).size != index.size:
            raise ValueError("OOF row_index contains duplicate rows")
        if not np.isin(truth, np.asarray(self.classes)).all():
            raise ValueError("OOF labels contain classes outside the class list")
        if set(self.probabilities) != set(self.members):
            raise ValueError("OOF probabilities must name exactly the members")
        for name in self.members:
            _probability_matrix(
                name,
                self.probabilities[name],
                rows=index.size,
                columns=len(self.classes),
                renormalise=False,
            )


def _aligned_probabilities(
    name: str,
    model: Any,
    X: np.ndarray,
    classes: tuple[int, ...],
    *,
    fold: int,
) -> np.ndarray:
    """Predict and reorder a member's columns into the shared class order."""
    predict = getattr(model, "predict_proba", None)
    if predict is None:
        raise TypeError(f"fold {fold}: member {name!r} has no predict_proba")
    matrix = np.asarray(predict(X), dtype=np.float64)
    model_classes = tuple(int(value) for value in getattr(model, "classes_", classes))
    if set(model_classes) != set(classes):
        raise ValueError(
            f"fold {fold}: member {name!r} classes {model_classes} do not match {classes}"
        )
    if matrix.shape != (X.shape[0], len(model_classes)):
        raise ValueError(
            f"fold {fold}: member {name!r} predict_proba returned shape {matrix.shape}, "
            f"expected {(X.shape[0], len(model_classes))}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"fold {fold}: member {name!r} returned non-finite probabilities")
    if (matrix < 0.0).any():
        raise ValueError(f"fold {fold}: member {name!r} returned negative probabilities")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError(f"fold {fold}: member {name!r} rows do not sum to a positive value")
    matrix = matrix / totals
    if model_classes != classes:
        order = [model_classes.index(label) for label in classes]
        matrix = matrix[:, order]
    return matrix


def fit_oof(
    members: Mapping[str, MemberFactory],
    bundle: DatasetBundle,
    config: CVConfig,
    *,
    include_flat: bool = False,
) -> OOFResult:
    """Produce one out-of-fold probability matrix per member.

    The fold boundaries come from :func:`scalping.ml.cv.fold_geometry` using the
    dataset manifest's horizon, purge and embargo, so they match what
    :func:`scalping.ml.cv.cross_validate` would use on the same bundle. Each
    member is built fresh for every fold; a factory that returns a fitted model
    would leak.

    The first fold block is only ever a training window, so it receives no OOF
    prediction. The meta-learner therefore sees the scored blocks only, which is
    the honest set. Every fold's training window must contain the full class
    set: a fold that cannot represent a class would force a fake zero-probability
    column, so it raises instead.
    """
    names = tuple(str(name) for name in members)
    if not names:
        raise ValueError("fit_oof needs at least one member")

    spec = TrainSpec(
        horizon=bundle.manifest.horizon,
        purge_bars=bundle.manifest.purge_bars,
        embargo_bars=bundle.manifest.embargo_bars,
        include_flat=include_flat,
    )
    plan = fold_geometry(bundle, spec=spec, config=config)

    train_mask = (bundle.split == 0) & plan.keep
    classes = tuple(sorted(int(value) for value in np.unique(bundle.y_direction[train_mask])))
    if len(classes) < 2:
        raise ValueError("the training split has a single class; an ensemble needs at least two")

    row_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    probability_parts: dict[str, list[np.ndarray]] = {name: [] for name in names}

    for fold in plan.folds:
        train_index = np.arange(fold.train_start, fold.train_end)[
            plan.keep[fold.train_start : fold.train_end]
        ]
        test_index = np.arange(fold.test_start, fold.test_end)[
            plan.keep[fold.test_start : fold.test_end]
        ]
        y_train = bundle.y_direction[train_index]
        y_test = bundle.y_direction[test_index]
        fold_classes = tuple(sorted(int(value) for value in np.unique(y_train)))
        if fold_classes != classes:
            raise ValueError(
                f"fold {fold.fold}: training window classes {fold_classes} do not match "
                f"the OOF class set {classes}"
            )

        for name in names:
            model = members[name]()
            model.fit(bundle.X[train_index], y_train)
            probability_parts[name].append(
                _aligned_probabilities(
                    name,
                    model,
                    bundle.X[test_index],
                    classes,
                    fold=fold.fold,
                )
            )
        row_parts.append(test_index)
        label_parts.append(y_test)

    row_index = np.concatenate(row_parts)
    if np.unique(row_index).size != row_index.size:
        raise RuntimeError("OOF fold test blocks overlap; the CV geometry is inconsistent")
    order = np.argsort(row_index, kind="stable")
    row_index = row_index[order]
    y_true = np.concatenate(label_parts)[order]

    return OOFResult(
        members=names,
        classes=classes,
        row_index=row_index,
        y_true=y_true,
        probabilities={
            name: np.concatenate(probability_parts[name], axis=0)[order] for name in names
        },
        folds=plan.folds,
    )


class Ensemble:
    """Shared validation and class bookkeeping for the combination rules."""

    mode: ClassVar[WeightingMode]

    def __init__(self, config: EnsembleConfig) -> None:
        if config.mode != self.mode:
            raise ValueError(f"{type(self).__name__} needs mode={self.mode!r}, got {config.mode!r}")
        self.config = config
        self.classes_: tuple[int, ...] | None = None

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(self.config.members)

    def fit(self, oof: OOFResult) -> Self:
        """Check that the OOF result describes this ensemble's members."""
        _check_members(self.config.members, oof.probabilities)
        for name in self.config.members:
            matrix = np.asarray(oof.probabilities[name])
            if matrix.shape[0] != np.asarray(oof.y_true).shape[0]:
                raise ValueError(f"member {name!r} has a different row count than the labels")
        if np.asarray(oof.y_true).shape[0] == 0:
            raise ValueError("cannot fit an ensemble on zero rows")
        self.classes_ = tuple(int(value) for value in oof.classes)
        return self

    def _prepare(
        self,
        probabilities: Mapping[str, np.ndarray],
    ) -> tuple[list[np.ndarray], int, int]:
        """Validate members, align shapes, apply the floor and normalise."""
        _check_members(self.config.members, probabilities)
        matrices: list[np.ndarray] = []
        rows: int | None = None
        columns: int | None = None
        for name in self.config.members:
            matrix = _probability_matrix(
                name,
                probabilities[name],
                rows=rows,
                columns=columns,
                renormalise=self.config.renormalise,
            )
            if rows is None:
                rows, columns = matrix.shape
            if self.config.probability_floor > 0.0:
                matrix = np.maximum(matrix, self.config.probability_floor)
                matrix = matrix / matrix.sum(axis=1, keepdims=True)
            matrices.append(matrix)
        if rows is None or columns is None:  # pragma: no cover - members are non-empty
            raise RuntimeError("no member probabilities were prepared")
        return matrices, rows, columns

    def predict_proba(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


def _check_members(
    expected: tuple[str, ...],
    probabilities: Mapping[str, np.ndarray],
) -> None:
    missing = [name for name in expected if name not in probabilities]
    extra = [name for name in probabilities if name not in expected]
    if missing:
        raise ValueError(f"missing member probabilities: {missing}")
    if extra:
        raise ValueError(f"unknown member probabilities: {extra}")


class SoftVotingEnsemble(Ensemble):
    """Unweighted probability average; every member gets the same vote."""

    mode: ClassVar[WeightingMode] = "equal"

    def predict_proba(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        matrices, _, _ = self._prepare(probabilities)
        return np.mean(np.stack(matrices, axis=0), axis=0)


class WeightedVotingEnsemble(Ensemble):
    """Probability average with fixed, non-negative member weights.

    Weights are normalised to sum to one, so only their ratio matters.
    """

    mode: ClassVar[WeightingMode] = "weighted"

    def __init__(self, config: EnsembleConfig) -> None:
        super().__init__(config)
        if config.weights is None:  # pragma: no cover - __post_init__ guarantees this
            raise ValueError("weighted mode requires weights")
        weights = np.asarray(
            [float(config.weights[name]) for name in config.members],
            dtype=np.float64,
        )
        self.weights_: np.ndarray = weights / weights.sum()

    def predict_proba(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        matrices, _, _ = self._prepare(probabilities)
        stacked = np.stack(matrices, axis=0)
        return np.tensordot(self.weights_, stacked, axes=(0, 0))


class HardVotingEnsemble(Ensemble):
    """Majority vote over each member's argmax class.

    A tie is not broken: :meth:`vote` returns the tie mask and
    :meth:`predict_proba` emits a uniform row, which the decision engine rejects
    as ``NO_TRADE``. A uniform row is the honest representation of "no majority",
    not a probability the members produced.
    """

    mode: ClassVar[WeightingMode] = "equal"

    def vote(
        self,
        probabilities: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(one_hot_or_uniform_probabilities, tie_mask)``."""
        matrices, rows, columns = self._prepare(probabilities)
        indices = np.stack([np.argmax(matrix, axis=1) for matrix in matrices], axis=0)
        counts = np.zeros((columns, rows), dtype=np.int64)
        for position in range(columns):
            counts[position] = (indices == position).sum(axis=0)
        top = counts.max(axis=0)
        ties = (counts == top).sum(axis=0) > 1
        output = np.zeros((rows, columns), dtype=np.float64)
        output[np.arange(rows), counts.argmax(axis=0)] = 1.0
        output[ties] = 1.0 / columns
        return output, ties

    def predict_proba(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.vote(probabilities)[0]


class StackedEnsemble(Ensemble):
    """Multinomial logistic regression on the members' OOF probabilities.

    The meta-learner sees out-of-fold predictions only. Columns are concatenated
    in ``config.members`` order and the target is encoded into the OOF class
    order, so ``predict_proba`` can map the meta-learner's columns back without
    guessing. Class weights are deliberately not balanced: the meta-learner
    should learn the calibration of the OOF rows, not re-balance them.
    """

    mode: ClassVar[WeightingMode] = "stacked"

    def __init__(self, config: EnsembleConfig) -> None:
        super().__init__(config)
        self.meta_: LogisticRegression | None = None

    def _features(
        self,
        probabilities: Mapping[str, np.ndarray],
        classes: tuple[int, ...],
    ) -> np.ndarray:
        matrices, _, columns = self._prepare(probabilities)
        if columns != len(classes):
            raise ValueError(
                f"stacked ensemble expects {len(classes)} classes, got {columns} columns"
            )
        return np.concatenate(matrices, axis=1)

    def fit(self, oof: OOFResult) -> Self:
        super().fit(oof)
        classes = tuple(int(value) for value in oof.classes)
        encoded = {label: position for position, label in enumerate(classes)}
        y = np.asarray([encoded[int(value)] for value in np.asarray(oof.y_true)], dtype=np.int64)
        if np.unique(y).size < len(classes):
            raise ValueError("the stacked meta-learner needs every class present in the OOF target")
        features = self._features(oof.probabilities, classes)
        meta = LogisticRegression(max_iter=1000, random_state=self.config.seed)
        meta.fit(features, y)
        self.meta_ = meta
        self.classes_ = classes
        return self

    def predict_proba(self, probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.meta_ is None or self.classes_ is None:
            raise ValueError("the stacked ensemble is not fitted; call fit(oof) first")
        features = self._features(probabilities, self.classes_)
        raw = np.asarray(self.meta_.predict_proba(features), dtype=np.float64)
        output = np.zeros((raw.shape[0], len(self.classes_)), dtype=np.float64)
        for position, encoded in enumerate(self.meta_.classes_):
            output[:, int(encoded)] = raw[:, position]
        return output


def disagreement(probabilities: Mapping[str, np.ndarray]) -> np.ndarray:
    """Mean pairwise total-variation distance between member distributions.

    For distributions ``p`` and ``q`` the total variation distance is
    ``0.5 * sum(|p - q|)``: the share of probability mass that would have to
    move to turn one into the other. It is symmetric, bounded in ``[0, 1]`` and
    equals 0 for identical members and 1 when two members put their mass on
    disjoint classes (opposite corners of the simplex).

    Mean pairwise TV is used instead of ``1 - max_vote_share`` because it does
    not reward a larger class set: a three-way split and a binary split with the
    same pairwise disagreement score the same. It is also independent of which
    class happens to be the argmax, so a confident minority cannot hide behind
    a quiet majority.

    Returns one score per row. A single member scores 0 everywhere.
    """
    names = tuple(probabilities)
    if not names:
        raise ValueError("disagreement needs at least one member")
    matrices: list[np.ndarray] = []
    rows: int | None = None
    columns: int | None = None
    for name in names:
        matrix = _probability_matrix(
            name,
            probabilities[name],
            rows=rows,
            columns=columns,
            renormalise=True,
        )
        if rows is None:
            rows, columns = matrix.shape
        matrices.append(matrix)
    if rows is None:  # pragma: no cover - names is non-empty
        raise RuntimeError("no member probabilities were provided")
    if len(matrices) == 1:
        return np.zeros(rows, dtype=np.float64)
    total = np.zeros(rows, dtype=np.float64)
    pairs = 0
    for left in range(len(matrices)):
        for right in range(left + 1, len(matrices)):
            total += 0.5 * np.abs(matrices[left] - matrices[right]).sum(axis=1)
            pairs += 1
    return total / pairs


def _comparison_row(name: str, metrics: ClassificationMetrics) -> dict[str, Any]:
    return {
        "name": name,
        "rows": metrics.rows,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
    }


def compare_to_best_member(
    y_true: np.ndarray,
    member_probabilities: Mapping[str, np.ndarray],
    ensemble_probabilities: np.ndarray,
    *,
    classes: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Score every member and the ensemble on the same rows.

    Uses :func:`scalping.nn.evaluate.classification_metrics` and sorts by the
    same key as :func:`scalping.nn.evaluate.compare_predictors` (accuracy
    descending, name ascending). The ensemble row is always named ``ensemble``;
    a member with that name is rejected to keep the table unambiguous.
    """
    if not member_probabilities:
        raise ValueError("comparison needs at least one member")
    if "ensemble" in member_probabilities:
        raise ValueError("'ensemble' is reserved for the combined row")
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if classes is None:
        classes = tuple(sorted({int(value) for value in truth.tolist()}))
    if len(classes) < 2:
        raise ValueError("need at least two classes to compare predictors")

    rows: list[dict[str, Any]] = []
    for name, values in member_probabilities.items():
        matrix = _probability_matrix(name, values, rows=truth.size, columns=len(classes))
        labels = np.asarray(classes)[np.argmax(matrix, axis=1)]
        rows.append(
            _comparison_row(
                name,
                classification_metrics(truth, labels, matrix, classes=classes),
            )
        )
    combined = _probability_matrix(
        "ensemble",
        ensemble_probabilities,
        rows=truth.size,
        columns=len(classes),
    )
    labels = np.asarray(classes)[np.argmax(combined, axis=1)]
    rows.append(
        _comparison_row(
            "ensemble",
            classification_metrics(truth, labels, combined, classes=classes),
        )
    )
    return sorted(rows, key=lambda row: (-row["accuracy"], row["name"]))
