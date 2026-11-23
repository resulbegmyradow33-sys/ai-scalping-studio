"""Purged, expanding-window cross-validation for model selection.

The dataset builder already cuts a chronological train/validation/test split and
purges the boundaries. This module only looks at the **training** split and cuts
it again into contiguous blocks, so a candidate model can be compared without
spending the validation or test rows. Fold ``k`` tests block ``k + 1`` and fits
on every earlier block, minus a purge (at least the label horizon) and an
optional embargo immediately before the test block.

Row offsets in :class:`CVFold` are positions in the ``DatasetBundle`` arrays
(``bundle.X``), not positions in the source frame. The first bundle row is the
source row ``features.valid_from``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from scalping.datasets.builder import build_dataset
from scalping.features.build import FeatureFrame
from scalping.ml.calibration import calibration_summary
from scalping.ml.models import build_model
from scalping.ml.train import TrainSpec

__all__ = ["CVConfig", "CVFold", "CVResult", "cross_validate"]

#: Metrics averaged across folds, in report order.
_AGGREGATE_KEYS: Final[tuple[str, ...]] = (
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "brier",
    "ece",
    "mce",
)


@dataclass(frozen=True, slots=True)
class CVConfig:
    """Fold geometry for one cross-validation run."""

    n_splits: int = 5
    purge_bars: int = 5
    embargo_bars: int = 0
    min_train_rows: int = 50

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if self.purge_bars < 0:
            raise ValueError("purge_bars must be non-negative")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")
        if self.min_train_rows < 1:
            raise ValueError("min_train_rows must be >= 1")


@dataclass(frozen=True, slots=True)
class CVFold:
    """One expanding-window fold, as ``[start, end)`` bundle row offsets.

    ``train_start`` is always the first training row. ``train_end`` is the
    boundary after the purge/embargo gap, so ``train_end <= test_start``.
    ``train_rows`` and ``test_rows`` are the geometric block sizes before any
    flat-label filtering; the fitted/scored counts are in the fold metrics.
    """

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CVResult:
    """Fold boundaries, per-fold metrics and their mean/std aggregate."""

    symbol: str
    timeframe: str
    dataset_id: str
    model: str
    config: CVConfig
    folds: tuple[CVFold, ...]
    fold_metrics: tuple[dict[str, Any], ...]
    aggregate: dict[str, dict[str, float]]
    split_counts: dict[str, int]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_id": self.dataset_id,
            "model": self.model,
            "config": asdict(self.config),
            "split_counts": self.split_counts,
            "folds": [
                {**fold.to_dict(), "metrics": metrics}
                for fold, metrics in zip(self.folds, self.fold_metrics, strict=True)
            ],
            "aggregate": self.aggregate,
            "created_at": self.created_at,
        }


def _block_edges(start: int, end: int, blocks: int) -> list[int]:
    """Contiguous ``[start, end)`` edges; the first blocks absorb the remainder."""
    total = end - start
    base, extra = divmod(total, blocks)
    sizes = [base + 1 if index < extra else base for index in range(blocks)]
    edges = [start]
    for size in sizes:
        edges.append(edges[-1] + size)
    if edges[-1] != end:  # pragma: no cover - guard only
        raise RuntimeError("fold blocks do not cover the training split")
    return edges


def _fold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
    *,
    train_rows: int,
) -> dict[str, Any]:
    """Classification and calibration metrics for one test block."""
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    one_hot = np.zeros_like(probabilities)
    for index, label in enumerate(classes):
        one_hot[y_true == label, index] = 1.0
    calibration = calibration_summary(y_true, probabilities, classes)
    metrics: dict[str, Any] = {
        "rows": int(y_true.size),
        "train_rows": int(train_rows),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(classes))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(calibration["ece"]),
        "mce": float(calibration["mce"]),
    }
    for key in _AGGREGATE_KEYS:
        if not np.isfinite(metrics[key]):
            raise ValueError(f"non-finite {key} in fold metrics")
    return metrics


def _aggregate(fold_metrics: tuple[dict[str, Any], ...]) -> dict[str, dict[str, float]]:
    """Mean and population std of every headline metric across folds."""
    aggregate: dict[str, dict[str, float]] = {}
    for key in _AGGREGATE_KEYS:
        values = np.asarray([float(metrics[key]) for metrics in fold_metrics])
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return aggregate


def cross_validate(
    features: FeatureFrame,
    *,
    symbol: str,
    timeframe: str,
    spec: TrainSpec,
    config: CVConfig,
) -> CVResult:
    """Run purged expanding-window CV over the training split only.

    The dataset is built once with ``spec`` (groups, horizon, fractions) and the
    validation/test rows are never selected. Every fold fits a fresh
    ``build_model(spec.model, seed=spec.seed)`` on its training window and scores
    the next contiguous block. Flat labels are dropped unless
    ``spec.include_flat`` is set, matching :func:`scalping.ml.train.train_classifier`.
    """
    bundle = build_dataset(
        features,
        symbol=symbol,
        timeframe=timeframe,
        groups=spec.groups,
        horizon=spec.horizon,
        train_fraction=spec.train_fraction,
        validation_fraction=spec.validation_fraction,
        purge_bars=max(spec.purge_bars, spec.horizon),
        embargo_bars=spec.embargo_bars,
    )

    train_positions = np.flatnonzero(bundle.split == 0)
    if train_positions.size == 0:
        raise ValueError("the dataset has no training rows")
    train_lo = int(train_positions[0])
    train_hi = int(train_positions[-1]) + 1
    if train_positions.size != train_hi - train_lo:
        raise ValueError("the training split is not contiguous")

    edges = _block_edges(train_lo, train_hi, config.n_splits + 1)
    purge = max(config.purge_bars, spec.horizon)
    gap = purge + config.embargo_bars
    keep = (
        np.ones(bundle.y_direction.shape[0], dtype=bool)
        if spec.include_flat
        else bundle.y_direction != 0
    )

    folds: list[CVFold] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(config.n_splits):
        test_start = edges[fold + 1]
        test_end = edges[fold + 2]
        train_end = test_start - gap
        if test_end <= test_start:
            raise ValueError(f"fold {fold}: empty test block")
        if train_end <= train_lo:
            raise ValueError(f"fold {fold}: purge/embargo ({gap} bars) leaves no training rows")
        train_rows = train_end - train_lo
        if train_rows < config.min_train_rows:
            raise ValueError(
                f"fold {fold}: {train_rows} training rows is below "
                f"min_train_rows={config.min_train_rows}"
            )

        train_index = np.arange(train_lo, train_end)[keep[train_lo:train_end]]
        test_index = np.arange(test_start, test_end)[keep[test_start:test_end]]
        if test_index.size == 0:
            raise ValueError(f"fold {fold}: empty test block after dropping flat rows")
        if train_index.size < 2:
            raise ValueError(
                f"fold {fold}: {train_index.size} training rows after dropping flat rows"
            )
        if train_index.size < config.min_train_rows:
            raise ValueError(
                f"fold {fold}: {train_index.size} usable training rows is below "
                f"min_train_rows={config.min_train_rows}"
            )

        y_train = bundle.y_direction[train_index]
        y_test = bundle.y_direction[test_index]
        model = build_model(spec.model, seed=spec.seed)
        model.fit(bundle.X[train_index], y_train)
        classes = tuple(int(value) for value in getattr(model, "classes_", ()))
        if len(classes) < 2:
            raise ValueError(f"fold {fold}: training window has fewer than two classes")
        missing = np.setdiff1d(y_test, np.asarray(classes))
        if missing.size:
            raise ValueError(
                f"fold {fold}: test labels {missing.tolist()} are absent from the training window"
            )
        probabilities = np.asarray(model.predict_proba(bundle.X[test_index]), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape != (y_test.size, len(classes)):
            raise ValueError(
                f"fold {fold}: predict_proba returned shape {probabilities.shape}, "
                f"expected {(y_test.size, len(classes))}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError(f"fold {fold}: predict_proba returned non-finite values")
        # float32 estimators (LightGBM) can sum to 1 +/- 1e-7. Re-normalise after
        # the float64 cast so sklearn's log_loss does not warn on rounding.
        totals = probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(totals).all() or np.any(totals <= 0.0):
            raise ValueError(f"fold {fold}: predict_proba rows do not sum to a positive value")
        probabilities = probabilities / totals

        folds.append(
            CVFold(
                fold=fold,
                train_start=train_lo,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_rows=train_rows,
                test_rows=test_end - test_start,
            )
        )
        fold_metrics.append(
            _fold_metrics(
                y_test,
                probabilities,
                classes,
                train_rows=int(train_index.size),
            )
        )

    return CVResult(
        symbol=symbol,
        timeframe=timeframe,
        dataset_id=bundle.manifest.dataset_id,
        model=spec.model,
        config=config,
        folds=tuple(folds),
        fold_metrics=tuple(fold_metrics),
        aggregate=_aggregate(tuple(fold_metrics)),
        split_counts={
            "train": bundle.rows("train"),
            "validation": bundle.rows("validation"),
            "test": bundle.rows("test"),
            "purged": int((bundle.split == -1).sum()),
            "flat_dropped": int((~keep).sum()),
        },
    )
