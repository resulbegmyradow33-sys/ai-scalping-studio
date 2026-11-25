"""Unsupervised anomaly detection over the causal feature matrix.

An anomaly score answers "how unlike the training rows is this row?". It is a
diagnostic, not a direction probability: there is no ``predict_proba``, no
class, no threshold that implies BUY or SELL, and nothing here is a promotion
candidate. The score is defined so that **higher means more anomalous**.

Fit and scoring are separated exactly like the classifier: fit on the training
split only, choose the operational threshold from a quantile of the *training*
scores, then score validation/test rows without refitting. A magnitude label
(for the optional AUC) is also defined from a training-only quantile, so the
separation number never peeks at the scored split's distribution.

Honest limits: an anomaly detector is not a predictor of returns. It can flag
unusual regimes, data problems or candidate event windows; a high score is not
evidence of a tradeable edge and must be turned into a falsifiable hypothesis
and tested on a new window before it means anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np
from sklearn.metrics import roc_auc_score

from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.features.build import FeatureFrame
from scalping.features.registry import DEFAULT_GROUPS
from scalping.ml.models import build_anomaly_model

__all__ = [
    "AnomalyDetector",
    "AnomalyReport",
    "AnomalyScore",
    "AnomalySpec",
    "detect_anomalies",
    "score_anomaly_split",
]

_SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")

#: Minimum training rows; an isolation forest on fewer is noise, not a model.
MIN_TRAIN_ROWS: Final[int] = 20


@dataclass(frozen=True, slots=True)
class AnomalySpec:
    """Dataset geometry and threshold quantile for one anomaly run.

    The data fields mirror :class:`scalping.ml.train.TrainSpec` because the same
    chronological purge/embargo rules apply. There is no ``model`` choice by
    default and no promotion flag: only isolation forest is implemented, and an
    anomaly run is never a candidate.
    """

    horizon: int = 5
    groups: tuple[str, ...] = DEFAULT_GROUPS
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    purge_bars: int = 5
    embargo_bars: int = 5
    seed: int = 0
    #: Training-score quantile used as the anomaly threshold (0.99 = top 1%).
    quantile: float = 0.99

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    """One split's anomaly scores and the training-derived threshold."""

    split: str
    rows: int
    threshold: float
    anomaly_rate: float
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    #: ``{"metric": "roc_auc", "auc": float | None, "positive_rate": float,
    #: "reason": ...}`` or ``None`` when no labels were supplied.
    separation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnomalyDetector:
    """A fitted detector plus the threshold chosen from training scores."""

    model: Any
    symbol: str
    timeframe: str
    dataset_id: str
    columns: tuple[str, ...]
    threshold: float
    quantile: float
    train_rows: int
    train_scores: np.ndarray

    @property
    def task(self) -> str:
        return "anomaly"

    @property
    def promotion_eligible(self) -> bool:
        """Always false: an anomaly detector is never a promotion candidate."""
        return False

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per row; higher means more unusual. Always finite."""
        features = np.asarray(X)
        raw = self.model.score_samples(features)
        values = -np.asarray(raw, dtype=np.float64).reshape(-1)
        if values.shape[0] != features.shape[0]:
            raise ValueError(
                f"score_samples returned {values.shape[0]} rows, expected {features.shape[0]}"
            )
        if not np.isfinite(values).all():
            raise ValueError("score_samples returned non-finite values")
        return values

    def flag(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask of rows whose score exceeds the training threshold."""
        return self.score(X) > self.threshold


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    """Detector, per-split scores and the geometry they were measured on."""

    detector: AnomalyDetector
    spec: AnomalySpec
    splits: dict[str, AnomalyScore]
    split_counts: dict[str, int]
    magnitude_quantile: float | None
    magnitude_cutoff: float | None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def task(self) -> str:
        return "anomaly"

    @property
    def promotion_eligible(self) -> bool:
        """Always false; see :class:`AnomalyDetector`."""
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "promotion_eligible": self.promotion_eligible,
            "symbol": self.detector.symbol,
            "timeframe": self.detector.timeframe,
            "dataset_id": self.detector.dataset_id,
            "model": "isolation_forest",
            "columns": list(self.detector.columns),
            "threshold": self.detector.threshold,
            "quantile": self.detector.quantile,
            "train_rows": self.detector.train_rows,
            "magnitude_quantile": self.magnitude_quantile,
            "magnitude_cutoff": self.magnitude_cutoff,
            "split_counts": self.split_counts,
            "splits": {name: score.to_dict() for name, score in self.splits.items()},
            "spec": asdict(self.spec),
            "created_at": self.created_at,
        }


def score_anomaly_split(
    detector: AnomalyDetector,
    bundle: DatasetBundle,
    *,
    split: str,
    labels: np.ndarray | None = None,
) -> AnomalyScore:
    """Score one split; report separation only when labels are supplied.

    ``labels`` is a boolean vector aligned with the split rows. When it is
    ``None`` the result carries no AUC: an unlabelled score has no ground truth
    to separate. A single-class label vector is reported with ``auc: None`` and
    a reason instead of inventing a number.
    """
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}; choose one of {_SPLITS}")
    mask = bundle.mask(split)
    rows = int(mask.sum())
    if rows == 0:
        raise ValueError(f"split {split!r} has no rows")
    scores = detector.score(bundle.X[mask])
    separation: dict[str, Any] | None = None
    if labels is not None:
        target = np.asarray(labels).reshape(-1)
        if target.shape[0] != rows:
            raise ValueError(
                f"labels have {target.shape[0]} entries, expected {rows} for split {split!r}"
            )
        if target.dtype.kind == "f" and not np.isfinite(target).all():
            raise ValueError("labels must not contain NaN or infinity")
        positive = target.astype(bool)
        positive_rate = float(positive.mean())
        if positive.all() or not positive.any():
            separation = {
                "metric": "roc_auc",
                "auc": None,
                "positive_rate": positive_rate,
                "reason": "single-class labels",
            }
        else:
            separation = {
                "metric": "roc_auc",
                "auc": float(roc_auc_score(positive, scores)),
                "positive_rate": positive_rate,
            }
    flagged = scores > detector.threshold
    return AnomalyScore(
        split=split,
        rows=rows,
        threshold=detector.threshold,
        anomaly_rate=float(flagged.mean()),
        score_mean=float(scores.mean()),
        score_std=float(scores.std(ddof=0)),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        separation=separation,
    )


def detect_anomalies(
    features: FeatureFrame,
    *,
    symbol: str,
    timeframe: str,
    spec: AnomalySpec | None = None,
    splits: tuple[str, ...] = ("validation",),
    magnitude_quantile: float | None = 0.9,
) -> AnomalyReport:
    """Fit an isolation forest on train rows and score the requested splits.

    ``splits`` may include ``validation`` and/or ``test`` (and ``train`` for a
    sanity check). ``magnitude_quantile`` defines the optional "large move"
    label as ``|future return| >=`` the same quantile of the *training* absolute
    returns; pass ``None`` to report scores without any separation number.
    Validation/test labels are used only as an after-the-fact comparison, never
    for fitting or threshold choice.
    """
    if not splits:
        raise ValueError("splits must not be empty")
    unknown = [name for name in splits if name not in _SPLITS]
    if unknown:
        raise ValueError(f"unknown splits {unknown}; choose from {_SPLITS}")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must not repeat")
    if magnitude_quantile is not None and not 0.0 < magnitude_quantile < 1.0:
        raise ValueError("magnitude_quantile must be in (0, 1) or None")

    resolved = spec or AnomalySpec()
    bundle = build_dataset(
        features,
        symbol=symbol,
        timeframe=timeframe,
        groups=resolved.groups,
        horizon=resolved.horizon,
        train_fraction=resolved.train_fraction,
        validation_fraction=resolved.validation_fraction,
        purge_bars=max(resolved.purge_bars, resolved.horizon),
        embargo_bars=resolved.embargo_bars,
    )
    train_mask = bundle.split == 0
    train_rows = int(train_mask.sum())
    if train_rows < MIN_TRAIN_ROWS:
        raise ValueError(
            f"not enough training rows for an anomaly fit ({train_rows} < {MIN_TRAIN_ROWS})"
        )

    model = build_anomaly_model("isolation_forest", seed=resolved.seed)
    model.fit(bundle.X[train_mask])
    train_scores = -np.asarray(model.score_samples(bundle.X[train_mask]), dtype=np.float64)
    if not np.isfinite(train_scores).all():
        raise ValueError("training anomaly scores are not finite")
    threshold = float(np.quantile(train_scores, resolved.quantile))

    cutoff: float | None = None
    if magnitude_quantile is not None:
        magnitudes = np.abs(bundle.y_return[train_mask].astype(np.float64))
        finite = magnitudes[np.isfinite(magnitudes)]
        if finite.size:
            cutoff = float(np.quantile(finite, magnitude_quantile))

    detector = AnomalyDetector(
        model=model,
        symbol=symbol,
        timeframe=timeframe,
        dataset_id=bundle.manifest.dataset_id,
        columns=bundle.manifest.columns,
        threshold=threshold,
        quantile=resolved.quantile,
        train_rows=train_rows,
        train_scores=train_scores,
    )

    scores: dict[str, AnomalyScore] = {}
    for split in splits:
        labels: np.ndarray | None = None
        if cutoff is not None:
            split_magnitudes = np.abs(bundle.y_return[bundle.mask(split)].astype(np.float64))
            labels = np.isfinite(split_magnitudes) & (split_magnitudes >= cutoff)
        scores[split] = score_anomaly_split(detector, bundle, split=split, labels=labels)

    return AnomalyReport(
        detector=detector,
        spec=resolved,
        splits=scores,
        split_counts={
            "train": train_rows,
            "validation": int((bundle.split == 1).sum()),
            "test": int((bundle.split == 2).sum()),
            "purged": int((bundle.split == -1).sum()),
        },
        magnitude_quantile=magnitude_quantile,
        magnitude_cutoff=cutoff,
    )
