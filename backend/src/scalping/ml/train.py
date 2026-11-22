"""Train and evaluate a classical direction classifier.

The dataset builder owns the chronological split and the purge/embargo. This
module fits on the training rows, reports validation metrics, and leaves the
test split untouched unless the caller explicitly asks for it. A model is only
an experiment artifact; promotion is a separate decision.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.features.build import FeatureFrame
from scalping.features.registry import DEFAULT_GROUPS
from scalping.ml.models import build_model

__all__ = [
    "MLResult",
    "TrainSpec",
    "load_model",
    "load_result",
    "save_result",
    "train_classifier",
]

#: Class labels emitted by the dataset builder.
CLASSES: tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """Everything that defines one classifier run."""

    model: str = "logistic"
    horizon: int = 5
    groups: tuple[str, ...] = DEFAULT_GROUPS
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    purge_bars: int = 5
    embargo_bars: int = 5
    seed: int = 0
    evaluate_test: bool = False
    include_flat: bool = False


@dataclass(frozen=True, slots=True)
class MLResult:
    """Fitted model plus the evidence needed to compare it."""

    model: Any
    spec: TrainSpec
    symbol: str
    timeframe: str
    dataset_id: str
    columns: tuple[str, ...]
    validation: dict[str, Any]
    test: dict[str, Any] | None
    baseline: dict[str, Any]
    split_counts: dict[str, int]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.spec.model,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_id": self.dataset_id,
            "columns": list(self.columns),
            "validation": self.validation,
            "test": self.test,
            "baseline": self.baseline,
            "split_counts": self.split_counts,
            "created_at": self.created_at,
            "spec": asdict(self.spec),
        }


def _metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
) -> dict[str, Any]:
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=list(classes),
        zero_division=0,
    )
    one_hot = np.zeros_like(probabilities)
    for index, label in enumerate(classes):
        one_hot[y_true == label, index] = 1.0
    return {
        "rows": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(classes))),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "precision": {
            str(label): float(precision[index]) for index, label in enumerate(classes)
        },
        "recall": {str(label): float(recall[index]) for index, label in enumerate(classes)},
        "f1": {str(label): float(f1[index]) for index, label in enumerate(classes)},
        "support": {str(label): int(support[index]) for index, label in enumerate(classes)},
        "confusion": confusion_matrix(y_true, predicted, labels=list(classes)).tolist(),
    }


def _baseline(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    classes: tuple[int, ...],
) -> dict[str, Any]:
    labels, counts = np.unique(y_train, return_counts=True)
    majority = int(labels[int(np.argmax(counts))])
    prior = np.zeros(len(classes), dtype=np.float64)
    for index, label in enumerate(classes):
        prior[index] = float(np.sum(y_train == label)) / max(1, y_train.size)
    prior = np.clip(prior, 1e-9, None)
    prior /= prior.sum()
    return {
        "majority_class": majority,
        "accuracy": float(np.mean(y_validation == majority)),
        "log_loss": float(log_loss(y_validation, np.tile(prior, (y_validation.size, 1)), labels=list(classes))),
        "class_prior": {str(label): float(prior[index]) for index, label in enumerate(classes)},
    }


def train_classifier(
    features: FeatureFrame,
    *,
    symbol: str,
    timeframe: str,
    spec: TrainSpec,
) -> MLResult:
    """Fit ``spec.model`` on the train split and measure validation."""
    bundle: DatasetBundle = build_dataset(
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
    # A 3-class target on H1 is effectively binary: an exact flat close is one
    # row in 4k. Keeping that class makes accuracy look better while the model
    # never predicts it. Drop flat rows by default and say how many were
    # dropped, instead of hiding the imbalance behind a misleading score.
    flat = bundle.y_direction == 0
    keep = np.ones(bundle.split.shape[0], dtype=bool) if spec.include_flat else ~flat
    train_mask = (bundle.split == 0) & keep
    validation_mask = (bundle.split == 1) & keep
    test_mask = (bundle.split == 2) & keep
    if int(train_mask.sum()) < 20:
        raise ValueError("not enough training rows; adjust fractions or horizon")
    if int(validation_mask.sum()) < 5:
        raise ValueError("not enough validation rows; adjust fractions or embargo")

    y_train = bundle.y_direction[train_mask]
    classes = tuple(int(value) for value in np.unique(y_train))
    model = build_model(spec.model, seed=spec.seed)
    model.fit(bundle.X[train_mask], y_train)
    probabilities = model.predict_proba(bundle.X[validation_mask])
    validation = _metrics(bundle.y_direction[validation_mask], probabilities, classes)
    test = None
    if spec.evaluate_test and int(test_mask.sum()) > 0:
        test = _metrics(
            bundle.y_direction[test_mask],
            model.predict_proba(bundle.X[test_mask]),
            classes,
        )
    return MLResult(
        model=model,
        spec=spec,
        symbol=symbol,
        timeframe=timeframe,
        dataset_id=bundle.manifest.dataset_id,
        columns=bundle.manifest.columns,
        validation=validation,
        test=test,
        baseline=_baseline(y_train, bundle.y_direction[validation_mask], classes),
        split_counts={
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
            "purged": int((bundle.split == -1).sum()),
            "flat_dropped": int(flat.sum()),
        },
    )


def save_result(result: MLResult, directory: str | Path) -> tuple[Path, Path]:
    """Persist the model and metadata; return ``(model_path, metadata_path)``."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.joblib"
    metadata_path = target / "metrics.json"
    joblib.dump(
        {
            "model": result.model,
            "classes": list(getattr(result.model, "classes_", CLASSES)),
            "columns": list(result.columns),
        },
        model_path,
    )
    temp = metadata_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, metadata_path)
    return model_path, metadata_path


def load_model(path: str | Path) -> tuple[Any, list[str]]:
    payload = joblib.load(Path(path))
    return payload["model"], payload["columns"]


def load_result(directory: str | Path) -> MLResult:
    """Rebuild an ``MLResult`` from a saved model directory."""
    target = Path(directory)
    metadata = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    spec_raw = metadata["spec"]
    spec = TrainSpec(
        model=str(spec_raw["model"]),
        horizon=int(spec_raw["horizon"]),
        groups=tuple(str(value) for value in spec_raw["groups"]),
        train_fraction=float(spec_raw["train_fraction"]),
        validation_fraction=float(spec_raw["validation_fraction"]),
        purge_bars=int(spec_raw["purge_bars"]),
        embargo_bars=int(spec_raw["embargo_bars"]),
        seed=int(spec_raw["seed"]),
        evaluate_test=bool(spec_raw.get("evaluate_test", False)),
        include_flat=bool(spec_raw.get("include_flat", False)),
    )
    model, columns = load_model(target / "model.joblib")
    return MLResult(
        model=model,
        spec=spec,
        symbol=str(metadata["symbol"]),
        timeframe=str(metadata["timeframe"]),
        dataset_id=str(metadata["dataset_id"]),
        columns=tuple(columns),
        validation=metadata["validation"],
        test=metadata.get("test"),
        baseline=metadata["baseline"],
        split_counts=metadata["split_counts"],
        created_at=str(metadata["created_at"]),
    )
