"""Shared prediction evaluator for neural models and the momentum baseline.

This module answers *prediction* questions: how often is the model right, is it
right for every class, and are its probabilities calibrated? It does **not**
claim an edge — a trading claim still requires the simulator and a cost stress
on the same rows.

The rule comparison is deliberately narrow: the repo's `ret_5` momentum
baseline (the same signal `scalping.cli` replays) is turned into a `-1/0/+1`
prediction and scored on the exact test rows the network saw. The full
`flow_momentum` card needs tick flow and is evaluated by the strategy layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import Tensor, nn

from scalping.datasets.builder import DatasetBundle
from scalping.nn.data import WindowedDataset
from scalping.nn.models import ModelName, build_model
from scalping.nn.train import DeviceRequest, load_checkpoint, select_device

__all__ = [
    "SPLIT_NAMES",
    "ClassificationMetrics",
    "classification_metrics",
    "compare_predictors",
    "evaluate_checkpoint",
    "momentum_rule_labels",
    "predict",
    "predict_checkpoint",
]

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")
SplitName = Literal["train", "validation", "test"]

_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Accuracy, balanced accuracy, calibration and per-class detail."""

    rows: int
    classes: tuple[int, ...]
    accuracy: float
    balanced_accuracy: float
    log_loss: float | None
    brier: float | None
    per_class: tuple[dict[str, float | int], ...]
    confusion: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "classes": list(self.classes),
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "per_class": list(self.per_class),
            "confusion": [list(row) for row in self.confusion],
        }


def _check_labels(y: np.ndarray, name: str) -> None:
    if y.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {y.shape}")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    *,
    classes: tuple[int, ...] | None = None,
) -> ClassificationMetrics:
    """Score labels (and optional probabilities) in a fixed class order."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    _check_labels(truth, "y_true")
    _check_labels(prediction, "y_pred")
    if truth.shape[0] != prediction.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if truth.shape[0] == 0:
        raise ValueError("cannot score an empty prediction set")
    if classes is None:
        classes = tuple(sorted({int(value) for value in truth.tolist()} | set(prediction.tolist())))
    if len(classes) < 2:
        raise ValueError("need at least two classes to score a classifier")
    index = {value: position for position, value in enumerate(classes)}
    unknown = sorted({int(value) for value in truth.tolist()} - set(classes))
    if unknown:
        raise ValueError(f"y_true contains classes outside the class list: {unknown}")
    truth_idx = np.array([index[int(value)] for value in truth.tolist()], dtype=np.int64)
    pred_idx = np.array([index.get(int(value), -1) for value in prediction.tolist()], dtype=np.int64)

    accuracy = float((truth_idx == pred_idx).mean())
    recalls: list[float] = []
    per_class: list[dict[str, float | int]] = []
    for position, value in enumerate(classes):
        support = int((truth_idx == position).sum())
        predicted = int((pred_idx == position).sum())
        hits = int(((truth_idx == position) & (pred_idx == position)).sum())
        precision = hits / predicted if predicted else 0.0
        recall = hits / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)
        per_class.append(
            {
                "class": int(value),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    balanced = float(np.mean(recalls)) if recalls else 0.0
    confusion = tuple(
        tuple(int(((truth_idx == row) & (pred_idx == column)).sum()) for column in range(len(classes)))
        for row in range(len(classes))
    )

    log_loss: float | None = None
    brier: float | None = None
    if proba is not None:
        matrix = np.asarray(proba, dtype=np.float64)
        if matrix.shape != (truth.shape[0], len(classes)):
            raise ValueError(
                f"proba must have shape ({truth.shape[0]}, {len(classes)}), got {matrix.shape}"
            )
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("proba contains non-finite or negative values")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("proba rows must sum to 1")
        clipped = np.clip(matrix, _EPS, 1.0)
        chosen = clipped[np.arange(truth.shape[0]), truth_idx]
        log_loss = float(-np.log(chosen).mean())
        one_hot = np.zeros_like(matrix)
        one_hot[np.arange(truth.shape[0]), truth_idx] = 1.0
        brier = float(((matrix - one_hot) ** 2).sum(axis=1).mean())

    return ClassificationMetrics(
        rows=int(truth.shape[0]),
        classes=tuple(int(value) for value in classes),
        accuracy=accuracy,
        balanced_accuracy=balanced,
        log_loss=log_loss,
        brier=brier,
        per_class=tuple(per_class),
        confusion=confusion,
    )


def _split_arrays(data: WindowedDataset, split: SplitName) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; choose from {SPLIT_NAMES}")
    X = getattr(data, f"X_{split}")
    y = getattr(data, f"y_{split}")
    ts = getattr(data, f"ts_{split}")
    if X.shape[0] == 0:
        raise ValueError(f"the {split} split has no windows")
    return X, y, ts


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    classes: tuple[int, ...],
    *,
    batch_size: int = 256,
    device: DeviceRequest = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities and decoded labels in ``classes`` order."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    target = select_device(device)
    model = model.to(target).eval()
    tensor = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    chunks: list[Tensor] = []
    for start in range(0, tensor.shape[0], batch_size):
        chunks.append(torch.softmax(model(tensor[start : start + batch_size].to(target)), dim=1).cpu())
    probabilities = torch.cat(chunks).numpy().astype(np.float64)
    decoded = np.array([classes[index] for index in probabilities.argmax(axis=1)], dtype=np.int64)
    return probabilities, decoded


def predict_checkpoint(
    checkpoint: str | Path,
    data: WindowedDataset,
    *,
    split: SplitName = "test",
    batch_size: int = 256,
    device: DeviceRequest = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild the checkpoint's model and score one split of ``data``."""
    payload = load_checkpoint(checkpoint)
    if payload["n_features"] != data.feature_count or payload["window"] != data.window:
        raise ValueError("checkpoint shape does not match the dataset")
    classes = tuple(int(value) for value in payload["classes"])
    model = build_model(
        payload["model"],  # type: ignore[arg-type]
        n_features=int(payload["n_features"]),
        window=int(payload["window"]),
        n_classes=len(classes),
        hidden=int(payload["config"]["hidden"]),
        dropout=float(payload["config"]["dropout"]),
        seed=int(payload["config"]["seed"]),
    )
    state = payload.get("best_state") or payload["model_state"]
    model.load_state_dict(state)
    X, y, ts = _split_arrays(data, split)
    probabilities, decoded = predict(model, X, classes, batch_size=batch_size, device=device)
    meta = {
        "model": payload["model"],
        "dataset_id": payload.get("dataset_id", data.dataset_id),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "classes": list(classes),
        "split": split,
    }
    return probabilities, decoded, y, ts, meta


def evaluate_checkpoint(
    checkpoint: str | Path,
    data: WindowedDataset,
    *,
    split: SplitName = "test",
    batch_size: int = 256,
    device: DeviceRequest = "auto",
) -> dict[str, Any]:
    """Metrics for one split, with the checkpoint identity attached."""
    probabilities, decoded, y, ts, meta = predict_checkpoint(
        checkpoint, data, split=split, batch_size=batch_size, device=device
    )
    metrics = classification_metrics(y, decoded, probabilities, classes=tuple(meta["classes"]))
    return {
        **meta,
        "metrics": metrics.to_dict(),
        "first_ts": str(ts[0]),
        "last_ts": str(ts[-1]),
        # Prediction metrics are not a trading result; the simulator gate is
        # still required before anything is called profitable.
        "trading_claim": False,
    }


def momentum_rule_labels(
    bundle: DatasetBundle,
    ts: np.ndarray,
    *,
    column: str = "ret_5",
) -> np.ndarray:
    """The CLI's `ret_5` momentum baseline as a `-1/0/+1` prediction.

    Scored on the same window-end timestamps as the network, so the comparison
    is row-for-row. This is the baseline rule, not the validated
    `flow_momentum` card.
    """
    if column not in bundle.manifest.columns:
        raise ValueError(f"dataset has no {column!r} column")
    position = bundle.manifest.columns.index(column)
    rows = np.searchsorted(bundle.ts, ts)
    if np.any(rows >= bundle.ts.shape[0]) or np.any(bundle.ts[rows] != ts):
        raise ValueError("timestamps do not align with the dataset rows")
    values = bundle.X[rows, position]
    return np.sign(values).astype(np.int64)


def compare_predictors(
    y_true: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray | None]],
    *,
    classes: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Score named predictors on one target vector; largest accuracy first."""
    rows: list[dict[str, Any]] = []
    for name, (labels, proba) in predictions.items():
        metrics = classification_metrics(y_true, labels, proba, classes=classes)
        rows.append(
            {
                "name": name,
                "rows": metrics.rows,
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "log_loss": metrics.log_loss,
                "brier": metrics.brier,
            }
        )
    return sorted(rows, key=lambda row: (-row["accuracy"], row["name"]))


def model_from_name(name: ModelName, **kwargs: Any) -> nn.Module:
    """Small alias so callers do not import the zoo module directly."""
    return build_model(name, **kwargs)
