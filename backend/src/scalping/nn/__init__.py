"""Neural-network research framework.

Phase 6 builds sequence classifiers on the leakage-safe dataset bundle:
:mod:`scalping.nn.data` provides windowed tensors with train-only
normalisation and :mod:`scalping.nn.models` the MLP/RNN/CNN/TCN zoo. The trainer
and shared evaluator arrive in later slices.
"""

from __future__ import annotations

from scalping.nn.benchmark import BenchmarkRow, benchmark_models
from scalping.nn.data import WindowConfig, WindowedDataset, WindowError, build_windowed_dataset
from scalping.nn.evaluate import (
    ClassificationMetrics,
    classification_metrics,
    compare_predictors,
    evaluate_checkpoint,
    momentum_rule_labels,
    predict,
    predict_checkpoint,
)
from scalping.nn.models import (
    MODEL_NAMES,
    CausalTCN,
    ModelSpec,
    build_model,
    model_names,
)
from scalping.nn.train import (
    JobControl,
    JobSlot,
    TrainConfig,
    TrainError,
    TrainResult,
    load_checkpoint,
    select_device,
    train_classifier,
)

__all__ = [
    "MODEL_NAMES",
    "BenchmarkRow",
    "CausalTCN",
    "ClassificationMetrics",
    "JobControl",
    "JobSlot",
    "ModelSpec",
    "TrainConfig",
    "TrainError",
    "TrainResult",
    "WindowConfig",
    "WindowError",
    "WindowedDataset",
    "benchmark_models",
    "build_model",
    "build_windowed_dataset",
    "classification_metrics",
    "compare_predictors",
    "evaluate_checkpoint",
    "load_checkpoint",
    "model_names",
    "momentum_rule_labels",
    "predict",
    "predict_checkpoint",
    "select_device",
    "train_classifier",
]
