"""Neural-network research framework.

Phase 6 builds sequence classifiers on the leakage-safe dataset bundle:
:mod:`scalping.nn.data` provides windowed tensors with train-only
normalisation and :mod:`scalping.nn.models` the MLP/RNN/CNN/TCN zoo. The trainer
and shared evaluator arrive in later slices.
"""

from __future__ import annotations

from scalping.nn.data import WindowConfig, WindowedDataset, WindowError, build_windowed_dataset
from scalping.nn.models import (
    MODEL_NAMES,
    CausalTCN,
    ModelSpec,
    build_model,
    model_names,
)

__all__ = [
    "MODEL_NAMES",
    "CausalTCN",
    "ModelSpec",
    "WindowConfig",
    "WindowError",
    "WindowedDataset",
    "build_model",
    "build_windowed_dataset",
    "model_names",
]
