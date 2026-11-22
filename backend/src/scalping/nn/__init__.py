"""Neural-network research framework.

Phase 6 builds sequence classifiers on the leakage-safe dataset bundle:
:mod:`scalping.nn.data` provides windowed tensors with train-only
normalisation. Models, trainer and the shared evaluator arrive in later slices.
"""

from __future__ import annotations

from scalping.nn.data import WindowConfig, WindowedDataset, WindowError, build_windowed_dataset

__all__ = ["WindowConfig", "WindowError", "WindowedDataset", "build_windowed_dataset"]
