"""Market-regime classification, evaluation and trade attribution.

The rule engine in :mod:`scalping.regime.rules` is the deterministic baseline;
:mod:`scalping.regime.models` adds K-Means, Gaussian mixture, HMM and a
supervised baseline on the same aligned-label contract.
:mod:`scalping.regime.strategy` joins closed trades to the label at their entry
bar. Regimes describe the past, never a forecast.
"""

from __future__ import annotations

from scalping.regime.evaluate import dwell_stats, transition_matrix
from scalping.regime.features import REGIME_FEATURES, build_regime_features
from scalping.regime.models import (
    MODEL_NAMES,
    ModelConfig,
    RegimeFit,
    RegimeModelError,
    fit_predict,
)
from scalping.regime.rules import REGIME_COLUMNS, RegimeConfig, classify
from scalping.regime.strategy import strategy_regime_table, trades_with_regime

__all__ = [
    "MODEL_NAMES",
    "REGIME_COLUMNS",
    "REGIME_FEATURES",
    "ModelConfig",
    "RegimeConfig",
    "RegimeFit",
    "RegimeModelError",
    "build_regime_features",
    "classify",
    "dwell_stats",
    "fit_predict",
    "strategy_regime_table",
    "trades_with_regime",
    "transition_matrix",
]
