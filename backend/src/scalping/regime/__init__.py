"""Market-regime classification.

The rule engine in :mod:`scalping.regime.rules` is the deterministic baseline;
statistical models (K-Means, GMM, HMM and a supervised baseline) build on the
same aligned-label contract. Regimes describe the past, never a forecast.
"""

from __future__ import annotations

from scalping.regime.rules import REGIME_COLUMNS, RegimeConfig, classify

__all__ = ["REGIME_COLUMNS", "RegimeConfig", "classify"]
