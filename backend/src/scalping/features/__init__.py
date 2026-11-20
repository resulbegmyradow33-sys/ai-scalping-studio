"""Causal market intelligence features.

Public surface:

* :func:`~scalping.features.build.build_features` returns a
  :class:`~scalping.features.build.FeatureFrame` for M1 bars.
* ``indicators`` (:func:`~scalping.features.indicators.atr`,
  :func:`~scalping.features.indicators.rsi`,
  :func:`~scalping.features.indicators.macd_hist`), ``structure``
  (:func:`~scalping.features.structure.market_structure`,
  :func:`~scalping.features.structure.swings`), ``liquidity``
  (:func:`~scalping.features.liquidity.fvg`,
  :func:`~scalping.features.liquidity.liquidity_sweep`) and ``context``
  (:func:`~scalping.features.context.session_flags`,
  :func:`~scalping.features.context.htf_context`) expose the building blocks.

Every row may use only data at or before its own bar; the 5m/15m context joins
the last closed bucket and swing labels sit at their confirmation bar.
"""

from __future__ import annotations

from scalping.features.build import FEATURE_COLUMNS, WARMUP, FeatureFrame, build_features

__all__ = ["FEATURE_COLUMNS", "WARMUP", "FeatureFrame", "build_features"]
