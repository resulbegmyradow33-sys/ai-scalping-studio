"""Named feature groups for configurable datasets.

The 59-column matrix is useful as a default, but not every experiment should
pay for every column. This registry maps each column to exactly one group and
returns column indices in the original order, so selecting a subset never
silently reorders features.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np

from scalping.features.build import FEATURE_COLUMNS

__all__ = [
    "DEFAULT_GROUPS",
    "FEATURE_ALIASES",
    "FEATURE_GROUPS",
    "canonical_name",
    "columns_for",
    "indices_for",
    "select_matrix",
]

#: Documented read-only aliases for canonical columns whose name is easy to
#: misread. An alias never enters :data:`~scalping.features.build.FEATURE_COLUMNS`,
#: so column order, the cached registry version and dataset checksums stay
#: stable. ``ret_3`` is a log return while the other ``ret_*`` columns are
#: simple returns; ``log_ret_3`` names that without renaming the column.
FEATURE_ALIASES: Final[dict[str, str]] = {
    "log_ret_3": "ret_3",
}

FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "price": (
        "ret_1",
        "ret_5",
        "ret_15",
        "ret_60",
        "ret_3",
        "body_atr",
        "upper_wick_atr",
        "lower_wick_atr",
        "range_atr",
        "dist_high_20",
        "dist_low_20",
        "pos_range_20",
        "eq_high_20",
        "eq_low_20",
    ),
    "momentum": (
        "ret_z20",
        "rsi_14",
        "macd_hist_atr",
        "ema_gap_9_21",
        "close_ema_60",
    ),
    "volatility": (
        "realized_vol_30",
        "vol_regime",
        "atr_norm",
        "atr_ratio_14_60",
        "spread_norm",
        "spread_z20",
    ),
    "volume": (
        "volume_z20",
        "volume_ratio_20",
    ),
    "structure": (
        "trend",
        "bos",
        "choch",
        "swing_high",
        "swing_low",
        "dist_structure_high",
        "dist_structure_low",
        "order_block_dist_atr",
        "liquidity_sweep",
        "fvg_up",
        "fvg_down",
        "fvg_dist",
    ),
    "session": (
        "tod_sin",
        "tod_cos",
        "asia",
        "london",
        "ny",
        "ny_open_30",
        "vwap_dist_atr",
        "day_position",
    ),
    "multi_timeframe": (
        "m5_ret_1",
        "m5_rsi_14",
        "m5_atr_norm",
        "m5_ema_gap_21",
        "m5_pos_range_20",
        "m15_ret_1",
        "m15_rsi_14",
        "m15_atr_norm",
        "m15_ema_gap_21",
        "m15_pos_range_20",
        "htf_bos",
        "htf_fvg_dist",
    ),
}

DEFAULT_GROUPS: Final[tuple[str, ...]] = tuple(FEATURE_GROUPS)


def canonical_name(name: str) -> str:
    """Resolve a documented alias to its canonical ``FEATURE_COLUMNS`` name.

    Canonical names pass through unchanged. Unknown names raise instead of
    resolving to themselves, so a typo cannot silently become a new feature.
    """
    resolved = FEATURE_ALIASES.get(name, name)
    if resolved not in FEATURE_COLUMNS:
        raise ValueError(f"unknown feature {name!r}; aliases: {', '.join(sorted(FEATURE_ALIASES))}")
    return resolved


def columns_for(groups: Iterable[str] | None = None) -> tuple[str, ...]:
    """Selected column names in the canonical matrix order.

    ``None`` means every registered group. Unknown groups raise instead of
    producing a silently smaller dataset.
    """
    names = list(DEFAULT_GROUPS if groups is None else groups)
    unknown = [name for name in names if name not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature groups: {', '.join(unknown)}; available: {', '.join(FEATURE_GROUPS)}"
        )
    selected = {column for name in names for column in FEATURE_GROUPS[name]}
    return tuple(column for column in FEATURE_COLUMNS if column in selected)


def indices_for(groups: Iterable[str] | None = None) -> list[int]:
    """Column indices for ``groups`` in the canonical order."""
    wanted = set(columns_for(groups))
    return [index for index, name in enumerate(FEATURE_COLUMNS) if name in wanted]


def select_matrix(X: np.ndarray, groups: Iterable[str] | None = None) -> np.ndarray:
    """Return ``X`` restricted to ``groups`` without changing row order."""
    indices = indices_for(groups)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return X[:, indices]
