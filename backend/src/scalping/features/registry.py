"""Named feature groups for configurable datasets.

The 59-column matrix is useful as a default, but not every experiment should
pay for every column. This registry maps each default column to exactly one
group and returns column indices in the original order, so selecting a subset
never silently reorders features.

Two kinds of group exist:

* groups in :data:`FEATURE_GROUPS` partition :data:`FEATURE_COLUMNS` and are
  selected by default;
* groups in :data:`OPTIONAL_FEATURE_GROUPS` add columns **beyond** the 59-column
  matrix and are built only when requested (``groups=("htf",)``). They never
  change :data:`FEATURE_COLUMNS` or the default matrix.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np

from scalping.features.build import FEATURE_COLUMNS
from scalping.features.context import MTF_CONTEXT_COLUMNS

__all__ = [
    "ALL_FEATURE_COLUMNS",
    "DEFAULT_GROUPS",
    "FEATURE_ALIASES",
    "FEATURE_GROUPS",
    "OPTIONAL_FEATURE_GROUPS",
    "build_columns",
    "canonical_name",
    "columns_for",
    "indices_for",
    "optional_groups",
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

#: Groups whose columns sit outside :data:`FEATURE_COLUMNS`. They are opt-in:
#: ``columns_for(None)`` and the 59-column matrix are unchanged. The ``htf``
#: group adds closed M30/H4 context (see ``features/context.py``); building it
#: requires M30/H4 resampling, hence the explicit request.
OPTIONAL_FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "htf": MTF_CONTEXT_COLUMNS,
}

#: Canonical order of every column the builder can emit: the 59 default columns
#: first, then optional groups in registry order. A default build is a prefix of
#: this tuple, so base-column indices never move.
ALL_FEATURE_COLUMNS: Final[tuple[str, ...]] = FEATURE_COLUMNS + tuple(
    column for columns in OPTIONAL_FEATURE_GROUPS.values() for column in columns
)

#: Every group name a caller may pass, default and optional.
_ALL_GROUP_NAMES: Final[tuple[str, ...]] = (*FEATURE_GROUPS, *OPTIONAL_FEATURE_GROUPS)


def _validate(groups: Iterable[str]) -> tuple[str, ...]:
    names = tuple(groups)
    unknown = [name for name in names if name not in _ALL_GROUP_NAMES]
    if unknown:
        raise ValueError(
            f"unknown feature groups: {', '.join(unknown)}; available: "
            f"{', '.join(_ALL_GROUP_NAMES)}"
        )
    return names


def optional_groups(groups: Iterable[str] | None = None) -> tuple[str, ...]:
    """The optional groups named by ``groups``, in registry order.

    ``None`` and base-only selections return ``()``, which is what a cache key
    and the builder need: the default matrix is independent of which base
    groups a downstream dataset selects.
    """
    if groups is None:
        return ()
    selected = set(_validate(groups))
    return tuple(name for name in OPTIONAL_FEATURE_GROUPS if name in selected)


def build_columns(groups: Iterable[str] | None = None) -> tuple[str, ...]:
    """Columns :func:`~scalping.features.build.build_features` emits for ``groups``.

    The 59 default columns are always built, because every existing consumer
    indexes them by position; requested optional groups are appended.
    """
    wanted = set(optional_groups(groups))
    return FEATURE_COLUMNS + tuple(
        column
        for name, columns in OPTIONAL_FEATURE_GROUPS.items()
        if name in wanted
        for column in columns
    )


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

    ``None`` means every default group, i.e. :data:`FEATURE_COLUMNS` exactly.
    Optional groups (``htf``) are included only when named. Unknown groups
    raise instead of producing a silently smaller dataset.
    """
    names = _validate(DEFAULT_GROUPS if groups is None else groups)
    selected = {
        column
        for name in names
        for column in (
            FEATURE_GROUPS[name] if name in FEATURE_GROUPS else OPTIONAL_FEATURE_GROUPS[name]
        )
    }
    return tuple(column for column in ALL_FEATURE_COLUMNS if column in selected)


def indices_for(groups: Iterable[str] | None = None) -> list[int]:
    """Column indices for ``groups`` in the canonical order.

    Indices address :data:`ALL_FEATURE_COLUMNS`, so a default selection is
    ``range(59)`` and optional columns follow at ``59`` onward. The matrix must
    have been built with those optional groups; :func:`select_matrix` and
    :func:`~scalping.datasets.builder.build_dataset` reject a narrower frame.
    """
    wanted = set(columns_for(groups))
    return [index for index, name in enumerate(ALL_FEATURE_COLUMNS) if name in wanted]


def select_matrix(X: np.ndarray, groups: Iterable[str] | None = None) -> np.ndarray:
    """Return ``X`` restricted to ``groups`` without changing row order."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    indices = indices_for(groups)
    if indices and X.shape[1] <= max(indices):
        raise ValueError(
            f"X has {X.shape[1]} columns but the selection needs column {max(indices)}; "
            "build the feature frame with the optional groups included"
        )
    return X[:, indices]
