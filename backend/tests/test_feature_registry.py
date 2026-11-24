"""Feature group registry."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.features.build import FEATURE_COLUMNS
from scalping.features.registry import (
    DEFAULT_GROUPS,
    FEATURE_ALIASES,
    FEATURE_GROUPS,
    canonical_name,
    columns_for,
    indices_for,
    select_matrix,
)

#: Exact matrix order. Changing it changes the cached registry version and every
#: stored dataset's column meaning, so an intentional change must edit this pin.
PINNED_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_60",
    "ret_3",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "range_atr",
    "realized_vol_30",
    "vol_regime",
    "ret_z20",
    "rsi_14",
    "macd_hist_atr",
    "ema_gap_9_21",
    "close_ema_60",
    "atr_norm",
    "atr_ratio_14_60",
    "volume_z20",
    "volume_ratio_20",
    "spread_norm",
    "spread_z20",
    "dist_high_20",
    "dist_low_20",
    "pos_range_20",
    "eq_high_20",
    "eq_low_20",
    "tod_sin",
    "tod_cos",
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
    "asia",
    "london",
    "ny",
    "ny_open_30",
    "vwap_dist_atr",
    "day_position",
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
)


def test_every_column_belongs_to_exactly_one_group() -> None:
    grouped = [column for group in FEATURE_GROUPS.values() for column in group]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(FEATURE_COLUMNS)


def test_default_selection_preserves_canonical_order() -> None:
    assert columns_for(None) == FEATURE_COLUMNS
    assert tuple(DEFAULT_GROUPS) == tuple(FEATURE_GROUPS)
    subset = columns_for(("momentum", "price"))
    assert subset[0] == "ret_1"
    assert subset.index("ret_1") < subset.index("rsi_14")


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="unknown feature groups"):
        columns_for(("price", "astrology"))


def test_select_matrix_uses_registry_indices() -> None:
    X = np.arange(5 * len(FEATURE_COLUMNS), dtype=np.float32).reshape(5, len(FEATURE_COLUMNS))
    selected = select_matrix(X, ("volume",))
    indices = indices_for(("volume",))
    assert selected.shape == (5, len(indices))
    assert np.array_equal(selected, X[:, indices])


def test_feature_column_order_is_pinned() -> None:
    assert FEATURE_COLUMNS == PINNED_COLUMNS
    assert len(FEATURE_COLUMNS) == 59


def test_log_return_alias_resolves_without_touching_the_matrix() -> None:
    # ``ret_3`` is a log return; its neighbours are simple returns. The alias
    # documents that, but the column keeps its name and position.
    assert FEATURE_ALIASES == {"log_ret_3": "ret_3"}
    assert canonical_name("log_ret_3") == "ret_3"
    assert canonical_name("ret_3") == "ret_3"
    assert "log_ret_3" not in FEATURE_COLUMNS
    assert FEATURE_COLUMNS.index("ret_3") == 4
    assert columns_for(("price",)) == FEATURE_GROUPS["price"]


def test_unknown_alias_raises_instead_of_becoming_a_column() -> None:
    with pytest.raises(ValueError, match="unknown feature"):
        canonical_name("log_ret_4")
