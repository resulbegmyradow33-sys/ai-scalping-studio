"""Feature group registry."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.features.build import FEATURE_COLUMNS
from scalping.features.registry import (
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    columns_for,
    indices_for,
    select_matrix,
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
