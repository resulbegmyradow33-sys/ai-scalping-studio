"""Canonical dataset ids: one comparison spelling without rewriting storage.

The search writes ``validation:<manifest id>`` trials while evidence stores the
manifest id. These tests pin the single allowed normalisation and the
collisions it must not create.
"""

from __future__ import annotations

import pytest

from scalping.discovery import SPLIT_PREFIX, canonical_dataset_id

PLAIN = "XAUUSD:H1:h1:b189a7fdbad6"
PREFIXED = f"validation:{PLAIN}"


def test_the_split_prefix_is_the_only_documented_marker() -> None:
    assert SPLIT_PREFIX == "validation:"


def test_validation_prefixed_and_manifest_ids_compare_equal() -> None:
    assert canonical_dataset_id(PREFIXED) == PLAIN
    assert canonical_dataset_id(PLAIN) == PLAIN
    assert canonical_dataset_id(PREFIXED) == canonical_dataset_id(PLAIN)


def test_the_marker_is_stripped_once_only() -> None:
    assert canonical_dataset_id(f"validation:{PREFIXED}") == PREFIXED
    assert canonical_dataset_id(f"validation:{PREFIXED}") != PLAIN


@pytest.mark.parametrize("prefix", ["test:", "holdout:", "holdout-test:", "locked:"])
def test_other_split_prefixes_are_preserved(prefix: str) -> None:
    candidate = f"{prefix}{PLAIN}"

    assert canonical_dataset_id(candidate) == candidate
    assert canonical_dataset_id(candidate) != canonical_dataset_id(PLAIN)


def test_symbol_timeframe_and_digest_stay_exact() -> None:
    variants = {
        "XAGUSD:H1:h1:b189a7fdbad6",
        "XAUUSD:M15:h1:b189a7fdbad6",
        "XAUUSD:H1:h4:b189a7fdbad6",
        "XAUUSD:H1:h1:000000000000",
    }
    canonical = {canonical_dataset_id(value) for value in variants}
    canonical.add(canonical_dataset_id(PLAIN))

    assert len(canonical) == len(variants) + 1


def test_surrounding_whitespace_is_not_a_different_dataset() -> None:
    assert canonical_dataset_id(f"  {PREFIXED}  ") == PLAIN


@pytest.mark.parametrize("value", ["", "   ", "validation:", "validation:   "])
def test_blank_ids_are_refused(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_dataset_id(value)


def test_non_string_ids_are_refused() -> None:
    with pytest.raises(TypeError):
        canonical_dataset_id(123)  # type: ignore[arg-type]
