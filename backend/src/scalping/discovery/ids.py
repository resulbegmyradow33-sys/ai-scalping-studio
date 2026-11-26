"""Canonical dataset identifiers for cross-producer comparison.

Two producers write the same dataset build under different spellings. The
search records a trial as ``validation:<manifest dataset id>`` (the evaluator
requires the split marker), while the evidence bundle records the manifest id
itself, ``<manifest dataset id>``. Comparing the raw strings therefore refuses
a promotion that has a perfectly good accepted trial: the trial exists, only
the spelling differs.

This module gives that comparison one spelling. :func:`canonical_dataset_id`
removes **exactly one** leading ``validation:`` marker and returns the id
unchanged otherwise. The stored id is never rewritten anywhere; callers keep
the original string for display and audit, and use the canonical form only for
equality tests.

Why only ``validation:``
------------------------
* ``validation:`` is the only split marker a producer prepends today. A holdout
  or test dataset is not renamed into the validation namespace; ids that carry
  ``test:``/``holdout:``-style tokens stay distinct and are still refused by
  the evidence and evaluator guards.
* The strip is single-pass, so ``validation:validation:X`` canonicalises to
  ``validation:X`` and cannot collide with ``X``.
* Symbol, timeframe, horizon and digest remain part of the compared id; no
  token splitting, case folding or interior normalisation happens here. Only
  surrounding whitespace is stripped, so ``"  X  "`` and ``X`` compare equal.

The function is intentionally strict: a missing, blank or prefix-only id is a
programmer error, not an empty dataset.
"""

from __future__ import annotations

from typing import Final

__all__ = ["SPLIT_PREFIX", "canonical_dataset_id"]

#: The only split marker stripped for comparison. Producers prepend it when a
#: trial must run on the validation split; evidence stores the manifest id.
SPLIT_PREFIX: Final[str] = "validation:"


def canonical_dataset_id(dataset_id: str) -> str:
    """Return ``dataset_id`` with one leading ``validation:`` marker removed.

    ``"validation:XAUUSD:H1:h1:abc"`` and ``"XAUUSD:H1:h1:abc"`` both return
    ``"XAUUSD:H1:h1:abc"``. Any other prefix, including ``test:``, ``holdout:``
    and a second ``validation:``, is preserved, so distinct datasets cannot
    become equal through this helper.

    Raises:
        TypeError: ``dataset_id`` is not a string.
        ValueError: ``dataset_id`` is blank, or is only the split marker.
    """
    if not isinstance(dataset_id, str):
        raise TypeError(f"dataset_id must be a str, got {type(dataset_id).__name__}")
    value = dataset_id.strip()
    if not value:
        raise ValueError("dataset_id must be a non-empty string")
    if value.startswith(SPLIT_PREFIX):
        value = value[len(SPLIT_PREFIX) :].strip()
        if not value:
            raise ValueError(f"dataset_id {dataset_id!r} carries no id after the split prefix")
    return value
