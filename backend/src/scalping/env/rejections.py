"""Deterministic broker-side order rejections.

Risk gates answer "may this account take the trade". Rejections answer "would
the broker have accepted the order at the fill bar". Two data-supported cases
are modelled:

* ``max_gap_atr`` — with execution delay, a fill bar whose close moved more than
  this many decision-bar ATRs is rejected instead of filled at the new price;
* ``reject_zero_volume`` — a bar with no traded volume has no liquidity to fill.

Both are deterministic and off by default. Rejections are not partial fills and
do not invent queue position; the project has no L2 data for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

__all__ = ["RejectionConfig"]


@dataclass(frozen=True, slots=True)
class RejectionConfig:
    """Entry-rejection thresholds; zero/false disables each rule."""

    max_gap_atr: float = 0.0
    reject_zero_volume: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.max_gap_atr) or self.max_gap_atr < 0.0:
            raise ValueError("max_gap_atr must be finite and >= 0")
        if not isinstance(self.reject_zero_volume, bool):
            raise TypeError("reject_zero_volume must be a bool")
        if self.max_gap_atr == 0.0 and not self.reject_zero_volume:
            raise ValueError("a rejection config must enable at least one rule")

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "max_gap_atr": self.max_gap_atr,
            "reject_zero_volume": self.reject_zero_volume,
        }
