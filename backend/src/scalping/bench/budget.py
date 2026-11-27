"""Reusable process-RSS budget.

The orchestrator has its own measured guard; this is the small building block
for scripts and trainers that need the same check without importing the
orchestrator. It reads the process high-water RSS (macOS bytes, Linux KiB),
subtracts an optional reserve and raises a typed error when the limit is
exceeded.

It is a stop signal, not an OS memory manager: it cannot reclaim memory and it
does not enforce one job per process.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from math import isfinite
from typing import Self

__all__ = ["MemoryBudgetError", "RssBudget"]


class MemoryBudgetError(RuntimeError):
    """The process exceeded its configured RSS budget."""


def process_rss_mb() -> float:
    """High-water RSS in MiB."""
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024.0 * 1024.0) if sys.platform == "darwin" else usage / 1024.0


@dataclass(frozen=True, slots=True)
class RssBudget:
    """A max RSS in MiB, with an optional reserve for the rest of the process."""

    max_mb: int
    reserve_mb: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_mb, bool) or not isinstance(self.max_mb, int):
            raise TypeError("max_mb must be an int")
        if self.max_mb < 1:
            raise ValueError("max_mb must be >= 1")
        if isinstance(self.reserve_mb, bool) or not isinstance(self.reserve_mb, int):
            raise TypeError("reserve_mb must be an int")
        if self.reserve_mb < 0 or self.reserve_mb >= self.max_mb:
            raise ValueError("reserve_mb must be in [0, max_mb)")

    @property
    def limit_mb(self) -> int:
        return self.max_mb - self.reserve_mb

    @property
    def used_mb(self) -> float:
        return process_rss_mb()

    @property
    def exceeded(self) -> bool:
        used = self.used_mb
        return isfinite(used) and used > self.limit_mb

    def check(self) -> None:
        if self.exceeded:
            raise MemoryBudgetError(
                f"process RSS {self.used_mb:.1f} MiB exceeds budget {self.limit_mb} MiB"
            )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_mb": self.max_mb,
            "reserve_mb": self.reserve_mb,
            "limit_mb": self.limit_mb,
            "used_mb": self.used_mb,
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None
