"""Calendar-month walk-forward splits.

A :class:`Split` carries three half-open windows. Folds are chronological,
they never overlap and each train window starts one calendar month after the
previous one. The last fold is the last one whose test window fits inside the
frame: ``test.end <= last timestamp``. The windows are date ranges; slicing the
frame is left to the caller, who can select ``ts >= start`` and ``ts < end``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

__all__ = ["Split", "Window", "walkforward_split"]


@dataclass(frozen=True, slots=True)
class Window:
    """Half-open time window ``[start, end)``."""

    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class Split:
    """One walk-forward fold: fit on ``train``, tune on ``val``, score on ``test``."""

    train: Window
    val: Window
    test: Window


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _add_months(value: datetime, months: int) -> datetime:
    """Shift by whole calendar months, keeping the time of day."""
    total = value.year * 12 + (value.month - 1) + months
    year, month = divmod(total, 12)
    return value.replace(year=year, month=month + 1)


def _check_positive(name: str, value: int) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return int(value)


def walkforward_split(
    df: pl.DataFrame,
    train_months: int = 6,
    val_months: int = 1,
    test_months: int = 1,
) -> list[Split]:
    """Return every fold that fits in ``df``'s timestamps.

    ``df`` needs a ``ts`` column. An empty frame yields no folds. A frame too
    short for one train+val+test window yields no folds rather than a partial
    split.
    """
    train_months = _check_positive("train_months", train_months)
    val_months = _check_positive("val_months", val_months)
    test_months = _check_positive("test_months", test_months)

    if df.is_empty():
        return []
    first = df["ts"].min()
    last = df["ts"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return []  # pragma: no cover - the column is validated by conform_bars
    first, last = _as_utc(first), _as_utc(last)

    splits: list[Split] = []
    cursor = first
    while True:
        train_end = _add_months(cursor, train_months)
        val_end = _add_months(train_end, val_months)
        test_end = _add_months(val_end, test_months)
        if test_end > last:
            return splits
        splits.append(
            Split(
                train=Window(cursor, train_end),
                val=Window(train_end, val_end),
                test=Window(val_end, test_end),
            )
        )
        cursor = _add_months(cursor, 1)
