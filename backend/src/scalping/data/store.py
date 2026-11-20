"""Partitioned Parquet storage for bar frames.

Layout: ``{root}/{symbol}/{timeframe}/{year:04d}/{month:02d}.parquet``.

Writes merge the incoming frame into the affected month partitions: rows are
deduplicated on ``ts`` keeping the *last* occurrence (so a re-fetch overrides
older values) and each partition is sorted ascending. :meth:`ParquetStore.write`
reports how many rows were *added*; a rewrite that only repeats known timestamps
reports ``0`` even though the partition file may be refreshed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from scalping.data.base import BAR_SCHEMA, conform_bars

__all__ = ["Coverage", "ParquetStore"]


@dataclass(frozen=True, slots=True)
class Coverage:
    """First/last stored timestamp and row count; timestamps are ``None`` if empty."""

    first_ts: datetime | None
    last_ts: datetime | None
    rows: int


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat naive bounds as UTC so they compare with the stored ``ts`` column."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


class ParquetStore:
    """Month-partitioned store keyed by ``(symbol, timeframe)``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # --- paths ---------------------------------------------------------
    def _partition_path(self, symbol: str, timeframe: str, year: int, month: int) -> Path:
        return self.root / symbol / timeframe / f"{year:04d}" / f"{month:02d}.parquet"

    def _partitions(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Path]:
        base = self.root / symbol / timeframe
        if not base.is_dir():
            return []
        found: list[Path] = []
        for path in sorted(base.rglob("*.parquet")):
            try:
                year, month = int(path.parent.name), int(path.stem)
            except ValueError:  # ignore files outside the year/month layout
                continue
            if start is not None and (year, month) < (start.year, start.month):
                continue
            if end is not None and (year, month) > (end.year, end.month):
                continue
            found.append(path)
        return found

    @staticmethod
    def _read_partition(path: Path) -> pl.DataFrame:
        if not path.exists():
            return _empty()
        return conform_bars(pl.read_parquet(path))

    # --- public API ----------------------------------------------------
    def write(self, symbol: str, timeframe: str, df: pl.DataFrame) -> int:
        """Merge ``df`` into the store and return the number of rows added.

        Rows whose ``ts`` already exists in the target partition are replaced
        rather than appended; only genuinely new timestamps count toward the
        return value, so replaying an identical batch returns ``0``.
        """
        frame = conform_bars(df)
        if frame.is_empty():
            return 0
        stamped = frame.with_columns(
            pl.col("ts").dt.year().alias("_year"),
            pl.col("ts").dt.month().alias("_month"),
        )
        added = 0
        for key, group in stamped.group_by(["_year", "_month"], maintain_order=True):
            year, month = int(key[0]), int(key[1])
            path = self._partition_path(symbol, timeframe, year, month)
            existing = self._read_partition(path)
            incoming = group.drop(["_year", "_month"])
            merged = (
                pl.concat([existing, incoming])
                .unique(subset=["ts"], keep="last")
                .sort("ts")
            )
            if not existing.equals(merged):
                path.parent.mkdir(parents=True, exist_ok=True)
                merged.write_parquet(path)
            added += merged.height - existing.height
        return added

    def read(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Read ``[start, end]`` (inclusive, UTC) sorted ascending.

        Returns a schema-correct empty frame when no partition intersects the
        range, so callers can slice the result without a height check.
        """
        start, end = _as_utc(start), _as_utc(end)
        if start is not None and end is not None and start > end:
            return _empty()
        paths = self._partitions(symbol, timeframe, start, end)
        if not paths:
            return _empty()
        frame = pl.concat([pl.read_parquet(path) for path in paths])
        if start is not None:
            frame = frame.filter(pl.col("ts") >= start)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= end)
        return conform_bars(frame).sort("ts")

    def coverage(self, symbol: str, timeframe: str = "M1") -> Coverage:
        """First/last timestamp and row count across every partition."""
        frame = self.read(symbol, timeframe)
        if frame.is_empty():
            return Coverage(first_ts=None, last_ts=None, rows=0)
        first = frame.get_column("ts").min()
        last = frame.get_column("ts").max()
        if not isinstance(first, datetime) or not isinstance(last, datetime):
            raise TypeError("ts column must hold datetimes")
        return Coverage(first_ts=first, last_ts=last, rows=frame.height)
