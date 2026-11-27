"""Partitioned Parquet storage for bar frames.

Layout: ``{root}/{symbol}/{timeframe}/{year:04d}/{month:02d}.parquet``.

Writes merge the incoming frame into the affected month partitions: rows are
deduplicated on ``ts`` keeping the *last* occurrence (so a re-fetch overrides
older values) and each partition is sorted ascending. :meth:`ParquetStore.write`
reports how many rows were *added*; a rewrite that only repeats known timestamps
reports ``0`` even though the partition file may be refreshed.
"""

from __future__ import annotations

import os
import tempfile
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


class _MetadataUnavailable(Exception):
    """A partition has no usable statistics; the caller should scan it."""


def _metadata_partition_coverage(path: Path) -> tuple[int, datetime | None, datetime | None]:
    """Rows and ``ts`` bounds from Parquet row-group statistics only.

    No column data is decoded. ``ts`` statistics are typed by pyarrow from the
    logical timestamp type, so the bounds come back as datetimes. A missing
    statistic or a non-timestamp value raises :class:`_MetadataUnavailable` and
    the caller falls back to a projection scan.
    """
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - pyarrow is a dependency
        raise _MetadataUnavailable("pyarrow is not available") from error
    metadata = pq.ParquetFile(path).metadata
    rows = int(metadata.num_rows)
    if rows == 0:
        return 0, None, None
    column = next(
        (index for index in range(metadata.num_columns) if metadata.schema.column(index).name == "ts"),
        None,
    )
    if column is None:
        raise _MetadataUnavailable("partition has no ts column")
    first: datetime | None = None
    last: datetime | None = None
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(column).statistics
        if statistics is None or not statistics.has_min_max:
            raise _MetadataUnavailable("partition has no ts statistics")
        low, high = statistics.min, statistics.max
        if not isinstance(low, datetime) or not isinstance(high, datetime):
            raise _MetadataUnavailable("ts statistics are not datetimes")
        low, high = _as_utc(low), _as_utc(high)
        if low is None or high is None:  # pragma: no cover - datetime input
            raise _MetadataUnavailable("ts statistics are empty")
        first = low if first is None or low < first else first
        last = high if last is None or high > last else last
    return rows, first, last


def _scan_partition_coverage(path: Path) -> tuple[int, datetime | None, datetime | None]:
    """Projection-scan fallback: read only ``ts``, never the full bar row."""
    frame = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("ts").max().alias("last_ts"),
        )
        .collect()
    )
    rows = int(frame["rows"][0])
    first = frame["first_ts"][0]
    last = frame["last_ts"][0]
    if first is not None and not isinstance(first, datetime):
        raise TypeError("ts column must hold datetimes")
    if last is not None and not isinstance(last, datetime):
        raise TypeError("ts column must hold datetimes")
    return rows, _as_utc(first), _as_utc(last)


def _partition_coverage(path: Path) -> tuple[int, datetime | None, datetime | None]:
    try:
        return _metadata_partition_coverage(path)
    except _MetadataUnavailable:
        return _scan_partition_coverage(path)


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat naive bounds as UTC so they compare with the stored ``ts`` column."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` without ever truncating the live file.

    The frame goes to a sibling temp file first and is moved onto ``path`` with
    an atomic replace, so an interrupt or full disk mid-write leaves the previous
    partition readable (the temp file is removed either way).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.write_parquet(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


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
                _write_atomic(merged, path)
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
        frame = pl.concat([self._read_partition(path) for path in paths])
        if start is not None:
            frame = frame.filter(pl.col("ts") >= start)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= end)
        return frame.sort("ts")

    def coverage(self, symbol: str, timeframe: str = "M1") -> Coverage:
        """First/last timestamp and row count across every partition.

        Reads Parquet row-group statistics; a partition without them is scanned
        for ``ts`` only. The full bar row is never materialised.
        """
        rows = 0
        first: datetime | None = None
        last: datetime | None = None
        for path in self._partitions(symbol, timeframe, None, None):
            part_rows, part_first, part_last = _partition_coverage(path)
            rows += part_rows
            if part_first is not None and (first is None or part_first < first):
                first = part_first
            if part_last is not None and (last is None or part_last > last):
                last = part_last
        return Coverage(first_ts=first, last_ts=last, rows=rows)

    def providers(self, symbol: str, timeframe: str = "M1") -> list[str]:
        """Sorted provider labels across the store, reading only ``provider``."""
        paths = self._partitions(symbol, timeframe, None, None)
        if not paths:
            return []
        frame = (
            pl.scan_parquet(paths)
            .select(pl.col("provider"))
            .unique()
            .collect()
        )
        return sorted(frame.get_column("provider").drop_nulls().to_list())
