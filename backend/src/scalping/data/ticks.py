"""Partitioned tick store: query raw ticks without pre-aggregating to bars.

The bar store answers "what did the candle look like"; this store keeps every
tick so a later query can reconstruct spreads, signed volume or event windows
without a lossy round-trip through M1.

Layout::

    <root>/<symbol>/<year:04d>/<month:02d>.parquet

Writes are atomic and merge by the full tick identity, so re-importing the same
file adds nothing. Reads are range-filtered and sorted; ``limit`` returns the
oldest rows in the range (use it for previews, not for sampling the middle).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import polars as pl
from polars.datatypes import DataType, DataTypeClass

__all__ = [
    "TICK_COLUMNS",
    "TICK_SCHEMA",
    "TickCoverage",
    "TickStore",
    "conform_ticks",
]

TICK_COLUMNS: Final[tuple[str, ...]] = (
    "ts",
    "bid",
    "ask",
    "bid_volume",
    "ask_volume",
    "provider",
)

_TICK_DTYPES: dict[str, DataType | DataTypeClass] = {
    "ts": pl.Datetime("us", "UTC"),
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_volume": pl.Float64,
    "ask_volume": pl.Float64,
    "provider": pl.String,
}

TICK_SCHEMA: Final[pl.Schema] = pl.Schema(_TICK_DTYPES)

_REQUIRED: Final[tuple[str, ...]] = ("ts", "bid", "ask", "bid_volume", "ask_volume")
_IDENTITY: Final[tuple[str, ...]] = ("ts", "bid", "ask", "bid_volume", "ask_volume", "provider")


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=TICK_SCHEMA)


def conform_ticks(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise a tick frame to :data:`TICK_SCHEMA`.

    Missing ``provider`` becomes ``"unknown"``; a naive timestamp is treated as
    UTC; extra columns are dropped. Structural validation stays in
    :func:`scalping.data.validate.validate_ticks` so a caller can store unknown
    data and still ask for a report.
    """
    missing = [name for name in _REQUIRED if name not in df.columns]
    if missing:
        raise ValueError(f"missing tick columns: {missing}")
    frame = df
    if "provider" not in frame.columns:
        frame = frame.with_columns(pl.lit("unknown", dtype=pl.String).alias("provider"))
    dtype = frame.schema["ts"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        frame = frame.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    return frame.select(TICK_COLUMNS).cast(TICK_SCHEMA).sort("ts")


@dataclass(frozen=True, slots=True)
class TickCoverage:
    """First/last tick timestamp and row count for one symbol."""

    first_ts: datetime | None
    last_ts: datetime | None
    rows: int


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.write_parquet(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class TickStore:
    """Month-partitioned raw tick store keyed by symbol."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _partition(self, symbol: str, year: int, month: int) -> Path:
        return self.root / symbol / f"{year:04d}" / f"{month:02d}.parquet"

    def _partitions(self, symbol: str) -> list[Path]:
        base = self.root / symbol
        if not base.is_dir():
            return []
        found: list[Path] = []
        for path in sorted(base.rglob("*.parquet")):
            try:
                int(path.parent.name)
                int(path.stem)
            except ValueError:
                continue
            found.append(path)
        return found

    def write(self, symbol: str, df: pl.DataFrame) -> int:
        """Merge ticks into the store; return newly added rows."""
        if not symbol:
            raise ValueError("symbol must be non-empty")
        frame = conform_ticks(df)
        if frame.is_empty():
            return 0
        stamped = frame.with_columns(
            pl.col("ts").dt.year().alias("_year"),
            pl.col("ts").dt.month().alias("_month"),
        )
        added = 0
        for key, group in stamped.group_by(["_year", "_month"], maintain_order=True):
            year, month = int(key[0]), int(key[1])
            path = self._partition(symbol, year, month)
            existing = (
                pl.read_parquet(path).pipe(conform_ticks) if path.exists() else _empty()
            )
            incoming = group.drop(["_year", "_month"])
            merged = (
                pl.concat([existing, incoming])
                .unique(subset=list(_IDENTITY), keep="last")
                .sort("ts")
            )
            if not existing.equals(merged):
                _write_atomic(merged, path)
            added += merged.height - existing.height
        return added

    def read(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Read ``[start, end]`` (inclusive, UTC), sorted; ``limit`` keeps oldest."""
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 or None")
        start = self._utc(start)
        end = self._utc(end)
        if start is not None and end is not None and start > end:
            return _empty()
        paths = self._partitions(symbol)
        if not paths:
            return _empty()
        frame = pl.concat(
            [pl.read_parquet(path).pipe(conform_ticks) for path in paths]
        ).sort("ts")
        if start is not None:
            frame = frame.filter(pl.col("ts") >= start)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= end)
        return frame.head(limit) if limit is not None else frame

    def coverage(self, symbol: str) -> TickCoverage:
        frame = self.read(symbol)
        if frame.is_empty():
            return TickCoverage(None, None, 0)
        return TickCoverage(
            first_ts=cast(datetime, frame["ts"].min()),
            last_ts=cast(datetime, frame["ts"].max()),
            rows=frame.height,
        )

    @staticmethod
    def _utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
