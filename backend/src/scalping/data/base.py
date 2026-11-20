"""Canonical bar schema and the provider protocol shared by every data source.

Every provider emits a frame that conforms to :data:`BAR_SCHEMA`; the store and
the resampler both normalise their input through :func:`conform_bars` so a
missing ``provider`` column or a naive timestamp can never poison a partition.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Protocol, runtime_checkable

import polars as pl
from polars.datatypes import DataType, DataTypeClass

__all__ = [
    "BAR_COLUMNS",
    "BAR_SCHEMA",
    "DataProvider",
    "conform_bars",
]

# Explicit union so mypy can match ``DataFrame.cast``'s ``SchemaDict`` input,
# which is invariant in the mapping key but accepts either form of dtype.
_BAR_DTYPES: dict[str, DataType | DataTypeClass] = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread": pl.Float64,
    "volume": pl.Int64,
    "provider": pl.String,
}

#: The single canonical M1 bar schema. ``ts`` is UTC; ``provider`` records the
#: source that produced the row (``synthetic``/``csv``/``dukascopy``).
BAR_SCHEMA: Final[pl.Schema] = pl.Schema(_BAR_DTYPES)

BAR_COLUMNS: Final[tuple[str, ...]] = tuple(BAR_SCHEMA)

#: Used when a caller hands us bars without a source label (test fixtures).
UNKNOWN_PROVIDER: Final[str] = "unknown"


def conform_bars(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``df`` restricted to :data:`BAR_SCHEMA`, with UTC timestamps.

    A missing ``provider`` column is filled with ``"unknown"`` and naive
    timestamps are assumed to already be UTC. Unknown extra columns are dropped
    so every parquet partition shares one schema.
    """
    missing = [name for name in BAR_COLUMNS if name != "provider" and name not in df.columns]
    if missing:
        raise ValueError(f"missing bar columns: {missing}")
    frame = df
    if "provider" not in frame.columns:
        frame = frame.with_columns(pl.lit(UNKNOWN_PROVIDER, dtype=pl.String).alias("provider"))
    ts_dtype = frame.schema["ts"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
        frame = frame.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    return frame.select(BAR_COLUMNS).cast(BAR_SCHEMA)


@runtime_checkable
class DataProvider(Protocol):
    """A source of M1 bars for one symbol.

    Providers are cheap to construct and stateless between calls; ``fetch``
    returns a frame conforming to :data:`BAR_SCHEMA` covering ``[start, end]``.

    ``start`` and ``end`` are ``datetime | None``. ``None`` means unbounded on
    that side, which a file-backed provider (CSV, parquet) honours by serving
    whatever its source covers. A provider that can only generate a finite
    window has nothing to fall back on, so it raises ``ValueError`` for a
    ``None`` bound; :class:`~scalping.data.synthetic.SyntheticProvider` is one
    such provider and needs both bounds.
    """

    name: str

    def fetch(
        self, symbol: str, start: datetime | None = None, end: datetime | None = None
    ) -> pl.DataFrame: ...
