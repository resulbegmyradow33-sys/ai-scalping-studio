"""Data layer: canonical bar schema, providers, partitioned store, resampling."""

from __future__ import annotations

from scalping.data.base import BAR_COLUMNS, BAR_SCHEMA, DataProvider, conform_bars
from scalping.data.resample import RESAMPLE_COLUMNS, RESAMPLE_SCHEMA, resample
from scalping.data.store import Coverage, ParquetStore
from scalping.data.synthetic import SyntheticProvider

__all__ = [
    "BAR_COLUMNS",
    "BAR_SCHEMA",
    "RESAMPLE_COLUMNS",
    "RESAMPLE_SCHEMA",
    "Coverage",
    "DataProvider",
    "ParquetStore",
    "SyntheticProvider",
    "conform_bars",
    "resample",
]
