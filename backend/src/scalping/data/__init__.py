"""Data layer: canonical bar schema, providers, partitioned store, resampling."""

from __future__ import annotations

from scalping.data.base import BAR_COLUMNS, BAR_SCHEMA, DataProvider, conform_bars
from scalping.data.csv_provider import CsvProvider, DataError
from scalping.data.dukascopy import DataUnavailable, DukascopyProvider, decode_ticks
from scalping.data.resample import RESAMPLE_COLUMNS, RESAMPLE_SCHEMA, resample
from scalping.data.store import Coverage, ParquetStore
from scalping.data.synthetic import SyntheticProvider

__all__ = [
    "BAR_COLUMNS",
    "BAR_SCHEMA",
    "RESAMPLE_COLUMNS",
    "RESAMPLE_SCHEMA",
    "Coverage",
    "CsvProvider",
    "DataError",
    "DataProvider",
    "DataUnavailable",
    "DukascopyProvider",
    "ParquetStore",
    "SyntheticProvider",
    "conform_bars",
    "decode_ticks",
    "resample",
]
