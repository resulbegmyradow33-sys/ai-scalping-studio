"""Read broker CSV exports into canonical, UTC-stamped M1 bars.

The accepted layout is the headerless six-column export used by Turkish
brokers: ``YYYY.MM.DD HH:MM,open,high,low,close,volume``. The timestamp is
broker wall-clock time, so it is shifted back by the broker's UTC offset
before it becomes a canonical ``ts``. Files that also carry bid/ask quotes
(``...,bid_close,ask_close,volume``) keep the real spread; a file without
them is labelled ``csv_no_spread`` and gets a null ask/spread so the
environment can apply its spread model.

Errors quote the 1-based source line, because the rows that reach the frame
are reordered and filtered and a line number is the only reference a user can
find in their file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Final

import polars as pl

from scalping.data.base import BAR_SCHEMA, DataProvider, conform_bars

__all__ = [
    "DEFAULT_BROKER_OFFSET_HOURS",
    "CsvProvider",
    "DataError",
]

logger = logging.getLogger(__name__)

#: Provider label for a file that carries bid/ask quotes.
PROVIDER_NAME: Final[str] = "csv"
#: Provider label when the file has no quotes; the env must add the spread.
PROVIDER_NAME_NO_SPREAD: Final[str] = "csv_no_spread"

#: Offset assumed when auto-detection has no DST gap to work with.
DEFAULT_BROKER_OFFSET_HOURS: Final[int] = 3

_TS_FORMAT: Final[str] = "%Y.%m.%d %H:%M"

#: Positional labels per accepted width; ``column_map`` renames these.
_LABELS: Final[dict[int, tuple[str, ...]]] = {
    6: ("datetime", "open", "high", "low", "close", "volume"),
    8: ("datetime", "open", "high", "low", "close", "bid_close", "ask_close", "volume"),
}

_DEFAULT_CANONICAL: Final[dict[str, str]] = {"datetime": "ts"}

_REQUIRED: Final[frozenset[str]] = frozenset({"ts", "open", "high", "low", "close", "volume"})
_QUOTES: Final[frozenset[str]] = frozenset({"bid_close", "ask_close"})

#: A DST spring-forward skips one wall-clock hour, so consecutive bars jump
#: 61 minutes instead of 1. Anything in this band is read as that gap.
_DST_GAP: Final[tuple[timedelta, timedelta]] = (timedelta(minutes=45), timedelta(minutes=75))


class DataError(ValueError):
    """A row that cannot become a canonical bar, located by source line.

    ``line`` is 1-based in the source file and ``column`` names the canonical
    column that failed (``None`` for row-level problems such as a wrong field
    count).
    """

    def __init__(self, line: int, reason: str, column: str | None = None) -> None:
        self.line = int(line)
        self.reason = reason
        self.column = column
        location = f"line {self.line}"
        if column is not None:
            location += f", column {column!r}"
        super().__init__(f"{location}: {reason}")


@dataclass(slots=True)
class _Record:
    """One parsed row; ``ts`` is still broker-local and naive."""

    line: int
    ts: datetime
    volume: int
    prices: dict[str, float] = field(default_factory=dict)


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat naive bounds as UTC so they compare with the stored ``ts``."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


def _parse_ts(text: str, line: int) -> datetime:
    try:
        return datetime.strptime(text, _TS_FORMAT)
    except ValueError:
        raise DataError(line, f"{text!r} is not a {_TS_FORMAT} timestamp", "ts") from None


def _parse_float(text: str, line: int, column: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise DataError(line, f"{text!r} is not a number", column) from None
    if not isfinite(value):
        raise DataError(line, f"{text!r} is not a finite number", column)
    return value


def _parse_volume(text: str, line: int) -> int:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        raise DataError(line, f"{text!r} is not a volume", "volume") from None
    if not isfinite(value) or not value.is_integer():
        raise DataError(line, f"{text!r} is not a whole volume", "volume")
    return int(value)


class CsvProvider(DataProvider):
    """Read broker M1 bars from ``path`` and normalise them to UTC.

    ``tz_offset_hours`` is the broker's UTC offset (``+3`` for Turkish
    brokers). When it is ``None`` the provider looks for a one-hour jump in
    consecutive timestamps, the signature of a DST spring-forward, and falls
    back to :data:`DEFAULT_BROKER_OFFSET_HOURS` with a warning otherwise.

    ``column_map`` renames the positional labels (``datetime``/``open``/...)
    to canonical names, so a broker that orders or names its fields
    differently can still feed the provider.
    """

    name: str = PROVIDER_NAME

    def __init__(
        self,
        path: Path,
        tz_offset_hours: int | None = None,
        column_map: dict[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.tz_offset_hours = tz_offset_hours
        self.column_map = dict(column_map or {})

    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        records = self._parse_records()
        if not records:
            return _empty()

        offset = self._resolve_offset(records)
        frame = self._build_frame(records, offset)

        start, end = _as_utc(start), _as_utc(end)
        if start is not None and end is not None and start > end:
            return _empty()
        if start is not None:
            frame = frame.filter(pl.col("ts") >= start)
        if end is not None:
            frame = frame.filter(pl.col("ts") <= end)
        return conform_bars(frame)

    # --- parsing -------------------------------------------------------
    def _canonical_columns(self, width: int, line: int) -> tuple[str, ...]:
        labels = _LABELS.get(width)
        if labels is None:
            widths = "/".join(str(value) for value in sorted(_LABELS))
            raise DataError(line, f"expected {widths} comma-separated fields, found {width}")
        mapped = tuple(
            self.column_map.get(label, _DEFAULT_CANONICAL.get(label, label)) for label in labels
        )
        if len(set(mapped)) != len(mapped):
            raise DataError(line, "column_map maps two fields to the same column")
        missing = _REQUIRED - set(mapped)
        if missing:
            raise DataError(line, f"column_map drops required column(s) {sorted(missing)}")
        unknown = set(mapped) - _REQUIRED - _QUOTES
        if unknown:
            raise DataError(line, f"column_map names unknown column(s) {sorted(unknown)}")
        if len(_QUOTES & set(mapped)) == 1:
            raise DataError(line, "bid_close and ask_close must be mapped together")
        return mapped

    def _parse_records(self) -> list[_Record]:
        text = self.path.read_text(encoding="utf-8")
        records: list[_Record] = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            fields = [field.strip() for field in raw.split(",")]
            columns = self._canonical_columns(len(fields), line_number)
            record = _Record(line=line_number, ts=datetime.min, volume=0)
            for column, value in zip(columns, fields, strict=True):
                if column == "ts":
                    record.ts = _parse_ts(value, line_number)
                elif column == "volume":
                    record.volume = _parse_volume(value, line_number)
                else:
                    record.prices[column] = _parse_float(value, line_number, column)
            records.append(record)
        return records

    # --- timezone ------------------------------------------------------
    def _resolve_offset(self, records: list[_Record]) -> int:
        if self.tz_offset_hours is not None:
            return int(self.tz_offset_hours)
        stamps = [record.ts for record in records]
        low, high = _DST_GAP
        for earlier, later in pairwise(stamps):
            if low <= later - earlier <= high:
                logger.info(
                    "csv: DST spring-forward gap in %s; using %+d h broker offset",
                    self.path,
                    DEFAULT_BROKER_OFFSET_HOURS,
                )
                return DEFAULT_BROKER_OFFSET_HOURS
        logger.warning(
            "csv: no DST gap in %s; assuming broker offset %+d h",
            self.path,
            DEFAULT_BROKER_OFFSET_HOURS,
        )
        return DEFAULT_BROKER_OFFSET_HOURS

    def _to_utc(self, records: list[_Record], offset: int) -> list[datetime]:
        """Convert local stamps to UTC and reject duplicate minutes."""
        shift = timedelta(hours=offset)
        seen: dict[datetime, int] = {}
        stamps: list[datetime] = []
        for record in records:
            stamp = (record.ts - shift).replace(tzinfo=UTC)
            first = seen.setdefault(stamp, record.line)
            if first != record.line:
                raise DataError(
                    record.line,
                    f"duplicate timestamp {stamp:%Y-%m-%d %H:%M} UTC (first seen on line {first})",
                    "ts",
                )
            stamps.append(stamp)
        return stamps

    # --- frame ---------------------------------------------------------
    def _build_frame(self, records: list[_Record], offset: int) -> pl.DataFrame:
        stamps = self._to_utc(records, offset)
        quoted = "ask_close" in records[0].prices
        closes = [record.prices["close"] for record in records]

        bids: list[float | None] = []
        asks: list[float | None] = []
        spreads: list[float | None] = []
        if quoted:
            for record in records:
                bid = record.prices["bid_close"]
                ask = record.prices["ask_close"]
                bids.append(bid)
                asks.append(ask)
                spreads.append(ask - bid)
            provider = PROVIDER_NAME
        else:
            # ``close`` is the only quote; mirror it as bid and leave ask null
            # so the env knows to apply its spread model.
            bids = list(closes)
            asks = [None] * len(records)
            spreads = [None] * len(records)
            provider = PROVIDER_NAME_NO_SPREAD

        def column(name: str, values: list[float] | list[int] | list[float | None]) -> pl.Series:
            dtype = pl.Int64 if name == "volume" else pl.Float64
            return pl.Series(name, values, dtype=dtype)

        return pl.DataFrame(
            {
                "ts": pl.Series("ts", stamps, dtype=pl.Datetime("us", "UTC")),
                "open": column("open", [record.prices["open"] for record in records]),
                "high": column("high", [record.prices["high"] for record in records]),
                "low": column("low", [record.prices["low"] for record in records]),
                "close": column("close", closes),
                "bid_close": column("bid_close", bids),
                "ask_close": column("ask_close", asks),
                "spread": column("spread", spreads),
                "volume": column("volume", [record.volume for record in records]),
            }
        ).with_columns(pl.lit(provider, dtype=pl.String).alias("provider"))
