"""Dukascopy tick feed with exponential backoff and a per-hour parquet cache.

The public feed serves one LZMA-compressed ``.bi5`` file per UTC hour::

    https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5

Each decompressed record is 20 bytes big-endian
``(ms_offset:u32, ask:u32, bid:u32, ask_vol:f32, bid_vol:f32)``. Prices are
stored in thousandths of a price unit, so a raw ``2351234`` is ``2351.234``.
``ms_offset`` is milliseconds since the top of the cached hour.

The endpoint rate-limits aggressively (HTTP 429 from a single machine is
normal) and occasionally returns 503. Both are retried with a delay of
``base_delay * 2**attempt`` (``1, 2, 4, 8, 16`` seconds by default) and then
reported as :class:`DataUnavailable`. A 404 means there is no file for that
hour and yields an empty frame without retrying. Any other status fails
immediately, because retrying a 403 cannot help.

A successful hour is cached at ``{cache_dir}/{symbol}/{yyyymmdd}/{hh}.parquet``
before it is returned. HTTP 200 with a zero-byte body means the market was
closed for that hour; the empty frame is cached too, so a closed hour is not
requested again. A 404 is *not* cached because it can also mean a mistyped
symbol, and caching would hide the mistake until the cache is cleared.
"""

from __future__ import annotations

import logging
import lzma
import os
import struct
import tempfile
import time
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Final

import httpx
import polars as pl

from scalping.data.base import BAR_COLUMNS, BAR_SCHEMA, DataProvider, conform_bars

__all__ = [
    "PROVIDER_NAME",
    "DataUnavailable",
    "DukascopyProvider",
    "decode_ticks",
]

logger = logging.getLogger(__name__)

PROVIDER_NAME: Final[str] = "dukascopy"

#: Public tick endpoint; ``month`` in the path is zero-based (January is ``00``).
BASE_URL: Final[str] = "https://datafeed.dukascopy.com/datafeed"

_RECORD: Final[struct.Struct] = struct.Struct(">IIIff")
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 503})
_REQUEST_TIMEOUT: Final[float] = 30.0
_USER_AGENT: Final[str] = "ai-scalping-studio/0.1"

#: LZMA-alone streams carry property byte 0x5d; a raw tick payload always starts
#: with the high byte of ``ms_offset``, which is 0x00 for any sub-hour offset.
_LZMA_PROPERTIES: Final[int] = 0x5D


def _sleep(seconds: float) -> None:
    """Backoff indirection so tests can record the schedule without waiting."""
    time.sleep(seconds)


def _as_utc(value: datetime) -> datetime:
    """Naive datetimes are assumed to already be UTC, matching ``conform_bars``."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` without ever truncating the live file.

    Same temp-file-then-replace pattern as :mod:`scalping.data.store`: an
    interrupted write leaves the previous hour readable instead of truncated.
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


def _decompress(payload: bytes) -> bytes:
    """Decompress a ``.bi5`` body; already-raw payloads pass through untouched."""
    if payload[:1] != bytes((_LZMA_PROPERTIES,)):
        return payload
    try:
        return lzma.decompress(payload)
    except lzma.LZMAError as exc:
        raise ValueError("dukascopy: tick payload is not valid LZMA data") from exc


def decode_ticks(payload: bytes) -> list[dict]:
    """Decode one hour's 20-byte big-endian tick records into dicts.

    Prices are scaled by ``1/1000``; volumes stay as the feed's ``f32`` value.
    An empty payload decodes to ``[]``. A trailing partial record is rejected
    rather than silently dropped, because it means the file is corrupt.
    """
    if not payload:
        return []
    if len(payload) % _RECORD.size:
        raise ValueError(
            f"dukascopy: tick payload is {len(payload)} bytes, "
            f"not a multiple of the {_RECORD.size}-byte record size"
        )
    return [
        {
            "ms": ms,
            "ask": ask / 1000.0,
            "bid": bid / 1000.0,
            "ask_vol": ask_vol,
            "bid_vol": bid_vol,
        }
        for ms, ask, bid, ask_vol, bid_vol in _RECORD.iter_unpack(payload)
    ]


class DataUnavailable(RuntimeError):
    """Dukascopy could not serve ``symbol`` at ``hour``.

    ``status`` is the last HTTP status seen, or ``None`` when every attempt
    failed at the transport level (connect/read errors).
    """

    def __init__(self, symbol: str, hour: datetime, status: int | None) -> None:
        self.symbol = symbol
        self.hour = hour
        self.status = status
        detail = "no response" if status is None else f"last HTTP status {status}"
        super().__init__(f"dukascopy: {symbol} {hour:%Y-%m-%d %H}:00 UTC unavailable ({detail})")


class DukascopyProvider(DataProvider):
    """Serve M1 bars from Dukascopy's public hourly tick files.

    ``cache_dir`` holds one parquet file per fetched hour. ``client`` lets a
    caller (or a test) supply an ``httpx.Client``; without one the provider
    creates its own and closes it in :meth:`close`. ``max_retries`` counts the
    attempts *after* the first request, so the default of 5 yields at most six
    requests per hour with delays of ``1, 2, 4, 8, 16`` seconds.

    ``fetch`` requires both bounds and covers ``[start, end]`` inclusive,
    walking hour by hour and dropping bars outside the requested window.
    """

    name: str = PROVIDER_NAME

    def __init__(
        self,
        cache_dir: Path,
        client: httpx.Client | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not isfinite(base_delay) or base_delay < 0:
            raise ValueError("base_delay must be a finite non-negative number")
        self.cache_dir = Path(cache_dir)
        self.max_retries = int(max_retries)
        self.base_delay = float(base_delay)
        self._client = client or httpx.Client(
            timeout=_REQUEST_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        )
        self._owns_client = client is None

    def close(self) -> None:
        """Close the internally created client; a caller-supplied one is left open."""
        if self._owns_client:
            self._client.close()

    # --- fetching ------------------------------------------------------
    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if start is None or end is None:
            raise ValueError("DukascopyProvider needs an explicit start and end")
        start, end = _as_utc(start), _as_utc(end)
        if end < start:
            return _empty()

        frames: list[pl.DataFrame] = []
        hour = start.replace(minute=0, second=0, microsecond=0)
        while hour <= end:
            frames.append(self._fetch_hour(symbol, hour))
            hour += timedelta(hours=1)
        frame = pl.concat(frames).filter((pl.col("ts") >= start) & (pl.col("ts") <= end))
        return frame.sort("ts").cast(BAR_SCHEMA)

    def _fetch_hour(self, symbol: str, hour_utc: datetime) -> pl.DataFrame:
        """Return one UTC hour of M1 bars, from cache when it is present.

        A freshly downloaded hour is written to the cache before returning,
        including an empty hour. A 404 returns an empty frame and leaves the
        cache untouched.
        """
        hour = _as_utc(hour_utc).replace(minute=0, second=0, microsecond=0)
        path = self._cache_path(symbol, hour)
        if path.exists():
            return conform_bars(pl.read_parquet(path))

        payload = self._download(symbol, hour)
        if payload is None:
            return _empty()
        frame = self._hour_bars(hour, decode_ticks(_decompress(payload)))
        _write_atomic(frame, path)
        return frame

    def _download(self, symbol: str, hour: datetime) -> bytes | None:
        """Fetch one hour, retrying 429/503; ``None`` means a 404 (no file)."""
        url = self._url(symbol, hour)
        status: int | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise DataUnavailable(symbol, hour, status) from exc
                self._backoff(symbol, hour, attempt, f"transport error: {exc}")
                continue

            status = response.status_code
            if status == 200:
                return response.content
            if status == 404:
                logger.info("dukascopy: no tick file for %s %s", symbol, hour)
                return None
            if status not in _RETRYABLE_STATUS:
                raise DataUnavailable(symbol, hour, status)
            if attempt >= self.max_retries:
                break
            self._backoff(symbol, hour, attempt, f"HTTP {status}")
        raise DataUnavailable(symbol, hour, status)

    def _backoff(self, symbol: str, hour: datetime, attempt: int, reason: str) -> None:
        delay = self.base_delay * 2**attempt
        logger.warning(
            "dukascopy: %s %s %s on attempt %d; retrying in %.2fs",
            symbol,
            hour,
            reason,
            attempt + 1,
            delay,
        )
        _sleep(delay)

    # --- paths ---------------------------------------------------------
    def _cache_path(self, symbol: str, hour: datetime) -> Path:
        if not symbol or symbol in {".", ".."} or "/" in symbol or os.sep in symbol:
            raise ValueError(f"symbol must be a plain name, got {symbol!r}")
        return self.cache_dir / symbol / f"{hour:%Y%m%d}" / f"{hour:%H}.parquet"

    @staticmethod
    def _url(symbol: str, hour: datetime) -> str:
        return (
            f"{BASE_URL}/{symbol}/{hour.year}/{hour.month - 1:02d}/"
            f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
        )

    # --- aggregation ---------------------------------------------------
    @staticmethod
    def _hour_bars(hour: datetime, ticks: list[dict]) -> pl.DataFrame:
        """Aggregate one hour of ticks into M1 bars.

        OHLC comes from the bid side, ``spread`` is the last ask minus the last
        bid, and ``volume`` is the number of ticks in the minute. Ticks are
        ordered by ``ms_offset`` in the file; the frame is sorted before
        grouping so the unnamed last tick of each minute wins.
        """
        if not ticks:
            return _empty()
        tick_frame = pl.DataFrame(
            {
                "ts": [hour + timedelta(milliseconds=tick["ms"]) for tick in ticks],
                "ask": [tick["ask"] for tick in ticks],
                "bid": [tick["bid"] for tick in ticks],
            },
            schema={"ts": pl.Datetime("us", "UTC"), "ask": pl.Float64, "bid": pl.Float64},
        )
        return (
            tick_frame.sort("ts")
            .group_by(pl.col("ts").dt.truncate("1m").alias("ts"), maintain_order=True)
            .agg(
                pl.col("bid").first().alias("open"),
                pl.col("bid").max().alias("high"),
                pl.col("bid").min().alias("low"),
                pl.col("bid").last().alias("close"),
                pl.col("bid").last().alias("bid_close"),
                pl.col("ask").last().alias("ask_close"),
                pl.len().cast(pl.Int64).alias("volume"),
            )
            .with_columns(
                (pl.col("ask_close") - pl.col("bid_close")).alias("spread"),
                pl.lit(PROVIDER_NAME, dtype=pl.String).alias("provider"),
            )
            .select(BAR_COLUMNS)
            .cast(BAR_SCHEMA)
        )
