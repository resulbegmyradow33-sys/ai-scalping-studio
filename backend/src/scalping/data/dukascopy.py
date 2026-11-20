"""Dukascopy tick feed with exponential backoff and a per-hour parquet cache.

The public feed serves one LZMA-compressed ``.bi5`` file per UTC hour::

    https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5

Each decompressed record is 20 bytes big-endian
``(ms_offset:u32, ask:u32, bid:u32, ask_vol:f32, bid_vol:f32)``. Integer prices
are quoted in points; :data:`POINT_DIVISORS` maps a symbol to its point divisor
(``1000`` for the metals), and a symbol without an entry is rejected at use
instead of silently returning prices that are off by a factor of ten or a
hundred. ``ms_offset`` is milliseconds since the top of the cached hour.

The endpoint rate-limits aggressively (HTTP 429 from a single machine is
normal) and occasionally returns 503. Both are retried with a delay of
``base_delay * 2**attempt`` (``1, 2, 4, 8, 16`` seconds by default) and then
reported as :class:`DataUnavailable`. A 404 means there is no file for that
hour and yields an empty frame without retrying. Any other status fails
immediately, because retrying a 403 cannot help.

A fully elapsed hour is cached at
``{cache_dir}/{symbol}/{yyyymmdd}/{hh}.parquet`` before it is returned. HTTP
200 with a zero-byte body means the market was closed; the empty frame is
cached too, so a closed hour is not requested again. A 404 for an elapsed hour
is cached the same way, because Dukascopy answers most weekend/holiday hours
with 404 and re-requesting them would hammer a rate-limited host. The current
and any future hour are never written to the cache: a partial hour would
freeze forever, and a 404 there may still turn into data.
"""

from __future__ import annotations

import logging
import lzma
import os
import struct
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Final, Self

import httpx
import polars as pl

from scalping.data.base import BAR_COLUMNS, BAR_SCHEMA, DataProvider, conform_bars

__all__ = [
    "POINT_DIVISORS",
    "PROVIDER_NAME",
    "DataUnavailable",
    "DukascopyProvider",
    "decode_ticks",
]

logger = logging.getLogger(__name__)

PROVIDER_NAME: Final[str] = "dukascopy"

#: Integer price fields are measured in 1/`divisor` of a price unit. Only
#: symbols listed here can be fetched; adding a 5-digit FX pair means adding
#: ``100000`` (or its actual divisor) rather than relying on a default.
POINT_DIVISORS: Final[dict[str, int]] = {"XAUUSD": 1000, "XAGUSD": 1000}

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


def _utc_now() -> datetime:
    """Default clock; injectable so cacheability never depends on the wall clock."""
    return datetime.now(UTC)


def _point_divisor(symbol: str) -> int:
    """Return the validated point divisor for ``symbol``.

    Refusing an unknown symbol is deliberate: dividing a 5-digit FX quote by
    the metals' ``1000`` would return a price that is two orders of magnitude
    wrong with no error, which is worse than failing the fetch.
    """
    try:
        return POINT_DIVISORS[symbol]
    except KeyError:
        raise ValueError(
            f"unsupported symbol {symbol!r}: add its point divisor to POINT_DIVISORS"
        ) from None


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

    Prices are scaled by ``1/1000`` to match the metals feed; the provider
    itself scales with the symbol's validated :data:`POINT_DIVISORS` entry.
    An empty payload decodes to ``[]``. A trailing partial record is rejected
    rather than silently dropped, because it means the file is corrupt.
    """
    return _decode_ticks(payload, 1000)


def _decode_ticks(payload: bytes, divisor: int) -> list[dict]:
    """Decode records, scaling integer prices by ``1/divisor``."""
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
            "ask": ask / divisor,
            "bid": bid / divisor,
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

    ``cache_dir`` holds one parquet file per fully elapsed hour. ``client``
    lets a caller (or a test) supply an ``httpx.Client``; without one the
    provider creates its own. Either way the client belongs to the provider
    while it is open: :meth:`close` (and ``with``) closes it.
    ``max_retries`` counts the attempts *after* the first request, so the
    default of 5 yields at most six requests per hour with delays of
    ``1, 2, 4, 8, 16`` seconds. ``now`` is the clock used to decide whether an
    hour has elapsed; tests inject a fixed one rather than depend on wall time.

    ``symbol`` is an optional up-front check: when given, an unsupported
    instrument fails at construction. Every call that names a symbol validates
    it as well, against :data:`POINT_DIVISORS`.

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
        now: Callable[[], datetime] = _utc_now,
        symbol: str | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not isfinite(base_delay) or base_delay < 0:
            raise ValueError("base_delay must be a finite non-negative number")
        if symbol is not None:
            _point_divisor(symbol)
        self.cache_dir = Path(cache_dir)
        self.max_retries = int(max_retries)
        self.base_delay = float(base_delay)
        self.symbol = symbol
        self._now = now
        self._client = client or httpx.Client(
            timeout=_REQUEST_TIMEOUT, headers={"User-Agent": _USER_AGENT}
        )

    def close(self) -> None:
        """Close the underlying HTTP client, whether injected or owned."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- fetching ------------------------------------------------------
    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        _point_divisor(symbol)
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

        The cache is written only when the hour has fully elapsed
        (``hour_end <= now``). That covers a downloaded hour, an empty 200, and
        a 404, so closed weekend/holiday hours are remembered. A partial or
        future hour is returned but never stored: caching it would freeze
        incomplete data forever.
        """
        divisor = _point_divisor(symbol)
        hour = _as_utc(hour_utc).replace(minute=0, second=0, microsecond=0)
        path = self._cache_path(symbol, hour)
        if path.exists():
            return conform_bars(pl.read_parquet(path))

        payload = self._download(symbol, hour)
        if payload is None:
            frame = _empty()
        else:
            frame = self._hour_bars(hour, _decode_ticks(_decompress(payload), divisor))
        if self._is_cacheable(hour):
            _write_atomic(frame, path)
        return frame

    def _is_cacheable(self, hour: datetime) -> bool:
        """An hour is cacheable once its last instant is at or before ``now``."""
        return hour + timedelta(hours=1) <= _as_utc(self._now())

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
