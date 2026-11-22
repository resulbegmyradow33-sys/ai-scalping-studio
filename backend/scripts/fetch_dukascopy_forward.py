"""Download Dukascopy ticks as M1 bars **and** bid/ask flow for forward tests.

The existing provider caches M1 bars only, throwing away the per-tick volumes
that the live ``flow_momentum`` card needs. This script reuses the provider's
HTTP client, retry policy and tick decoder, then writes one hourly parquet with
both the bar columns and the minute's aggressive buy/sell volume:

    flow = (ask_volume_sum - bid_volume_sum) / (ask_volume_sum + bid_volume_sum)

The hourly cache makes a multi-hour crawl resumable. ``--aggregate`` combines
it into a ParquetStore plus a flow side store, and resamples the higher
timeframes the panel expects.

Usage:
    python -m scripts.fetch_dukascopy_forward \
        --start 2026-06-01 --end 2026-11-21 \
        --root ../data-forward-store --flow-root ../data-forward-flow \
        --cache ../data-forward-cache --hour-sleep 1.0
    python -m scripts.fetch_dukascopy_forward --aggregate --root ... --flow-root ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from scalping.data.base import BAR_COLUMNS, BAR_SCHEMA
from scalping.data.dukascopy import (
    DataUnavailable,
    DukascopyProvider,
    _decode_ticks,
    _decompress,
    _point_divisor,
)
from scalping.data.provenance import build_provenance, save_provenance
from scalping.data.resample import resample
from scalping.data.store import ParquetStore

_CACHE_SCHEMA: dict[str, pl.DataType] = {
    **BAR_SCHEMA,
    "buy_vol": pl.Float64,
    "sell_vol": pl.Float64,
    "flow": pl.Float64,
}


class RateLimited(RuntimeError):
    """The feed kept refusing after every retry; pause the whole crawl."""


def _download(
    provider: DukascopyProvider,
    symbol: str,
    hour: datetime,
    max_wait: float,
    attempts: int,
) -> bytes | None:
    """One hour with `Retry-After`-aware backoff, returning None on a 404.

    The provider's own retry caps out at a few seconds, which is far too short
    once the feed has issued a 429 ban. This loop waits up to ``max_wait`` and
    raises :class:`RateLimited` when even that is not enough, so the caller can
    pause instead of hammering the host.
    """
    wait = 5.0
    url = provider._url(symbol, hour)
    for _ in range(attempts):
        response = provider._client.get(url)
        status = response.status_code
        if status == 200:
            return response.content
        if status == 404:
            return None
        if status in (429, 503):
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except ValueError:
                delay = wait
            delay = min(max(delay, 5.0), max_wait)
            print(
                json.dumps(
                    {
                        "event": "rate_limited",
                        "hour": hour.isoformat(),
                        "status": status,
                        "wait": delay,
                    }
                ),
                flush=True,
            )
            time.sleep(delay)
            wait = min(wait * 2, max_wait)
            continue
        raise DataUnavailable(symbol, hour, status)
    raise RateLimited(f"still refused after {attempts} attempts at {hour.isoformat()}")


def _empty_hour() -> pl.DataFrame:
    return pl.DataFrame(schema=_CACHE_SCHEMA)


def _flow_frame(hour: datetime, ticks: list[dict]) -> pl.DataFrame:
    if not ticks:
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime("us", "UTC"),
                "buy_vol": pl.Float64,
                "sell_vol": pl.Float64,
                "flow": pl.Float64,
            }
        )
    ticks_frame = pl.DataFrame(
        {
            "ts": [hour + timedelta(milliseconds=tick["ms"]) for tick in ticks],
            "buy_vol": [tick["ask_vol"] for tick in ticks],
            "sell_vol": [tick["bid_vol"] for tick in ticks],
        },
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "buy_vol": pl.Float64,
            "sell_vol": pl.Float64,
        },
    )
    return (
        ticks_frame.sort("ts")
        .group_by(pl.col("ts").dt.truncate("1m").alias("ts"), maintain_order=True)
        .agg(
            pl.col("buy_vol").sum().alias("buy_vol"),
            pl.col("sell_vol").sum().alias("sell_vol"),
        )
        .with_columns(
            (
                (pl.col("buy_vol") - pl.col("sell_vol"))
                / (pl.col("buy_vol") + pl.col("sell_vol")).clip(1e-12)
            ).alias("flow")
        )
    )


def _hour_frame(hour: datetime, ticks: list[dict]) -> pl.DataFrame:
    if not ticks:
        return _empty_hour()
    bars = DukascopyProvider._hour_bars(hour, ticks)
    flow = _flow_frame(hour, ticks).select("ts", "buy_vol", "sell_vol", "flow")
    return (
        bars.join(flow, on="ts", how="left")
        .with_columns(
            pl.col("buy_vol").fill_null(0.0),
            pl.col("sell_vol").fill_null(0.0),
            pl.col("flow").fill_null(0.0),
        )
        .select(list(_CACHE_SCHEMA))
        .cast(_CACHE_SCHEMA)
    )


def _cache_path(cache: Path, symbol: str, hour: datetime) -> Path:
    return cache / symbol / f"{hour:%Y%m%d}" / f"{hour:%H}.parquet"


def _aggregate(root: Path, flow_root: Path, cache: Path, symbol: str) -> dict[str, object]:
    files = sorted((cache / symbol).glob("*/*.parquet"))
    frames = [pl.read_parquet(path) for path in files]
    frames = [frame for frame in frames if frame.height]
    if not frames:
        raise ValueError(f"no cached hours under {cache / symbol}")
    every = pl.concat(frames).unique("ts", keep="last").sort("ts")
    bars = every.select(BAR_COLUMNS).cast(BAR_SCHEMA)
    store = ParquetStore(root)
    store.write(symbol, "M1", bars)

    flow = every.select("ts", "buy_vol", "sell_vol", "flow")
    flow_dir = flow_root / symbol / "M1"
    flow_dir.mkdir(parents=True, exist_ok=True)
    flow.write_parquet(flow_dir / "part-0000.parquet")

    m1 = store.read(symbol, "M1")
    save_provenance(
        store.root,
        build_provenance(
            m1,
            symbol=symbol,
            timeframe="M1",
            source="dukascopy:tick-crawl",
        ),
    )
    for rule, timeframe in (("5m", "M5"), ("15m", "M15"), ("30m", "M30"), ("1h", "H1"), ("4h", "H4"), ("1d", "D1")):
        derived = resample(m1, rule).rename({"spread_mean": "spread"})
        store.write(symbol, timeframe, derived)
        save_provenance(
            store.root,
            build_provenance(
                derived,
                symbol=symbol,
                timeframe=timeframe,
                source="derived:M1",
            ),
        )
    coverage = store.coverage(symbol)
    return {
        "event": "aggregated",
        "hours": len(files),
        "rows": coverage.rows,
        "first": coverage.first_ts.isoformat() if coverage.first_ts else None,
        "last": coverage.last_ts.isoformat() if coverage.last_ts else None,
    }


def _hours(start: datetime, end: datetime) -> list[datetime]:
    hour = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    while hour <= last:
        out.append(hour)
        hour += timedelta(hours=1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dukascopy forward crawl with flow")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--root", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--hour-sleep", type=float, default=1.0)
    parser.add_argument("--stall-wait", type=float, default=30.0)
    parser.add_argument("--passes", type=int, default=1, help="retry missed hours this many times")
    parser.add_argument("--pass-pause", type=float, default=120.0)
    parser.add_argument("--ban-wait", type=float, default=300.0, help="pause after a hard rate limit")
    parser.add_argument("--ban-attempts", type=int, default=4)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)

    root, flow_root, cache = Path(args.root), Path(args.flow_root), Path(args.cache)
    if args.aggregate:
        print(json.dumps(_aggregate(root, flow_root, cache, args.symbol)), flush=True)
        return 0
    if args.start is None or args.end is None:
        parser.error("--start and --end are required unless --aggregate is set")

    _point_divisor(args.symbol)
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) + timedelta(days=1)
    hours = _hours(start, end)
    provider = DukascopyProvider(
        cache_dir=cache,
        max_retries=5,
        base_delay=2.0,
        symbol=args.symbol,
    )
    done = empty = 0
    missed = 0
    try:
        for pass_index in range(max(1, args.passes)):
            pass_missed = 0
            paused = False
            for index, hour in enumerate(hours, start=1):
                path = _cache_path(cache, args.symbol, hour)
                if path.exists():
                    if pass_index == 0:
                        done += 1
                    continue
                try:
                    payload = _download(
                        provider, args.symbol, hour, args.ban_wait, args.ban_attempts
                    )
                except RateLimited as exc:
                    print(
                        json.dumps(
                            {
                                "event": "pause",
                                "pass": pass_index + 1,
                                "hour": hour.isoformat(),
                                "error": str(exc),
                                "wait": args.ban_wait,
                            }
                        ),
                        flush=True,
                    )
                    time.sleep(args.ban_wait)
                    paused = True
                    break
                except DataUnavailable as exc:
                    pass_missed += 1
                    print(
                        json.dumps(
                            {
                                "event": "miss",
                                "pass": pass_index + 1,
                                "hour": hour.isoformat(),
                                "error": str(exc)[:160],
                            }
                        ),
                        flush=True,
                    )
                    time.sleep(min(args.stall_wait, 60.0))
                    continue
                ticks = (
                    _decode_ticks(_decompress(payload), _point_divisor(args.symbol))
                    if payload is not None
                    else []
                )
                frame = _hour_frame(hour, ticks)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(path)
                if pass_index == 0:
                    done += 1
                    if frame.height == 0:
                        empty += 1
                if index % 48 == 0 or index == len(hours):
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "pass": pass_index + 1,
                                "i": index,
                                "n": len(hours),
                                "cached": done,
                                "empty": empty,
                                "missed": pass_missed,
                                "last": hour.isoformat(),
                            }
                        ),
                        flush=True,
                    )
                time.sleep(args.hour_sleep)
            missed = pass_missed
            if paused and pass_index + 1 < args.passes:
                print(
                    json.dumps(
                        {"event": "resume_after_pause", "pass": pass_index + 2}
                    ),
                    flush=True,
                )
                time.sleep(args.pass_pause)
                continue
            if missed == 0 or pass_index + 1 >= args.passes:
                break
            print(
                json.dumps(
                    {"event": "retry_pause", "missing": missed, "wait": args.pass_pause}
                ),
                flush=True,
            )
            time.sleep(args.pass_pause)
    finally:
        provider.close()

    present = sum(_cache_path(cache, args.symbol, hour).exists() for hour in hours)
    summary = _aggregate(root, flow_root, cache, args.symbol)
    summary.update(
        {
            "cached": done,
            "empty": empty,
            "missed": missed,
            "present": present,
            "expected": len(hours),
            "complete": present == len(hours),
        }
    )
    print(json.dumps(summary), flush=True)
    return 0 if missed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
