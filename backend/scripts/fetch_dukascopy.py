"""Batch Dukascopy downloader: ticks -> M1 -> ParquetStore, polite edition.

The public datafeed answers in three ways on this network:

* a keep-alive connection serves an hour in well under a second,
* a fresh TLS handshake can stall for ~30 s,
* a burst of requests is answered with 429/503 for minutes.

So this script fetches **one hour at a time**, keeps a persistent client, paces
itself, and when the feed starts refusing it waits out the ban instead of
skipping the day. Progress is one JSON line per day, and the cache plus the
store make an interrupted run resumable.

Usage:
    python -m scripts.fetch_dukascopy --symbol XAUUSD \
        --start 2025-10-01 --end 2026-09-30 --root ../data-real
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from scalping.data.dukascopy import DataUnavailable, DukascopyProvider
from scalping.data.store import ParquetStore

_CLIENT_TIMEOUT = 90.0


def _days(start: datetime, end: datetime) -> list[datetime]:
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    while day <= last:
        if day.weekday() < 5:  # XAUUSD does not trade on weekends
            out.append(day)
        day += timedelta(days=1)
    return out


def _fetch_hour(
    provider: DukascopyProvider,
    symbol: str,
    hour: datetime,
    stall_wait: float,
    stall_tries: int,
) -> pl.DataFrame | None:
    """Fetch one hour, waiting out refusals. Returns None when it gives up."""
    wait = stall_wait
    for attempt in range(1, stall_tries + 1):
        try:
            return provider._fetch_hour(symbol, hour)
        except DataUnavailable as exc:
            if attempt == stall_tries:
                print(
                    json.dumps(
                        {"event": "hour_failed", "hour": hour.isoformat(), "error": str(exc)}
                    ),
                    flush=True,
                )
                return None
            print(
                json.dumps(
                    {
                        "event": "stall",
                        "hour": hour.isoformat(),
                        "attempt": attempt,
                        "wait": wait,
                        "error": str(exc)[:120],
                    }
                ),
                flush=True,
            )
            time.sleep(wait)
            wait = min(wait * 2, 300.0)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Dukascopy ticks as M1 parquet")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--root", required=True, help="ParquetStore root")
    parser.add_argument("--cache", default=None, help="tick cache dir (default: <root>/_cache)")
    parser.add_argument("--hour-sleep", type=float, default=1.0, help="pause between hours")
    parser.add_argument("--stall-wait", type=float, default=60.0, help="first wait after a refusal")
    parser.add_argument("--stall-tries", type=int, default=8, help="attempts per hour before skipping")
    args = parser.parse_args(argv)

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) + timedelta(days=1)
    root = Path(args.root)
    cache = Path(args.cache) if args.cache else root / "_cache"
    store = ParquetStore(root)
    client = httpx.Client(
        timeout=_CLIENT_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        limits=httpx.Limits(max_keepalive_connections=4, max_connections=4),
    )
    provider = DukascopyProvider(cache, client=client, max_retries=4, base_delay=1.0)
    days = _days(start, end)
    written = 0
    failed_days: list[str] = []

    print(json.dumps({"event": "start", "days": len(days), "symbol": args.symbol}), flush=True)
    try:
        for index, day in enumerate(days, start=1):
            frames: list[pl.DataFrame] = []
            day_failed = False
            for hour_offset in range(24):
                hour = day + timedelta(hours=hour_offset)
                frame = _fetch_hour(provider, args.symbol, hour, args.stall_wait, args.stall_tries)
                if frame is None:
                    day_failed = True
                elif frame.height:
                    frames.append(frame)
                time.sleep(args.hour_sleep)
            day_frame = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
            rows = store.write(args.symbol, "M1", day_frame) if day_frame.height else 0
            written += rows
            if day_failed:
                failed_days.append(day.date().isoformat())
            print(
                json.dumps(
                    {
                        "event": "day",
                        "i": index,
                        "n": len(days),
                        "day": day.date().isoformat(),
                        "rows": rows,
                        "total": written,
                        "degraded": day_failed,
                    }
                ),
                flush=True,
            )
    finally:
        provider.close()

    summary = {
        "event": "done",
        "rows": written,
        "days_ok": len(days) - len(failed_days),
        "days_degraded": failed_days,
        "root": str(root),
    }
    print(json.dumps(summary), flush=True)
    return 0 if not failed_days else 2


if __name__ == "__main__":
    sys.exit(main())
