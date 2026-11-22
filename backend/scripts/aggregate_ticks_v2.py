"""Aggregate the monthly XAUUSD tick parquet archive into M1 bars.

Schema of the source::

    timestamp (naive, UTC), bid_price, ask_price, bid_volume, ask_volume

The first tick of a month lands on the Sunday 22:00 UTC open, so the naive
timestamps are UTC. OHLC follows the bid side, ``ask_close`` is the last ask of
the minute, ``spread`` is the mean ask-bid, and ``volume`` is the tick count.
``volume_real`` keeps the summed quote volume for reference.

Usage:
    python -m scripts.aggregate_ticks_v2 --dir ../data-ticks5 \
        --symbol XAUUSD --root ../data-ticks5-store
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from scalping.data.base import BAR_SCHEMA
from scalping.data.provenance import build_provenance, save_provenance
from scalping.data.store import ParquetStore


def _m1(path: Path) -> pl.DataFrame:
    ticks = (
        pl.read_parquet(path, columns=["timestamp", "bid_price", "ask_price", "bid_volume"])
        .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").alias("ts"))
        .with_columns(
            (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
            pl.lit(1, dtype=pl.Int64).alias("n"),
        )
    )
    bars = (
        ticks.group_by_dynamic("ts", every="1m", closed="left", label="left")
        .agg(
            pl.col("bid_price").first().alias("open"),
            pl.col("bid_price").max().alias("high"),
            pl.col("bid_price").min().alias("low"),
            pl.col("bid_price").last().alias("close"),
            pl.col("bid_price").last().alias("bid_close"),
            pl.col("ask_price").last().alias("ask_close"),
            pl.col("spread").mean().alias("spread"),
            pl.col("n").sum().alias("volume"),
        )
        .with_columns(pl.lit("ticks5").alias("provider"))
    )
    return bars.cast(BAR_SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate monthly ticks into M1 bars")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    files = sorted(Path(args.dir).glob("*.parquet"))
    if not files:
        print(json.dumps({"error": f"no parquet under {args.dir}"}), file=sys.stderr)
        return 2
    store = ParquetStore(Path(args.root))
    total = 0
    for index, path in enumerate(files, start=1):
        frame = _m1(path)
        if frame.height == 0:
            continue
        total += store.write(args.symbol, "M1", frame)
        print(
            json.dumps(
                {
                    "i": index,
                    "n": len(files),
                    "file": path.name,
                    "bars": frame.height,
                    "first": frame["ts"][0].isoformat(),
                    "last": frame["ts"][-1].isoformat(),
                    "added": total,
                }
            ),
            flush=True,
        )
    coverage = store.coverage(args.symbol)
    final = store.read(args.symbol, "M1")
    if not final.is_empty():
        save_provenance(
            store.root,
            build_provenance(
                final,
                symbol=args.symbol,
                timeframe="M1",
                source=f"ticks:{Path(args.dir).name}",
            ),
        )
    print(
        json.dumps(
            {
                "event": "done",
                "rows": coverage.rows,
                "first": coverage.first_ts.isoformat() if coverage.first_ts else None,
                "last": coverage.last_ts.isoformat() if coverage.last_ts else None,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
