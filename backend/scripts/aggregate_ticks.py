"""Aggregate daily Tickmill tick parquet files into M1 bars.

The export looks like::

    bid, ask, last, volume, time_msc, flags, volume_real, spread, spread_pips, time

``time`` is the broker's wall clock (EET/EEST), so it is parsed in
``Europe/Kyiv`` and converted to UTC. OHLC follows the project's bid-side
convention, ``ask_close`` is the last ask of the minute, ``spread`` is the mean
ask-bid, and ``volume`` is the tick count (the export's trade volume is 0 for
gold quotes).

Usage:
    python -m scripts.aggregate_ticks --dir ../data-ticks \
        --symbol XAUUSD --root ../data-rich-store
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from scalping.data.base import BAR_SCHEMA
from scalping.data.store import ParquetStore

_BROKER_TZ = "Europe/Kyiv"


def _m1(path: Path) -> pl.DataFrame:
    ticks = pl.read_parquet(path).select(["time", "bid", "ask"]).sort("time")
    ticks = ticks.with_columns(
        pl.col("time")
        .dt.replace_time_zone(_BROKER_TZ, ambiguous="earliest", non_existent="null")
        .dt.convert_time_zone("UTC")
        .alias("ts")
    )
    ticks = ticks.with_columns(
        (pl.col("ask") - pl.col("bid")).alias("spread"), pl.lit(1, dtype=pl.Int64).alias("n")
    )
    bars = (
        ticks.group_by_dynamic("ts", every="1m", closed="left", label="left")
        .agg(
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.col("bid").last().alias("bid_close"),
            pl.col("ask").last().alias("ask_close"),
            pl.col("spread").mean().alias("spread"),
            pl.col("n").sum().alias("volume"),
        )
        .drop_nulls("ts")
        .with_columns(pl.lit("tickmill").alias("provider"))
    )
    return bars.cast(BAR_SCHEMA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate tick parquet into M1 bars")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    files = sorted(Path(args.dir).glob("XAUUSD_*.parquet"))
    if not files:
        print(json.dumps({"error": f"no tick files under {args.dir}"}), file=sys.stderr)
        return 2
    store = ParquetStore(Path(args.root))
    total = 0
    for path in files:
        frame = _m1(path)
        if frame.height == 0:
            continue
        total += store.write(args.symbol, "M1", frame)
        print(
            json.dumps(
                {
                    "file": path.name,
                    "bars": frame.height,
                    "first": frame["ts"][0].isoformat(),
                    "last": frame["ts"][-1].isoformat(),
                }
            ),
            flush=True,
        )
    coverage = store.coverage(args.symbol)
    print(
        json.dumps(
            {
                "event": "done",
                "added": total,
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
