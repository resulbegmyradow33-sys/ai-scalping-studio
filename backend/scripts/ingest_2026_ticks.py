"""Turn the weekly 2026 Dukascopy tick CSVs into M1 bars plus signed flow.

The GitHub packaging (``Touhid-HZ/xauusd-tick-data-2026``) decodes the public
Dukascopy ``.bi5`` feed into weekly gzip CSVs:

    timestamp (UTC ms), ask, bid, ask_volume, bid_volume

Bars use the bid side for OHLC, the last ask for ``ask_close``, the mean
ask-bid for ``spread`` and the tick count for ``volume``. The flow table keeps
the per-minute sums and the scale-free imbalance
``(ask_volume - bid_volume) / (ask_volume + bid_volume)``.

Usage:
    python -m scripts.ingest_2026_ticks --dir ../data-2026-ticks \
        --root ../data-forward-store --flow-root ../data-forward-flow \
        --cache ../data-2026-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from scalping.data.base import BAR_SCHEMA
from scalping.data.resample import resample
from scalping.data.store import ParquetStore

_CACHE_SCHEMA: dict[str, pl.DataType] = {
    **BAR_SCHEMA,
    "buy_vol": pl.Float64,
    "sell_vol": pl.Float64,
    "flow": pl.Float64,
}


def _week(path: Path) -> pl.DataFrame:
    ticks = (
        pl.read_csv(
            path,
            columns=["timestamp", "ask", "bid", "ask_volume", "bid_volume"],
            schema_overrides={
                "timestamp": pl.String,
                "ask": pl.Float64,
                "bid": pl.Float64,
                "ask_volume": pl.Float64,
                "bid_volume": pl.Float64,
            },
        )
        .with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("ts")
        )
        .drop_nulls("ts")
        .sort("ts")
    )
    if ticks.height == 0:
        return pl.DataFrame(schema=_CACHE_SCHEMA)
    bars = (
        ticks.group_by_dynamic("ts", every="1m", closed="left", label="left")
        .agg(
            pl.col("bid").first().alias("open"),
            pl.col("bid").max().alias("high"),
            pl.col("bid").min().alias("low"),
            pl.col("bid").last().alias("close"),
            pl.col("bid").last().alias("bid_close"),
            pl.col("ask").last().alias("ask_close"),
            (pl.col("ask") - pl.col("bid")).mean().alias("spread"),
            pl.len().cast(pl.Int64).alias("volume"),
            pl.col("ask_volume").sum().alias("buy_vol"),
            pl.col("bid_volume").sum().alias("sell_vol"),
            pl.lit("dukascopy-2026").alias("provider"),
        )
        .with_columns(
            (
                (pl.col("buy_vol") - pl.col("sell_vol"))
                / (pl.col("buy_vol") + pl.col("sell_vol")).clip(1e-12)
            ).alias("flow")
        )
        .select(list(_CACHE_SCHEMA))
        .cast(_CACHE_SCHEMA)
    )
    return bars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate 2026 tick CSVs")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default="2026-05-23")
    args = parser.parse_args(argv)

    source = Path(args.dir)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.csv.gz"))
    if not files:
        print(json.dumps({"error": f"no csv.gz under {source}"}), file=sys.stderr)
        return 2
    for index, path in enumerate(files, start=1):
        target = cache / f"{path.stem}.parquet"
        if target.exists():
            print(json.dumps({"i": index, "n": len(files), "file": path.name, "skip": True}), flush=True)
            continue
        frame = _week(path)
        frame.write_parquet(target)
        print(
            json.dumps(
                {
                    "i": index,
                    "n": len(files),
                    "file": path.name,
                    "bars": frame.height,
                    "first": str(frame["ts"][0]) if frame.height else None,
                    "last": str(frame["ts"][-1]) if frame.height else None,
                }
            ),
            flush=True,
        )

    parts = sorted(cache.glob("*.parquet"))
    every = (
        pl.concat([pl.read_parquet(path) for path in parts])
        .unique("ts", keep="last")
        .sort("ts")
    )
    start = pl.Series([args.start]).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")[0]
    every = every.filter(pl.col("ts") >= start)
    store = ParquetStore(Path(args.root))
    store.write(args.symbol, "M1", every.select(list(BAR_SCHEMA)).cast(BAR_SCHEMA))
    flow = every.select("ts", "buy_vol", "sell_vol", "flow")
    flow_dir = Path(args.flow_root) / args.symbol / "M1"
    flow_dir.mkdir(parents=True, exist_ok=True)
    flow.write_parquet(flow_dir / "part-0000.parquet")
    m1 = store.read(args.symbol, "M1")
    for rule, timeframe in (("5m", "M5"), ("15m", "M15"), ("1h", "H1"), ("1d", "D1")):
        store.write(
            args.symbol,
            timeframe,
            resample(m1, rule).rename({"spread_mean": "spread"}),
        )
    coverage = store.coverage(args.symbol)
    print(
        json.dumps(
            {
                "event": "done",
                "rows": coverage.rows,
                "first": coverage.first_ts.isoformat() if coverage.first_ts else None,
                "last": coverage.last_ts.isoformat() if coverage.last_ts else None,
                "flow_rows": flow.height,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
