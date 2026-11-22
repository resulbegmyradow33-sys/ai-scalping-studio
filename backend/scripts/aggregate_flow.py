"""Aggregate quote volume from the 5-year tick archive into a per-minute flow table.

The archive carries ``bid_volume`` (volume printed at the bid, i.e. sell-side
aggression) and ``ask_volume`` (buy-side aggression) on every tick. Summing them
per minute gives a microstructure signal that is not derived from price:

    flow = (buy_vol - sell_vol) / (buy_vol + sell_vol)   in [-1, 1]

Output is a small parquet table per month with ``ts, buy_vol, sell_vol, flow``
under ``<root>/<symbol>/M1``. It is not part of BAR_SCHEMA on purpose: this is a
research side-channel, not a bar column.

Usage:
    python -m scripts.aggregate_flow --dir ../data-ticks5 \
        --symbol XAUUSD --root ../data-flow-store
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl


def _flow(path: Path) -> pl.DataFrame:
    ticks = (
        pl.read_parquet(path, columns=["timestamp", "bid_volume", "ask_volume"])
        .with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").alias("ts"))
        .group_by_dynamic("ts", every="1m", closed="left", label="left")
        .agg(
            pl.col("bid_volume").sum().alias("sell_vol"),
            pl.col("ask_volume").sum().alias("buy_vol"),
        )
        .with_columns(
            (
                (pl.col("buy_vol") - pl.col("sell_vol"))
                / (pl.col("buy_vol") + pl.col("sell_vol")).clip(1e-12)
            ).alias("flow")
        )
    )
    return ticks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate tick volume imbalance per minute")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    files = sorted(Path(args.dir).glob("*.parquet"))
    if not files:
        print(json.dumps({"error": f"no parquet under {args.dir}"}), file=sys.stderr)
        return 2
    root = Path(args.root) / args.symbol / "M1"
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    frames: list[pl.DataFrame] = []
    for index, path in enumerate(files, start=1):
        frame = _flow(path)
        frames.append(frame)
        total += frame.height
        if len(frames) >= 12:
            pl.concat(frames).unique("ts").sort("ts").write_parquet(
                root / f"part-{index:04d}.parquet"
            )
            frames = []
        if index % 10 == 0:
            print(json.dumps({"i": index, "n": len(files), "rows": total}), flush=True)
    if frames:
        pl.concat(frames).unique("ts").sort("ts").write_parquet(root / "part-final.parquet")
    print(json.dumps({"event": "done", "rows": total, "root": str(root)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
