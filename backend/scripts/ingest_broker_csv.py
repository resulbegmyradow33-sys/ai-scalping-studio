"""Ingest broker-style M1 CSVs (one date and one time column) into the store.

The public HuggingFace mirror of MT4/MT5 exports looks like:

    2024.01.01,18:00,2062.598,2064.525,2062.405,2064.235,0

Two things differ from ``CsvProvider``'s HistData layout: the timestamp is split
across two columns, and the broker clock is EET/EEST (GMT+2 winter, GMT+3
summer), not a fixed offset. Parsing the wall clock in ``Europe/Kyiv`` and
converting to UTC gives the right instant on both sides of a DST change.

Bid is the OHLC series; ask and spread are left null so the spread model in
``features.prepare`` prices the fills (the export has no quotes).

Usage:
    python -m scripts.ingest_broker_csv --dir ../data-real-csv \
        --symbol XAUUSD --root ../data-real-store
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from scalping.data.base import BAR_SCHEMA
from scalping.data.store import ParquetStore


def _load(path: Path, zone: ZoneInfo) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        has_header=False,
        separator=",",
        new_columns=["date", "time", "open", "high", "low", "close", "volume"],
        schema_overrides={
            "date": pl.String,
            "time": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
        },
    )
    stamps = [
        datetime.strptime(f"{day} {clock}", "%Y.%m.%d %H:%M").replace(tzinfo=zone).astimezone(
            ZoneInfo("UTC")
        )
        for day, clock in zip(frame["date"].to_list(), frame["time"].to_list(), strict=True)
    ]
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "bid_close": frame["close"],
            "ask_close": [None] * frame.height,
            "spread": [None] * frame.height,
            "volume": frame["volume"],
            "provider": ["csv_eet"] * frame.height,
        },
        schema=BAR_SCHEMA,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest broker M1 CSVs into the parquet store")
    parser.add_argument("--dir", required=True, help="directory holding *.csv files")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--root", required=True, help="ParquetStore root")
    parser.add_argument("--tz", default="Europe/Kyiv", help="broker wall-clock zone")
    args = parser.parse_args(argv)

    zone = ZoneInfo(args.tz)
    source = Path(args.dir)
    files = sorted(source.glob("*.csv"))
    if not files:
        print(json.dumps({"error": f"no csv files under {source}"}), file=sys.stderr)
        return 2

    store = ParquetStore(Path(args.root))
    total = 0
    for path in files:
        frame = _load(path, zone)
        rows = store.write(args.symbol, "M1", frame)
        total += rows
        print(
            json.dumps(
                {
                    "file": path.name,
                    "raw": frame.height,
                    "added": rows,
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
                "rows": total,
                "coverage_rows": coverage.rows,
                "first_ts": coverage.first_ts.isoformat() if coverage.first_ts else None,
                "last_ts": coverage.last_ts.isoformat() if coverage.last_ts else None,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
