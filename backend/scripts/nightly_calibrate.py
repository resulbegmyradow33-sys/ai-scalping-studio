"""Nightly recalibration loop for the live strategy card.

Run once (cron/launchd) or keep it alive with ``--loop``. Every cycle:

1. reads the newest bars and the tick-flow side table,
2. selects rule parameters on the training slice,
3. gates the winner on the holdout,
4. writes ``strategy_card.json`` only when it passes and beats the incumbent,
5. appends the full report to ``strategy_history.jsonl`` either way.

The panel reloads the card on its next ``start``, so a promotion goes live
without a restart and a rejection changes nothing.

Usage:
    python -m scripts.nightly_calibrate --store ../data-ticks5-store \\
        --flow-root ../data-flow-store --card runs/strategy_card.json

    # keep it running: one cycle every 24 hours
    python -m scripts.nightly_calibrate --loop --interval-hours 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.strategy.calibrate import (
    AUDIT_FILENAME,
    calibrate,
    card_from_report,
    record_calibration,
)
from scalping.strategy.card import load_card, save_card
from scalping.strategy.flow import flow_series

_NOTE = "holdout reused across nightly runs; treat later promotions as weaker evidence"


def _cycle(args: argparse.Namespace) -> dict[str, object]:
    store = ParquetStore(Path(args.store))
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise ValueError(f"no {args.symbol} {args.timeframe} bars under {args.store}")
    flow = flow_series(Path(args.flow_root), args.symbol, args.timeframe, frame)
    card_path = Path(args.card)
    incumbent = load_card(card_path)
    report = calibrate(
        frame,
        flow,
        symbol=args.symbol,
        timeframe=args.timeframe,
        base=Settings.load(args.settings) if args.settings else Settings(),
        holdout_fraction=args.holdout_fraction,
        min_train_trades=args.min_train_trades,
        min_holdout_trades=args.min_holdout_trades,
        incumbent=None if incumbent is None else incumbent.params,
    )
    if report.promoted:
        save_card(
            card_path,
            card_from_report(report, symbol=args.symbol, note=_NOTE),
        )
    audit_path = Path(args.audit) if args.audit else card_path.parent / AUDIT_FILENAME
    record_calibration(
        audit_path,
        actor="nightly_calibrate",
        dry_run=False,
        report=report,
        symbol=args.symbol,
        timeframe=args.timeframe,
        card_written=report.promoted,
        card_path=card_path,
    )
    result = {
        "event": "calibrated",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bars": frame.height,
        "last_bar": str(frame["ts"][-1]),
        **report.to_dict(),
        "card_path": str(card_path),
        "active": load_card(card_path) is not None,
    }
    history = Path(args.history)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly rule recalibration")
    parser.add_argument("--store", default="../data-ticks5-store")
    parser.add_argument("--flow-root", default="../data-flow-store")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1", choices=("M5", "M15", "M30", "H1", "H4", "D1"))
    parser.add_argument("--card", default="runs/strategy_card.json")
    parser.add_argument("--history", default="runs/strategy_history.jsonl")
    parser.add_argument(
        "--audit",
        default=None,
        help=f"calibration audit log (default: next to the card, {AUDIT_FILENAME})",
    )
    parser.add_argument("--settings", default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-trades", type=int, default=100)
    parser.add_argument("--min-holdout-trades", type=int, default=30)
    parser.add_argument("--loop", action="store_true", help="keep recalibrating forever")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    if args.interval_hours <= 0:
        parser.error("--interval-hours must be positive")

    while True:
        try:
            print(json.dumps(_cycle(args), sort_keys=True), flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the loop
            print(
                json.dumps(
                    {
                        "event": "error",
                        "at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "error": str(exc),
                        "trace": traceback.format_exc(limit=3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not args.loop:
                return 1
        if not args.loop:
            return 0
        time.sleep(args.interval_hours * 3600.0)


if __name__ == "__main__":
    sys.exit(main())
