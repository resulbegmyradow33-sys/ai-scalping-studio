"""The RL lab: retry imitation + PPO whenever the data actually changes.

The rule card is the product. This script is the controlled experiment that
keeps asking whether reinforcement learning can beat it, without ever touching
the live card:

* if the data digest matches the last recorded cycle, it does nothing;
* otherwise it trains one BC+PPO policy per split and evaluates the same rule
  on the same split;
* a cycle is only marked ``beats_rule`` when the agent is positive and ahead of
  the rule in *every* split;
* results go to ``runs/rl_history.jsonl``; promotion stays manual because two
  splits are still a thin basis for real money.

Usage:
    python -m scripts.nightly_rl --splits 2024-01-01 2025-01-01
    python -m scripts.nightly_rl --loop --interval-hours 168
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scalping.backtest.engine import run_backtest
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.features.build import build_features
from scalping.strategy.card import frame_digest, load_card
from scalping.strategy.flow import flow_series, flow_thresholds
from scalping.strategy.rule import FlowMomentumRule
from scalping.strategy.signals import strategy_signals

__all__ = ["main"]


def _rule_split(
    frame: pl.DataFrame,
    flow: np.ndarray,
    threshold: np.ndarray,
    signal: np.ndarray,
    settings: Settings,
    split: str,
) -> dict[str, float]:
    """Sequential rule performance from ``split`` to the end of the frame."""
    stamp = pl.Series([split]).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")[0]
    index = int(frame["ts"].search_sorted(stamp))
    sliced = frame.slice(index)
    features = build_features(sliced, settings.costs)
    rule = FlowMomentumRule.build(sliced, flow[index:], threshold[index:], signal[index:])
    rule.reset(features.valid_from + 59)
    result = run_backtest(
        lambda obs: rule.act(),
        features,
        settings,
        policy_name="flow_momentum",
        window=60,
        max_steps=len(features.ts) + 1,
    )
    metrics = result.metrics
    return {
        "trades": float(metrics.trades),
        "net_profit": float(metrics.net_profit),
        "profit_factor": float(metrics.profit_factor),
        "expectancy_r": float(metrics.expectancy_r),
        "max_drawdown": float(metrics.max_drawdown),
    }


def _run_agent(args: argparse.Namespace, split: str) -> tuple[str, dict[str, float]]:
    """Train one BC+PPO run and return its id and evaluation metrics."""
    command = [
        sys.executable,
        "-m",
        "scripts.train_flow",
        "--store",
        args.store,
        "--flow-root",
        args.flow_root,
        "--symbol",
        args.symbol,
        "--tf",
        args.timeframe,
        "--train-end",
        split,
        "--steps",
        str(args.steps),
        "--envs",
        str(args.envs),
        "--bc-steps",
        str(args.bc_steps),
        "--bc-epochs",
        str(args.bc_epochs),
        "--bc-lr",
        str(args.bc_lr),
        "--no-bc-executed",
        "--out-dir",
        args.out_dir,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"training failed for split {split}: {completed.stderr.strip()[-800:]}"
        )
    run_id = ""
    metrics: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "run_id" in payload:
            run_id = str(payload["run_id"])
        if "eval_trades" in payload:
            metrics = {
                "trades": float(payload["eval_trades"]),
                "net_profit": float(payload["net"]),
                "profit_factor": float(payload["pf"]),
                "expectancy_r": float(payload["expectancy_r"]),
                "max_drawdown": float(payload["dd"]),
            }
    if not run_id or not metrics:
        raise RuntimeError(f"training produced no parseable result for split {split}")
    return run_id, metrics


def _last_history(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _cycle(args: argparse.Namespace) -> dict[str, Any]:
    store = ParquetStore(Path(args.store))
    frame = store.read(args.symbol, args.timeframe)
    if frame.is_empty():
        raise ValueError(f"no {args.symbol} {args.timeframe} bars under {args.store}")
    digest = frame_digest(frame)
    history = Path(args.history)
    previous = _last_history(history)
    if previous is not None and previous.get("digest") == digest:
        return {
            "event": "skipped",
            "reason": "data digest unchanged",
            "digest": digest,
            "last_bar": str(frame["ts"][-1]),
        }

    flow = flow_series(Path(args.flow_root), args.symbol, args.timeframe, frame)
    card = load_card(args.card)
    if card is None:
        raise ValueError(f"no strategy card at {args.card}; calibrate the rule first")
    params = card.params
    base = Settings()
    settings = dataclasses.replace(
        base,
        risk=dataclasses.replace(
            base.risk,
            sl_atr_mult=params.sl_atr_mult,
            tp_atr_mult=params.tp_atr_mult,
            max_bars_in_trade=params.max_bars_in_trade,
        ),
    )
    threshold = flow_thresholds(flow, params.flow_window, params.flow_quantile)
    signal = strategy_signals(frame)["sig_momentum"].to_numpy()

    splits: list[dict[str, Any]] = []
    for split in args.splits:
        rule = _rule_split(frame, flow, threshold, signal, settings, split)
        run_id, agent = _run_agent(args, split)
        splits.append({"split": split, "run_id": run_id, "agent": agent, "rule": rule})

    beats = all(
        item["agent"]["expectancy_r"] > 0
        and item["agent"]["expectancy_r"] > item["rule"]["expectancy_r"]
        for item in splits
    )
    result = {
        "event": "rl_cycle",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "digest": digest,
        "last_bar": str(frame["ts"][-1]),
        "card_digest": card.digest,
        "splits": splits,
        "beats_rule": beats,
        "steps": args.steps,
        "bc_steps": args.bc_steps,
        "bc_epochs": args.bc_epochs,
    }
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous RL experiment")
    parser.add_argument("--store", default="../data-ticks5-store")
    parser.add_argument("--flow-root", default="../data-flow-store")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1", choices=("M15", "H1"))
    parser.add_argument("--card", default="runs/strategy_card.json")
    parser.add_argument("--history", default="runs/rl_history.jsonl")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--splits", nargs="+", default=["2024-01-01", "2025-01-01"])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--bc-steps", type=int, default=100_000)
    parser.add_argument("--bc-epochs", type=int, default=4)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-hours", type=float, default=168.0)
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
