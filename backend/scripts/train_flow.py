"""Train and evaluate with the tick-volume flow gate on a tick-derived store.

The flow channel is a side table (``<flow_root>/<symbol>/M1``) produced by
``scripts.aggregate_flow``. It is resampled to the run's timeframe (mean) and
aligned by timestamp; the gate threshold is the causal 240-bar 80th percentile
of ``|flow|`` shifted by one bar, so a bar is only tradable when its imbalance
is extreme relative to the recent past.

Usage:
    python -m scripts.train_flow --tf H1 --train-end 2024-01-01 \
        --steps 300000 --flow-quantile 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from scalping.backtest.engine import run_backtest
from scalping.config import ContractSpec, CostConfig, RiskConfig, Settings
from scalping.data.store import ParquetStore
from scalping.features.build import build_features
from scalping.rl.policy import load_policy
from scalping.rl.train import TrainConfig, train
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM, bandit_training_data, label_events

_RULE = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h", "D1": "1d"}


def _flow_series(flow_root: Path, symbol: str, rule: str, frame: pl.DataFrame) -> np.ndarray:
    parts = sorted((flow_root / symbol / "M1").glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no flow partitions under {flow_root / symbol / 'M1'}")
    flow = pl.concat([pl.read_parquet(p) for p in parts]).unique("ts").sort("ts")
    if rule != "1m":
        flow = flow.group_by_dynamic("ts", every=rule, closed="left", label="left").agg(
            pl.col("flow").mean()
        )
    flow = flow.with_columns(pl.col("ts").dt.cast_time_unit("us"))
    joined = frame.select("ts").with_columns(pl.col("ts").dt.cast_time_unit("us")).join(
        flow, on="ts", how="left"
    )
    return joined["flow"].to_numpy()


def _thresholds(flow: np.ndarray, window: int, quantile: float) -> np.ndarray:
    series = pl.Series("f", np.abs(flow))
    thr = (
        series.rolling_quantile(quantile, window_size=window, min_samples=min(60, window))
        .shift(1)
        .to_numpy()
    )
    return np.asarray(thr, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate with the flow gate")
    parser.add_argument("--store", default="../data-ticks5-store")
    parser.add_argument("--flow-root", default="../data-flow-store")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--tf", default="H1")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--sl-atr", type=float, default=3.0)
    parser.add_argument("--tp-atr", type=float, default=6.0)
    parser.add_argument("--max-bars", type=int, default=48)
    parser.add_argument("--flow-quantile", type=float, default=0.8)
    parser.add_argument("--flow-window", type=int, default=240)
    parser.add_argument("--flow-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args(argv)

    rule = _RULE[args.tf]
    settings = Settings(
        contract=ContractSpec(),
        risk=RiskConfig(
            max_bars_in_trade=args.max_bars,
            sl_atr_mult=args.sl_atr,
            tp_atr_mult=args.tp_atr,
        ),
        costs=CostConfig(),
    )
    store = ParquetStore(Path(args.store))
    frame = store.read(args.symbol, args.tf)
    split = pl.Series([args.train_end]).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")[0]
    train_frame = frame.filter(pl.col("ts") < split)
    eval_frame = frame.filter(pl.col("ts") >= split)
    print(json.dumps({"tf": args.tf, "train": train_frame.height, "eval": eval_frame.height}))

    flow_all = _flow_series(Path(args.flow_root), args.symbol, rule, frame)
    thr_all = _thresholds(flow_all, args.flow_window, args.flow_quantile)
    train_flow, train_thr = flow_all[: train_frame.height], thr_all[: train_frame.height]
    eval_flow, eval_thr = flow_all[train_frame.height :], thr_all[train_frame.height :]

    features = build_features(train_frame, settings.costs)
    labels = label_events(train_frame, settings)
    X, arms, rewards = bandit_training_data(features, labels)
    selector = LinUCB(dim=CONTEXT_DIM).fit_offline(X, arms, rewards).freeze()
    print(json.dumps({"bandit_labels": int(X.shape[0])}))

    config = TrainConfig(
        run_name=f"flow-{args.tf}",
        total_steps=args.steps,
        n_envs=args.envs,
        seed=args.seed,
        window=args.window,
        timeframe=args.tf,
        gate_signals=True,
        entry_penalty=0.05,
        random_start=True,
    )
    result = train(
        config,
        features,
        settings,
        selector=selector,
        out_dir=Path(args.out_dir),
        labels=labels,
        train_end=args.train_end,
        flow=train_flow,
        flow_thr=train_thr,
        flow_obs=args.flow_obs,
    )
    print(json.dumps({"run_id": result.run_id, "reward_mean": result.reward_mean}))

    eval_features = build_features(eval_frame, settings.costs)
    policy = load_policy(Path(args.out_dir) / result.run_id)
    backtest = run_backtest(
        policy,
        eval_features,
        settings,
        strategy_selector=selector,
        labels=label_events(eval_frame, settings),
        window=args.window,
        flow=eval_flow,
        flow_thr=eval_thr,
        flow_obs=args.flow_obs,
        max_steps=len(eval_features.ts) + 1,
    )
    metrics = backtest.metrics
    print(
        json.dumps(
            {
                "eval_trades": metrics.trades,
                "net": round(metrics.net_profit, 2),
                "pf": round(metrics.profit_factor, 3),
                "dd": round(metrics.max_drawdown, 3),
                "win": round(metrics.win_rate, 3),
                "expectancy_r": round(metrics.expectancy_r, 3),
                "costs": round(metrics.total_costs, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
