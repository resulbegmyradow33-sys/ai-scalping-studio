"""Run the promoted card, unchanged, on a genuinely unseen forward window.

The nightly calibrator is allowed to re-select parameters, so its holdout is
not a clean forward test: it shares history with the selection. This script
loads the card as-is, concatenates the historical and forward stores for
warm-up, and starts the simulator at the forward boundary. No parameter is
fitted on the forward bars.

Flow comes from two side stores (historical Tickmill-derived and the forward
Dukascopy crawl); both expose the same scale-free imbalance
``(buy - sell) / (buy + sell)``, so the causal quantile gate transfers without
recalibration.

Usage:
    python -m scripts.forward_test --start 2026-06-01 --end 2026-11-21
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from scalping.backtest.engine import run_backtest
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.features.build import build_features
from scalping.strategy.card import load_card
from scalping.strategy.flow import flow_thresholds
from scalping.strategy.rule import FlowMomentumRule
from scalping.strategy.signals import strategy_signals

_RULE = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}


def _flow_h1(root: Path, symbol: str, timeframe: str) -> pl.DataFrame:
    """Mean imbalance per bar from one side store, sorted by timestamp."""
    parts = sorted((root / symbol / "M1").glob("*.parquet"))
    if not parts:
        raise ValueError(f"no flow partitions under {root / symbol / 'M1'}")
    flow = pl.concat([pl.read_parquet(path) for path in parts]).unique("ts").sort("ts")
    rule = _RULE[timeframe]
    if rule != "1m":
        flow = flow.group_by_dynamic("ts", every=rule, closed="left", label="left").agg(
            pl.col("flow").mean()
        )
    return flow.with_columns(pl.col("ts").dt.cast_time_unit("us"))


def _combined_flow(
    roots: list[Path], symbol: str, timeframe: str, frame: pl.DataFrame
) -> np.ndarray:
    frames = [_flow_h1(root, symbol, timeframe) for root in roots]
    merged = pl.concat(frames).unique("ts", keep="last").sort("ts")
    aligned = (
        frame.select("ts")
        .with_columns(pl.col("ts").dt.cast_time_unit("us"))
        .join(merged, on="ts", how="left")
    )
    return aligned["flow"].to_numpy()


def _settings_for(card_path: Path) -> tuple[object, Settings]:
    card = load_card(card_path)
    if card is None:
        raise ValueError(f"no strategy card at {card_path}")
    base = Settings()
    settings = dataclasses.replace(
        base,
        risk=dataclasses.replace(
            base.risk,
            sl_atr_mult=card.params.sl_atr_mult,
            tp_atr_mult=card.params.tp_atr_mult,
            max_bars_in_trade=card.params.max_bars_in_trade,
        ),
    )
    return card, settings


def _run_rule(
    frame: pl.DataFrame,
    flow: np.ndarray,
    threshold: np.ndarray,
    signal: np.ndarray,
    settings: Settings,
    start: str,
    end: str | None = None,
) -> dict[str, float]:
    stamp = pl.Series([start]).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")[0]
    work = frame
    if end is not None:
        cutoff = pl.Series([end]).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")[0]
        work = frame.filter(pl.col("ts") <= cutoff)
    index = int(work["ts"].search_sorted(stamp))
    sliced = work.slice(index)
    if sliced.height < 200:
        raise ValueError(f"only {sliced.height} bars from {start}; need at least 200")
    features = build_features(sliced, settings.costs)
    stop = index + sliced.height
    rule = FlowMomentumRule.build(
        sliced, flow[index:stop], threshold[index:stop], signal[index:stop]
    )
    rule.reset(features.valid_from + 59)
    result = run_backtest(
        lambda obs: rule.act(),
        features,
        settings,
        policy_name="flow_momentum_forward",
        window=60,
        max_steps=len(features.ts) + 1,
    )
    metrics = result.metrics
    return {
        "trades": float(metrics.trades),
        "net_profit": float(metrics.net_profit),
        "profit_factor": float(metrics.profit_factor),
        "win_rate": float(metrics.win_rate),
        "expectancy_r": float(metrics.expectancy_r),
        "max_drawdown": float(metrics.max_drawdown),
        "costs": float(metrics.total_costs),
    }


def _stress(settings: Settings, multiplier: float) -> Settings:
    costs = settings.costs
    return dataclasses.replace(
        settings,
        costs=dataclasses.replace(
            costs,
            spread_floor=costs.spread_floor * multiplier,
            slippage_atr_frac=costs.slippage_atr_frac * multiplier,
            commission_per_lot=costs.commission_per_lot * multiplier,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forward test of the live card")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--card", default="runs/strategy_card.json")
    parser.add_argument("--history-store", default="../data-ticks5-store")
    parser.add_argument("--history-flow", default="../data-flow-store")
    parser.add_argument("--forward-store", default="../data-forward-store")
    parser.add_argument("--forward-flow", default="../data-forward-flow")
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-11-21")
    parser.add_argument("--baseline-start", default="2025-12-01")
    args = parser.parse_args(argv)

    card, settings = _settings_for(Path(args.card))
    history = ParquetStore(Path(args.history_store)).read(args.symbol, args.timeframe)
    forward = ParquetStore(Path(args.forward_store)).read(args.symbol, args.timeframe)
    if forward.is_empty():
        raise ValueError(f"no forward bars under {args.forward_store}")
    end = pl.Series([args.end]).str.to_datetime("%Y-%m-%d").dt.replace_time_zone("UTC")[0]
    forward = forward.filter(pl.col("ts") <= end)
    frame = (
        pl.concat([history, forward])
        .unique("ts", keep="last")
        .sort("ts")
        .filter(pl.col("ts") <= end)
    )
    flow = _combined_flow(
        [Path(args.history_flow), Path(args.forward_flow)],
        args.symbol,
        args.timeframe,
        frame,
    )
    threshold = flow_thresholds(flow, card.params.flow_window, card.params.flow_quantile)
    signal = strategy_signals(frame)["sig_momentum"].to_numpy()

    history_end = str(history["ts"][-1])[:10]
    baseline = _run_rule(
        frame, flow, threshold, signal, settings, args.baseline_start, history_end
    )
    forward_metrics = _run_rule(frame, flow, threshold, signal, settings, args.start)
    stress = {}
    for multiplier in (2.0,):
        stress[f"x{multiplier:g}"] = _run_rule(
            frame,
            flow,
            threshold,
            signal,
            _stress(settings, multiplier),
            args.start,
        )
    payload = {
        "event": "forward_test",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start": args.start,
        "end": args.end,
        "card": card.to_dict(),
        "forward_bars": int(
            frame.filter(
                pl.col("ts")
                >= pl.Series([args.start])
                .str.to_datetime("%Y-%m-%d")
                .dt.replace_time_zone("UTC")[0]
            ).height
        ),
        "baseline_previous_six_months": baseline,
        "forward": forward_metrics,
        "stress": stress,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
