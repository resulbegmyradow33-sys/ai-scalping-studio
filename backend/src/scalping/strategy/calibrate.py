"""Nightly recalibration of the validated flow-momentum rule.

The research conclusion was blunt: the parameters that survived five years
were a *rule*, not a PPO policy. The rule still has free parameters (flow
quantile, trailing window, stop/target/timeout), and markets drift, so the
useful form of "keep learning" is to re-select those parameters as data
accumulates.

Method, stated explicitly because it is easy to fool yourself here:

* the grid is scored on the **training slice only**;
* the winner is then evaluated **once** on the untouched holdout slice;
* it is promoted only when the holdout has enough trades, a positive mean R,
  and beats the incumbent card evaluated on that same holdout.

The holdout is not reused forever: each nightly run extends the end of the
data, so the newest bars were never selectable. Over many nights the same
historical holdout does get seen repeatedly, which weakens it; the card's
``note`` records that caveat rather than pretending the gate is a clean
out-of-sample test forever.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from scalping.config import RiskConfig, Settings
from scalping.strategy.card import (
    CARD_CLASSIFICATION,
    PolicyCard,
    RuleMetrics,
    RuleParams,
    frame_digest,
)
from scalping.strategy.flow import flow_thresholds
from scalping.strategy.labeling import label_events
from scalping.strategy.signals import strategy_signals

__all__ = [
    "AUDIT_FILENAME",
    "DEFAULT_RISK_OPTIONS",
    "CalibrationReport",
    "Candidate",
    "calibrate",
    "evaluate_params",
    "record_calibration",
    "summarise",
]

#: Append-only audit next to the card. One JSON object per calibration attempt.
AUDIT_FILENAME: str = "strategy_calibration_audit.jsonl"

#: ``(sl_atr_mult, tp_atr_mult, max_bars_in_trade)`` candidates.
DEFAULT_RISK_OPTIONS: tuple[tuple[float, float, int], ...] = (
    (2.0, 4.0, 48),
    (3.0, 6.0, 48),
    (4.0, 8.0, 96),
)

#: Flow quantiles tried on the training slice.
DEFAULT_QUANTILES: tuple[float, ...] = (0.75, 0.80, 0.85)

#: Trailing windows for the causal threshold, in bars.
DEFAULT_WINDOWS: tuple[int, ...] = (120, 240, 480)

#: Equity fraction risked per trade when turning R into a drawdown.
_RISK_PER_TRADE: float = 0.005


@dataclass(frozen=True, slots=True)
class Candidate:
    """One parameter set with its training and holdout evidence."""

    params: RuleParams
    train: RuleMetrics
    holdout: RuleMetrics


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """What the calibrator looked at, chose, and whether it promoted it."""

    candidate: Candidate
    incumbent: RuleMetrics | None
    promoted: bool
    reason: str
    digest: str
    evaluated: int

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "digest": self.digest,
            "evaluated": self.evaluated,
            "candidate": {
                "params": {
                    "timeframe": self.candidate.params.timeframe,
                    "flow_quantile": self.candidate.params.flow_quantile,
                    "flow_window": self.candidate.params.flow_window,
                    "sl_atr_mult": self.candidate.params.sl_atr_mult,
                    "tp_atr_mult": self.candidate.params.tp_atr_mult,
                    "max_bars_in_trade": self.candidate.params.max_bars_in_trade,
                },
                "train": self.candidate.train.to_dict(),
                "holdout": self.candidate.holdout.to_dict(),
            },
            "incumbent": None if self.incumbent is None else self.incumbent.to_dict(),
        }


def summarise(
    r_multiples: Iterable[float] | np.ndarray, risk: float = _RISK_PER_TRADE
) -> RuleMetrics:
    """Turn a sequence of realised R into the panel's metrics.

    Drawdown uses equal-fraction compounding at ``risk`` per trade, matching
    the way the labels were priced, so a card's DD is comparable across runs.
    """
    values = np.asarray(
        list(r_multiples) if not isinstance(r_multiples, np.ndarray) else r_multiples
    )
    values = values[np.isfinite(values)]
    trades = int(values.size)
    if trades == 0:
        return RuleMetrics()
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in values:
        equity *= 1.0 + risk * float(value)
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return RuleMetrics(
        trades=trades,
        expectancy_r=float(values.mean()),
        profit_factor=float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0,
        win_rate=float((values > 0).mean()),
        max_drawdown=float(drawdown),
        total_r=float(values.sum()),
    )


def _risk_settings(params: RuleParams, costs: Settings) -> Settings:
    return Settings(
        contract=costs.contract,
        costs=costs.costs,
        risk=RiskConfig(
            risk_pct=costs.risk.risk_pct,
            daily_loss_pct=costs.risk.daily_loss_pct,
            max_notional_x=costs.risk.max_notional_x,
            max_positions=costs.risk.max_positions,
            max_bars_in_trade=params.max_bars_in_trade,
            sl_atr_mult=params.sl_atr_mult,
            tp_atr_mult=params.tp_atr_mult,
        ),
    )


def _momentum_labels(
    frame: pl.DataFrame,
    params: RuleParams,
    base: Settings,
) -> pl.DataFrame:
    """Momentum labels with the signal sign; independent of the flow gate."""
    labels = label_events(frame, _risk_settings(params, base))
    momentum = labels.filter(pl.col("strategy") == "momentum").select("ts", "r_multiple")
    if momentum.is_empty():
        return momentum.with_columns(pl.lit(0, dtype=pl.Int8).alias("sig_momentum"))
    signals = strategy_signals(frame).select("ts", "sig_momentum")
    return momentum.join(signals, on="ts", how="inner").sort("ts")


def _apply_gate(
    momentum: pl.DataFrame,
    frame: pl.DataFrame,
    flow: np.ndarray,
    threshold: np.ndarray,
) -> pl.DataFrame:
    """Keep the labels whose bar had a signal and an extreme flow imbalance."""
    if momentum.is_empty():
        return momentum.select("ts", "r_multiple")
    gate = pl.DataFrame(
        {
            "ts": frame["ts"],
            "gate": np.isfinite(flow) & np.isfinite(threshold) & (np.abs(flow) > threshold),
        }
    )
    return (
        momentum.join(gate, on="ts", how="inner")
        .filter(pl.col("gate") & (pl.col("sig_momentum") != 0))
        .select("ts", "r_multiple")
        .sort("ts")
    )


def evaluate_params(
    frame: pl.DataFrame,
    flow: np.ndarray,
    params: RuleParams,
    base: Settings,
    holdout_start: datetime,
) -> tuple[RuleMetrics, RuleMetrics]:
    """Train and holdout metrics for one parameter set."""
    threshold = flow_thresholds(flow, params.flow_window, params.flow_quantile)
    momentum = _momentum_labels(frame, params, base)
    joined = _apply_gate(momentum, frame, flow, threshold)
    if joined.is_empty():
        return RuleMetrics(), RuleMetrics()
    train = joined.filter(pl.col("ts") < holdout_start)["r_multiple"].to_numpy()
    holdout = joined.filter(pl.col("ts") >= holdout_start)["r_multiple"].to_numpy()
    return summarise(train), summarise(holdout)


def _holdout_start(frame: pl.DataFrame, fraction: float) -> datetime:
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    index = max(1, min(frame.height - 1, int(frame.height * (1.0 - fraction))))
    return frame["ts"][index]


def calibrate(
    frame: pl.DataFrame,
    flow: np.ndarray,
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "H1",
    base: Settings | None = None,
    holdout_fraction: float = 0.2,
    risk_options: Iterable[tuple[float, float, int]] = DEFAULT_RISK_OPTIONS,
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    min_train_trades: int = 100,
    min_holdout_trades: int = 30,
    incumbent: RuleParams | None = None,
) -> CalibrationReport:
    """Select on the training slice, then evaluate once on the holdout."""
    if frame.height < 200:
        raise ValueError("calibrate needs at least 200 bars")
    if flow.shape != (frame.height,):
        raise ValueError(f"flow must have shape ({frame.height},), got {flow.shape}")
    settings = base if base is not None else Settings()
    holdout_start = _holdout_start(frame, holdout_fraction)
    digest = frame_digest(frame)

    best: Candidate | None = None
    evaluated = 0
    for sl, tp, hold in risk_options:
        risk_params = RuleParams(
            timeframe=timeframe,
            sl_atr_mult=float(sl),
            tp_atr_mult=float(tp),
            max_bars_in_trade=int(hold),
        )
        momentum = _momentum_labels(frame, risk_params, settings)
        for quantile in quantiles:
            for window in windows:
                trial = RuleParams(
                    timeframe=timeframe,
                    flow_quantile=float(quantile),
                    flow_window=int(window),
                    sl_atr_mult=float(sl),
                    tp_atr_mult=float(tp),
                    max_bars_in_trade=int(hold),
                )
                threshold = flow_thresholds(flow, trial.flow_window, trial.flow_quantile)
                joined = _apply_gate(momentum, frame, flow, threshold)
                evaluated += 1
                if joined.is_empty():
                    continue
                train = summarise(
                    joined.filter(pl.col("ts") < holdout_start)["r_multiple"].to_numpy()
                )
                if train.trades < min_train_trades:
                    continue
                if best is None or train.expectancy_r > best.train.expectancy_r:
                    holdout = summarise(
                        joined.filter(pl.col("ts") >= holdout_start)["r_multiple"].to_numpy()
                    )
                    best = Candidate(params=trial, train=train, holdout=holdout)

    if best is None:
        empty = Candidate(
            params=RuleParams(timeframe=timeframe),
            train=RuleMetrics(),
            holdout=RuleMetrics(),
        )
        return CalibrationReport(
            candidate=empty,
            incumbent=None,
            promoted=False,
            reason="no parameter set reached min_train_trades",
            digest=digest,
            evaluated=evaluated,
        )

    incumbent_metrics: RuleMetrics | None = None
    if incumbent is not None:
        _, incumbent_metrics = evaluate_params(frame, flow, incumbent, settings, holdout_start)

    reason = ""
    promoted = False
    if best.holdout.trades < min_holdout_trades:
        reason = f"holdout trades {best.holdout.trades} < {min_holdout_trades}"
    elif best.holdout.expectancy_r <= 0.0:
        reason = f"holdout expectancy {best.holdout.expectancy_r:+.4f}R is not positive"
    elif (
        incumbent_metrics is not None
        and best.holdout.expectancy_r <= incumbent_metrics.expectancy_r
    ):
        reason = (
            f"holdout {best.holdout.expectancy_r:+.4f}R does not beat "
            f"incumbent {incumbent_metrics.expectancy_r:+.4f}R"
        )
    else:
        promoted = True
        reason = "holdout positive and at least as good as the incumbent"

    return CalibrationReport(
        candidate=best,
        incumbent=incumbent_metrics,
        promoted=promoted,
        reason=reason,
        digest=digest,
        evaluated=evaluated,
    )


def card_from_report(
    report: CalibrationReport,
    *,
    symbol: str = "XAUUSD",
    note: str = "",
) -> PolicyCard:
    """Wrap a promoted report as the card the runner reads."""
    if not report.promoted:
        raise ValueError("cannot build a card from a rejected report")
    return PolicyCard.build(
        report.candidate.params,
        report.candidate.train,
        report.candidate.holdout,
        report.digest,
        symbol=symbol,
        note=note,
    )


def record_calibration(
    path: str | Path,
    *,
    actor: str,
    dry_run: bool,
    report: CalibrationReport,
    symbol: str,
    timeframe: str,
    card_written: bool,
    card_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one audit record for a calibration attempt and return it.

    The record is the answer to "who ran what, and what happened": actor, UTC
    timestamp, ``dry_run`` and the report. It is written for rejected and
    dry-run attempts too, because the decision not to promote is evidence as
    well. ``report.to_dict`` is embedded under ``result`` rather than copied
    into the top level so a future report field cannot collide with the audit
    envelope.
    """
    row: dict[str, Any] = {
        "event": "calibration",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "actor": actor,
        "symbol": symbol,
        "timeframe": timeframe,
        "dry_run": bool(dry_run),
        "promoted": bool(report.promoted),
        "result": report.to_dict(),
        "card_written": bool(card_written),
        "card_path": None if card_path is None else str(card_path),
        "classification": CARD_CLASSIFICATION,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row
