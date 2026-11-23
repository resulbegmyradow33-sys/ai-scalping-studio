"""Trade the classifier in the paper simulator.

Prediction metrics are not trading metrics. This module turns a fitted
classifier into a deterministic policy, runs it on one chronological split with
the existing PaperBroker/RiskEngine, and reports the same costs every other
strategy pays.

The policy is deliberately simple: when flat, enter in the direction whose
probability clears ``threshold``; when in a position, close if the opposite
probability clears ``exit_threshold``. The simulator still owns sizing, stops,
targets, the time stop and every risk block. A classifier cannot bypass them.

The optional regime breakdown describes the *base* run only: it groups the same
closed trades the headline metrics were computed from by the regime label that
was current at each trade's entry. It is a descriptive table, not a second
evaluation and not a promotion gate. Cost-stress runs are headline-only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.backtest.engine import BacktestResult, run_backtest
from scalping.config import Settings
from scalping.datasets.builder import build_dataset
from scalping.features.build import FeatureFrame
from scalping.ml.train import MLResult
from scalping.regime.rules import classify
from scalping.regime.strategy import UNKNOWN_REGIME, strategy_regime_table
from scalping.types import Trade

__all__ = ["EvaluationResult", "evaluate_model"]

_HOLD: Final[int] = 0
_BUY: Final[int] = 1
_SELL: Final[int] = 2
_CLOSE: Final[int] = 3
_SPLITS: Final[dict[str, int]] = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    split: str
    threshold: float
    exit_threshold: float
    metrics: dict[str, Any]
    hold: dict[str, Any]
    stress: dict[str, dict[str, Any]]
    #: Base-run regime table: ``{"rows": [...], "unmatched_trades": n}``.
    #: Empty when no labels were supplied and rule labels were not requested.
    regime: dict[str, Any] = field(default_factory=dict)


def _slice_features(features: FeatureFrame, start: int) -> FeatureFrame:
    return FeatureFrame(
        X=features.X[start:],
        ts=features.ts[start:],
        ohlc=features.ohlc[start:],
        spread=features.spread[start:],
        atr=features.atr[start:],
        valid_from=0,
        columns=features.columns,
    )


def _bars_frame(features: FeatureFrame) -> pl.DataFrame:
    """Minimal bar frame for the rule engine.

    ``volume`` and the bid/ask closes are placeholders: :func:`classify` reads
    only ``ts`` and OHLC, and :func:`~scalping.data.base.conform_bars` requires
    the full schema. Filling them here keeps the call on the public rule engine
    instead of a private copy of its window math.
    """
    ohlc = np.asarray(features.ohlc, dtype=np.float64)
    close = ohlc[:, 3]
    spread = np.nan_to_num(
        np.asarray(features.spread, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    spread = np.clip(spread, 0.0, None)
    return pl.DataFrame(
        {
            "ts": np.asarray(features.ts, dtype="datetime64[us]"),
            "open": ohlc[:, 0],
            "high": ohlc[:, 1],
            "low": ohlc[:, 2],
            "close": close,
            "bid_close": close - spread / 2.0,
            "ask_close": close + spread / 2.0,
            "spread": spread,
            "volume": np.zeros(len(close), dtype=np.int64),
        }
    )


def _rule_labels(features: FeatureFrame) -> pl.DataFrame:
    """Causal rule labels for the evaluated window, one row per bar.

    The labels are computed from the sliced frame, so a bar's label uses only
    bars at or before it. The cost is that the rule windows warm up inside the
    split: the first bars fall back to ``normal`` volatility and ``sideways``
    trend rather than borrowing history from before the split.
    """
    return classify(_bars_frame(features)).select("ts", "regime")


def _normalise_labels(labels: pl.DataFrame) -> pl.DataFrame:
    """Reduce external labels to ``ts`` + a non-null string regime.

    Nulls, ``NaN`` and the usual string spellings of "missing" collapse to
    :data:`UNKNOWN_REGIME`, so a broken label becomes an explicit unmatched
    bucket instead of a regime named ``"NaN"``.
    """
    missing = [name for name in ("ts", "regime") if name not in labels.columns]
    if missing:
        raise ValueError(f"regime labels are missing columns: {missing}")
    frame = labels.select("ts", "regime")
    dtype = frame.schema["ts"]
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"regime labels 'ts' must be a datetime, got {dtype}")
    if dtype.time_zone is None:
        frame = frame.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    else:
        frame = frame.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    text = pl.col("regime").cast(pl.String)
    missing_value = text.is_null() | text.is_in(["", "nan", "NaN", "null", "None"])
    return frame.with_columns(
        pl.when(missing_value).then(pl.lit(UNKNOWN_REGIME)).otherwise(text).alias("regime")
    )


def _regime_breakdown(trades: Sequence[Trade], labels: pl.DataFrame) -> dict[str, Any]:
    """Per-regime table over the closed trades of one run.

    ``rows`` excludes the ``unknown`` bucket; those trades are counted in
    ``unmatched_trades`` instead, so ``sum(rows.trades) + unmatched_trades``
    always equals the headline trade count.
    """
    table = strategy_regime_table(trades, labels)
    if table.is_empty():
        return {"rows": [], "unmatched_trades": 0}

    unknown = table.filter(pl.col("regime") == UNKNOWN_REGIME)
    unmatched = int(unknown["trades"].sum()) if unknown.height else 0
    records: list[dict[str, Any]] = (
        table.filter(pl.col("regime") != UNKNOWN_REGIME).sort("regime").to_dicts()
    )
    rows: list[dict[str, Any]] = []
    assigned = 0
    for row in records:
        count = int(row["trades"])
        assigned += count
        rows.append(
            {
                "regime": str(row["regime"]),
                "trades": count,
                "win_rate": float(row["win_rate"]),
                "expectancy_r": float(row["expectancy_r"]),
                "total_r": float(row["total_r"]),
                "pnl": float(row["total_pnl"]),
                "profit_factor": (
                    None if row["profit_factor"] is None else float(row["profit_factor"])
                ),
            }
        )
    if assigned + unmatched != len(trades):
        raise RuntimeError(
            "regime breakdown does not cover the headline trades: "
            f"{assigned} assigned + {unmatched} unmatched != {len(trades)}"
        )
    return {"rows": rows, "unmatched_trades": unmatched}


def _metrics(result: Any) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "trades": int(metrics.trades),
        "net_profit": float(metrics.net_profit),
        "profit_factor": float(metrics.profit_factor),
        "win_rate": float(metrics.win_rate),
        "expectancy_r": float(metrics.expectancy_r),
        "max_drawdown": float(metrics.max_drawdown),
        "total_costs": float(metrics.total_costs),
    }


class _MLPolicy:
    """Probability threshold policy with a monotonic bar cursor."""

    def __init__(
        self,
        p_up: np.ndarray,
        p_down: np.ndarray,
        *,
        threshold: float,
        exit_threshold: float,
        window: int,
    ) -> None:
        self._p_up = p_up
        self._p_down = p_down
        self._threshold = threshold
        self._exit = exit_threshold
        self._index = max(0, window - 1)

    def __call__(self, obs: np.ndarray) -> int:
        index = self._index
        self._index += 1
        if index >= len(self._p_up):
            return _HOLD
        side = round(float(obs[-5]))  # position features: side is first
        up = float(self._p_up[index])
        down = float(self._p_down[index])
        if side == 0:
            if up >= self._threshold:
                return _BUY
            if down >= self._threshold:
                return _SELL
            return _HOLD
        if side > 0:  # long
            return _CLOSE if down >= self._exit else _HOLD
        return _CLOSE if up >= self._exit else _HOLD


def _probabilities(model: Any, result: MLResult, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(X)
    classes = [int(value) for value in model.classes_]
    p_up = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    p_down = probabilities[:, classes.index(-1)] if -1 in classes else np.zeros(len(X))
    return np.asarray(p_up, dtype=np.float64), np.asarray(p_down, dtype=np.float64)


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    """Abort between simulator passes when the caller asked to cancel."""
    if cancel_check is not None and cancel_check():
        raise InterruptedError("evaluation cancelled")


def evaluate_model(
    result: MLResult,
    features: FeatureFrame,
    settings: Settings,
    *,
    split: str = "validation",
    threshold: float = 0.55,
    exit_threshold: float = 0.55,
    window: int = 60,
    stress: tuple[float, ...] = (),
    regime_labels: pl.DataFrame | None = None,
    compute_regime: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> EvaluationResult:
    """Run the fitted classifier on one chronological split.

    ``regime_labels`` is an optional frame with ``ts`` and ``regime`` columns,
    joined as-of each trade's entry. When it is omitted and ``compute_regime``
    is true, the rule engine labels the same sliced bars the policy trades, so
    no label can use data after its bar. With ``compute_regime=False`` and no
    labels, ``EvaluationResult.regime`` stays empty; the CLI opts out that way
    to keep its default JSON output unchanged.

    The breakdown covers the base run only. Cost-stress runs repeat the same
    policy at higher costs and remain headline-only, because a regime bucket
    computed from stressed PnL would mix two different cost assumptions in one
    table.

    ``cancel_check`` is polled between simulator passes (before the base run,
    before hold, and between stress runs). A backtest pass itself is not
    interruptible; a caller that needs a hard stop must wait for the current
    pass. ``InterruptedError`` means the evaluation stopped on request and no
    result should be published.
    """
    if split not in _SPLITS:
        raise ValueError(f"unknown split {split!r}; choose train/validation/test")
    if not 0.5 <= threshold < 1.0 or not 0.5 <= exit_threshold <= 1.0:
        raise ValueError("thresholds must be in [0.5, 1)")
    if not 0.0 < window:
        raise ValueError("window must be positive")

    spec = result.spec
    bundle = build_dataset(
        features,
        symbol=result.symbol,
        timeframe=result.timeframe,
        groups=spec.groups,
        horizon=spec.horizon,
        train_fraction=spec.train_fraction,
        validation_fraction=spec.validation_fraction,
        purge_bars=max(spec.purge_bars, spec.horizon),
        embargo_bars=spec.embargo_bars,
    )
    mask = bundle.split == _SPLITS[split]
    if not mask.any():
        raise ValueError(f"the {split} split is empty")
    first = int(np.argmax(mask))
    start = int(features.valid_from) + first
    sliced = _slice_features(features, start)
    p_up, p_down = _probabilities(result.model, result, bundle.X)
    p_up = p_up[first:]
    p_down = p_down[first:]

    def run(multiplier: float) -> BacktestResult:
        active = sliced
        costs = settings.costs
        if multiplier != 1.0:
            active = replace(active, spread=active.spread * multiplier)
            costs = replace(
                costs,
                spread_floor=costs.spread_floor * multiplier,
                slippage_atr_frac=costs.slippage_atr_frac * multiplier,
                commission_per_lot=costs.commission_per_lot * multiplier,
            )
        config = replace(settings, costs=costs)
        policy = _MLPolicy(
            p_up,
            p_down,
            threshold=threshold,
            exit_threshold=exit_threshold,
            window=window,
        )
        return run_backtest(
            policy,
            active,
            config,
            policy_name=f"ml-{result.spec.model}-{split}",
            window=window,
            max_steps=len(active.ts) + 1,
        )

    _check_cancel(cancel_check)
    base = run(1.0)
    _check_cancel(cancel_check)
    hold = _metrics(
        run_backtest(
            lambda obs: 0,
            sliced,
            settings,
            policy_name="hold",
            window=window,
            max_steps=len(sliced.ts) + 1,
        )
    )
    _check_cancel(cancel_check)

    breakdown: dict[str, Any] = {}
    if regime_labels is not None:
        breakdown = _regime_breakdown(base.trades, _normalise_labels(regime_labels))
    elif compute_regime:
        breakdown = _regime_breakdown(base.trades, _rule_labels(sliced))

    stressed: dict[str, dict[str, Any]] = {}
    for value in stress:
        _check_cancel(cancel_check)
        stressed[f"x{value:g}"] = _metrics(run(value))

    return EvaluationResult(
        split=split,
        threshold=threshold,
        exit_threshold=exit_threshold,
        metrics=_metrics(base),
        hold=hold,
        stress=stressed,
        regime=breakdown,
    )
