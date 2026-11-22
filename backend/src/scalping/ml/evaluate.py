"""Trade the classifier in the paper simulator.

Prediction metrics are not trading metrics. This module turns a fitted
classifier into a deterministic policy, runs it on one chronological split with
the existing PaperBroker/RiskEngine, and reports the same costs every other
strategy pays.

The policy is deliberately simple: when flat, enter in the direction whose
probability clears ``threshold``; when in a position, close if the opposite
probability clears ``exit_threshold``. The simulator still owns sizing, stops,
targets, the time stop and every risk block. A classifier cannot bypass them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from scalping.backtest.engine import run_backtest
from scalping.config import Settings
from scalping.datasets.builder import build_dataset
from scalping.features.build import FeatureFrame
from scalping.ml.train import MLResult

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
) -> EvaluationResult:
    """Run the fitted classifier on one chronological split."""
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

    def run(multiplier: float) -> dict[str, Any]:
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
        return _metrics(
            run_backtest(
                policy,
                active,
                config,
                policy_name=f"ml-{result.spec.model}-{split}",
                window=window,
                max_steps=len(active.ts) + 1,
            )
        )

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
    return EvaluationResult(
        split=split,
        threshold=threshold,
        exit_threshold=exit_threshold,
        metrics=run(1.0),
        hold=hold,
        stress={f"x{value:g}": run(value) for value in stress},
    )
