"""Monte Carlo trade-sequence analysis over one closed-trade list.

A backtest is a single ordering of a finite set of trades. That ordering decided
the equity path, and therefore the drawdown, a fill sequence the strategy did
not choose. This module asks a narrow question: *given the trades we observed,
how much does their order matter?* It reorders (``shuffle``) or resamples
(``bootstrap``) the trades, compounds each path from a starting balance and
reports the distribution of final P&L, maximum drawdown and ruin.

Conventions
-----------
* Each trade's P&L is treated as a return on the **starting** balance:
  ``r_i = pnl_i / starting_balance``. A path compounds multiplicatively,
  ``balance_k = starting_balance * prod(1 + r_i)``. That keeps the observed
  dollar P&L scale while letting gains and losses compound; it is not a
  re-simulation with a different lot size.
* A balance that reaches zero stops compounding: the path stays at zero and the
  remaining trades are no-ops. The broker model can go that far only if the
  caller passes hand-made trades; a real run is protected by the risk layer.
* ``max_drawdown`` is the deepest drop from a running peak as a positive
  fraction of that peak, measured from the starting balance through every
  post-trade balance, clamped to ``[0, 1]``. It matches
  :func:`~scalping.backtest.metrics.compute`.
* A path is **ruined** when any post-trade balance is at or below
  ``ruin_fraction * starting_balance``. Ruin is a touch, not an ending state:
  a path that recovers still counts, because the account would have breached
  the floor on the way.
* ``n_paths`` paths are generated sequentially from one
  ``np.random.default_rng(seed)``, so a seed fully determines every path and
  repeated calls return identical results.

Order invariance of terminal wealth
-----------------------------------
With multiplicative compounding, a shuffled path ends at the **same** balance
as the original: the product of ``(1 + r_i)`` does not depend on order. That is
arithmetic, not a bug. Under ``shuffle`` the terminal-P&L percentiles are
therefore flat and the information is in ``max_drawdown`` and the ruin rate;
under ``bootstrap`` the sampled multiset changes, so terminal P&L varies too.
Read the shuffle report as "how bad could the path have been with the same
trades", never as a P&L confidence interval.

IID vs block bootstrap
----------------------
``bootstrap`` draws single trades with replacement (IID). It is the standard
answer to "what if the winners and losers came in a different order", and it is
wrong in one specific way: it destroys **streak clustering**. If the strategy's
losses arrive in runs (a losing regime, a widening spread, a news session),
IID resampling scatters them and makes the drawdown distribution look milder
than the data supports. ``bootstrap`` with a ``block_size`` greater than one
uses a circular block bootstrap: it samples contiguous runs of trades and
concatenates them, so short-range dependence survives inside a block. It still
cannot preserve dependence longer than the block, and picking the block size is
a modelling choice, not a measurement. A block size of one is exactly the IID
bootstrap.

What this does not prove
------------------------
The trades are the ones a single historical run produced. Resampling them
cannot tell you how the strategy behaves on unseen data, cannot repair a
selection bias in which run was chosen, and reuses each trade's realised costs
instead of re-pricing them. A narrow Monte Carlo distribution means the result
is insensitive to ordering, not that the strategy is profitable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from scalping.backtest.engine import BacktestResult
from scalping.types import Trade

__all__ = [
    "MonteCarloConfig",
    "MonteCarloResult",
    "PathOutcome",
    "run_montecarlo",
]

Mode = Literal["shuffle", "bootstrap"]

_MODES: Final[tuple[str, ...]] = ("shuffle", "bootstrap")


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """How to resample, how many paths, and where ruin sits.

    ``block_size`` is only read by ``mode="bootstrap"``; one means the IID
    bootstrap, larger values use a circular block bootstrap. ``ruin_fraction``
    is measured against ``starting_balance`` and must be in ``[0, 1)``: at
    ``1.0`` every path would be ruined before the first trade. Confidence
    levels are quantiles in ``(0, 1)`` and become ``p05``/``p50``/``p95`` style
    keys.
    """

    n_paths: int = 1_000
    seed: int = 0
    mode: Mode = "shuffle"
    block_size: int = 5
    confidence_levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    starting_balance: float = 10_000.0
    ruin_fraction: float = 0.0

    def __post_init__(self) -> None:
        if int(self.n_paths) < 1:
            raise ValueError(f"n_paths must be >= 1, got {self.n_paths!r}")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if int(self.block_size) < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size!r}")
        if not np.isfinite(self.starting_balance) or self.starting_balance <= 0.0:
            raise ValueError(
                f"starting_balance must be finite and > 0, got {self.starting_balance!r}"
            )
        if not 0.0 <= self.ruin_fraction < 1.0:
            raise ValueError(f"ruin_fraction must be in [0, 1), got {self.ruin_fraction!r}")
        levels = tuple(float(level) for level in self.confidence_levels)
        if not levels:
            raise ValueError("confidence_levels must not be empty")
        if any(not 0.0 < level < 1.0 for level in levels):
            raise ValueError(f"confidence levels must be in (0, 1), got {self.confidence_levels!r}")
        labels = [_level_label(level) for level in levels]
        if len(set(labels)) != len(labels):
            raise ValueError(f"confidence levels collide in percentile labels: {labels!r}")


@dataclass(frozen=True, slots=True)
class PathOutcome:
    """One resampled sequence and the balance path it produced."""

    index: int
    final_pnl: float
    ending_balance: float
    max_drawdown: float
    ruined: bool


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Distribution over the resampled paths, plus the raw per-path outcomes.

    ``paths`` is the number of paths actually generated: ``0`` for an empty
    trade list, otherwise ``config.n_paths``. ``degenerate`` marks an input that
    has no sequence risk: no trades, or exactly one, where every path is the
    same sequence. ``worst_path`` is the path with the lowest ending balance
    (ties go to the lowest index). Percentile dicts use the keys from
    ``confidence_levels`` and are ``0.0`` for an empty input.
    """

    mode: Mode
    seed: int
    paths: int
    trades: int
    block_size: int
    confidence_levels: tuple[float, ...]
    starting_balance: float
    ruin_threshold: float
    probability_of_ruin: float
    outcomes: tuple[PathOutcome, ...]
    final_pnl_percentiles: dict[str, float]
    max_drawdown_percentiles: dict[str, float]
    ending_balance_percentiles: dict[str, float]
    worst_path: PathOutcome | None
    degenerate: bool


def _level_label(level: float) -> str:
    """``0.05`` -> ``"p05"``; the label is what the JSON report exposes."""
    return f"p{round(level * 100):02d}"


def _percentiles(values: np.ndarray, levels: Sequence[float]) -> dict[str, float]:
    """Linear-interpolated quantiles, keyed by their percentile label."""
    return {_level_label(level): float(np.percentile(values, level * 100.0)) for level in levels}


def _equity_path(pnls: np.ndarray, starting_balance: float) -> np.ndarray:
    """Compound ``pnls`` in order as returns on ``starting_balance``.

    The first non-positive balance ends the path: the remaining entries are
    zeros, because a broker account cannot trade through zero.
    """
    balances = np.zeros(pnls.size, dtype=np.float64)
    balance = float(starting_balance)
    for position, pnl in enumerate(pnls):
        if balance <= 0.0:
            break
        balance *= 1.0 + float(pnl) / starting_balance
        balance = max(balance, 0.0)
        balances[position] = balance
    return balances


def _max_drawdown(balances: np.ndarray, starting_balance: float) -> float:
    """Deepest peak-to-balance drop as a positive fraction of the peak."""
    curve = np.concatenate(([float(starting_balance)], balances))
    peaks = np.maximum.accumulate(curve)
    drawdowns = np.where(
        peaks > 0.0,
        (peaks - curve) / np.where(peaks > 0.0, peaks, 1.0),
        0.0,
    )
    return float(np.clip(np.max(drawdowns), 0.0, 1.0))


def _path_indices(n: int, config: MonteCarloConfig) -> list[np.ndarray]:
    """Index sequences for every path; deterministic for ``config.seed``.

    ``shuffle`` draws a permutation of the whole list. ``bootstrap`` draws
    ``n`` indices with replacement. With ``block_size > 1`` the same mode draws
    circular contiguous blocks of that size and truncates the concatenation to
    ``n``; a block longer than the sample is clamped to the sample length, so
    the block never wraps over the same trade twice in a row.
    """
    if n < 1:
        return []
    rng = np.random.default_rng(config.seed)
    paths: list[np.ndarray] = []
    if config.mode == "shuffle":
        for _ in range(config.n_paths):
            paths.append(rng.permutation(n))
        return paths

    block = min(int(config.block_size), n)
    for _ in range(config.n_paths):
        chunks: list[np.ndarray] = []
        assembled = 0
        while assembled < n:
            start = int(rng.integers(0, n))
            chunks.append((start + np.arange(block)) % n)
            assembled += block
        paths.append(np.concatenate(chunks)[:n])
    return paths


def _empty_result(config: MonteCarloConfig, levels: tuple[float, ...]) -> MonteCarloResult:
    """Documented degenerate report for an input with no trades."""
    zeros = {_level_label(level): 0.0 for level in levels}
    return MonteCarloResult(
        mode=config.mode,
        seed=int(config.seed),
        paths=0,
        trades=0,
        block_size=int(config.block_size),
        confidence_levels=levels,
        starting_balance=float(config.starting_balance),
        ruin_threshold=float(config.ruin_fraction * config.starting_balance),
        probability_of_ruin=0.0,
        outcomes=(),
        final_pnl_percentiles=dict(zeros),
        max_drawdown_percentiles=dict(zeros),
        ending_balance_percentiles=dict(zeros),
        worst_path=None,
        degenerate=True,
    )


def run_montecarlo(
    trades: Sequence[Trade] | BacktestResult,
    config: MonteCarloConfig | None = None,
) -> MonteCarloResult:
    """Resample ``trades`` and report the distribution of path outcomes.

    ``trades`` may be the ordered closed-trade list or a
    :class:`~scalping.backtest.engine.BacktestResult`, whose ``trades`` are
    used. The caller's list is never mutated or reordered in place. An empty
    list returns a documented zero-path result instead of raising; a single
    trade returns ``n_paths`` identical paths and sets ``degenerate`` because
    there is no sequence left to vary.
    """
    settings = config if config is not None else MonteCarloConfig()
    ordered = list(trades.trades) if isinstance(trades, BacktestResult) else list(trades)
    levels = tuple(float(level) for level in settings.confidence_levels)
    if not ordered:
        return _empty_result(settings, levels)

    pnls = np.asarray([trade.pnl for trade in ordered], dtype=np.float64)
    starting = float(settings.starting_balance)
    threshold = float(settings.ruin_fraction * settings.starting_balance)

    outcomes: list[PathOutcome] = []
    for index, indices in enumerate(_path_indices(pnls.size, settings)):
        balances = _equity_path(pnls[indices], starting)
        ending = float(balances[-1])
        outcomes.append(
            PathOutcome(
                index=index,
                final_pnl=ending - starting,
                ending_balance=ending,
                max_drawdown=_max_drawdown(balances, starting),
                ruined=bool(float(np.min(balances)) <= threshold),
            )
        )

    final_pnl = np.asarray([outcome.final_pnl for outcome in outcomes], dtype=np.float64)
    drawdowns = np.asarray([outcome.max_drawdown for outcome in outcomes], dtype=np.float64)
    endings = np.asarray([outcome.ending_balance for outcome in outcomes], dtype=np.float64)
    worst = min(outcomes, key=lambda outcome: (outcome.ending_balance, outcome.index))
    return MonteCarloResult(
        mode=settings.mode,
        seed=int(settings.seed),
        paths=len(outcomes),
        trades=int(pnls.size),
        block_size=int(settings.block_size),
        confidence_levels=levels,
        starting_balance=starting,
        ruin_threshold=threshold,
        probability_of_ruin=sum(outcome.ruined for outcome in outcomes) / len(outcomes),
        outcomes=tuple(outcomes),
        final_pnl_percentiles=_percentiles(final_pnl, levels),
        max_drawdown_percentiles=_percentiles(drawdowns, levels),
        ending_balance_percentiles=_percentiles(endings, levels),
        worst_path=worst,
        degenerate=pnls.size <= 1,
    )
