"""Train independent seeds through one injected trainer.

The trainer itself is a callable ``seed -> result``; this module only owns the
seed list contract, ordering and duplicate checks. It deliberately does not
import stable-baselines3 or build a PPO model, so it can be unit-tested and
reused by scripts without a GPU or a run directory.

Training N seeds is not an ensemble by itself: the caller still evaluates the
saved policies with :func:`scalping.rl.seed_eval.evaluate_seed_ensemble` and
checks out-of-sample/cost-stress evidence before treating the average as better.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

__all__ = ["SeedRun", "train_seed_runs"]

@dataclass(frozen=True, slots=True)
class SeedRun[T]:
    """One completed seed and whatever its trainer returned."""

    seed: int
    result: T


def train_seed_runs[T](
    train: Callable[[int], T],
    seeds: Iterable[int],
) -> tuple[SeedRun[T], ...]:
    """Call ``train(seed)`` once per unique seed, in the given order.

    Seeds must be non-negative integers; bools and duplicates are rejected
    rather than silently collapsed, because two runs with the same seed are not
    independent evidence. A trainer exception propagates unchanged after the
    earlier runs have already happened; this function does not retry or hide it.
    """
    if not callable(train):
        raise TypeError("train must be callable")
    ordered: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        if seed in seen:
            raise ValueError(f"duplicate seed {seed}; every member needs its own seed")
        seen.add(seed)
        ordered.append(seed)
    if not ordered:
        raise ValueError("need at least one seed")

    runs: list[SeedRun[T]] = []
    for seed in ordered:
        runs.append(SeedRun(seed=seed, result=train(seed)))
    return tuple(runs)
