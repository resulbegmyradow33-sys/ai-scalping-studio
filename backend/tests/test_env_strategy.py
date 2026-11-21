"""Tests for the optional strategy context appended to an observation.

With a ``strategy_selector`` the environment appends six values to every
observation: a one-hot of the arm the selector chose for the decision bar plus
that arm's signal strength on the same bar. Without a selector the observation
and its dimension are unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
import pytest

from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.trading_env import TradingEnv
from scalping.features.build import FeatureFrame, build_features
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM, context_matrix, strength_matrix
from scalping.strategy.signals import ALL_STRATEGIES

ARMS = len(ALL_STRATEGIES)


@lru_cache(maxsize=1)
def make() -> FeatureFrame:
    df = SyntheticProvider(seed=5).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )
    return build_features(df)


@lru_cache(maxsize=1)
def strengths() -> np.ndarray:
    return strength_matrix(make())


@lru_cache(maxsize=1)
def contexts() -> np.ndarray:
    return context_matrix(make())


def test_obs_dim_grows_by_six_with_selector():
    plain = TradingEnv(make(), Settings())
    with_sel = TradingEnv(make(), Settings(), strategy_selector=LinUCB(dim=CONTEXT_DIM))
    assert with_sel.obs_dim == plain.obs_dim + 6


def test_selected_strategy_is_reported():
    env = TradingEnv(make(), Settings(), strategy_selector=LinUCB(dim=CONTEXT_DIM))
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert info["strategy"] in {"momentum", "reversal", "sweep", "fvg", "trend"}


def test_obs_remains_finite():
    env = TradingEnv(make(), Settings(), strategy_selector=LinUCB(dim=CONTEXT_DIM))
    obs, _ = env.reset(seed=1)
    assert np.isfinite(obs).all()


def test_onehot_matches_the_selected_arm():
    env = TradingEnv(make(), Settings(), strategy_selector=lambda ctx: 2)
    obs, _ = env.reset(seed=1)
    # No decision has been made yet, so reset carries no arm.
    assert obs[-6:-1].tolist() == [0.0] * ARMS

    obs, _, _, _, info = env.step(0)
    assert obs[-6:-1].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert info["strategy"] == "sweep"


def test_selected_arm_strength_is_appended():
    env = TradingEnv(make(), Settings(), strategy_selector=lambda ctx: 0)
    env.reset(seed=1)
    bar = env.sim.bar_index
    obs, _, _, _, info = env.step(0)
    assert info["strategy"] == "momentum"
    assert obs[-1] == pytest.approx(strengths()[bar, 0])


def test_stronger_arm_appends_its_own_strength():
    envs = [
        TradingEnv(make(), Settings(), strategy_selector=lambda ctx, arm=arm: arm) for arm in (0, 1)
    ]
    for env in envs:
        env.reset(seed=1)

    # A reachable bar where momentum and reversal disagree, momentum firing.
    matrix = strengths()
    start = envs[0].sim.bar_index
    bar, momentum, reversal = next(
        (index, row[0], row[1])
        for index, row in enumerate(matrix)
        if index >= start and row[0] > 0.0 and row[0] != row[1]
    )
    while envs[0].sim.bar_index < bar:
        for env in envs:
            env.step(0)

    first, *_ = envs[0].step(0)
    second, *_ = envs[1].step(0)
    assert first[-1] == pytest.approx(momentum)
    assert second[-1] == pytest.approx(reversal)
    assert first[-1] != second[-1]


def test_selector_is_called_once_per_step_with_the_current_bar():
    calls: list[tuple[int, np.ndarray]] = []

    def selector(ctx: np.ndarray) -> int:
        calls.append((env.sim.bar_index, ctx.copy()))
        return 0

    env = TradingEnv(make(), Settings(), strategy_selector=selector)
    env.reset(seed=1)
    assert calls == []  # reset makes no decision

    env.step(0)
    first = env.sim.bar_index
    env.step(0)
    second = env.sim.bar_index

    assert [bar for bar, _ in calls] == [first - 1, first]
    assert np.array_equal(calls[0][1], contexts()[first - 1])
    assert np.array_equal(calls[1][1], contexts()[second - 1])
