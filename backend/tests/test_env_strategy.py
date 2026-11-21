"""Tests for the optional strategy context appended to an observation.

With a ``strategy_selector`` the environment appends six values to every
observation: a one-hot of the arm chosen for the bar the observation describes
plus that arm's signal strength on the same bar. Without a selector the
observation and its dimension are unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.trading_env import TradingEnv
from scalping.features.build import FeatureFrame, build_features
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import (
    CONTEXT_DIM,
    context_matrix,
    label_events,
    strength_matrix,
)
from scalping.strategy.signals import ALL_STRATEGIES

ARMS = len(ALL_STRATEGIES)


@lru_cache(maxsize=1)
def bars() -> pl.DataFrame:
    return SyntheticProvider(seed=5).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )


@lru_cache(maxsize=1)
def make() -> FeatureFrame:
    return build_features(bars())


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
    # The reset observation describes the first decision bar, so its arm is set.
    assert obs[-6:-1].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    obs, _, _, _, info = env.step(0)
    assert obs[-6:-1].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert info["strategy"] == "sweep"


def test_selected_arm_strength_is_appended():
    env = TradingEnv(make(), Settings(), strategy_selector=lambda ctx: 0)
    obs, _ = env.reset(seed=1)
    assert obs[-1] == pytest.approx(strengths()[env.sim.bar_index, 0])

    obs, _, _, _, info = env.step(0)
    assert info["strategy"] == "momentum"
    assert obs[-1] == pytest.approx(strengths()[env.sim.bar_index, 0])


def test_returned_obs_describes_the_current_bar():
    def pick(ctx: np.ndarray) -> int:
        return int(np.argmax(np.abs(ctx))) % ARMS

    env = TradingEnv(make(), Settings(), strategy_selector=pick)
    obs, _ = env.reset(seed=1)
    for _ in range(15):
        bar = env.sim.bar_index
        arm = pick(contexts()[bar])
        assert obs[-6:-1].sum() == pytest.approx(1.0)
        assert int(np.argmax(obs[-6:-1])) == arm
        assert obs[-1] == pytest.approx(strengths()[bar, arm])
        obs, *_ = env.step(0)


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
    obs: list[np.ndarray] = []
    while envs[0].sim.bar_index < bar:
        obs = [env.step(0)[0] for env in envs]

    assert obs[0][-1] == pytest.approx(momentum)
    assert obs[1][-1] == pytest.approx(reversal)
    assert obs[0][-1] != obs[1][-1]


def test_signal_firing_on_the_decision_bar_is_visible_immediately():
    matrix = strengths()
    env = TradingEnv(make(), Settings(), strategy_selector=lambda ctx: 0)
    env.reset(seed=1)
    start = env.sim.bar_index
    # A bar where momentum fires right after a bar where it did not: a one-bar
    # lag would show the previous bar's 0.0 instead of this bar's strength.
    bar = next(
        index
        for index in range(start + 1, len(matrix))
        if matrix[index, 0] > 0.0 and matrix[index - 1, 0] == 0.0
    )
    while env.sim.bar_index < bar:
        obs, *_ = env.step(0)

    assert env.sim.bar_index == bar
    assert obs[-1] > 0.0
    assert obs[-1] == pytest.approx(matrix[bar, 0])


def test_selector_is_called_once_per_decision_bar():
    calls: list[np.ndarray] = []

    def selector(ctx: np.ndarray) -> int:
        calls.append(ctx.copy())
        return 0

    # ``reset`` runs inside the constructor and already chooses the first bar,
    # so the selector must not depend on ``env`` existing yet.
    env = TradingEnv(make(), Settings(), strategy_selector=selector)
    first = env.sim.bar_index
    env.step(0)
    second = env.sim.bar_index
    env.step(0)
    third = env.sim.bar_index

    assert len(calls) == 3  # reset plus one per step, no duplicates
    assert (first, second, third) == (first, first + 1, first + 2)
    for ctx, bar in zip(calls, (first, second, third)):
        assert np.array_equal(ctx, contexts()[bar])


def test_labels_price_the_selector_context():
    seen: list[np.ndarray] = []

    def selector(ctx: np.ndarray) -> int:
        seen.append(ctx.copy())
        return 0

    env = TradingEnv(
        make(),
        Settings(),
        strategy_selector=selector,
        labels=label_events(bars(), Settings()).head(0),
    )
    env.reset(seed=1)

    assert np.allclose(seen[0][-ARMS:], 0.5)
    # The same row built from the realised labels is not neutral.
    assert not np.allclose(contexts()[env.sim.bar_index][-ARMS:], 0.5)


def test_strength_matrix_rejects_a_misaligned_frame():
    frame = make()
    order = np.arange(len(frame.X))[::-1]
    shuffled = FeatureFrame(
        X=frame.X[order],
        ts=frame.ts[order],
        ohlc=frame.ohlc[order],
        spread=frame.spread[order],
        atr=frame.atr[order],
        valid_from=frame.valid_from,
        columns=list(frame.columns),
    )
    with pytest.raises(ValueError, match="not aligned"):
        strength_matrix(shuffled)
