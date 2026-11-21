"""LinUCB bandit: closed-form ridge, deterministic selection, safe persistence.

The load-bearing properties: ``select`` is a pure argmax (lowest arm index on a
tie), a rewarded arm takes over the picks, the offline fit recovers the arm the
history paid for, and a saved model reloads into an identical one without
touching the live object.
"""

from __future__ import annotations

import numpy as np
import pytest

from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM


def test_selects_in_range():
    b = LinUCB(dim=4)
    assert 0 <= b.select(np.ones(4)) < 5


def test_learns_dominant_arm_online():
    rng = np.random.default_rng(0)
    b = LinUCB(dim=4, alpha=0.1)
    for _ in range(200):
        x = rng.normal(size=4)
        a = b.select(x)
        b.update(a, x, 1.0 if a == 2 else 0.0)
    picks = [b.select(rng.normal(size=4)) for _ in range(50)]
    assert picks.count(2) >= 40


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_learns_dominant_arm_online_across_seeds(seed):
    rng = np.random.default_rng(seed)
    b = LinUCB(dim=4, alpha=0.1)
    for _ in range(200):
        x = rng.normal(size=4)
        a = b.select(x)
        b.update(a, x, 1.0 if a == 2 else 0.0)
    picks = [b.select(rng.normal(size=4)) for _ in range(50)]
    assert picks.count(2) >= 40


def test_offline_fit_recovers_best_arm():
    rng = np.random.default_rng(1)
    X, arms, rewards = [], [], []
    for _ in range(1000):
        arm = rng.integers(0, 5)
        x = rng.normal(size=4)
        X.append(x)
        arms.append(arm)
        rewards.append(1.0 if arm == 3 else 0.0)
    b = LinUCB(dim=4).fit_offline(np.array(X), np.array(arms), np.array(rewards))
    assert b.select(np.array([0.1, 0.2, 0.3, 0.4])) == 3


def test_save_load_roundtrip(tmp_path):
    b = LinUCB(dim=4)
    b.update(0, np.ones(4), 1.0)
    p = tmp_path / "bandit.npz"
    b.save(p)
    c = LinUCB.load(p)
    assert np.allclose(b.weights(), c.weights())


def test_empty_fit_does_not_crash():
    b = LinUCB(dim=4).fit_offline(np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))
    assert 0 <= b.select(np.ones(4)) < 5


def test_nan_context_rejected():
    b = LinUCB(dim=4)
    bad = np.ones(4)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        b.select(bad)
    with pytest.raises(ValueError, match="non-finite"):
        b.update(0, bad, 1.0)
    with pytest.raises(ValueError, match="finite"):
        b.update(0, np.ones(4), np.inf)
    bad_batch = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, np.inf, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        b.fit_offline(bad_batch, np.array([0, 1]), np.array([1.0, 0.0]))


def test_dim_mismatch_rejected():
    b = LinUCB(dim=4)
    with pytest.raises(ValueError, match="dim=4"):
        b.select(np.ones(5))
    with pytest.raises(ValueError, match="dim=4"):
        b.update(0, np.ones(3), 1.0)
    with pytest.raises(ValueError, match="dim=4"):
        b.fit_offline(np.zeros((2, 5)), np.zeros(2, dtype=int), np.zeros(2))


def test_select_tie_break_is_lowest_arm():
    b = LinUCB(dim=4, n_arms=5)
    x = np.array([0.3, -0.2, 0.5, 0.1])
    # A fresh bandit has five identical arms: every score ties.
    assert {b.select(x) for _ in range(20)} == {0}


def test_untouched_arms_are_explored_in_order():
    b = LinUCB(dim=4, n_arms=5, alpha=0.5)
    x = np.array([0.3, -0.2, 0.5, 0.1])
    seen = []
    for _ in range(5):
        a = b.select(x)
        seen.append(a)
        b.update(a, x, 0.0)
    assert seen == [0, 1, 2, 3, 4]


def test_arm_out_of_range_rejected():
    b = LinUCB(dim=4)
    with pytest.raises(ValueError, match="arm"):
        b.update(5, np.ones(4), 1.0)
    with pytest.raises(ValueError, match="arm"):
        b.update(-1, np.ones(4), 1.0)


def test_weights_shape_and_arm_stats():
    b = LinUCB(dim=4)
    b.update(1, np.ones(4), 1.0)
    b.update(1, np.full(4, 2.0), 0.0)
    assert b.weights().shape == (5, 4)
    stats = b.arm_stats()
    assert len(stats) == 5
    assert stats[1]["pulls"] == 2
    assert stats[1]["mean_reward"] == pytest.approx(0.5)
    assert stats[0]["pulls"] == 0
    assert stats[0]["mean_reward"] == 0.0


def test_default_dim_is_context_dim():
    assert LinUCB(n_arms=5).weights().shape == (5, CONTEXT_DIM)


def test_load_into_different_dim_raises(tmp_path):
    source = LinUCB(dim=4)
    source.update(0, np.ones(4), 1.0)
    p = source.save(tmp_path / "bandit.npz")

    with pytest.raises(ValueError, match="dim"):
        LinUCB(dim=5).load_into(p)
    with pytest.raises(ValueError, match="n_arms"):
        LinUCB(dim=4, n_arms=4).load_into(p)


def test_load_rejects_metadata_shape_mismatch(tmp_path):
    b = LinUCB(dim=4)
    payload = {
        "dim": np.array(3),
        "n_arms": np.array(5),
        "alpha": np.array(0.5),
        "ridge": np.array(1.0),
        "A": b._A.copy(),
        "b": b._b.copy(),
        "pulls": b._pulls.copy(),
        "reward_sum": b._reward_sum.copy(),
    }
    p = tmp_path / "tampered.npz"
    np.savez(p, **payload)
    with pytest.raises(ValueError, match="dim"):
        LinUCB.load(p)


def test_save_does_not_mutate_and_load_is_independent(tmp_path):
    b = LinUCB(dim=4)
    b.update(2, np.ones(4), 1.0)
    before = (b._A.copy(), b._b.copy(), b._pulls.copy(), b._reward_sum.copy())

    p = b.save(tmp_path / "bandit.npz")
    loaded = LinUCB.load(p)

    for after, expected in zip((b._A, b._b, b._pulls, b._reward_sum), before, strict=True):
        assert np.array_equal(after, expected)

    loaded.update(2, np.ones(4), 5.0)
    assert not np.array_equal(loaded.weights(), b.weights())


def test_save_appends_missing_npz_suffix(tmp_path):
    b = LinUCB(dim=4)
    b.update(0, np.ones(4), 1.0)
    p = b.save(tmp_path / "bandit")
    assert p.name == "bandit.npz"
    assert np.allclose(b.weights(), LinUCB.load(tmp_path / "bandit").weights())


def test_save_load_preserves_selection(tmp_path):
    rng = np.random.default_rng(4)
    b = LinUCB(dim=4, alpha=0.1)
    for _ in range(50):
        x = rng.normal(size=4)
        b.update(3, x, 1.0)
    p = b.save(tmp_path / "bandit.npz")
    c = LinUCB.load(p)
    probes = [rng.normal(size=4) for _ in range(20)]
    assert [b.select(x) for x in probes] == [c.select(x) for x in probes]


def test_offline_fit_accumulates_each_arm_in_closed_form():
    x = np.array([1.0, 0.0])
    X = np.array([x, 2.0 * x])
    arms = np.array([0, 0])
    rewards = np.array([1.0, 1.0])
    b = LinUCB(dim=2, n_arms=2, alpha=0.0, ridge=1.0).fit_offline(X, arms, rewards)
    # Designs are [1, 0, 1] and [2, 0, 1] (the bias column stays 1.0), so
    # A = I + z1 z1' + z2 z2' and b = z1 + z2, theta = A^-1 b.
    z1 = np.array([1.0, 0.0, 1.0])
    z2 = np.array([2.0, 0.0, 1.0])
    A = np.eye(3) + np.outer(z1, z1) + np.outer(z2, z2)
    expected = np.linalg.solve(A, z1 + z2)
    assert np.allclose(b.weights()[0], expected[:2])
    assert b.arm_stats()[0]["intercept"] == pytest.approx(expected[2])


def test_invalid_hyperparameters_rejected():
    with pytest.raises(ValueError):
        LinUCB(dim=0)
    with pytest.raises(ValueError):
        LinUCB(n_arms=0)
    with pytest.raises(ValueError):
        LinUCB(alpha=-0.1)
    with pytest.raises(ValueError):
        LinUCB(ridge=0.0)
