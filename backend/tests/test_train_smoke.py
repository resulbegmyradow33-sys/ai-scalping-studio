"""End-to-end PPO smoke: train a tiny model, then load and use its run directory."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.trading_env import TradingEnv
from scalping.features.build import FeatureFrame, build_features
from scalping.rl.policy import load_policy
from scalping.rl.registry import run_id_for
from scalping.rl.train import TrainConfig, TrainResult, data_hash_for, train
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM
from scalping.strategy.signals import ALL_STRATEGIES

SYMBOL = "XAUUSD"
START = datetime(2026, 1, 5, tzinfo=UTC)
#: Monday 00:00 through Thursday 00:00 — ~4 320 M1 bars, above the 4 000 floor.
END = datetime(2026, 1, 8, tzinfo=UTC)

METRIC_KEYS = (
    "step",
    "reward_mean",
    "ep_len_mean",
    "entropy_loss",
    "policy_loss",
    "value_loss",
    "action_dist",
)


def _frame(seed: int = 5) -> FeatureFrame:
    return build_features(SyntheticProvider(seed=seed).fetch(SYMBOL, START, END))


@pytest.mark.slow
def test_training_produces_loadable_model(tmp_path):
    df = SyntheticProvider(seed=5).fetch(
        "XAUUSD",
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 8, tzinfo=UTC),
    )
    f = build_features(df)
    cfg = TrainConfig(
        run_name="smoke",
        total_steps=1024,
        n_envs=1,
        n_steps=256,
        batch_size=64,
        checkpoint_every=512,
        data_hash="smoke",
    )
    res = train(cfg, f, Settings(), out_dir=tmp_path)
    assert (tmp_path / res.run_id / "model.zip").exists()
    metrics = json.loads((tmp_path / res.run_id / "metrics.json").read_text())
    assert np.isfinite(metrics["reward_mean"])
    assert sum(metrics["action_dist"]) == pytest.approx(1.0, abs=0.05)


@pytest.mark.slow
def test_training_refuses_tiny_dataset(tmp_path):
    df = SyntheticProvider(seed=5).fetch(
        "XAUUSD",
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 5, 2, tzinfo=UTC),
    )
    f = build_features(df)
    with pytest.raises(ValueError, match="not enough bars"):
        train(
            TrainConfig(run_name="tiny", total_steps=256, data_hash="x"),
            f,
            Settings(),
            out_dir=tmp_path,
        )


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> tuple[Path, FeatureFrame, TrainResult]:
    """One short training run shared by the inference tests."""
    out = tmp_path_factory.mktemp("rl-policy")
    features = _frame()
    cfg = TrainConfig(
        run_name="policy",
        total_steps=512,
        n_envs=1,
        n_steps=256,
        batch_size=64,
        checkpoint_every=256,
    )
    return out, features, train(cfg, features, Settings(), out_dir=out)


@pytest.mark.slow
def test_load_policy_act_is_bounded_and_deterministic(trained_run):
    _, features, result = trained_run
    policy = load_policy(result.model_path.parent)
    env = TradingEnv(features, Settings())
    obs, _ = env.reset(seed=0)
    first, second = policy.act(obs), policy.act(obs)
    assert first == second
    assert 0 <= first <= 3
    assert policy.act(obs.reshape(1, -1)) == first
    assert policy(obs) == first


@pytest.mark.slow
def test_load_policy_works_from_a_fresh_process(trained_run):
    _, _, result = trained_run
    script = """
import json, sys
from pathlib import Path

import numpy as np

from scalping.rl.policy import load_policy

run = Path(sys.argv[1])
config = json.loads((run / "config.json").read_text())
policy = load_policy(run)
obs = np.zeros(config["obs_dim"], dtype=np.float32)
first, second = policy.act(obs), policy.act(obs)
assert first == second and 0 <= first <= 3, (first, second)
print(first)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(result.model_path.parent)],
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.slow
def test_metrics_snapshot_has_every_documented_key(trained_run):
    _, _, result = trained_run
    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["step"] == result.steps == 512
    for key in METRIC_KEYS:
        assert key in metrics, key
    assert len(metrics["action_dist"]) == 4
    assert sum(metrics["action_dist"]) == pytest.approx(1.0)
    assert np.isfinite(metrics["reward_mean"])
    # Two 256-step chunks: a cumulative callback sees 512 steps in one episode,
    # a callback reset by each ``learn`` call would report 256.
    assert metrics["ep_len_mean"] == pytest.approx(512.0)


@pytest.mark.slow
def test_run_id_is_reproducible_from_the_written_config(trained_run):
    _, features, result = trained_run
    config = json.loads((result.model_path.parent / "config.json").read_text())
    assert config["data_hash"] == data_hash_for(features)
    assert run_id_for(config["params"], config["data_hash"], config["seed"]) == result.run_id


@pytest.mark.slow
def test_frozen_bandit_is_saved_beside_the_model(tmp_path):
    features = _frame()
    bandit = LinUCB(dim=CONTEXT_DIM)
    before = bandit.arm_stats()
    cfg = TrainConfig(
        run_name="bandit",
        total_steps=256,
        n_envs=1,
        n_steps=256,
        batch_size=64,
        checkpoint_every=256,
    )
    res = train(cfg, features, Settings(), selector=bandit, out_dir=tmp_path)
    run_dir = res.model_path.parent

    bandit_path = run_dir / "bandit.npz"
    assert bandit_path.exists()
    assert LinUCB.load(bandit_path).arm_stats() == before, "PPO must not train the bandit"

    config = json.loads((run_dir / "config.json").read_text())
    expected_dim = 60 * features.X.shape[1] + 5 + len(ALL_STRATEGIES) + 1
    assert config["obs_dim"] == expected_dim
    assert load_policy(run_dir).obs_dim == expected_dim
