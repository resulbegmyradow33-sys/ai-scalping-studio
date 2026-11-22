"""End-to-end PPO smoke: train tiny models, inspect the run, load and use them.

Every training test is ``slow``; the pure environment/identity tests stay fast.
Runs are small (256-1 024 steps) so the whole file stays around ten seconds on
the target Mac mini.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scalping.config import CostConfig, Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.env.trading_env import TradingEnv
from scalping.features.build import FeatureFrame, build_features
from scalping.rl.policy import load_policy
from scalping.rl.registry import BANDIT_NAME, load_run, run_id_for
from scalping.rl.train import (
    TrainConfig,
    TrainResult,
    data_hash_for,
    env_starts_for,
    run_identity,
    train,
)
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM, bandit_training_data, label_events
from scalping.strategy.signals import ALL_STRATEGIES

SYMBOL = "XAUUSD"
START = datetime(2026, 1, 5, tzinfo=UTC)
#: Monday 00:00 through Thursday 00:00 — ~4 320 M1 bars, above the 4 000 floor.
END = datetime(2026, 1, 8, tzinfo=UTC)

METRIC_KEYS = (
    "step",
    "reward_mean",
    "reward_mean_interval",
    "ep_len_mean",
    "entropy_loss",
    "policy_loss",
    "value_loss",
    "action_dist",
)


def _frame(seed: int = 5) -> FeatureFrame:
    return build_features(SyntheticProvider(seed=seed).fetch(SYMBOL, START, END))


def _train(tmp_path: Path, **overrides) -> TrainResult:
    train_keys = {
        "stop_event",
        "pause_event",
        "train_end",
        "eval_start",
        "selector",
        "labels",
        "teacher_actions",
    }
    train_kwargs = {key: overrides.pop(key) for key in list(overrides) if key in train_keys}
    fields = {
        "run_name": "smoke",
        "total_steps": 256,
        "n_envs": 1,
        "n_steps": 256,
        "batch_size": 64,
        "checkpoint_every": 256,
    }
    fields.update(overrides)
    return train(TrainConfig(**fields), _frame(), Settings(), out_dir=tmp_path, **train_kwargs)


def _wait_for_status(root: Path, status: str, timeout: float = 20.0) -> Path:
    """Poll ``root`` until one run directory declares ``status``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in root.iterdir():
            if path.is_dir():
                manifest = load_run(path)
                if manifest is not None and manifest.get("status") == status:
                    return path
        time.sleep(0.01)
    raise AssertionError(f"no run reached status {status!r} within {timeout}s")


def _wait_for_steps(root: Path, minimum: int, timeout: float = 20.0) -> Path:
    """Poll ``root`` until one run's manifest reports at least ``minimum`` steps."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path in root.iterdir():
            if path.is_dir():
                manifest = load_run(path)
                if manifest is not None and manifest.get("steps", 0) >= minimum:
                    return path
        time.sleep(0.01)
    raise AssertionError(f"no run reached {minimum} steps within {timeout}s")


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


def test_env_starts_split_the_usable_range():
    features = _frame()
    starts = env_starts_for(features, 2)
    first = features.valid_from + 60 - 1
    usable = len(features.X) - first
    assert starts == [first, first + usable // 2]
    assert len(set(env_starts_for(features, 4))) == 4


def test_two_env_starts_reset_and_reward_differently():
    features = _frame()
    starts = env_starts_for(features, 2)
    left = TradingEnv(features, Settings(), start=starts[0])
    right = TradingEnv(features, Settings(), start=starts[1])
    obs_left, _ = left.reset(seed=0)
    obs_right, _ = right.reset(seed=0)
    assert left.sim.bar_index == starts[0]
    assert right.sim.bar_index == starts[1]
    assert not np.array_equal(obs_left, obs_right)
    # Action 1 is a buy: the entry marks to the next bar, so a different slice of
    # the same price path must produce a different first reward.
    assert left.step(1).reward != right.step(1).reward


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
def test_vecnormalize_is_saved_loaded_and_required(trained_run, tmp_path):
    _, _, result = trained_run
    run_dir = result.model_path.parent
    assert (run_dir / "vecnormalize.pkl").exists()

    policy = load_policy(run_dir)
    raw = np.zeros(policy.obs_dim, dtype=np.float32)
    normalized = policy.normalize_obs(raw)
    assert normalized.shape == raw.shape
    assert np.isfinite(normalized).all()
    assert not np.allclose(normalized, raw), "observation normalisation was not applied"

    # A copied run that lost vecnormalize.pkl must fail loudly instead of
    # predicting on unnormalised observations.
    broken = tmp_path / result.run_id
    broken.mkdir()
    for name in ("model.zip", "config.json"):
        shutil.copy(run_dir / name, broken / name)
    with pytest.raises(FileNotFoundError, match="vecnormalize.pkl"):
        load_policy(broken)


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
def test_metrics_history_and_losses_after_the_first_update(trained_run):
    _, _, result = trained_run
    history_path = result.model_path.parent / "metrics.jsonl"
    lines = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(lines) >= 2
    assert [line["step"] for line in lines] == sorted(line["step"] for line in lines)
    for line in lines:
        assert "reward_mean_interval" in line
        assert len(line["action_dist"]) == 4

    # The first checkpoint happens before PPO's first update: losses are null,
    # not a misleading zero. The later snapshot must carry real losses.
    assert lines[0]["policy_loss"] is None
    latest = lines[-1]
    for key in ("entropy_loss", "policy_loss", "value_loss"):
        value = latest[key]
        assert value is not None, key
        assert np.isfinite(value), key
        assert value != 0.0, key

    snapshot = json.loads(result.metrics_path.read_text())
    assert snapshot["step"] == latest["step"] == result.steps
    assert snapshot["reward_mean_interval"] == pytest.approx(latest["reward_mean_interval"])


@pytest.mark.slow
def test_config_records_settings_data_and_env_starts(trained_run):
    _, features, result = trained_run
    config = json.loads((result.model_path.parent / "config.json").read_text())
    assert config["data_hash"] == data_hash_for(features)
    assert config["valid_from"] == features.valid_from
    assert config["n_features"] == features.X.shape[1]
    assert config["feature_columns"] == features.columns
    assert config["env_starts"] == env_starts_for(features, 1)
    assert config["initial_balance"] == Settings().contract.initial_balance
    assert config["selector"] == {"enabled": False, "type": None, "digest": None}
    assert config["use_strategy"] is False
    assert config["random_start"] is True
    for block in ("contract", "risk", "costs"):
        assert block in config, block
    assert config["normalisation"] == {
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
    }
    assert run_id_for(config["identity"], config["data_hash"], config["seed"]) == result.run_id


@pytest.mark.slow
def test_manifest_completes_with_steps_and_reward(trained_run):
    _, _, result = trained_run
    manifest = load_run(result.model_path.parent)
    assert manifest["status"] == "completed"
    assert manifest["steps"] == result.steps == 512
    assert manifest["reward_mean"] == pytest.approx(result.reward_mean)


@pytest.mark.slow
def test_run_id_covers_settings_and_selector(tmp_path):
    features = _frame()
    cfg = {
        "run_name": "identity",
        "total_steps": 256,
        "n_envs": 1,
        "n_steps": 256,
        "batch_size": 64,
        "checkpoint_every": 256,
    }
    base = train(TrainConfig(**cfg), features, Settings(), out_dir=tmp_path / "base")
    costs = replace(Settings(), costs=CostConfig(spread_floor=0.4))
    other_settings = train(TrainConfig(**cfg), features, costs, out_dir=tmp_path / "costs")
    with_bandit = train(
        TrainConfig(**cfg),
        features,
        Settings(),
        selector=LinUCB(dim=CONTEXT_DIM),
        out_dir=tmp_path / "bandit",
    )
    assert len({base.run_id, other_settings.run_id, with_bandit.run_id}) == 3


@pytest.mark.slow
def test_existing_run_with_a_different_config_is_refused(tmp_path):
    result = _train(tmp_path)
    config_path = result.model_path.parent / "config.json"
    payload = json.loads(config_path.read_text())
    payload["seed"] = 12_345
    config_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="run directory already holds a different configuration"):
        _train(tmp_path)
    assert json.loads(config_path.read_text())["seed"] == 12_345


@pytest.mark.slow
def test_stale_bandit_is_removed_without_a_selector(tmp_path):
    result = _train(tmp_path)
    stale = result.model_path.parent / "bandit.npz"
    stale.write_bytes(b"stale")
    _train(tmp_path)
    assert not stale.exists()


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
    assert config["selector"] == {
        "enabled": True,
        "type": "LinUCB",
        "digest": bandit.state_digest(),
    }
    assert load_policy(run_dir).obs_dim == expected_dim


@pytest.mark.slow
def test_stop_event_interrupts_and_leaves_a_loadable_run(tmp_path):
    features = _frame()
    stop = threading.Event()
    results: dict[str, TrainResult] = {}
    cfg = TrainConfig(
        run_name="stop",
        total_steps=100_000,
        n_envs=1,
        n_steps=256,
        batch_size=64,
        checkpoint_every=256,
    )

    def run() -> None:
        results["result"] = train(cfg, features, Settings(), out_dir=tmp_path, stop_event=stop)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    run_dir = _wait_for_status(tmp_path, "running")
    _wait_for_steps(tmp_path, 256)
    stop.set()
    thread.join(timeout=120)
    assert not thread.is_alive()

    manifest = load_run(run_dir)
    assert manifest["status"] == "interrupted"
    assert manifest["steps"] >= 256
    policy = load_policy(run_dir)
    assert 0 <= policy.act(np.zeros(policy.obs_dim, dtype=np.float32)) <= 3
    assert results["result"].steps == manifest["steps"]


@pytest.mark.slow
def test_pause_event_freezes_then_resumes(tmp_path):
    features = _frame()
    pause = threading.Event()
    results: dict[str, TrainResult] = {}
    cfg = TrainConfig(
        run_name="pause",
        total_steps=2000,
        n_envs=1,
        n_steps=256,
        batch_size=64,
        checkpoint_every=256,
    )

    def run() -> None:
        results["result"] = train(cfg, features, Settings(), out_dir=tmp_path, pause_event=pause)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    _wait_for_steps(tmp_path, 256)
    pause.set()
    run_dir = _wait_for_status(tmp_path, "paused")
    paused = load_run(run_dir)
    assert paused["steps"] >= 256
    assert 256 <= paused["steps"] < 2000

    # A paused run is a complete, loadable checkpoint.
    assert (run_dir / "model.zip").exists()
    assert (run_dir / "vecnormalize.pkl").exists()
    policy = load_policy(run_dir)
    assert 0 <= policy.act(np.zeros(policy.obs_dim, dtype=np.float32)) <= 3

    pause.clear()
    thread.join(timeout=180)
    assert not thread.is_alive()
    manifest = load_run(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["steps"] == results["result"].steps
    assert manifest["steps"] >= 2000
    assert paused["steps"] < manifest["steps"]


def _selector(features: FeatureFrame, labels) -> LinUCB:
    X, arms, rewards = bandit_training_data(features, labels)
    return LinUCB(dim=CONTEXT_DIM).fit_offline(X, arms, rewards).freeze()


@pytest.mark.slow
def test_selector_adds_the_strategy_context_and_saves_the_bandit(tmp_path):
    df = SyntheticProvider(seed=5).fetch(SYMBOL, START, END)
    features = build_features(df)
    labels = label_events(df, Settings())
    selector = _selector(features, labels)
    cfg = TrainConfig(
        run_name="strategy",
        total_steps=256,
        n_envs=1,
        n_steps=256,
        batch_size=64,
        checkpoint_every=256,
    )
    res = train(cfg, features, Settings(), selector=selector, labels=labels, out_dir=tmp_path)
    run_dir = res.model_path.parent
    config = json.loads((run_dir / "config.json").read_text())
    assert config["obs_dim"] == 60 * features.X.shape[1] + 5 + len(ALL_STRATEGIES) + 1
    assert config["use_strategy"] is True
    assert config["selector"]["digest"] == selector.state_digest()
    assert (run_dir / BANDIT_NAME).is_file()
    assert load_policy(run_dir).obs_dim == config["obs_dim"]


@pytest.mark.slow
def test_no_selector_keeps_the_plain_observation_width(tmp_path):
    res = _train(tmp_path)
    run_dir = res.model_path.parent
    config = json.loads((run_dir / "config.json").read_text())
    features = _frame()
    assert config["obs_dim"] == 60 * features.X.shape[1] + 5
    assert config["use_strategy"] is False
    assert not (run_dir / BANDIT_NAME).exists()


@pytest.mark.slow
def test_train_records_the_split_provenance(tmp_path):
    res = _train(
        tmp_path,
        train_end="2026-01-07T00:00:00+00:00",
        eval_start="2026-01-07T00:01:00+00:00",
    )
    config = json.loads((res.model_path.parent / "config.json").read_text())
    manifest = load_run(res.model_path.parent)
    assert config["train_end"] == "2026-01-07T00:00:00+00:00"
    assert config["eval_start"] == "2026-01-07T00:01:00+00:00"
    assert manifest["train_end"] == config["train_end"]
    assert manifest["eval_start"] == config["eval_start"]
    assert config["identity"]["split"] == {
        "train_end": config["train_end"],
        "eval_start": config["eval_start"],
    }


@pytest.mark.slow
def test_bandit_state_changes_the_run_id():
    features = _frame()
    cfg = TrainConfig(run_name="digest", total_steps=256, n_envs=1, n_steps=256)
    rng = np.random.default_rng(0)
    arms = rng.integers(0, 5, size=50)
    rewards = rng.normal(size=50)
    a = LinUCB(dim=CONTEXT_DIM).fit_offline(rng.normal(size=(50, CONTEXT_DIM)), arms, rewards)
    b = LinUCB(dim=CONTEXT_DIM).fit_offline(rng.normal(size=(50, CONTEXT_DIM)), arms, rewards)
    id_a, identity_a = run_identity(cfg, features, Settings(), selector=a)
    id_b, identity_b = run_identity(cfg, features, Settings(), selector=b)
    assert id_a != id_b
    assert identity_a["selector"]["digest"] == a.state_digest()
    assert identity_b["selector"]["digest"] == b.state_digest()


@pytest.mark.slow
def test_zero_step_run_has_null_action_distribution_and_is_degenerate(tmp_path):
    stop = threading.Event()
    stop.set()
    res = _train(tmp_path, stop_event=stop)
    metrics = json.loads(res.metrics_path.read_text())
    manifest = load_run(res.model_path.parent)
    assert metrics["step"] == 0
    assert metrics["action_dist"] is None
    assert metrics["reward_mean_normalized"] == metrics["reward_mean"]
    assert manifest["degenerate"] is True
    assert manifest["baseline_hold_net_profit"] == pytest.approx(0.0)
    assert manifest["reward_mean_normalized"] == metrics["reward_mean"]


@pytest.mark.slow
def test_manifest_records_random_start_and_sampled_start(trained_run, tmp_path):
    _, features, result = trained_run
    config = json.loads((result.model_path.parent / "config.json").read_text())
    manifest = load_run(result.model_path.parent)
    assert config["random_start"] is True
    assert manifest["random_start"] is True
    assert isinstance(manifest["sampled_start"], int)
    first = features.valid_from + 60 - 1
    assert first <= manifest["sampled_start"] <= len(features.X) - 1


@pytest.mark.slow
def test_unexpected_failure_writes_a_failed_manifest(tmp_path, monkeypatch):
    import importlib

    train_module = importlib.import_module("scalping.rl.train")

    def boom(self, *args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(train_module.PPO, "learn", boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        _train(tmp_path)

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    manifest = load_run(run_dirs[0])
    assert manifest["status"] == "failed"
    assert "kaboom" in manifest["error"]


def test_bc_epochs_need_steps() -> None:
    with pytest.raises(ValueError, match="bc_epochs requires"):
        TrainConfig(run_name="bc", bc_epochs=2)


@pytest.mark.slow
def test_behaviour_cloning_is_recorded(tmp_path):
    """PPO can start from the rule's actions instead of pure exploration."""
    frame = _frame()
    teacher = np.ones(frame.X.shape[0], dtype=np.int64)
    result = _train(
        tmp_path,
        total_steps=256,
        bc_steps=512,
        bc_epochs=1,
        bc_lr=1e-3,
        teacher_actions=teacher,
    )
    config = json.loads((tmp_path / result.run_id / "config.json").read_text())
    cloning = config["behaviour_cloning"]
    assert cloning["samples"] == 512
    assert cloning["epochs"] == 1
    assert 0.0 <= cloning["accuracy"] <= 1.0
    assert 0.0 <= cloning["entry_recall"] <= 1.0
    assert 0.0 <= cloning["predicted_entry_rate"] <= 1.0


@pytest.mark.slow
def test_bc_only_saves_a_clone_without_ppo(tmp_path):
    frame = _frame()
    teacher = np.ones(frame.X.shape[0], dtype=np.int64)
    result = _train(
        tmp_path,
        total_steps=256,
        bc_steps=512,
        bc_epochs=1,
        bc_only=True,
        teacher_actions=teacher,
    )
    assert result.steps == 0
    assert (tmp_path / result.run_id / "model.zip").is_file()
