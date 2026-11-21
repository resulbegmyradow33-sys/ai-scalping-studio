"""Reinforcement-learning layer: PPO training, run registry and inference.

``train`` turns a feature frame into a self-contained ``runs/<run_id>/``;
``load_policy`` turns that directory back into a greedy policy; ``registry``
holds the id and file conventions the API and evaluation layers share.
"""

from __future__ import annotations

from scalping.rl.callbacks import (
    CheckpointCallback,
    ControlCallback,
    MetricsCallback,
    TrainingHalted,
)
from scalping.rl.policy import PolicyWrapper, load_policy
from scalping.rl.registry import (
    BANDIT_NAME,
    CONFIG_NAME,
    MANIFEST_NAME,
    METRICS_HISTORY_NAME,
    METRICS_NAME,
    MODEL_NAME,
    VECNORMALIZE_NAME,
    ManifestWriter,
    append_jsonl,
    load_run,
    run_id_for,
    save_manifest,
    save_model,
    save_vecnormalize,
    write_json,
)
from scalping.rl.train import (
    MIN_BARS,
    TrainConfig,
    TrainResult,
    data_hash_for,
    env_starts_for,
    train,
)

__all__ = [
    "BANDIT_NAME",
    "CONFIG_NAME",
    "MANIFEST_NAME",
    "METRICS_HISTORY_NAME",
    "METRICS_NAME",
    "MIN_BARS",
    "MODEL_NAME",
    "VECNORMALIZE_NAME",
    "CheckpointCallback",
    "ControlCallback",
    "ManifestWriter",
    "MetricsCallback",
    "PolicyWrapper",
    "TrainConfig",
    "TrainResult",
    "TrainingHalted",
    "append_jsonl",
    "data_hash_for",
    "env_starts_for",
    "load_policy",
    "load_run",
    "run_id_for",
    "save_manifest",
    "save_model",
    "save_vecnormalize",
    "train",
    "write_json",
]
