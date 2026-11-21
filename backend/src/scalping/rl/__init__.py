"""Reinforcement-learning layer: PPO training, run registry and inference.

``train`` turns a feature frame into a self-contained ``runs/<run_id>/``;
``load_policy`` turns that directory back into a greedy policy; ``registry``
holds the id and file conventions the API and evaluation layers share.
"""

from __future__ import annotations

from scalping.rl.callbacks import AtomicCheckpointCallback, MetricsCallback
from scalping.rl.policy import PolicyWrapper, load_policy
from scalping.rl.registry import (
    MANIFEST_NAME,
    MODEL_NAME,
    load_run,
    run_id_for,
    save_manifest,
    save_model,
    write_json,
)
from scalping.rl.train import MIN_BARS, TrainConfig, TrainResult, data_hash_for, train

__all__ = [
    "MANIFEST_NAME",
    "MIN_BARS",
    "MODEL_NAME",
    "AtomicCheckpointCallback",
    "MetricsCallback",
    "PolicyWrapper",
    "TrainConfig",
    "TrainResult",
    "data_hash_for",
    "load_policy",
    "load_run",
    "run_id_for",
    "save_manifest",
    "save_model",
    "train",
    "write_json",
]
