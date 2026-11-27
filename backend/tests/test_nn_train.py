"""Trainer: learnability, checkpoint round-trip, resume, cancel and slots."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from scalping.nn import WindowConfig, WindowedDataset
from scalping.nn.train import (
    _RUN_SLOT,
    JobControl,
    TrainConfig,
    TrainError,
    load_checkpoint,
    select_device,
    train_classifier,
)

WINDOW = 10
FEATURES = 3


def dataset(n_train: int = 240, n_val: int = 80, seed: int = 5) -> WindowedDataset:
    """Learnable toy data: class -1 when the window's first feature sums negative."""
    rng = np.random.default_rng(seed)

    def split(rows: int):
        X = rng.normal(size=(rows, WINDOW, FEATURES)).astype(np.float32)
        y = np.where(X[:, :, 0].sum(axis=1) > 0, 1, -1).astype(np.int64)
        ts = np.arange(rows).astype("datetime64[m]")
        return X, y, ts

    X_train, y_train, ts_train = split(n_train)
    X_val, y_val, ts_val = split(n_val)
    return WindowedDataset(
        X_train=X_train,
        y_train=y_train,
        ts_train=ts_train,
        X_validation=X_val,
        y_validation=y_val,
        ts_validation=ts_val,
        X_test=np.empty((0, WINDOW, FEATURES), dtype=np.float32),
        y_test=np.empty(0, dtype=np.int64),
        ts_test=np.empty(0, dtype="datetime64[m]"),
        mean=np.zeros(FEATURES, dtype=np.float32),
        std=np.ones(FEATURES, dtype=np.float32),
        columns=("a", "b", "c"),
        dataset_id="toy:w10",
        window=WINDOW,
    )


def config(**overrides) -> TrainConfig:
    base = {
        "model": "mlp",
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 5e-3,
        "hidden": 32,
        "dropout": 0.0,
        "device": "cpu",
        "seed": 1,
    }
    return TrainConfig(**{**base, **overrides})  # type: ignore[arg-type]


def test_trainer_learns_a_learnable_signal(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    result = train_classifier(dataset(), config(), checkpoint=checkpoint)

    assert result.stopped in {"epochs", "early_stop"}
    assert result.epochs_run >= 2
    assert result.history[0]["train_loss"] > result.history[-1]["train_loss"]
    assert result.val_accuracy >= 0.8
    assert result.device == "cpu"
    assert checkpoint.exists()


def test_checkpoint_round_trip_and_resume(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    first = train_classifier(dataset(), config(epochs=2), checkpoint=checkpoint)
    assert first.epochs_run == 2

    payload = load_checkpoint(checkpoint)
    assert payload["model"] == "mlp"
    assert payload["epoch"] == 1
    assert payload["classes"] == [-1, 1]
    assert payload["n_features"] == FEATURES
    assert len(payload["history"]) == 2

    resumed = train_classifier(
        dataset(), config(epochs=4), checkpoint=checkpoint, resume_from=checkpoint
    )
    assert resumed.epochs_run == 4
    assert len(resumed.history) == 4
    assert resumed.best_epoch >= first.best_epoch


def test_cancellation_keeps_a_loadable_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "cancel.pt"
    control = JobControl()
    control.cancel()

    result = train_classifier(
        dataset(), config(epochs=10), checkpoint=checkpoint, control=control
    )

    assert result.stopped == "cancelled"
    assert result.epochs_run == 0
    assert load_checkpoint(checkpoint)["model"] == "mlp"


def test_one_job_slot_rejects_a_second_training(tmp_path) -> None:
    assert _RUN_SLOT.acquire()
    try:
        with pytest.raises(TrainError, match="already running"):
            train_classifier(dataset(), config(epochs=1), checkpoint=tmp_path / "busy.pt")
    finally:
        _RUN_SLOT.release()


def test_device_selection_is_explicit() -> None:
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cpu", "mps"}
    if not torch.backends.mps.is_available():
        with pytest.raises(TrainError, match="not available"):
            select_device("mps")


def test_resume_rejects_a_different_model(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    train_classifier(dataset(), config(epochs=1), checkpoint=checkpoint)

    with pytest.raises(TrainError, match="trained as"):
        train_classifier(
            dataset(),
            config(epochs=2, model="gru", hidden=8),
            checkpoint=checkpoint,
            resume_from=checkpoint,
        )


def test_validation_needs_a_known_class() -> None:
    data = dataset()
    bad = replace(
        data,
        y_validation=np.full(10, 7, dtype=np.int64),
        X_validation=data.X_validation[:10],
    )

    with pytest.raises(TrainError, match="absent from training"):
        train_classifier(bad, config(epochs=1), checkpoint="/tmp/never.pt")


def test_bad_config_is_rejected() -> None:
    with pytest.raises(TrainError, match="epochs"):
        config(epochs=0)
    with pytest.raises(TrainError, match="unknown model"):
        config(model="spectral")
    with pytest.raises(TrainError, match="device"):
        config(device="tpu")
    with pytest.raises(TrainError, match="max_minutes"):
        config(max_minutes=0.0)
    with pytest.raises(ValueError, match="window"):
        WindowConfig(window=0)


def test_gradient_clipping_is_applied_every_batch(tmp_path, monkeypatch) -> None:
    import torch

    calls: list[float] = []
    original = torch.nn.utils.clip_grad_norm_

    def recorder(parameters, max_norm, *args, **kwargs):
        calls.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recorder)
    config = TrainConfig(
        model="mlp",
        epochs=1,
        batch_size=32,
        learning_rate=5e-3,
        hidden=16,
        dropout=0.0,
        device="cpu",
        seed=1,
        grad_clip=0.7,
    )

    train_classifier(dataset(), config, checkpoint=tmp_path / "clip.pt")

    assert calls
    assert set(calls) == {0.7}


def test_early_stopping_stops_when_validation_stalls(tmp_path, monkeypatch) -> None:
    import scalping.nn.train as trainer

    monkeypatch.setattr(trainer, "_evaluate", lambda *args, **kwargs: (1.0, 0.5))
    config = TrainConfig(
        model="mlp",
        epochs=10,
        batch_size=32,
        learning_rate=5e-3,
        hidden=16,
        dropout=0.0,
        device="cpu",
        seed=1,
        patience=2,
        min_delta=1e-4,
    )

    result = train_classifier(dataset(), config, checkpoint=tmp_path / "early.pt")

    assert result.stopped == "early_stop"
    assert result.epochs_run == 3
    assert len(result.history) == 3
    assert result.best_epoch == 1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available on this machine"
)
def test_explicit_mps_is_selected_when_available() -> None:
    assert select_device("mps").type == "mps"


def test_resume_rejects_a_different_dataset_identity(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    train_classifier(dataset(), config(epochs=1), checkpoint=checkpoint)
    other = replace(dataset(), dataset_id="toy:other")

    with pytest.raises(TrainError, match="dataset_id"):
        train_classifier(
            other,
            config(epochs=2),
            checkpoint=checkpoint,
            resume_from=checkpoint,
        )
