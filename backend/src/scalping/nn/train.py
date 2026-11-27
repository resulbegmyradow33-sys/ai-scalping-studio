"""One-job neural trainer with MPS/CPU selection, checkpoints and cancellation.

The trainer is deliberately small and synchronous. It fits a model chosen from
:mod:`scalping.nn.models` on a :class:`~scalping.nn.data.WindowedDataset`, keeps
the best validation loss, and writes a resumable checkpoint after every epoch.

Guarantees:

* **one heavy job per process** — a non-blocking slot rejects a second call;
* **explicit device policy** — ``auto`` prefers MPS and falls back to CPU, while
  an explicit ``mps`` request on a machine without it is an error, not a silent
  slowdown;
* **bounded work** — gradient clipping, early stopping and an optional
  wall-clock budget;
* **cancellation** — :class:`JobControl.cancel` is checked every batch and the
  last complete checkpoint stays loadable;
* **checkpoints** — atomic writes (temp file + replace), so an interrupt never
  leaves a half-written file.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from scalping.nn.data import WindowedDataset
from scalping.nn.models import MODEL_NAMES, ModelName, build_model

__all__ = [
    "JobControl",
    "JobSlot",
    "TrainConfig",
    "TrainError",
    "TrainResult",
    "device_label",
    "load_checkpoint",
    "select_device",
    "train_classifier",
]

DeviceRequest = Literal["auto", "cpu", "mps"]


class TrainError(ValueError):
    """The request cannot be trained (bad config, busy slot, bad checkpoint)."""


class JobSlot:
    """A non-blocking process-wide slot for one heavy job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    def busy(self) -> bool:
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._lock.release()
        return not acquired


#: Process-wide slot: the 8 GB Mac mini should run one training job at a time.
_RUN_SLOT = JobSlot()


class JobControl:
    """Cooperative cancellation flag shared with the caller."""

    def __init__(self) -> None:
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    def cancelled(self) -> bool:
        return self._stop.is_set()


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training hyper-parameters and resource guards."""

    model: ModelName = "gru"
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    hidden: int = 64
    dropout: float = 0.1
    device: DeviceRequest = "auto"
    seed: int = 0
    checkpoint_every: int = 1
    max_minutes: float | None = None

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise TrainError(f"unknown model {self.model!r}; choose from {MODEL_NAMES}")
        if self.epochs < 1:
            raise TrainError("epochs must be >= 1")
        if self.batch_size < 1:
            raise TrainError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise TrainError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise TrainError("weight_decay must be >= 0")
        if self.grad_clip <= 0:
            raise TrainError("grad_clip must be > 0")
        if self.patience < 1:
            raise TrainError("patience must be >= 1")
        if self.min_delta < 0:
            raise TrainError("min_delta must be >= 0")
        if self.hidden < 1:
            raise TrainError("hidden must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise TrainError("dropout must be in [0, 1)")
        if self.device not in ("auto", "cpu", "mps"):
            raise TrainError(f"unknown device {self.device!r}")
        if self.seed < 0:
            raise TrainError("seed must be >= 0")
        if self.checkpoint_every < 1:
            raise TrainError("checkpoint_every must be >= 1")
        if self.max_minutes is not None and self.max_minutes <= 0:
            raise TrainError("max_minutes must be positive when set")


@dataclass(frozen=True, slots=True)
class TrainResult:
    """History plus the best checkpoint; the model itself is written to disk."""

    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_val_loss: float
    val_accuracy: float
    epochs_run: int
    device: str
    checkpoint: Path
    stopped: str
    dataset_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": list(self.history),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "val_accuracy": self.val_accuracy,
            "epochs_run": self.epochs_run,
            "device": self.device,
            "checkpoint": str(self.checkpoint),
            "stopped": self.stopped,
            "dataset_id": self.dataset_id,
        }


def select_device(request: DeviceRequest) -> torch.device:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""
    if request == "cpu":
        return torch.device("cpu")
    if request == "mps":
        if not torch.backends.mps.is_available():
            raise TrainError("MPS was requested but is not available on this machine")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_label(device: torch.device) -> str:
    return str(device)


# Public alias used by the CLI/API later; keeps the underscore name private.
device_label = _device_label


def _class_index(y: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    values = sorted({int(value) for value in y.tolist()})
    if not values:
        raise TrainError("the training split has no labels")
    index = {value: position for position, value in enumerate(values)}
    mapped = np.array([index[int(value)] for value in y.tolist()], dtype=np.int64)
    return mapped, tuple(values)


def _validate_split(
    data: WindowedDataset, config: TrainConfig
) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[int, ...]]:
    if data.X_train.shape[0] == 0:
        raise TrainError("the training split has no windows")
    if data.X_validation.shape[0] == 0:
        raise TrainError("the validation split has no windows; early stopping needs one")
    if data.feature_count != data.X_train.shape[2]:
        raise TrainError("feature count is inconsistent inside the dataset")
    train_y, classes = _class_index(data.y_train)
    known = set(classes)
    unknown = sorted({int(value) for value in data.y_validation.tolist()} - known)
    if unknown:
        raise TrainError(f"validation contains labels absent from training: {unknown}")
    index = {value: position for position, value in enumerate(classes)}
    val_y = np.array([index[int(value)] for value in data.y_validation.tolist()], dtype=np.int64)
    if len(classes) < 2:
        raise TrainError("the training split has a single class; nothing to learn")

    x_train = torch.from_numpy(np.ascontiguousarray(data.X_train, dtype=np.float32))
    x_val = torch.from_numpy(np.ascontiguousarray(data.X_validation, dtype=np.float32))
    return x_train, torch.from_numpy(train_y), x_val, torch.from_numpy(val_y), classes


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _evaluate(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb)
            total_loss += float(criterion(logits, yb).item()) * xb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().item())
    rows = max(1, x.shape[0])
    return total_loss / rows, correct / rows


def train_classifier(
    data: WindowedDataset,
    config: TrainConfig | None = None,
    *,
    checkpoint: str | Path,
    control: JobControl | None = None,
    resume_from: str | Path | None = None,
) -> TrainResult:
    """Fit one classifier, checkpointing each epoch; resumable and cancellable."""
    cfg = TrainConfig() if config is None else config
    control = JobControl() if control is None else control
    path = Path(checkpoint)
    if not _RUN_SLOT.acquire():
        raise TrainError("another training job is already running in this process")
    rng_state = torch.random.get_rng_state()
    try:
        x_train, y_train, x_val, y_val, classes = _validate_split(data, cfg)
        device = select_device(cfg.device)
        torch.manual_seed(cfg.seed)
        model = build_model(
            cfg.model,
            n_features=data.feature_count,
            window=data.window,
            n_classes=len(classes),
            hidden=cfg.hidden,
            dropout=cfg.dropout,
            seed=cfg.seed,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        start_epoch = 0
        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_val_accuracy = 0.0
        best_state: dict[str, Tensor] | None = None

        if resume_from is not None:
            payload = load_checkpoint(resume_from)
            if payload["model"] != cfg.model:
                raise TrainError(
                    f"checkpoint was trained as {payload['model']!r}, not {cfg.model!r}"
                )
            if payload["n_features"] != data.feature_count or payload["window"] != data.window:
                raise TrainError("checkpoint shape does not match the dataset")
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            start_epoch = int(payload["epoch"]) + 1
            history = list(payload["history"])
            best_val_loss = float(payload["best_val_loss"])
            best_epoch = int(payload["best_epoch"])
            best_val_accuracy = float(payload.get("best_val_accuracy", 0.0))
            best_state = payload.get("best_state")

        deadline = None if cfg.max_minutes is None else time.monotonic() + cfg.max_minutes * 60.0
        generator = torch.Generator().manual_seed(cfg.seed)
        epochs_run = start_epoch
        stopped = "epochs"

        for epoch in range(start_epoch, cfg.epochs):
            if control.cancelled():
                stopped = "cancelled"
                break
            if deadline is not None and time.monotonic() >= deadline:
                stopped = "time_limit"
                break
            model.train()
            order = torch.randperm(x_train.shape[0], generator=generator)
            train_loss = 0.0
            for start in range(0, x_train.shape[0], cfg.batch_size):
                if control.cancelled():
                    stopped = "cancelled"
                    break
                batch = order[start : start + cfg.batch_size]
                xb = x_train[batch].to(device)
                yb = y_train[batch].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                train_loss += float(loss.item()) * xb.shape[0]
            if stopped == "cancelled":
                break
            train_loss /= max(1, x_train.shape[0])
            val_loss, val_accuracy = _evaluate(
                model, x_val, y_val, criterion, device, cfg.batch_size
            )
            epochs_run = epoch + 1
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            if val_loss < best_val_loss - cfg.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_val_accuracy = val_accuracy
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }

            if epochs_run % cfg.checkpoint_every == 0 or control.cancelled():
                _atomic_save(
                    path,
                    {
                        "model": cfg.model,
                        "epoch": epoch,
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "best_val_accuracy": best_val_accuracy,
                        "best_state": best_state,
                        "classes": list(classes),
                        "n_features": data.feature_count,
                        "window": data.window,
                        "dataset_id": data.dataset_id,
                        "config": asdict(cfg),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                )

            if epoch + 1 - best_epoch >= cfg.patience:
                stopped = "early_stop"
                break

        if not path.exists() or stopped in {"cancelled", "time_limit"}:
            _atomic_save(
                path,
                {
                    "model": cfg.model,
                    "epoch": max(0, epochs_run - 1),
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "best_state": best_state,
                    "classes": list(classes),
                    "n_features": data.feature_count,
                    "window": data.window,
                    "dataset_id": data.dataset_id,
                    "config": asdict(cfg),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
            )

        return TrainResult(
            history=tuple(history),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            val_accuracy=best_val_accuracy,
            epochs_run=epochs_run,
            device=_device_label(device),
            checkpoint=path,
            stopped=stopped,
            dataset_id=data.dataset_id,
        )
    finally:
        torch.random.set_rng_state(rng_state)
        _RUN_SLOT.release()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`train_classifier`."""
    source = Path(path)
    if not source.exists():
        raise TrainError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise TrainError(f"not a training checkpoint: {source}")
    return payload
