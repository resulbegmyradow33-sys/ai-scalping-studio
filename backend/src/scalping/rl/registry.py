"""Run registry: stable run ids and crash-safe artefacts under ``runs/<run_id>/``.

Every training run owns one directory named after :func:`run_id_for`. The
directory holds the model, the observation/reward normaliser, the resolved
config, the latest metrics snapshot, a JSONL metrics history, the optional bandit
and a small ``manifest.json`` that lists the run for the API.

Writes never happen in place. A file is written to a sibling temp file, flushed
to disk and moved onto the target with :func:`os.replace`, so a crash or a
Ctrl-C at any point leaves the previous version readable instead of a truncated
JSON or a half-written zip.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "BANDIT_NAME",
    "CONFIG_NAME",
    "MANIFEST_NAME",
    "METRICS_HISTORY_NAME",
    "METRICS_NAME",
    "MODEL_NAME",
    "VECNORMALIZE_NAME",
    "ManifestWriter",
    "append_jsonl",
    "load_run",
    "run_id_for",
    "save_manifest",
    "save_model",
    "save_vecnormalize",
    "write_json",
]

#: File names inside a run directory.
MANIFEST_NAME: str = "manifest.json"
CONFIG_NAME: str = "config.json"
METRICS_NAME: str = "metrics.json"
METRICS_HISTORY_NAME: str = "metrics.jsonl"
MODEL_NAME: str = "model.zip"
VECNORMALIZE_NAME: str = "vecnormalize.pkl"
BANDIT_NAME: str = "bandit.npz"

#: Length of a run id in hex characters.
_ID_LENGTH: int = 12


class _Savable(Protocol):
    """The slice of a stable-baselines3 object that :func:`save_model` needs."""

    def save(self, path: str) -> None: ...


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """One spelling per mapping: sorted keys, no spaces, no NaN.

    ``3e-4`` and ``0.0003`` are the same float and JSON writes them the same way,
    so the digest depends on the values, not on the insertion order or on how
    the caller spelled the literal.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def run_id_for(config: Mapping[str, Any], data_hash: str, seed: int) -> str:
    """Return the 12-hex-character id of a run.

    ``run_id = sha256(canonical_json(config) + "\\0" + data_hash + "\\0" + seed)[:12]``
    — the same hyperparameters, data and seed always name the same directory, so
    a resumed run overwrites its own artefacts instead of piling up duplicates.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not isinstance(data_hash, str):
        raise TypeError(f"data_hash must be a str, got {type(data_hash).__name__}")
    material = f"{_canonical_json(config)}\0{data_hash}\0{seed}".encode()
    return hashlib.sha256(material).hexdigest()[:_ID_LENGTH]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _fsync(path: Path) -> None:
    """Flush an already written file to disk before it is renamed."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temp_sibling(target: Path, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=suffix)
    os.close(fd)
    return Path(temp_name)


def _atomic_write_text(target: Path, text: str) -> Path:
    """Write ``text`` to ``target`` through a flushed sibling temp file."""
    temp = _temp_sibling(target, ".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Serialise ``payload`` to ``path`` atomically.

    ``allow_nan=False`` makes a NaN metric fail here, at the write, instead of
    producing a file that a strict JSON reader later rejects.
    """
    target = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _atomic_write_text(target, text)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Append one JSON line, flushed and fsynced.

    Appending keeps the history cheap as it grows. A single short ``O_APPEND``
    write is not torn by a concurrent reader; a crash before the fsync can at
    worst drop the newest line, which a training UI can tolerate.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _save_atomic(obj: _Savable, path: str | Path, suffix: str) -> Path:
    """``obj.save`` into a temp file, fsync it, then move it onto ``path``."""
    target = Path(path)
    temp = _temp_sibling(target, suffix)
    try:
        obj.save(str(temp))
        _fsync(temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def save_model(model: _Savable, path: str | Path) -> Path:
    """Save a stable-baselines3 model without ever truncating the old file.

    ``model.save`` appends ``.zip`` only when the suffix is different, so the
    temp file already ends in ``.zip`` and lands exactly where
    :func:`os.replace` expects it.
    """
    return _save_atomic(model, path, ".zip")


def save_vecnormalize(vec_normalize: _Savable, path: str | Path) -> Path:
    """Save a ``VecNormalize`` pickle atomically; its ``save`` writes the path."""
    return _save_atomic(vec_normalize, path, ".pkl")


def save_manifest(run_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``manifest.json`` into ``run_dir`` and return its path."""
    return write_json(Path(run_dir) / MANIFEST_NAME, manifest)


def load_run(run_dir: str | Path) -> dict[str, Any] | None:
    """Return the manifest of ``run_dir``, or ``None`` when there is none.

    A directory without a manifest is simply a run that never finished its
    first write. A manifest that exists but is not a JSON object is corruption,
    so it raises ``ValueError`` naming the file instead of masquerading as an
    empty run.
    """
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


class ManifestWriter:
    """Rewrite a run's manifest with a new status, step count and reward.

    The API polls this file, so it is replaced atomically on every call. The
    writer owns the immutable part of the manifest (id, seed, paths) and the
    caller supplies the part that changes while training.
    """

    def __init__(self, run_dir: str | Path, base: Mapping[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.base = dict(base)
        self.status = "running"

    def write(
        self,
        *,
        status: str | None = None,
        steps: int = 0,
        reward_mean: float | None = None,
    ) -> Path:
        if status is not None:
            self.status = status
        payload = dict(self.base)
        payload.update(
            {
                "status": self.status,
                "steps": int(steps),
                "reward_mean": reward_mean,
                "updated_at": _now(),
            }
        )
        return save_manifest(self.run_dir, payload)
