"""Run registry: stable run ids and crash-safe artefacts under ``runs/<run_id>/``.

Every training run owns one directory named after :func:`run_id_for`. The
directory holds the model, the resolved config, the last metrics snapshot, the
optional bandit and a small ``manifest.json`` that lists the run for the API.

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
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "MANIFEST_NAME",
    "MODEL_NAME",
    "load_run",
    "run_id_for",
    "save_manifest",
    "save_model",
    "write_json",
]

#: File name of the run manifest inside a run directory.
MANIFEST_NAME: str = "manifest.json"

#: File name of the stable-baselines3 checkpoint inside a run directory.
MODEL_NAME: str = "model.zip"

#: Length of a run id in hex characters.
_ID_LENGTH: int = 12


class _Savable(Protocol):
    """The slice of a stable-baselines3 model that :func:`save_model` needs."""

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


def _atomic_write_text(target: Path, text: str) -> Path:
    """Write ``text`` to ``target`` through a flushed sibling temp file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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


def save_model(model: _Savable, path: str | Path) -> Path:
    """Save a stable-baselines3 model without ever truncating the old file.

    ``model.save`` appends ``.zip`` only when the suffix is different, so the
    temp file already ends in ``.zip`` and lands exactly where
    :func:`os.replace` expects it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".zip")
    os.close(fd)
    temp = Path(temp_name)
    try:
        model.save(str(temp))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


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
