"""Persisted dataset arrays for large matrices.

A dataset bundle lives as Python arrays in memory. Training repeatedly should
not rebuild it from Parquet, and a multi-GB matrix must not be copied just to be
read. This module writes one ``.npy`` file per array plus an ``artifact.json``
manifest and loads ``X`` as a read-only memory map when asked.

Layout::

    <directory>/artifact.json     metadata + per-file checksums (written last)
    <directory>/X.npy
    <directory>/y_direction.npy
    <directory>/y_return.npy
    <directory>/ts.npy
    <directory>/split.npy

``artifact.json`` is the commit marker: arrays written before a crash without
it are not a dataset. Checksums are file digests checked on load by default;
pass ``verify=False`` to skip the read on a trusted artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from scalping.datasets.builder import SPLIT_NAMES, DatasetBundle

__all__ = [
    "ARRAY_NAMES",
    "ARTIFACT_FILE",
    "ArrayArtifact",
    "LoadedArrays",
    "load_dataset_arrays",
    "save_dataset_arrays",
]

ARTIFACT_FILE: Final[str] = "artifact.json"
ARRAY_NAMES: Final[tuple[str, ...]] = ("X", "y_direction", "y_return", "ts", "split")
_ATTRIBUTE: Final[dict[str, str]] = {
    "X": "X",
    "y_direction": "y_direction",
    "y_return": "y_return",
    "ts": "ts",
    "split": "split",
}
_CHUNK: Final[int] = 1024 * 1024


class ArtifactError(ValueError):
    """The artifact is missing, old or corrupt."""


@dataclass(frozen=True, slots=True)
class ArrayArtifact:
    """Lightweight view of an artifact on disk."""

    directory: Path
    dataset_id: str
    arrays: dict[str, dict[str, Any]]

    def path_for(self, name: str) -> Path:
        if name not in self.arrays:
            raise ArtifactError(f"artifact has no array {name!r}")
        return self.directory / self.arrays[name]["file"]


@dataclass(frozen=True, slots=True)
class LoadedArrays:
    """Arrays plus the manifest metadata they were saved with."""

    directory: Path
    dataset_id: str
    X: np.ndarray
    y_direction: np.ndarray
    y_return: np.ndarray
    ts: np.ndarray
    split: np.ndarray
    manifest: dict[str, Any]
    memmapped: bool

    def mask(self, name: str) -> np.ndarray:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        return self.split == SPLIT_NAMES.index(name)

    def rows(self, name: str) -> int:
        return int(self.mask(name).sum())


def _digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_array(path: Path, array: np.ndarray) -> None:
    # np.save appends ``.npy`` when the target does not end with it, so write
    # through a handle instead of trusting the temp file name.
    tmp = path.with_name(path.name + ".tmp.npy")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_dataset_arrays(directory: str | Path, bundle: DatasetBundle) -> Path:
    """Write ``bundle`` to ``directory``; return the manifest path."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, dict[str, Any]] = {}
    for name in ARRAY_NAMES:
        array = np.asarray(getattr(bundle, _ATTRIBUTE[name]))
        path = root / f"{name}.npy"
        _write_array(path, array)
        arrays[name] = {
            "file": path.name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "checksum": _digest(path),
        }

    payload = {
        "version": 1,
        "dataset_id": bundle.manifest.dataset_id,
        "manifest": bundle.manifest.to_dict(),
        "arrays": arrays,
    }
    target = root / ARTIFACT_FILE
    tmp = target.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _read_artifact(directory: str | Path) -> ArrayArtifact:
    root = Path(directory)
    meta = root / ARTIFACT_FILE
    if not meta.is_file():
        raise ArtifactError(f"no {ARTIFACT_FILE} under {root}; not a saved dataset")
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"artifact manifest is not valid JSON: {exc}") from exc
    if payload.get("version") != 1:
        raise ArtifactError(f"unsupported artifact version {payload.get('version')!r}")
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict) or not all(
        isinstance(entry, dict) and "file" in entry for entry in arrays.values()
    ):
        raise ArtifactError("artifact manifest has no usable array table")
    return ArrayArtifact(
        directory=root,
        dataset_id=str(payload.get("dataset_id", "")),
        arrays=arrays,
    )


def load_dataset_arrays(
    directory: str | Path,
    *,
    mmap: bool = True,
    verify: bool = True,
) -> LoadedArrays:
    """Load a saved artifact; ``X`` is a read-only memory map when ``mmap``."""
    artifact = _read_artifact(directory)
    loaded: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        entry = artifact.arrays.get(name)
        if entry is None:
            raise ArtifactError(f"artifact is missing the {name!r} array")
        path = artifact.directory / str(entry["file"])
        if not path.is_file():
            raise ArtifactError(f"artifact array file is missing: {path}")
        if verify and _digest(path) != entry.get("checksum"):
            raise ArtifactError(f"checksum mismatch for {path.name}")
        if name == "X" and mmap:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            values = np.load(path, allow_pickle=False)
        expected_dtype = np.dtype(str(entry["dtype"]))
        expected_shape = tuple(int(value) for value in entry["shape"])
        if values.dtype != expected_dtype or values.shape != expected_shape:
            raise ArtifactError(
                f"{path.name} is {values.dtype}{values.shape}, manifest says "
                f"{expected_dtype}{expected_shape}"
            )
        loaded[name] = values

    return LoadedArrays(
        directory=artifact.directory,
        dataset_id=artifact.dataset_id,
        X=loaded["X"],
        y_direction=loaded["y_direction"],
        y_return=loaded["y_return"],
        ts=loaded["ts"],
        split=loaded["split"],
        manifest=_read_manifest(artifact.directory),
        memmapped=bool(mmap),
    )


def _read_manifest(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / ARTIFACT_FILE).read_text(encoding="utf-8"))
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ArtifactError("artifact manifest has no dataset manifest")
    return manifest
