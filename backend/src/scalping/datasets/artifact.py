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

The manifest is untrusted input. Every ``file`` entry must be a plain file name
and its resolved path must stay inside the artifact directory, so a crafted
manifest cannot make the loader read arbitrary files. A manifest missing a
required field raises :class:`ArtifactError` at parse time, never ``KeyError``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final

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
_REQUIRED_FIELDS: Final[tuple[str, ...]] = ("file", "dtype", "shape", "checksum")
_CHUNK: Final[int] = 1024 * 1024


class ArtifactError(ValueError):
    """The artifact is missing, old or corrupt."""


@dataclass(frozen=True, slots=True)
class ArrayArtifact:
    """Lightweight, validated view of an artifact on disk.

    ``arrays`` maps array names to normalised entries: ``file`` and ``checksum``
    are strings, ``dtype`` is a canonical dtype name and ``shape`` is a tuple of
    non-negative ints.
    """

    directory: Path
    dataset_id: str
    manifest: dict[str, Any]
    arrays: dict[str, dict[str, Any]]

    def path_for(self, name: str) -> Path:
        """Return the on-disk path of array ``name``, confined to the artifact."""
        if name not in self.arrays:
            raise ArtifactError(f"artifact has no array {name!r}")
        return _member_path(self.directory, self.arrays[name]["file"])


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


def _check_member_name(name: str) -> None:
    """Reject anything that is not a plain file name inside the directory."""
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).is_absolute()
    ):
        raise ArtifactError(f"artifact file name {name!r} must be a plain file name")


def _member_path(directory: Path, name: str) -> Path:
    """Resolve a manifest ``file`` value to a path inside ``directory``.

    The name check rejects traversal syntax outright; the resolve check then
    catches a symlink inside the directory that points elsewhere.
    """
    _check_member_name(name)
    root = directory.resolve()
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root):
        raise ArtifactError(f"artifact file name {name!r} escapes {root}")
    return path


def _parse_dtype(name: str, dtype: Any) -> str:
    if not isinstance(dtype, str):
        raise ArtifactError(f"artifact entry {name!r} has a non-string dtype")
    try:
        return str(np.dtype(dtype))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact entry {name!r} has an unknown dtype {dtype!r}") from exc


def _parse_shape(name: str, shape: Any) -> tuple[int, ...]:
    if not isinstance(shape, (list, tuple)):
        raise ArtifactError(f"artifact entry {name!r} has a non-list shape")
    dimensions: list[int] = []
    for value in shape:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArtifactError(f"artifact entry {name!r} has an invalid shape {shape!r}")
        dimensions.append(value)
    return tuple(dimensions)


def _parse_entry(name: str, entry: Any) -> dict[str, Any]:
    """Validate one ``arrays`` entry; normalise its types."""
    if not isinstance(entry, dict):
        raise ArtifactError(f"artifact entry {name!r} is not an object")
    missing = [field for field in _REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ArtifactError(f"artifact entry {name!r} is missing {', '.join(missing)}")
    file_name = entry["file"]
    if not isinstance(file_name, str):
        raise ArtifactError(f"artifact entry {name!r} has a non-string file")
    _check_member_name(file_name)
    checksum = entry["checksum"]
    if not isinstance(checksum, str) or not checksum:
        raise ArtifactError(f"artifact entry {name!r} has an invalid checksum")
    return {
        "file": file_name,
        "dtype": _parse_dtype(name, entry["dtype"]),
        "shape": _parse_shape(name, entry["shape"]),
        "checksum": checksum,
    }


def _digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_atomic_bytes(path: Path, writer: Callable[[BinaryIO], object]) -> None:
    """Write through a unique sibling temp file, then move it onto ``path``.

    ``tempfile.mkstemp`` gives every save its own scratch file: two concurrent
    saves into one directory must not replace or unlink each other's temp file.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_array(path: Path, array: np.ndarray) -> None:
    # np.save appends ``.npy`` when given a path, so write through the handle.
    _write_atomic_bytes(
        path,
        lambda handle: np.save(handle, np.ascontiguousarray(array), allow_pickle=False),
    )


def _write_manifest(root: Path, payload: dict[str, Any]) -> Path:
    target = root / ARTIFACT_FILE
    _write_atomic_bytes(
        target,
        lambda handle: handle.write((json.dumps(payload, indent=2) + "\n").encode("utf-8")),
    )
    return target


def save_dataset_arrays(directory: str | Path, bundle: DatasetBundle) -> Path:
    """Write ``bundle`` to ``directory``; return the manifest path.

    Each array is written to a unique temp file and moved into place; the
    manifest is written last and is the commit marker. Concurrent saves into the
    same directory are safe: the last manifest to land wins and every array file
    is a complete ``.npy``.
    """
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
    return _write_manifest(root, payload)


def _load_payload(root: Path) -> dict[str, Any]:
    meta = root / ARTIFACT_FILE
    if not meta.is_file():
        raise ArtifactError(f"no {ARTIFACT_FILE} under {root}; not a saved dataset")
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"artifact manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("artifact manifest must be a JSON object")
    if payload.get("version") != 1:
        raise ArtifactError(f"unsupported artifact version {payload.get('version')!r}")
    return payload


def _read_artifact(directory: str | Path) -> ArrayArtifact:
    root = Path(directory)
    payload = _load_payload(root)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ArtifactError("artifact manifest has no dataset manifest")
    arrays = payload.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise ArtifactError("artifact manifest has no usable array table")
    parsed = {name: _parse_entry(name, entry) for name, entry in arrays.items()}
    return ArrayArtifact(
        directory=root,
        dataset_id=str(payload.get("dataset_id", "")),
        manifest=manifest,
        arrays=parsed,
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
        if name not in artifact.arrays:
            raise ArtifactError(f"artifact is missing the {name!r} array")
        entry = artifact.arrays[name]
        path = artifact.path_for(name)
        if not path.is_file():
            raise ArtifactError(f"artifact array file is missing: {path}")
        if verify and _digest(path) != entry["checksum"]:
            raise ArtifactError(f"checksum mismatch for {path.name}")
        if name == "X" and mmap:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            values = np.load(path, allow_pickle=False)
        expected_dtype = np.dtype(entry["dtype"])
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
        manifest=artifact.manifest,
        memmapped=bool(mmap),
    )
