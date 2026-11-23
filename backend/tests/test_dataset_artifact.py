"""Persisted dataset arrays: round-trip, memmap and integrity checks."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.artifact import (
    ARTIFACT_FILE,
    ArtifactError,
    load_dataset_arrays,
    save_dataset_arrays,
)
from scalping.datasets.builder import build_dataset
from scalping.features.build import build_features


def bundle():
    frame = SyntheticProvider(seed=7).fetch(
        "XAUUSD", datetime(2026, 1, 5, tzinfo=UTC), datetime(2026, 1, 6, tzinfo=UTC)
    )
    return build_dataset(build_features(frame), horizon=1, purge_bars=1)


def test_round_trip_preserves_arrays_and_id(tmp_path) -> None:
    source = bundle()
    manifest_path = save_dataset_arrays(tmp_path, source)

    assert manifest_path == tmp_path / ARTIFACT_FILE
    loaded = load_dataset_arrays(tmp_path)

    np.testing.assert_array_equal(loaded.X, source.X)
    np.testing.assert_array_equal(loaded.y_direction, source.y_direction)
    np.testing.assert_array_equal(loaded.y_return, source.y_return)
    np.testing.assert_array_equal(loaded.ts, source.ts)
    np.testing.assert_array_equal(loaded.split, source.split)
    assert loaded.dataset_id == source.manifest.dataset_id
    assert loaded.manifest["dataset_id"] == source.manifest.dataset_id
    assert loaded.rows("train") == source.rows("train")
    assert loaded.X.dtype == source.X.dtype


def test_x_is_a_read_only_memory_map_by_default(tmp_path) -> None:
    save_dataset_arrays(tmp_path, bundle())

    mapped = load_dataset_arrays(tmp_path)
    plain = load_dataset_arrays(tmp_path, mmap=False)

    assert isinstance(mapped.X, np.memmap)
    assert mapped.memmapped is True
    assert not isinstance(plain.X, np.memmap)
    assert plain.memmapped is False
    with pytest.raises(ValueError):
        mapped.X[0, 0] = 1.0


def test_tampering_is_caught_by_the_checksum(tmp_path) -> None:
    save_dataset_arrays(tmp_path, bundle())
    path = tmp_path / "y_direction.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_dataset_arrays(tmp_path)
    load_dataset_arrays(tmp_path, verify=False)  # explicit opt-out


def test_missing_manifest_or_array_is_rejected(tmp_path) -> None:
    with pytest.raises(ArtifactError, match="no artifact.json"):
        load_dataset_arrays(tmp_path)

    save_dataset_arrays(tmp_path, bundle())
    (tmp_path / "ts.npy").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_dataset_arrays(tmp_path)


def test_saving_twice_replaces_the_artifact(tmp_path) -> None:
    source = bundle()
    save_dataset_arrays(tmp_path, source)
    save_dataset_arrays(tmp_path, source)

    assert len(list(tmp_path.glob("*.npy"))) == 5
    assert len(list(tmp_path.glob("*.tmp"))) == 0
    load_dataset_arrays(tmp_path)
