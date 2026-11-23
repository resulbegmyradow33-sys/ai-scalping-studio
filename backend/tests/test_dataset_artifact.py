"""Persisted dataset arrays: round-trip, memmap, integrity and manifest safety."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

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


def payload_of(directory: Path) -> dict:
    return json.loads((directory / ARTIFACT_FILE).read_text(encoding="utf-8"))


def write_payload(directory: Path, payload: dict) -> None:
    (directory / ARTIFACT_FILE).write_text(json.dumps(payload), encoding="utf-8")


def temp_leftovers(directory: Path) -> list[str]:
    """Every scratch file left behind, hidden dotfiles included."""
    return sorted(path.name for path in directory.iterdir() if ".tmp" in path.name)


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
    assert temp_leftovers(tmp_path) == []
    load_dataset_arrays(tmp_path)


def test_concurrent_saves_keep_distinct_temp_files(tmp_path) -> None:
    source = bundle()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: save_dataset_arrays(tmp_path, source), range(4)))

    assert temp_leftovers(tmp_path) == []
    assert sorted(path.name for path in tmp_path.glob("*.npy")) == [
        "X.npy",
        "split.npy",
        "ts.npy",
        "y_direction.npy",
        "y_return.npy",
    ]
    load_dataset_arrays(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "../../../../etc/hosts",
        "/etc/hosts",
        "../X.npy",
        "nested/X.npy",
        "..\\X.npy",
        ".",
        "..",
        "",
    ],
)
def test_manifest_file_must_be_a_plain_name_inside_the_artifact(tmp_path, name) -> None:
    save_dataset_arrays(tmp_path, bundle())
    payload = payload_of(tmp_path)
    payload["arrays"]["X"]["file"] = name
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match="plain file name"):
        load_dataset_arrays(tmp_path)


def test_manifest_file_cannot_follow_a_symlink_out_of_the_artifact(tmp_path) -> None:
    outside = Path("/etc/hosts")
    if not outside.is_file():
        pytest.skip("/etc/hosts is not available on this platform")

    save_dataset_arrays(tmp_path, bundle())
    target = tmp_path / "y_direction.npy"
    target.unlink()
    os.symlink(outside, target)
    payload = payload_of(tmp_path)
    payload["arrays"]["y_direction"]["file"] = "y_direction.npy"
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match="escapes"):
        load_dataset_arrays(tmp_path)


@pytest.mark.parametrize("field", ["file", "dtype", "shape", "checksum"])
def test_manifest_entry_missing_a_required_field_is_an_artifact_error(tmp_path, field) -> None:
    save_dataset_arrays(tmp_path, bundle())
    payload = payload_of(tmp_path)
    del payload["arrays"]["X"][field]
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match=field):
        load_dataset_arrays(tmp_path)


def test_manifest_entry_must_be_an_object(tmp_path) -> None:
    save_dataset_arrays(tmp_path, bundle())
    payload = payload_of(tmp_path)
    payload["arrays"]["X"] = "X.npy"
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match="not an object"):
        load_dataset_arrays(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dtype", "not-a-dtype", "unknown dtype"),
        ("dtype", 3, "non-string dtype"),
        ("shape", "many", "non-list shape"),
        ("shape", [-1], "invalid shape"),
        ("shape", [True], "invalid shape"),
    ],
)
def test_manifest_entry_with_invalid_metadata_is_rejected(tmp_path, field, value, match) -> None:
    save_dataset_arrays(tmp_path, bundle())
    payload = payload_of(tmp_path)
    payload["arrays"]["X"][field] = value
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match=match):
        load_dataset_arrays(tmp_path)


def test_shape_mismatch_on_load_is_rejected(tmp_path) -> None:
    save_dataset_arrays(tmp_path, bundle())
    payload = payload_of(tmp_path)
    payload["arrays"]["y_direction"]["shape"] = [1]
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match="manifest says"):
        load_dataset_arrays(tmp_path)


def test_dtype_mismatch_on_load_is_rejected(tmp_path) -> None:
    save_dataset_arrays(tmp_path, bundle())
    payload = payload_of(tmp_path)
    assert payload["arrays"]["y_return"]["dtype"].startswith("float")
    payload["arrays"]["y_return"]["dtype"] = "int8"
    write_payload(tmp_path, payload)

    with pytest.raises(ArtifactError, match="manifest says"):
        load_dataset_arrays(tmp_path)
