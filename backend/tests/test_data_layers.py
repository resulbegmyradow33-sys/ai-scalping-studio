"""Explicit data layers: files, manifest, checksums and fail-closed validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.datasets.layers import LAYER_NAMES, build_layers
from scalping.features.build import FEATURE_COLUMNS

START = datetime(2026, 1, 5, tzinfo=UTC)


def seed_store(root) -> ParquetStore:
    store = ParquetStore(root)
    frame = SyntheticProvider(seed=7).fetch("XAUUSD", START, START + timedelta(days=2))
    store.write("XAUUSD", "M1", frame)
    return store


def test_writes_all_six_layers_with_a_manifest(tmp_path) -> None:
    store_root = tmp_path / "store"
    out = tmp_path / "layers"
    seed_store(store_root)

    manifest = build_layers(store_root, out, "XAUUSD", "M1")
    manifest_json = json.loads((out / "layers.json").read_text())

    assert {item.layer for item in manifest.files} == set(LAYER_NAMES)
    assert manifest_json["dataset_id"] == manifest.dataset_id
    assert manifest.dataset_id.startswith("XAUUSD:M1:")
    assert manifest.raw_rows == 2_881
    assert manifest.purged_rows > 0
    for item in manifest.files:
        assert int(item.to_dict()["rows"]) == manifest_json["files"][LAYER_NAMES.index(item.layer)]["rows"]

    raw = pl.read_parquet(out / "raw" / "XAUUSD" / "M1.parquet")
    cleaned = pl.read_parquet(out / "cleaned" / "XAUUSD" / "M1.parquet")
    assert raw.equals(cleaned)
    raw_manifest = next(item for item in manifest.files if item.layer == "raw")
    clean_manifest = next(item for item in manifest.files if item.layer == "cleaned")
    assert raw_manifest.checksum == clean_manifest.checksum


def test_feature_and_split_layers_are_row_consistent(tmp_path) -> None:
    store_root = tmp_path / "store"
    out = tmp_path / "layers"
    seed_store(store_root)

    manifest = build_layers(store_root, out, "XAUUSD", "M1")

    features = pl.read_parquet(out / "features" / "XAUUSD" / "M1.parquet")
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert {"ts", "y_direction", "y_return", "split"} <= set(features.columns)
    assert features.height < manifest.raw_rows  # warm-up and tail horizon are absent

    split_rows = 0
    for index, name in enumerate(("training", "validation", "test")):
        view = pl.read_parquet(out / name / "XAUUSD" / "M1.parquet")
        assert set(view["split"].unique().to_list()) <= {index}
        split_rows += view.height
    assert split_rows + manifest.purged_rows == features.height


def test_invalid_slice_fails_before_any_layer_is_written(tmp_path) -> None:
    store_root = tmp_path / "store"
    seed_store(store_root)
    path = store_root / "XAUUSD" / "M1" / "2026" / "01.parquet"
    frame = pl.read_parquet(path)
    broken = frame.with_columns((pl.col("bid_close") - 1.0).alias("ask_close"))
    broken.write_parquet(path)
    out = tmp_path / "layers"

    with pytest.raises(ValueError, match="not structurally valid"):
        build_layers(store_root, out, "XAUUSD", "M1")

    assert not (out / "cleaned").exists()
    assert not (out / "layers.json").exists()


def test_empty_store_and_missing_symbol_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="no XAUUSD M1 bars"):
        build_layers(tmp_path / "empty", tmp_path / "layers", "XAUUSD", "M1")


def test_symbol_and_timeframe_must_be_safe_components(tmp_path) -> None:
    store_root = tmp_path / "store"
    seed_store(store_root)
    out = tmp_path / "layers"

    for bad in ("../escape", "a/b", "..", ""):
        with pytest.raises(ValueError, match="invalid symbol"):
            build_layers(store_root, out, bad, "M1")
        with pytest.raises(ValueError, match="invalid timeframe"):
            build_layers(store_root, out, "XAUUSD", bad)
