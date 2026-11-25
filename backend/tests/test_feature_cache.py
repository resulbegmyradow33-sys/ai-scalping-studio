"""Feature-frame cache tests.

The load-bearing property is **bit identity**: a cache hit must return the exact
arrays ``build_features`` would have produced, because every downstream artefact
(dataset id, ``data_hash``, run id) hashes those bytes. The tests below cover the
key inputs (data, costs, registry), corruption and quarantine, concurrency, a
cross-process key, the CLI wiring and the ML runner wiring.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scalping.cli import main
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.features import cache as cache_module
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame, build_features
from scalping.features.cache import (
    CACHE_VERSION,
    FEATURE_CACHE_DIRNAME,
    FeatureCache,
    FrameMeta,
    cached_build_features,
    frame_meta,
    registry_version,
)
from scalping.types import CostConfig

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
DAYS = 3
_ARRAYS = ("X", "ts", "ohlc", "spread", "atr")


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return SyntheticProvider(seed=7).fetch(SYMBOL, START, START + timedelta(days=DAYS))


@pytest.fixture(scope="module")
def built(bars: pl.DataFrame) -> FeatureFrame:
    return build_features(bars, Settings().costs)


@pytest.fixture
def cache(tmp_path: Path) -> FeatureCache:
    return FeatureCache(tmp_path / FEATURE_CACHE_DIRNAME)


def same_frame(left: FeatureFrame, right: FeatureFrame) -> bool:
    """Exact equality of every array, dtype, column name and ``valid_from``.

    ``np.array_equal`` is not enough: the warm-up head carries NaN, and NaN is
    not equal to itself. Comparing bytes is the actual contract.
    """
    if left.valid_from != right.valid_from or left.columns != right.columns:
        return False
    for name in _ARRAYS:
        a, b = getattr(left, name), getattr(right, name)
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            return False
    return True


class BuildCounter:
    """Wrap the real builder and count how often it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: pl.DataFrame, costs: CostConfig | None = None) -> FeatureFrame:
        self.calls += 1
        return build_features(frame, costs)


def _key(cache: FeatureCache, frame: pl.DataFrame, settings: Settings | None = None) -> str:
    return cache.key_for(
        SYMBOL, TIMEFRAME, frame_meta(frame), Settings() if settings is None else settings
    )


# --- hits, misses and bit identity -------------------------------------------


def test_second_call_hits_without_rebuilding(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)
    first = cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )
    second = cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )

    assert counter.calls == 1
    assert same_frame(first, built)
    assert same_frame(second, built)
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["saves"] == 1
    assert stats["entries"] == 1


def test_a_miss_rebuilds_and_invalidation_forces_another(
    cache: FeatureCache, bars: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)
    key = _key(cache, bars)
    assert cache.load(key) is None

    cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )
    assert counter.calls == 1

    assert cache.invalidate(key) is True
    assert cache.load(key) is None
    cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )
    assert counter.calls == 2


def test_disabling_the_cache_always_builds(
    bars: pl.DataFrame, built: FeatureFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)
    for _ in range(2):
        frame = cached_build_features(
            None, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
        )
        assert same_frame(frame, built)
    assert counter.calls == 2
    assert not any(tmp_path.rglob("*.npz"))


def test_saved_entry_restores_dtypes_and_layout(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame
) -> None:
    key = _key(cache, bars)
    path = cache.save(key, built)
    assert path.is_file()
    loaded = cache.load(key)
    assert loaded is not None
    assert same_frame(loaded, built)
    assert loaded.X.dtype == np.float32
    assert loaded.ts.dtype == np.dtype("datetime64[us]")
    assert loaded.ohlc.dtype == np.float64
    assert loaded.columns == list(FEATURE_COLUMNS)
    assert loaded.X.flags["C_CONTIGUOUS"]
    assert loaded.valid_from == built.valid_from


def test_no_pickle_is_used(cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame) -> None:
    key = _key(cache, bars)
    array_path = cache.save(key, built)
    with np.load(array_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(_ARRAYS)
        for name in archive.files:
            assert archive[name].dtype.kind != "O"
    manifest = json.loads(cache._manifest_path(key).read_text(encoding="utf-8"))
    assert manifest["version"] == CACHE_VERSION
    assert set(manifest["arrays"]) == set(_ARRAYS)
    assert "pickle" not in json.dumps(manifest)


# --- key inputs ---------------------------------------------------------------


def test_key_covers_symbol_timeframe_and_data(cache: FeatureCache, bars: pl.DataFrame) -> None:
    base = _key(cache, bars)
    assert base == _key(cache, bars), "the same bars must give the same key"
    assert base != cache.key_for(SYMBOL, "H1", frame_meta(bars), Settings())
    assert base != cache.key_for("EURUSD", TIMEFRAME, frame_meta(bars), Settings())
    assert base != _key(cache, bars.head(-1))
    assert base == _key(cache, bars.reverse()), "row order is normalised before hashing"

    moved = bars.with_columns(
        pl.when(pl.int_range(pl.len()) == 10)
        .then(pl.col("close") + 0.01)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert base != _key(cache, moved)


def test_key_covers_costs_and_registry(
    cache: FeatureCache, bars: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _key(cache, bars)
    assert base != _key(cache, bars, Settings(costs=CostConfig(spread_floor=0.31)))
    assert base != _key(cache, bars, Settings(costs=CostConfig(spread_vol_mult=0.5)))

    monkeypatch.setattr(cache_module, "registry_version", lambda: "different-registry")
    assert base != _key(cache, bars)


def test_a_settings_change_rebuilds_and_is_not_a_hit(
    cache: FeatureCache, bars: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different cost block is a different input, even mid-run."""
    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)
    changed = Settings(costs=CostConfig(spread_floor=0.31))

    first = cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )
    second = cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=changed
    )

    assert counter.calls == 2, "a changed cost block must not reuse the old entry"
    assert same_frame(first, build_features(bars, Settings().costs))
    assert same_frame(second, build_features(bars, changed.costs))
    stats = cache.stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 2


def test_a_data_change_rebuilds_and_is_not_a_hit(
    cache: FeatureCache, bars: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One edited bar changes the data identity and therefore the key."""
    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)
    edited = bars.with_columns(
        pl.when(pl.int_range(pl.len()) == 10)
        .then(pl.col("close") + 0.01)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )
    cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=edited, settings=Settings()
    )

    assert counter.calls == 2, "an edited bar must not hit the old entry"
    assert cache.stats()["misses"] == 2
    assert cache.stats()["entries"] == 2


def test_a_version_bump_rejects_an_existing_entry(
    cache: FeatureCache,
    bars: pl.DataFrame,
    built: FeatureFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bumping CACHE_VERSION changes the key *and* invalidates an old manifest."""
    key = _key(cache, bars)
    cache.save(key, built)
    assert cache.load(key) is not None

    monkeypatch.setattr(cache_module, "CACHE_VERSION", CACHE_VERSION + 1)
    assert _key(cache, bars) != key, "a version bump must change the key"
    assert cache.load(key) is None, "the manifest version check must reject the old entry"
    assert cache.stats()["corrupt"] == 1
    assert _bad_files(cache, key)


def test_registry_version_is_a_stable_digest() -> None:
    assert registry_version() == registry_version()
    assert len(registry_version()) == 16


def test_meta_describes_the_data_identity(bars: pl.DataFrame) -> None:
    meta = frame_meta(bars)
    assert isinstance(meta, FrameMeta)
    assert meta.rows == bars.height
    assert meta.first_ts == "2026-01-05T00:00:00+00:00"
    assert meta.last_ts == "2026-01-08T00:00:00+00:00"
    assert len(meta.fingerprint) == 40


def test_key_is_stable_across_a_process_restart(tmp_path: Path) -> None:
    """A key computed in a fresh interpreter must equal the in-process one."""
    code = f"""
from datetime import UTC, datetime, timedelta
from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.features.cache import FeatureCache, frame_meta

start = datetime(2026, 1, 5, tzinfo=UTC)
frame = SyntheticProvider(seed=7).fetch(
    "{SYMBOL}", start, start + timedelta(days=1)
)
cache = FeatureCache({str(tmp_path / "unused")!r})
print(cache.key_for("{SYMBOL}", "{TIMEFRAME}", frame_meta(frame), Settings()))
"""
    env = {**os.environ, "PYTHONHASHSEED": "123"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    frame = SyntheticProvider(seed=7).fetch(SYMBOL, START, START + timedelta(days=1))
    local = FeatureCache(tmp_path / "local").key_for(
        SYMBOL, TIMEFRAME, frame_meta(frame), Settings()
    )
    assert result.stdout.strip() == local


# --- corruption ---------------------------------------------------------------


def _bad_files(cache: FeatureCache, key: str) -> list[Path]:
    return sorted(cache._directory(key).glob(f"*{key}*.bad"))


def test_a_truncated_npz_is_a_miss_and_is_quarantined(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame
) -> None:
    key = _key(cache, bars)
    path = cache.save(key, built)
    path.write_bytes(path.read_bytes()[:100])

    assert cache.load(key) is None
    assert cache.stats()["corrupt"] == 1
    assert not cache._array_path(key).exists()
    assert _bad_files(cache, key)
    assert cache.stats()["entries"] == 0


def test_a_broken_manifest_is_quarantined(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame
) -> None:
    key = _key(cache, bars)
    cache.save(key, built)
    cache._manifest_path(key).write_text("{not json", encoding="utf-8")

    assert cache.load(key) is None
    assert cache.stats()["corrupt"] == 1
    assert _bad_files(cache, key)


def test_a_wrong_checksum_is_rejected(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame
) -> None:
    key = _key(cache, bars)
    cache.save(key, built)
    manifest_path = cache._manifest_path(key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksum"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert cache.load(key) is None
    assert cache.stats()["corrupt"] == 1
    assert not cache._manifest_path(key).exists()
    assert _bad_files(cache, key)


def test_mismatched_shapes_are_rejected(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame
) -> None:
    key = _key(cache, bars)
    array_path = cache.save(key, built)
    with open(array_path, "wb") as handle:
        np.savez(
            handle, X=built.X[:10], ts=built.ts, ohlc=built.ohlc, spread=built.spread, atr=built.atr
        )

    assert cache.load(key) is None
    assert cache.stats()["corrupt"] == 1


def test_a_rejected_entry_is_rebuilt_not_reused(
    cache: FeatureCache,
    bars: pl.DataFrame,
    built: FeatureFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _key(cache, bars)
    path = cache.save(key, built)
    path.write_bytes(b"broken")
    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)

    frame = cached_build_features(
        cache, symbol=SYMBOL, timeframe=TIMEFRAME, frame=bars, settings=Settings()
    )
    assert counter.calls == 1
    assert same_frame(frame, built)
    assert cache.load(key) is not None
    assert _bad_files(cache, key)


# --- concurrency and maintenance ---------------------------------------------


def test_concurrent_save_and_load_never_reads_a_partial_entry(
    cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame
) -> None:
    key = _key(cache, bars)
    cache.save(key, built)
    failures: list[str] = []

    def reader() -> None:
        for _ in range(10):
            frame = cache.load(key)
            if frame is None:
                failures.append("miss on a live entry")
            elif not same_frame(frame, built):
                failures.append("arrays differ from the build")

    def writer() -> None:
        for _ in range(5):
            cache.save(key, built)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(reader) for _ in range(4)]
        futures += [pool.submit(writer) for _ in range(2)]
        for future in futures:
            future.result()

    assert failures == []
    assert cache.stats()["corrupt"] == 0
    assert cache.load(key) is not None


def test_clear_and_stats(cache: FeatureCache, bars: pl.DataFrame, built: FeatureFrame) -> None:
    cache.save(_key(cache, bars), built)
    assert cache.stats()["entries"] == 1
    assert cache.clear() == 2
    assert cache.stats()["entries"] == 0
    assert cache.clear() == 0


# --- integration --------------------------------------------------------------


def test_cli_dataset_build_reuses_then_skips_the_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "data"
    assert (
        main(
            [
                "data",
                "synth",
                "--symbol",
                SYMBOL,
                "--start",
                "2026-01-05",
                "--end",
                "2026-01-07",
                "--root",
                str(root),
            ]
        )
        == 0
    )
    cache_dir = root / FEATURE_CACHE_DIRNAME

    def build(out: Path, *extra: str) -> str:
        assert (
            main(
                [
                    "dataset",
                    "build",
                    "--symbol",
                    SYMBOL,
                    "--timeframe",
                    TIMEFRAME,
                    "--root",
                    str(root),
                    "--out",
                    str(out),
                    *extra,
                ]
            )
            == 0
        )
        return str(json.loads(out.read_text(encoding="utf-8"))["dataset_id"])

    first = build(tmp_path / "first.json")
    assert list(cache_dir.rglob("*.npz"))
    second = build(tmp_path / "second.json")
    assert first == second

    shutil.rmtree(cache_dir)
    third = build(tmp_path / "third.json", "--no-cache")
    assert first == third
    assert not cache_dir.exists()
    capsys.readouterr()


def test_ml_runner_wires_the_cache_and_can_disable_it(
    tmp_path: Path, bars: pl.DataFrame, built: FeatureFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scalping.api.ml_runner import MLRunner

    store = ParquetStore(tmp_path / "store")
    runs = tmp_path / "runs"
    runner = MLRunner(store, Settings(), runs)
    assert runner.cache is not None
    assert runner.cache.root == runs / FEATURE_CACHE_DIRNAME
    assert MLRunner(store, Settings(), runs, use_cache=False).cache is None

    counter = BuildCounter()
    monkeypatch.setattr(cache_module, "build_features", counter)
    first = runner._features(bars, SYMBOL, TIMEFRAME)
    second = runner._features(bars, SYMBOL, TIMEFRAME)
    assert counter.calls == 1
    assert same_frame(first, built)
    assert same_frame(second, built)

    disabled = MLRunner(store, Settings(), tmp_path / "runs-off", use_cache=False)
    disabled._features(bars, SYMBOL, TIMEFRAME)
    assert counter.calls == 2
