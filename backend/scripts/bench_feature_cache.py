"""Measure the feature cache on real bars: hit vs miss, save/load, key cost.

This script measures; it does not validate a strategy, prove profitability or
accept anything. It reads real bars from an existing parquet store, bounded by
``--case`` rows, and never writes to the store. Cache entries go to a temporary
directory that is removed before exit.

One case is ``label=store_root:timeframe:rows`` where ``rows=0`` means the whole
store. Every stage is timed with
:func:`scalping.bench.pipeline.measure_stage`, so the report carries the same
wall/CPU/RSS shape as ``scripts/bench_pipeline.py``. On top of that the script
records the entry size, the cold-miss/first-hit split through
``cached_build_features``, corruption quarantine and key invalidation.

Examples
--------
    cd backend && uv run python scripts/bench_feature_cache.py \\
        --case m1_20k=../../../data-ticks5-store:M1:20000 \\
        --case m1_200k=../../../data-ticks5-store:M1:200000 \\
        --case h1_full=../../../data-real-store:H1:0 \\
        --out ../docs/evidence/cache-2026-11-24.json

The report is written to ``--out`` and the same JSON object is printed on
stdout as one line. Errors print one JSON line to stderr: exit code 2 for an
unusable input or a failed stage, 1 for an unexpected failure.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from scalping.bench.pipeline import StageError, measure_stage

# The suite keeps the one environment/claims block; reuse it so a cache report
# carries the same schema and the same "not acceptance" wording as the pipeline.
from scalping.bench.suite import _claims, _environment
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.features import cache as cache_module
from scalping.features.build import build_features
from scalping.features.cache import (
    CACHE_VERSION,
    FeatureCache,
    cached_build_features,
    frame_meta,
)
from scalping.types import CostConfig

SCHEMA: Final[str] = "scalping.bench.cache/1"

#: Repeated cache calls after one cold miss; the first call builds, the rest hit.
CALL_LOOP: Final[int] = 10

#: A changed cost value used to prove settings invalidation; small but non-zero.
_INVALIDATING_FLOOR: Final[float] = 0.31


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="LABEL=ROOT:TIMEFRAME:ROWS",
        help="one measured slice; ROWS=0 reads the whole store (repeatable)",
    )
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--repeats", type=int, default=3, help="measured repeats per stage (>= 1)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded calls per stage (>= 0)")
    parser.add_argument("--out", default="runs/bench/feature-cache.json")
    return parser


def _parse_case(raw: str) -> tuple[str, Path, str, int]:
    label, separator, spec = raw.partition("=")
    if not separator or not label or not spec:
        raise ValueError(f"case must be LABEL=ROOT:TIMEFRAME:ROWS, got {raw!r}")
    root, timeframe, rows_text = spec.rsplit(":", 2)
    frames = {"M1", "H1"}
    if timeframe not in frames:
        raise ValueError(f"timeframe must be one of {sorted(frames)}, got {timeframe!r}")
    rows = int(rows_text)
    if rows < 0:
        raise ValueError(f"rows must be >= 0, got {rows}")
    return label, Path(root), timeframe, rows


def _read_frame(
    root: Path, symbol: str, timeframe: str, rows: int
) -> tuple[pl.DataFrame, int, float]:
    started = time.perf_counter()
    frame = ParquetStore(root).read(symbol, timeframe)
    elapsed = time.perf_counter() - started
    if frame.is_empty():
        raise FileNotFoundError(f"no {symbol} {timeframe} bars under {root}")
    available = frame.height
    return (frame if rows == 0 else frame.head(rows)), available, elapsed


def _entry_bytes(cache: FeatureCache, key: str, frame: Any) -> dict[str, int]:
    npz = cache._array_path(key).stat().st_size
    manifest = cache._manifest_path(key).stat().st_size
    arrays = sum(int(getattr(frame, name).nbytes) for name in ("X", "ts", "ohlc", "spread", "atr"))
    return {
        "npz_bytes": npz,
        "manifest_bytes": manifest,
        "entry_bytes": npz + manifest,
        "array_bytes": arrays,
        "x_bytes": int(frame.X.nbytes),
    }


def _corruption_check(cache: FeatureCache, key: str, frame: Any) -> dict[str, Any]:
    """Save, truncate the ``.npz`` and confirm a rejected load quarantines it."""
    path = cache.save(key, frame)
    path.write_bytes(path.read_bytes()[:100])
    before = cache.stats()
    loaded = cache.load(key)
    after = cache.stats()
    quarantined = sorted(cache._directory(key).glob("*.bad"))
    return {
        "load_returns_none": loaded is None,
        "corrupt_delta": int(after["corrupt"]) - int(before["corrupt"]),
        "miss_delta": int(after["misses"]) - int(before["misses"]),
        "quarantined_files": len(quarantined),
        "live_entries": int(after["entries"]),
    }


def _invalidation_check(
    cache: FeatureCache,
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame,
    settings: Settings,
    base_key: str,
) -> dict[str, Any]:
    """Prove a settings edit and a single-bar edit each miss instead of hitting."""
    changed_settings = Settings(costs=CostConfig(spread_floor=_INVALIDATING_FLOOR))
    settings_key = cache.key_for(symbol, timeframe, frame_meta(frame), changed_settings)

    edited = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("close") + 0.01)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    data_key = cache.key_for(symbol, timeframe, frame_meta(edited), settings)

    # One entry built under the base key, then re-requested with a changed cost
    # block and with one edited bar: both must be misses, not silent hits.
    first = cached_build_features(
        cache, symbol=symbol, timeframe=timeframe, frame=frame, settings=settings
    )
    after_first = cache.stats()
    same = cached_build_features(
        cache, symbol=symbol, timeframe=timeframe, frame=frame, settings=settings
    )
    after_same = cache.stats()
    cached_build_features(
        cache, symbol=symbol, timeframe=timeframe, frame=frame, settings=changed_settings
    )
    after_settings = cache.stats()
    cached_build_features(
        cache, symbol=symbol, timeframe=timeframe, frame=edited, settings=settings
    )
    after_data = cache.stats()

    return {
        "settings_key_differs": settings_key != base_key,
        "data_key_differs": data_key != base_key,
        "first_call_miss": int(after_first["misses"]) == 1 and int(after_first["hits"]) == 0,
        "second_call_hits": int(after_same["hits"]) == 1,
        "settings_change_misses": int(after_settings["misses"]) - int(after_same["misses"]) == 1,
        "data_change_misses": int(after_data["misses"]) - int(after_settings["misses"]) == 1,
        "entries": int(after_data["entries"]),
        "same_arrays": first.X.tobytes() == same.X.tobytes(),
    }


def _version_check(
    cache: FeatureCache,
    key: str,
    frame: Any,
    *,
    symbol: str,
    timeframe: str,
    bars: pl.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    """Save under the live version, then load with a bumped one."""
    cache.save(key, frame)
    original = cache_module.CACHE_VERSION
    try:
        cache_module.CACHE_VERSION = original + 1  # type: ignore[misc]
        bumped_key = cache.key_for(symbol, timeframe, frame_meta(bars), settings)
        loaded = cache.load(key)
    finally:
        cache_module.CACHE_VERSION = original  # type: ignore[misc]
    return {
        "old_entry_rejected": loaded is None,
        "bumped_key_differs": bumped_key != key,
        "version": original,
    }


def _call_loop(
    cache_root: Path,
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame,
    settings: Settings,
    calls: int,
) -> dict[str, Any]:
    cache = FeatureCache(cache_root)
    seconds: list[float] = []
    for _ in range(calls):
        started = time.perf_counter()
        cached_build_features(
            cache, symbol=symbol, timeframe=timeframe, frame=frame, settings=settings
        )
        seconds.append(time.perf_counter() - started)
    stats = cache.stats()
    total = int(stats["hits"]) + int(stats["misses"])
    return {
        "calls": calls,
        "hits": int(stats["hits"]),
        "misses": int(stats["misses"]),
        "hit_ratio": round(int(stats["hits"]) / total, 4) if total else None,
        "cold_seconds": round(seconds[0], 6),
        "warm_seconds_median": round(float(statistics.median(seconds[1:])), 6)
        if len(seconds) > 1
        else None,
        "warm_seconds_min": round(min(seconds[1:]), 6) if len(seconds) > 1 else None,
        "warm_seconds_max": round(max(seconds[1:]), 6) if len(seconds) > 1 else None,
        "entries": int(stats["entries"]),
    }


def _load_average() -> str:
    load = os.getloadavg()
    return f"{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}"


def _measure_case(
    *,
    label: str,
    root: Path,
    timeframe: str,
    rows: int,
    symbol: str,
    repeats: int,
    warmup: int,
    workdir: Path,
) -> dict[str, Any]:
    load_at_case_start = _load_average()
    frame, available, store_read = _read_frame(root, symbol, timeframe, rows)
    settings = Settings()
    costs = settings.costs
    cache = FeatureCache(workdir / f"{label}-live")
    meta = frame_meta(frame)
    key = cache.key_for(symbol, timeframe, meta, costs)

    stages: dict[str, Any] = {}

    meta_stage = measure_stage(
        "frame_meta", lambda: {"rows": frame_meta(frame).rows}, repeats=repeats, warmup=warmup
    )
    if not meta_stage.ok:
        raise StageError(meta_stage)
    stages["frame_meta"] = meta_stage.to_dict()

    built = build_features(frame, costs)

    build_stage = measure_stage(
        "build_features",
        lambda: {"rows": int(build_features(frame, costs).X.shape[0])},
        repeats=repeats,
        warmup=warmup,
    )
    if not build_stage.ok:
        raise StageError(build_stage)
    stages["build_features"] = build_stage.to_dict()

    def compute_key() -> dict[str, int]:
        cache.key_for(symbol, timeframe, meta, costs)
        return {"rows": meta.rows}

    key_measure = measure_stage("key_for", compute_key, repeats=repeats, warmup=warmup)
    if not key_measure.ok:
        raise StageError(key_measure)
    stages["key_for"] = key_measure.to_dict()

    def do_save() -> dict[str, int]:
        cache.save(key, built)
        return {"rows": int(built.X.shape[0])}

    save_measure = measure_stage("save", do_save, repeats=repeats, warmup=1, disk_path=cache.root)
    if not save_measure.ok:
        raise StageError(save_measure)
    stages["save"] = save_measure.to_dict()

    def do_load() -> dict[str, int]:
        loaded = cache.load(key)
        if loaded is None:
            raise RuntimeError("live entry missed")
        return {"rows": int(loaded.X.shape[0])}

    load_measure = measure_stage("load", do_load, repeats=repeats, warmup=warmup)
    if not load_measure.ok:
        raise StageError(load_measure)
    stages["load"] = load_measure.to_dict()

    entry = _entry_bytes(cache, key, built)

    call_loop = _call_loop(
        workdir / f"{label}-loop",
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        settings=settings,
        calls=CALL_LOOP,
    )

    corruption_cache = FeatureCache(workdir / f"{label}-corrupt")
    corruption = _corruption_check(corruption_cache, key, built)

    invalidation_cache = FeatureCache(workdir / f"{label}-invalidate")
    invalidation = _invalidation_check(
        invalidation_cache,
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        settings=settings,
        base_key=key,
    )

    version_cache = FeatureCache(workdir / f"{label}-version")
    version = _version_check(
        version_cache,
        key,
        built,
        symbol=symbol,
        timeframe=timeframe,
        bars=frame,
        settings=settings,
    )

    meta_median = float(stages["frame_meta"]["wall_seconds"]["median"])
    key_median = float(stages["key_for"]["wall_seconds"]["median"])
    build_median = float(stages["build_features"]["wall_seconds"]["median"])
    load_median = float(stages["load"]["wall_seconds"]["median"])
    save_median = float(stages["save"]["wall_seconds"]["median"])
    return {
        "label": label,
        "store_root": str(root),
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(frame.height),
        "rows_available": int(available),
        "first_ts": meta.first_ts,
        "last_ts": meta.last_ts,
        "load_average_at_case_start": load_at_case_start,
        "store_read_seconds": round(store_read, 6),
        "key": key,
        "fingerprint_chars": len(meta.fingerprint),
        "stages": stages,
        "derived": {
            "fingerprint_share_of_build_plus_meta": round(
                meta_median / (meta_median + build_median), 4
            ),
            "key_total_share_of_build_plus_meta": round(
                (meta_median + key_median) / (meta_median + build_median), 4
            ),
            "meta_to_build_ratio": round(meta_median / build_median, 4) if build_median else None,
            "load_to_build_ratio": round(load_median / build_median, 4) if build_median else None,
            "save_to_build_ratio": round(save_median / build_median, 4) if build_median else None,
            "hit_speedup_vs_build": round(build_median / load_median, 2) if load_median else None,
        },
        "entry": entry,
        "call_loop": call_loop,
        "corruption": corruption,
        "invalidation": invalidation,
        "version": version,
    }


def _build_report(cases: Sequence[dict[str, Any]], load_at_start: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_commit": _git_head(),
        "branch": _git_branch(),
        "host": {
            **_environment(),
            "load_average_at_start": load_at_start,
            "note": (
                "local measurements on one machine at one time, from a real parquet "
                "store; load_average_at_start and load_average_at_case_start are "
                "recorded because other processes change wall time. When load is high, "
                "compare same-process ratios and the min/CPU columns before absolute "
                "seconds. One heavy job at a time is the intended condition."
            ),
        },
        "cache_version": CACHE_VERSION,
        "repeats": None,
        "cases": list(cases),
        "claims": _claims(["real_store"]),
    }


def _git_head() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is optional in this report
        return None


def _git_branch() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        parsed = [_parse_case(raw) for raw in args.case]
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        if args.warmup < 0:
            raise ValueError("--warmup must be >= 0")
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 2

    report: dict[str, Any] = {}
    cases: list[dict[str, Any]] = []
    load_at_start = _load_average()
    try:
        with tempfile.TemporaryDirectory(prefix="scalping-cache-bench-") as holder:
            workdir = Path(holder)
            for label, root, timeframe, rows in parsed:
                case = _measure_case(
                    label=label,
                    root=root,
                    timeframe=timeframe,
                    rows=rows,
                    symbol=args.symbol,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    workdir=workdir,
                )
                cases.append(case)
                del case
                gc.collect()
        report = _build_report(cases, load_at_start)
        report["repeats"] = args.repeats
        report["warmup"] = args.warmup
        report["call_loop"] = CALL_LOOP
        _write_report(Path(args.out), report)
    except (FileNotFoundError, StageError, ValueError) as exc:
        message = str(exc)
        if isinstance(exc, StageError):
            message = exc.measurement.error
        print(json.dumps({"error": message, "type": type(exc).__name__}), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - one JSON line beats a traceback
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        return 1

    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess
    raise SystemExit(main())
