"""On-disk cache for built :class:`~scalping.features.build.FeatureFrame` matrices.

``build_features`` is the expensive step repeated by training, walk-forward
evaluation and (later) candidate search: the same ``(symbol, timeframe, bars,
costs, registry)`` input is rebuilt for every experiment. This cache materialises
the result once and returns the identical arrays afterwards.

The key is a :func:`hashlib.blake2b` digest of a canonical JSON payload:

* ``symbol`` and ``timeframe``,
* the **data identity**: row count, first/last timestamp and a digest of the raw
  bar columns that feed the builder (``ts``, OHLC, quotes, spread, volume),
* a *registry version*: the canonical ``FEATURE_COLUMNS`` tuple plus ``WARMUP``,
  so a column added, removed or reordered misses instead of mixing matrices,
* the cache format version, and
* the full :class:`~scalping.types.CostConfig` block. Only ``spread_floor`` and
  ``spread_vol_mult`` actually reach :func:`~scalping.features.prepare.prepare_bars`,
  but keying on the whole block is the conservative choice: a changed cost
  config is treated as a different input instead of a silent hit. Bump
  :data:`CACHE_VERSION` when a feature *formula* changes without changing the
  column list.

One entry is a pair of files under ``{root}/{key[:2]}/``: an uncompressed
``.npz`` with ``X``/``ts``/``ohlc``/``spread``/``atr`` (``allow_pickle=False``)
and a JSON manifest with shapes, dtypes, columns, ``valid_from`` and a BLAKE2b
checksum. Both are written to a sibling temp file and moved into place with
``os.replace``, so a reader never opens a partially written file.

A load validates the manifest, the key it claims to belong to, array names,
shapes, dtypes, a recomputed checksum, ``valid_from`` and finiteness. Any
mismatch is a **miss**, and the bad pair is renamed ``*.bad`` (quarantined) so
the next build overwrites a clean entry. The cache is an optimisation, never a
correctness proof: the bit-identity tests are. A same-key write always carries
the same arrays, so a reader racing two writers still sees a checksum-consistent
file even if the manifest and the ``.npz`` come from different writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.config import Settings
from scalping.features.build import FEATURE_COLUMNS, WARMUP, FeatureFrame, build_features
from scalping.types import CostConfig

__all__ = [
    "CACHE_VERSION",
    "FEATURE_CACHE_DIRNAME",
    "FeatureCache",
    "FrameMeta",
    "cached_build_features",
    "frame_meta",
    "registry_version",
]

logger = logging.getLogger(__name__)

#: Version of the on-disk layout *and* of the feature formulas the key assumes.
#: Bump it when a formula changes without a change to ``FEATURE_COLUMNS``.
CACHE_VERSION: Final[int] = 1

#: Sub-directory a CLI/API caller uses under its data or runs root.
FEATURE_CACHE_DIRNAME: Final[str] = ".feature-cache"

_ARRAY_SUFFIX: Final[str] = ".npz"
_MANIFEST_SUFFIX: Final[str] = ".json"
_BAD_SUFFIX: Final[str] = ".bad"
_ARRAY_NAMES: Final[tuple[str, ...]] = ("X", "ts", "ohlc", "spread", "atr")

#: Raw bar columns that can influence the matrix, in a fixed order. ``provider``
#: is excluded: the source label does not change a single feature value.
_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "bid_close",
    "ask_close",
    "spread",
    "volume",
)

#: Local-time stamp appended to a quarantined file's original name.
_BAD_STAMP_FORMAT: Final[str] = "%Y%m%dT%H%M%S%f"


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Cheap identity of the raw bars a feature frame would be built from."""

    rows: int
    first_ts: str | None
    last_ts: str | None
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "fingerprint": self.fingerprint,
        }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def frame_meta(frame: pl.DataFrame) -> FrameMeta:
    """Describe ``frame`` well enough to detect any change that reaches the matrix.

    The frame is sorted by ``ts`` first (``build_features`` sorts too), so the
    same bars in a different row order share one key. Every input column is
    hashed with its name, dtype and length in the digest, which keeps the
    framing unambiguous.
    """
    missing = [name for name in _INPUT_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing bar columns: {missing}")
    ordered = frame if frame.get_column("ts").is_sorted() else frame.sort("ts")
    digest = hashlib.blake2b(digest_size=20)
    for name in _INPUT_COLUMNS:
        values = ordered.get_column(name).to_numpy()
        raw = np.ascontiguousarray(values)
        digest.update(f"{name}:{values.dtype.str}:{values.size}".encode())
        if raw.dtype.kind == "M":
            raw = raw.view("i8")
        digest.update(raw.tobytes())
    if ordered.is_empty():
        return FrameMeta(rows=0, first_ts=None, last_ts=None, fingerprint=digest.hexdigest())
    return FrameMeta(
        rows=ordered.height,
        first_ts=_iso(ordered.get_column("ts").min()),
        last_ts=_iso(ordered.get_column("ts").max()),
        fingerprint=digest.hexdigest(),
    )


def registry_version() -> str:
    """Digest of the canonical feature layout; changes with columns or warm-up."""
    payload = {"warmup": WARMUP, "columns": list(FEATURE_COLUMNS)}
    return hashlib.sha256(_canonical(payload)).hexdigest()[:16]


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _costs_of(settings: Settings | CostConfig) -> CostConfig:
    costs = settings.costs if isinstance(settings, Settings) else settings
    if not isinstance(costs, CostConfig):  # pragma: no cover - caller contract
        raise TypeError(f"expected Settings or CostConfig, got {type(settings).__name__}")
    return costs


def _checksum(frame: FeatureFrame) -> str:
    """Digest every array plus the scalar fields that define the result."""
    digest = hashlib.blake2b(digest_size=32)
    digest.update(f"columns:{','.join(frame.columns)}".encode())
    digest.update(f"valid_from:{frame.valid_from}".encode())
    for name in _ARRAY_NAMES:
        values = np.ascontiguousarray(getattr(frame, name))
        digest.update(f"{name}:{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


class FeatureCache:
    """Disk cache of built feature frames keyed by data, costs and registry.

    The class is safe to share between threads: files are written atomically and
    the counters are guarded by one lock. It holds no frame in memory.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "corrupt": 0,
            "errors": 0,
        }
        self._load_seconds = 0.0
        self._save_seconds = 0.0

    # --- paths ---------------------------------------------------------
    def _directory(self, key: str) -> Path:
        return self.root / key[:2]

    def _array_path(self, key: str) -> Path:
        return self._directory(key) / f"{key}{_ARRAY_SUFFIX}"

    def _manifest_path(self, key: str) -> Path:
        return self._directory(key) / f"{key}{_MANIFEST_SUFFIX}"

    # --- keys ----------------------------------------------------------
    @staticmethod
    def _payload(
        symbol: str,
        timeframe: str,
        meta: FrameMeta,
        costs: CostConfig,
    ) -> dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "symbol": symbol,
            "timeframe": timeframe,
            "registry": registry_version(),
            "costs": asdict(costs),
            "data": meta.to_dict(),
        }

    def key_for(
        self,
        symbol: str,
        timeframe: str,
        frame_meta: FrameMeta,
        settings: Settings | CostConfig,
    ) -> str:
        """Stable cache key for one input combination.

        Pure function of its arguments (plus the module-level registry digest),
        so two processes computing it for the same bars agree byte for byte.
        """
        payload = self._payload(symbol, timeframe, frame_meta, _costs_of(settings))
        return hashlib.blake2b(_canonical(payload), digest_size=32).hexdigest()

    # --- read ----------------------------------------------------------
    def load(self, key: str) -> FeatureFrame | None:
        """Return the cached frame, or ``None`` on any miss or invalid entry."""
        started = time.perf_counter()
        try:
            frame = self._read(key)
        except Exception:  # a cache read must never fail a build
            logger.exception("feature cache entry %s could not be read", key)
            self._count("errors")
            self._quarantine(key)
            frame = None
        elapsed = time.perf_counter() - started
        with self._lock:
            self._load_seconds += elapsed
            self._counts["hits" if frame is not None else "misses"] += 1
        return frame

    def _read(self, key: str) -> FeatureFrame | None:
        array_path = self._array_path(key)
        manifest_path = self._manifest_path(key)
        if not array_path.exists() and not manifest_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return self._reject(key, f"manifest unreadable: {exc}")
        if not isinstance(manifest, dict):
            return self._reject(key, "manifest is not a JSON object")
        if manifest.get("version") != CACHE_VERSION:
            return self._reject(
                key, f"cache version {manifest.get('version')!r} != {CACHE_VERSION}"
            )
        if manifest.get("key") != key:
            return self._reject(key, "manifest key does not match the requested key")

        columns = manifest.get("columns")
        if not isinstance(columns, list) or not all(isinstance(name, str) for name in columns):
            return self._reject(key, "manifest columns are not a list of names")
        if tuple(columns) != FEATURE_COLUMNS:
            return self._reject(key, "manifest columns differ from the registry")

        declared = manifest.get("arrays")
        if not isinstance(declared, dict) or set(declared) != set(_ARRAY_NAMES):
            return self._reject(key, "manifest does not describe exactly the cached arrays")

        try:
            with np.load(array_path, allow_pickle=False) as archive:
                if set(archive.files) != set(_ARRAY_NAMES):
                    return self._reject(key, "npz does not contain exactly the cached arrays")
                arrays = {name: np.array(archive[name], copy=True) for name in _ARRAY_NAMES}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            return self._reject(key, f"npz unreadable: {exc}")

        valid_from = manifest.get("valid_from")
        checksum = manifest.get("checksum")
        if not isinstance(valid_from, int) or isinstance(valid_from, bool):
            return self._reject(key, "manifest has no integer valid_from")
        if not isinstance(checksum, str) or not checksum:
            return self._reject(key, "manifest has no checksum")

        frame = FeatureFrame(
            X=arrays["X"],
            ts=arrays["ts"],
            ohlc=arrays["ohlc"],
            spread=arrays["spread"],
            atr=arrays["atr"],
            valid_from=valid_from,
            columns=list(columns),
        )
        reason = self._validate(frame, declared, manifest)
        if reason is not None:
            return self._reject(key, reason)
        if _checksum(frame) != checksum:
            return self._reject(key, "checksum mismatch")
        return frame

    @staticmethod
    def _validate(
        frame: FeatureFrame,
        declared: dict[str, Any],
        manifest: dict[str, Any],
    ) -> str | None:
        """Return why ``frame`` is unusable, or ``None`` when it is consistent."""
        rows = int(frame.X.shape[0]) if frame.X.ndim == 2 else -1
        if frame.X.ndim != 2 or frame.X.shape[1] != len(frame.columns):
            return f"X shape {frame.X.shape} does not match {len(frame.columns)} columns"
        for name in _ARRAY_NAMES:
            values = getattr(frame, name)
            info = declared.get(name)
            if not isinstance(info, dict):
                return f"manifest has no metadata for {name}"
            if list(values.shape) != info.get("shape"):
                return f"{name} shape {list(values.shape)} != {info.get('shape')}"
            if str(values.dtype) != info.get("dtype"):
                return f"{name} dtype {values.dtype} != {info.get('dtype')}"
        if frame.ts.shape != (rows,) or frame.spread.shape != (rows,):
            return "ts/spread length does not match X"
        if frame.ohlc.shape != (rows, 4):
            return f"ohlc shape {frame.ohlc.shape} is not ({rows}, 4)"
        if frame.atr.shape != (rows,):
            return f"atr shape {frame.atr.shape} is not ({rows},)"
        if not 0 <= frame.valid_from <= rows:
            return f"valid_from {frame.valid_from} is outside [0, {rows}]"
        for name in ("ohlc", "spread", "atr"):
            if not np.isfinite(getattr(frame, name)).all():
                return f"{name} contains a non-finite value"
        # Only rows at or after valid_from are guaranteed finite; the warm-up
        # head is allowed to hold NaN, exactly as ``build_features`` returns it.
        if not np.isfinite(frame.X[frame.valid_from :]).all():
            return "X is non-finite after valid_from"
        if manifest.get("rows") != rows:
            return f"manifest rows {manifest.get('rows')!r} != {rows}"
        return None

    def _reject(self, key: str, reason: str) -> FeatureFrame | None:
        """Log, count and quarantine one unusable entry; always a miss."""
        logger.warning("feature cache entry %s rejected: %s", key, reason)
        self._count("corrupt")
        self._quarantine(key)
        return None

    def _quarantine(self, key: str) -> None:
        """Rename the pair to ``*.bad`` so a later build starts from a clean key."""
        stamp = datetime.now(UTC).strftime(_BAD_STAMP_FORMAT)
        for path in (self._array_path(key), self._manifest_path(key)):
            if not path.exists():
                continue
            try:
                path.replace(path.with_name(f"{path.name}.{stamp}{_BAD_SUFFIX}"))
            except OSError:  # pragma: no cover - read-only or vanished directory
                logger.exception("cannot quarantine feature cache file %s", path)

    # --- write ---------------------------------------------------------
    def save(self, key: str, frame: FeatureFrame) -> Path:
        """Persist ``frame`` under ``key`` atomically and return the ``.npz`` path."""
        started = time.perf_counter()
        self._check_frame(frame)
        directory = self._directory(key)
        directory.mkdir(parents=True, exist_ok=True)
        array_path = self._array_path(key)
        manifest_path = self._manifest_path(key)

        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{key}.", suffix=_ARRAY_SUFFIX)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(
                    handle,
                    X=frame.X,
                    ts=frame.ts,
                    ohlc=frame.ohlc,
                    spread=frame.spread,
                    atr=frame.atr,
                )
            os.replace(tmp, array_path)
        finally:
            tmp.unlink(missing_ok=True)

        manifest = {
            "version": CACHE_VERSION,
            "key": key,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows": int(frame.X.shape[0]),
            "valid_from": int(frame.valid_from),
            "columns": list(frame.columns),
            "checksum": _checksum(frame),
            "arrays": {
                name: {
                    "shape": list(getattr(frame, name).shape),
                    "dtype": str(getattr(frame, name).dtype),
                }
                for name in _ARRAY_NAMES
            },
        }
        self._write_manifest(manifest_path, manifest)
        self._count("saves")
        with self._lock:
            self._save_seconds += time.perf_counter() - started
        return array_path

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        """Write the manifest to a sibling temp file and replace the live one."""
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    # --- maintenance ---------------------------------------------------
    def invalidate(self, key: str) -> bool:
        """Delete one entry (live files only); ``True`` when something existed."""
        removed = False
        for path in (self._array_path(key), self._manifest_path(key)):
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
        return removed

    def clear(self) -> int:
        """Delete every cache file under ``root`` and return how many were removed."""
        if not self.root.is_dir():
            return 0
        removed = 0
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {_ARRAY_SUFFIX, _MANIFEST_SUFFIX, ".tmp"} or path.name.endswith(
                _BAD_SUFFIX
            ):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        """Counters plus the time spent in ``load``/``save`` and the entry count."""
        with self._lock:
            counts = dict(self._counts)
            load_seconds = self._load_seconds
            save_seconds = self._save_seconds
        return {
            **counts,
            "load_seconds": round(load_seconds, 6),
            "save_seconds": round(save_seconds, 6),
            "entries": self._entries(),
            "root": str(self.root),
        }

    def _entries(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(
            1
            for path in self.root.rglob(f"*{_ARRAY_SUFFIX}")
            if not path.name.endswith(_BAD_SUFFIX)
        )

    def _count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    @staticmethod
    def _check_frame(frame: FeatureFrame) -> None:
        """Refuse to persist a frame the loader would immediately reject."""
        rows = int(frame.X.shape[0]) if frame.X.ndim == 2 else -1
        if frame.X.ndim != 2 or frame.X.shape[1] != len(frame.columns):
            raise ValueError(f"X shape {frame.X.shape} does not match {len(frame.columns)} columns")
        if frame.ts.shape != (rows,) or frame.ohlc.shape != (rows, 4):
            raise ValueError("ts/ohlc shapes do not match X")
        if frame.spread.shape != (rows,) or frame.atr.shape != (rows,):
            raise ValueError("spread/atr shapes do not match X")
        if not 0 <= frame.valid_from <= rows:
            raise ValueError(f"valid_from {frame.valid_from} is outside [0, {rows}]")
        if not np.isfinite(frame.X[frame.valid_from :]).all():
            raise ValueError("X is non-finite after valid_from")


def cached_build_features(
    cache: FeatureCache | None,
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame,
    settings: Settings | CostConfig,
) -> FeatureFrame:
    """Return the cached matrix for ``frame``, building and saving it on a miss.

    ``cache=None`` disables the path entirely and is the exact historical
    behaviour. A failed save is logged and ignored: a full or read-only cache
    directory must not fail a training run.
    """
    costs = _costs_of(settings)
    if cache is None or frame.is_empty():
        return build_features(frame, costs)
    meta = frame_meta(frame)
    key = cache.key_for(symbol, timeframe, meta, costs)
    cached = cache.load(key)
    if cached is not None:
        return cached
    features = build_features(frame, costs)
    try:
        cache.save(key, features)
    except OSError:
        logger.exception("cannot save feature cache entry %s", key)
    return features
