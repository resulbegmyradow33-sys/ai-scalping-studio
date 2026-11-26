"""Explicit raw → cleaned → features → split layers on disk.

The Parquet store is already canonical, but training still reads one flat frame
and builds everything in memory. This module materialises the layers the
project documents:

```
<out>/raw/<symbol>/<timeframe>.parquet        exact stored bars
<out>/cleaned/<symbol>/<timeframe>.parquet    validated bars; never repaired
<out>/features/<symbol>/<timeframe>.parquet   causal features + labels + split
<out>/{training,validation,test}/...          split views (purged rows excluded)
<out>/layers.json                             checksums, rows and dataset id
```

Fail-closed: a structurally invalid slice raises instead of writing a cleaned
layer. "Cleaned" here means *validated*, not *modified* — the raw and cleaned
files are identical when the slice is clean, and that is recorded in the
manifest rather than hidden.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from scalping.data.store import ParquetStore
from scalping.data.validate import validate_bars
from scalping.datasets.builder import SPLIT_NAMES, build_dataset
from scalping.features.build import build_features

__all__ = ["LAYER_NAMES", "LayerFile", "LayerManifest", "build_layers"]

LAYER_NAMES: Final[tuple[str, ...]] = (
    "raw",
    "cleaned",
    "features",
    "training",
    "validation",
    "test",
)


@dataclass(frozen=True, slots=True)
class LayerFile:
    """One written layer file."""

    layer: str
    path: str
    rows: int
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "path": self.path,
            "rows": self.rows,
            "checksum": self.checksum,
        }


@dataclass(frozen=True, slots=True)
class LayerManifest:
    """Everything a later run needs to identify the layer set."""

    root: str
    symbol: str
    timeframe: str
    dataset_id: str
    raw_rows: int
    purged_rows: int
    files: tuple[LayerFile, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_id": self.dataset_id,
            "raw_rows": self.raw_rows,
            "purged_rows": self.purged_rows,
            "files": [item.to_dict() for item in self.files],
            "created_at": self.created_at,
        }


def _digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        frame.write_parquet(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _feature_frame(features, bundle) -> pl.DataFrame:
    """Feature matrix + labels + split marker, row-aligned with the bundle.

    The bundle starts at ``valid_from`` and ends one horizon before the tail, so
    the feature layer carries exactly those usable rows (no warm-up rows).
    """
    start = bundle.manifest.train.start
    rows = bundle.X.shape[0]
    frame = pl.DataFrame({"ts": features.ts[start : start + rows]})
    for index, name in enumerate(features.columns):
        frame = frame.with_columns(
            pl.Series(name, features.X[start : start + rows, index])
        )
    return frame.with_columns(
        pl.Series("y_direction", bundle.y_direction),
        pl.Series("y_return", bundle.y_return),
        pl.Series("split", bundle.split),
    )


def build_layers(
    store_root: str | Path,
    out_root: str | Path,
    symbol: str,
    timeframe: str = "M1",
    *,
    horizon: int = 1,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_bars: int = 0,
) -> LayerManifest:
    """Materialise all six layers; raise instead of fabricating a clean slice."""
    store = ParquetStore(store_root)
    frame = store.read(symbol, timeframe)
    if frame.is_empty():
        raise ValueError(f"no {symbol} {timeframe} bars under {store.root}")
    report = validate_bars(frame, symbol=symbol, timeframe=timeframe)
    if not report.ok:
        codes = ", ".join(issue.code for issue in report.issues if issue.severity == "error")
        raise ValueError(f"{symbol} {timeframe} is not structurally valid: {codes}")

    root = Path(out_root)
    files: list[LayerFile] = []
    for layer, content in (("raw", frame), ("cleaned", frame)):
        path = root / layer / symbol / f"{timeframe}.parquet"
        _write(path, content)
        files.append(LayerFile(layer, str(path), content.height, _digest(path)))

    features = build_features(frame)
    bundle = build_dataset(
        features,
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        purge_bars=purge_bars,
    )
    feature_frame = _feature_frame(features, bundle)
    feature_path = root / "features" / symbol / f"{timeframe}.parquet"
    _write(feature_path, feature_frame)
    files.append(LayerFile("features", str(feature_path), feature_frame.height, _digest(feature_path)))

    for index, name in enumerate(SPLIT_NAMES):
        view = feature_frame.filter(pl.col("split") == index)
        layer = "training" if name == "train" else name
        path = root / layer / symbol / f"{timeframe}.parquet"
        _write(path, view)
        files.append(LayerFile(layer, str(path), view.height, _digest(path)))

    manifest = LayerManifest(
        root=str(root),
        symbol=symbol,
        timeframe=timeframe,
        dataset_id=bundle.manifest.dataset_id,
        raw_rows=frame.height,
        purged_rows=int((bundle.split == -1).sum()),
        files=tuple(files),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    target = root / "layers.json"
    tmp = target.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return manifest
