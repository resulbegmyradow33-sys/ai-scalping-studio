"""Dataset provenance: what a store partition contains and where it came from.

Parquet alone cannot answer "is this real or synthetic, which file produced it,
and has it changed since?". Provenance is a small JSON sidecar written next to
the store, with a content checksum over timestamps, prices, spread and volume.
It is metadata, not a repair mechanism: a changed frame gets a new checksum,
never a silent overwrite of the old record.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

__all__ = [
    "DatasetProvenance",
    "checksum",
    "load_provenance",
    "provenance_path",
    "save_provenance",
]


def checksum(df: pl.DataFrame) -> str:
    """Content hash of the columns that define a bar dataset."""
    if df.is_empty():
        return "empty"
    digest = hashlib.blake2b(digest_size=16)
    for name in ("ts", "open", "high", "low", "close", "spread", "volume"):
        if name not in df.columns:
            continue
        series = df.get_column(name)
        if name == "ts":
            values = series.dt.cast_time_unit("us").cast(pl.Int64).to_numpy()
        else:
            values = series.cast(pl.Float64).to_numpy()
        digest.update(f"{name}:{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Metadata for one ``(symbol, timeframe)`` dataset."""

    symbol: str
    timeframe: str
    source: str
    synthetic: bool
    rows: int
    first_ts: str | None
    last_ts: str | None
    checksum: str
    created_at: str
    notes: str = ""

    @property
    def dataset_id(self) -> str:
        return f"{self.symbol}:{self.timeframe}:{self.checksum[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "dataset_id": self.dataset_id}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetProvenance:
        return cls(
            symbol=str(raw["symbol"]),
            timeframe=str(raw["timeframe"]),
            source=str(raw["source"]),
            synthetic=bool(raw["synthetic"]),
            rows=int(raw["rows"]),
            first_ts=None if raw.get("first_ts") is None else str(raw["first_ts"]),
            last_ts=None if raw.get("last_ts") is None else str(raw["last_ts"]),
            checksum=str(raw["checksum"]),
            created_at=str(raw["created_at"]),
            notes=str(raw.get("notes", "")),
        )


def build_provenance(
    df: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    source: str,
    synthetic: bool = False,
    notes: str = "",
) -> DatasetProvenance:
    """Describe ``df``; the caller decides whether to persist it."""
    first = df["ts"].min() if not df.is_empty() else None
    last = df["ts"].max() if not df.is_empty() else None
    return DatasetProvenance(
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        synthetic=bool(synthetic),
        rows=df.height,
        first_ts=first.isoformat() if isinstance(first, datetime) else None,
        last_ts=last.isoformat() if isinstance(last, datetime) else None,
        checksum=checksum(df),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=notes,
    )


def provenance_path(root: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(root) / "_meta" / symbol / f"{timeframe}.json"


def save_provenance(root: str | Path, provenance: DatasetProvenance) -> Path:
    """Write atomically so a reader never sees a half-written sidecar."""
    target = provenance_path(root, provenance.symbol, provenance.timeframe)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, target)
    return target


def load_provenance(root: str | Path, symbol: str, timeframe: str) -> DatasetProvenance | None:
    target = provenance_path(root, symbol, timeframe)
    if not target.exists():
        return None
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("provenance sidecar must be a JSON object")
    return DatasetProvenance.from_dict(raw)
