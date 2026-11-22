"""The persisted strategy card: the parameters the live panel replays.

The card is the product artifact of the research. It holds the validated
``flow_momentum`` parameters plus the training and holdout metrics that earned
them a promotion. The runner reads it at every ``start``, so a nightly
recalibration that writes a new card is picked up without a restart.

A card is only written by :mod:`scalping.strategy.calibrate` after the holdout
gate passes; loading validates the shape so a hand-edited or truncated file is
rejected instead of silently changing how the panel trades.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

__all__ = [
    "CARD_VERSION",
    "STRATEGY",
    "PolicyCard",
    "RuleMetrics",
    "RuleParams",
    "frame_digest",
    "load_card",
    "save_card",
]

#: Bumped when the on-disk meaning of a field changes.
CARD_VERSION: int = 1

#: The strategy the card describes.
STRATEGY: str = "flow_momentum"

_TIMEFRAMES: frozenset[str] = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D1"})


@dataclass(frozen=True, slots=True)
class RuleParams:
    """Everything the runner needs to rebuild the rule on a frame."""

    timeframe: str = "H1"
    flow_quantile: float = 0.80
    flow_window: int = 240
    sl_atr_mult: float = 3.0
    tp_atr_mult: float = 6.0
    max_bars_in_trade: int = 48

    def validate(self) -> None:
        if self.timeframe not in _TIMEFRAMES:
            raise ValueError(f"unsupported timeframe {self.timeframe!r}")
        if not 0.5 <= self.flow_quantile < 1.0:
            raise ValueError("flow_quantile must be in [0.5, 1.0)")
        if self.flow_window < 20:
            raise ValueError("flow_window must be at least 20 bars")
        if self.sl_atr_mult <= 0 or self.tp_atr_mult <= 0:
            raise ValueError("sl/tp multipliers must be positive")
        if self.max_bars_in_trade < 1:
            raise ValueError("max_bars_in_trade must be positive")


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    """A rule's realised behaviour on one slice."""

    trades: int = 0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_r: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> RuleMetrics:
        return cls(
            trades=int(raw.get("trades", 0)),
            expectancy_r=float(raw.get("expectancy_r", 0.0)),
            profit_factor=float(raw.get("profit_factor", 0.0)),
            win_rate=float(raw.get("win_rate", 0.0)),
            max_drawdown=float(raw.get("max_drawdown", 0.0)),
            total_r=float(raw.get("total_r", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyCard:
    """A promoted rule plus the evidence that promoted it."""

    params: RuleParams
    train: RuleMetrics
    holdout: RuleMetrics
    digest: str
    created_at: str
    version: int = CARD_VERSION
    strategy: str = STRATEGY
    symbol: str = "XAUUSD"
    note: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "params": asdict(self.params),
            "train": self.train.to_dict(),
            "holdout": self.holdout.to_dict(),
            "digest": self.digest,
            "created_at": self.created_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyCard:
        version = int(raw.get("version", 0))
        if version != CARD_VERSION:
            raise ValueError(f"unsupported card version {version}")
        if raw.get("strategy", STRATEGY) != STRATEGY:
            raise ValueError(f"unsupported strategy {raw.get('strategy')!r}")
        params = RuleParams(**raw["params"])
        params.validate()
        return cls(
            params=params,
            train=RuleMetrics.from_mapping(raw.get("train", {})),
            holdout=RuleMetrics.from_mapping(raw.get("holdout", {})),
            digest=str(raw["digest"]),
            created_at=str(raw["created_at"]),
            version=version,
            strategy=STRATEGY,
            symbol=str(raw.get("symbol", "XAUUSD")),
            note=str(raw.get("note", "")),
        )

    @classmethod
    def build(
        cls,
        params: RuleParams,
        train: RuleMetrics,
        holdout: RuleMetrics,
        digest: str,
        *,
        symbol: str = "XAUUSD",
        note: str = "",
    ) -> PolicyCard:
        params.validate()
        return cls(
            params=params,
            train=train,
            holdout=holdout,
            digest=digest,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            symbol=symbol,
            note=note,
        )


def frame_digest(frame: pl.DataFrame) -> str:
    """A stable fingerprint of the rows a calibration actually saw.

    Hashing every close would be slow on five years of M1; the digest covers
    the span, the row count and a strided sample of timestamps and closes. Two
    frames with the same digest are the same dataset for practical purposes,
    and a rewritten or extended store changes it.
    """
    if frame.is_empty():
        return "empty"
    height = frame.height
    stride = max(1, height // 64)
    stamps = frame["ts"].gather(list(range(0, height, stride))).cast(pl.Int64).to_list()
    closes = frame["close"].gather(list(range(0, height, stride))).to_list()
    payload = {
        "rows": height,
        "stride": stride,
        "first": str(frame["ts"][0]),
        "last": str(frame["ts"][-1]),
        "ts": stamps,
        "close": [round(float(value), 5) for value in closes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def save_card(path: str | Path, card: PolicyCard) -> None:
    """Write the card atomically so a reader never sees half a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, target)


def load_card(path: str | Path | None) -> PolicyCard | None:
    """Load a card, or return ``None`` when it does not exist.

    A malformed card raises: silently falling back to an unknown rule is worse
    than refusing to start the validated strategy.
    """
    if path is None:
        return None
    target = Path(path)
    if not target.exists():
        return None
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("strategy card must be a JSON object")
    return PolicyCard.from_dict(raw)
