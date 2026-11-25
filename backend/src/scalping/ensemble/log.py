"""Append-only JSONL decision log.

`Decision.to_dict()` already defines the row schema; this module is the durable
sink. Each line is one decision with a `schema_version`, written with a single
append write and `fsync` so a crash cannot leave a half-written trailing line
unnoticed (the reader rejects it instead of guessing).

The log is local research state (`runs/` or similar, gitignored). It never
places an order and is not read by the trading path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

from scalping.ensemble.decision import Decision

__all__ = ["SCHEMA_VERSION", "DecisionLog", "DecisionLogError"]

SCHEMA_VERSION: Final[int] = 1


class DecisionLogError(ValueError):
    """The log contains a row that is not a valid decision record."""


class DecisionLog:
    """Append and read decision rows under one JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, decision: Decision) -> None:
        if not isinstance(decision, Decision):
            raise TypeError(f"decision must be a Decision, got {type(decision).__name__}")
        row = {"schema_version": SCHEMA_VERSION, **decision.to_dict()}
        line = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self, *, limit: int | None = None) -> list[Decision]:
        """Read rows in order; ``limit`` keeps the oldest rows."""
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 or None")
        if not self.path.exists():
            return []
        decisions: list[Decision] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                if limit is not None and len(decisions) >= limit:
                    break
                try:
                    decisions.append(self._parse(text))
                except (ValueError, KeyError, TypeError) as exc:
                    raise DecisionLogError(
                        f"{self.path}: line {line_number} is not a decision row: {exc}"
                    ) from exc
        return decisions

    @staticmethod
    def _parse(text: str) -> Decision:
        row = json.loads(text)
        if not isinstance(row, dict):
            raise TypeError("row must be a JSON object")
        version = row.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {version!r}")
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict):
            raise TypeError("probabilities must be an object keyed by class")
        classes = tuple(int(value) for value in row.get("classes", ()))
        values = tuple(float(probabilities[str(label)]) for label in classes)
        return Decision(
            action=row["action"],
            probabilities=values,
            classes=classes,
            disagreement=float(row["disagreement"]),
            regime=row.get("regime"),
            reason=str(row["reason"]),
            model_version=str(row["model_version"]),
            ts=str(row["ts"]),
        )
