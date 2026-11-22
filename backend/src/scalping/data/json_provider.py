"""Import bars from a JSON file.

Accepted shapes:

* ``[{"ts": "...", "open": ...}, ...]``
* ``{"bars": [...]}`` or ``{"data": [...]}``

Each row may use ``ts``/``timestamp``/``time``; ``bid_close``, ``ask_close``,
``spread``, ``volume`` and ``provider`` are optional and become null/0 when
absent, which lets the feature preparer apply its spread model. Naive
timestamps are treated as UTC; epoch numbers are accepted as seconds or
milliseconds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from scalping.data.base import BAR_SCHEMA, DataProvider, conform_bars

__all__ = ["JsonProvider"]

_TS_KEYS = ("ts", "timestamp", "time")
_PRICE_KEYS = ("open", "high", "low", "close")


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1e11:  # milliseconds
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


def _number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class JsonProvider(DataProvider):
    """Read a JSON array of bars and return a conformed frame."""

    name = "json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _rows(self) -> list[dict[str, Any]]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key in ("bars", "data", "rows"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
            else:
                raise ValueError("JSON object must contain a 'bars'/'data'/'rows' array")
        if not isinstance(raw, list):
            raise TypeError("JSON input must be a list of bar objects")
        return [row for row in raw if isinstance(row, dict)]

    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        records: list[dict[str, Any]] = []
        problems = 0
        for row in self._rows():
            stamp_value = next((row[key] for key in _TS_KEYS if key in row), None)
            stamp = _parse_ts(stamp_value)
            if stamp is None:
                problems += 1
                continue
            if start is not None and stamp < start:
                continue
            if end is not None and stamp > end:
                continue
            prices = {key: _number(row, key) for key in _PRICE_KEYS}
            if any(value is None for value in prices.values()):
                problems += 1
                continue
            records.append(
                {
                    "ts": stamp,
                    **prices,
                    "bid_close": _number(row, "bid_close"),
                    "ask_close": _number(row, "ask_close"),
                    "spread": _number(row, "spread"),
                    "volume": int(_number(row, "volume") or 0),
                    "provider": str(row.get("provider") or self.name),
                }
            )
        if not records:
            if problems:
                raise ValueError(f"no usable bars in {self.path}; {problems} malformed rows")
            return pl.DataFrame(schema=BAR_SCHEMA)
        frame = conform_bars(pl.DataFrame(records, infer_schema_length=None))
        return frame.sort("ts")
