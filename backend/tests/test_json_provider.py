"""JSON bar import."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from scalping.data.json_provider import JsonProvider


def _rows() -> list[dict[str, object]]:
    return [
        {
            "timestamp": "2026-01-05T00:00:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10,
        },
        {
            "timestamp": "2026-01-05T00:01:00Z",
            "open": 100.5,
            "high": 101.5,
            "low": 100.0,
            "close": 101.0,
        },
    ]


def test_imports_bars_and_fills_optional_columns(tmp_path) -> None:
    path = tmp_path / "bars.json"
    path.write_text(json.dumps({"bars": _rows()}), encoding="utf-8")
    frame = JsonProvider(path).fetch("XAUUSD")
    assert frame.height == 2
    assert frame["volume"].to_list() == [10, 0]
    assert frame["provider"].unique().to_list() == ["json"]
    assert frame["spread"].null_count() == 2


def test_filters_by_range(tmp_path) -> None:
    path = tmp_path / "bars.json"
    path.write_text(json.dumps(_rows()), encoding="utf-8")
    frame = JsonProvider(path).fetch(
        "XAUUSD",
        start=datetime(2026, 1, 5, 0, 1, tzinfo=UTC),
    )
    assert frame.height == 1


def test_rejects_an_object_without_a_bar_array(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"foo": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        JsonProvider(path).fetch("XAUUSD")
