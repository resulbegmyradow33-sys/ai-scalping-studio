"""Append-only decision log: round-trip, limits and corrupt rows."""

from __future__ import annotations

import json

import pytest

from scalping.ensemble import Decision, DecisionLog, DecisionLogError
from scalping.ensemble.log import SCHEMA_VERSION


def decision(action: str = "BUY", ts: str = "2026-01-05T10:00:00+00:00") -> Decision:
    return Decision(
        action=action,  # type: ignore[arg-type]
        probabilities=(0.05, 0.15, 0.80),
        classes=(-1, 0, 1),
        disagreement=0.10,
        regime="up",
        reason="entry_long",
        model_version="test-1",
        ts=ts,
    )


def test_round_trip_preserves_every_field(tmp_path) -> None:
    log = DecisionLog(tmp_path / "nested" / "decisions.jsonl")
    first = decision()
    second = decision("NO_TRADE", ts="2026-01-05T10:01:00+00:00")

    log.append(first)
    log.append(second)

    assert log.read() == [first, second]
    raw = json.loads(log.path.read_text().splitlines()[0])
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["probabilities"] == {"-1": 0.05, "0": 0.15, "1": 0.8}


def test_limit_and_missing_file(tmp_path) -> None:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    assert log.read() == []

    log.append(decision(ts="a"))
    log.append(decision(ts="b"))
    assert [row.ts for row in log.read(limit=1)] == ["a"]
    with pytest.raises(ValueError, match="limit"):
        log.read(limit=0)


def test_corrupt_rows_fail_closed(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text('{"schema_version": 1, "action": "BUY"}\n', encoding="utf-8")

    with pytest.raises(DecisionLogError, match="line 1"):
        DecisionLog(path).read()

    path.write_text('{"schema_version": 99}\n', encoding="utf-8")
    with pytest.raises(DecisionLogError, match="schema_version"):
        DecisionLog(path).read()

    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DecisionLogError, match="line 1"):
        DecisionLog(path).read()


def test_blank_lines_are_skipped_and_type_is_enforced(tmp_path) -> None:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    log.append(decision())
    log.path.write_text("\n" + log.path.read_text(), encoding="utf-8")
    assert len(log.read()) == 1

    with pytest.raises(TypeError, match="Decision"):
        log.append({"action": "BUY"})  # type: ignore[arg-type]
