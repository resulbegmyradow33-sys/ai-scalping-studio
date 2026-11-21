"""Run registry: stable ids and an atomic manifest next to the artefacts."""

from __future__ import annotations

import json

import pytest

from scalping.rl.registry import (
    ManifestWriter,
    append_jsonl,
    load_run,
    run_id_for,
    save_manifest,
)


def test_run_id_stable_and_seeded():
    a = run_id_for({"lr": 3e-4, "total_steps": 100}, "abc", 7)
    b = run_id_for({"lr": 3e-4, "total_steps": 100}, "abc", 7)
    c = run_id_for({"lr": 3e-4, "total_steps": 100}, "abc", 8)
    assert a == b and a != c and len(a) == 12


def test_manifest_roundtrip(tmp_path):
    save_manifest(tmp_path, {"run_id": "x", "steps": 10, "metrics": {"reward_mean": 1.2}})
    assert load_run(tmp_path)["metrics"]["reward_mean"] == 1.2


def test_missing_manifest_returns_none(tmp_path):
    assert load_run(tmp_path) is None


def test_run_id_ignores_key_order_but_not_data():
    forward = {"lr": 3e-4, "total_steps": 100}
    reversed_ = {"total_steps": 100, "lr": 3e-4}
    assert run_id_for(forward, "abc", 7) == run_id_for(reversed_, "abc", 7)
    assert run_id_for(forward, "abc", 7) != run_id_for(forward, "abd", 7)


def test_manifest_is_valid_json_and_leaves_no_temp_file(tmp_path):
    path = save_manifest(tmp_path, {"run_id": "x", "steps": 10})
    assert path.name == "manifest.json"
    assert json.loads(path.read_text())["steps"] == 10
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_corrupt_manifest_names_the_file(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="manifest.json"):
        load_run(tmp_path)


def test_manifest_writer_tracks_status(tmp_path):
    writer = ManifestWriter(tmp_path, {"run_id": "x", "created_at": "t"})
    writer.write(status="running", steps=0, reward_mean=None)
    assert load_run(tmp_path)["status"] == "running"

    writer.write(status="paused", steps=10, reward_mean=0.5)
    paused = load_run(tmp_path)
    assert paused["status"] == "paused"
    assert paused["steps"] == 10
    assert paused["reward_mean"] == 0.5
    assert "updated_at" in paused
    assert paused["run_id"] == "x"


def test_append_jsonl_appends_one_line_per_call(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"step": 1})
    append_jsonl(path, {"step": 2, "reward_mean": 0.25})
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["step"] for line in lines] == [1, 2]
    assert lines[1]["reward_mean"] == 0.25
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metrics.jsonl"]
