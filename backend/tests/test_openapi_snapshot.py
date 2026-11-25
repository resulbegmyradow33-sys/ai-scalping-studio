"""Schema drift guard for the public HTTP API.

The committed snapshot at ``tests/snapshots/openapi.json`` is a normalized view
of ``/openapi.json``: every ``/api/`` path and method, the request/response
models, query parameters, required fields and enum values. A diff here means the
panel's contract changed; the failure message lists each added, removed or
changed path and field instead of dumping two JSON blobs.

Intentional changes are recorded by rewriting the snapshot, then reviewing the
diff:

    cd backend
    UPDATE_OPENAPI=1 uv run pytest tests/test_openapi_snapshot.py
    git diff tests/snapshots/openapi.json

See ``docs/api-contracts.md``.
"""

from __future__ import annotations

import copy
import json
import os

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.api.openapi_contract import (
    DEFAULT_SNAPSHOT_PATH,
    build_snapshot,
    diff_snapshots,
    load_snapshot,
    write_snapshot,
)

UPDATE_HINT = (
    "If this change is intentional, rewrite the snapshot and review the diff:\n"
    "    cd backend && UPDATE_OPENAPI=1 uv run pytest tests/test_openapi_snapshot.py\n"
    "    git diff tests/snapshots/openapi.json"
)


def _live_snapshot(tmp_path) -> dict:
    app = create_app(data_root=tmp_path, runs_root=tmp_path / "runs")
    response = TestClient(app).get("/openapi.json")
    assert response.status_code == 200
    return build_snapshot(response.json())


def test_public_openapi_matches_snapshot(tmp_path) -> None:
    current = _live_snapshot(tmp_path)

    if os.environ.get("UPDATE_OPENAPI") == "1":
        write_snapshot(DEFAULT_SNAPSHOT_PATH, current)
        pytest.skip(f"rewrote {DEFAULT_SNAPSHOT_PATH}; review the git diff and commit it")

    committed = load_snapshot(DEFAULT_SNAPSHOT_PATH)
    drift = diff_snapshots(committed, current)
    assert not drift, (
        f"public API schema drifted from the committed snapshot:\n\n{drift}\n\n{UPDATE_HINT}"
    )


def test_snapshot_is_deterministic(tmp_path) -> None:
    """Two app factories must normalize to byte-identical JSON."""
    first = _live_snapshot(tmp_path)
    second = _live_snapshot(tmp_path)
    assert first == second
    serialized = json.dumps(first, indent=2, sort_keys=True)
    assert json.loads(serialized) == first


def test_diff_reports_paths_fields_and_enums() -> None:
    """The guard must actually detect the drift classes it claims to cover."""
    expected = {
        "operations": {
            "GET /api/health": {
                "request": None,
                "query": {},
                "responses": {"200": "HealthResponse"},
            },
            "POST /api/paper/sessions": {
                "request": "PaperSessionCreate",
                "query": {"limit": "integer"},
                "responses": {"201": "PaperSessionCreated"},
            },
        },
        "models": {
            "HealthResponse": {
                "required": ["status"],
                "properties": {"status": "string", "version": "string"},
            },
            "PaperSessionCreate": {
                "required": ["symbol"],
                "properties": {"symbol": "string", "timeframe": "enum[M1|H1]"},
                "enums": {"timeframe": ["M1", "H1"]},
            },
        },
    }
    current = copy.deepcopy(expected)

    # Added and removed operations.
    current["operations"]["GET /api/regime"] = {
        "request": None,
        "query": {},
        "responses": {"200": "RegimeResponse"},
    }
    del current["operations"]["POST /api/paper/sessions"]

    # Changed response model, required field and enum values.
    current["operations"]["GET /api/health"]["responses"]["200"] = "HealthResponseV2"
    current["models"]["HealthResponse"]["required"] = ["status", "version"]
    current["models"]["HealthResponse"]["properties"]["device"] = "string"
    current["models"]["PaperSessionCreate"]["properties"]["timeframe"] = "enum[M1|H1|H4]"
    current["models"]["PaperSessionCreate"]["enums"]["timeframe"] = ["M1", "H1", "H4"]
    current["models"]["PaperSessionCreate"]["properties"]["close_on_stop"] = "boolean | null"

    drift = diff_snapshots(expected, current)

    assert "+ GET /api/regime" in drift
    assert "- POST /api/paper/sessions" in drift
    assert "~ response 200: HealthResponse -> HealthResponseV2" in drift
    assert "+ required version" in drift
    assert "+ field device: string" in drift
    assert "+ field close_on_stop: boolean | null" in drift
    assert "+ enum timeframe: H4" in drift


def test_diff_is_empty_for_identical_snapshots() -> None:
    assert diff_snapshots({"operations": {}, "models": {}}, {"operations": {}, "models": {}}) == ""


def test_diff_reports_snapshot_format_change() -> None:
    """A layout version bump must fail the guard, not pass as an empty diff."""
    expected = {"schema": "scalping-public-api", "version": 1, "operations": {}, "models": {}}
    current = {"schema": "scalping-public-api", "version": 2, "operations": {}, "models": {}}
    drift = diff_snapshots(expected, current)
    assert drift == "header: snapshot version 1 -> 2"
