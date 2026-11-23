"""Boundary checks: HTTP ids and import paths must not escape the data root.

A symbol becomes a directory name and the CSV path becomes an open() argument,
so both are validated at the request boundary. Every rejection here is a 4xx
raised before the store is touched; a 500 or a directory outside the root is a
failure of the test, not an acceptable outcome.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app

BODY = {
    "provider": "synthetic",
    "symbol": "XAUUSD",
    "seed": 7,
    "start": "2026-01-05T00:00:00Z",
    "end": "2026-01-07T00:00:00Z",
}

FIXTURE = Path(__file__).parent / "fixtures" / "xauusd_gmt3.csv"


def test_download_rejects_symbol_traversal_before_touching_the_store(tmp_path) -> None:
    """``symbol`` is a path segment, so ``../escaped`` is a 422 and writes nothing."""
    root = tmp_path / "data"
    with TestClient(create_app(data_root=root)) as client:
        response = client.post("/api/data/download", json={**BODY, "symbol": "../escaped"})

    assert response.status_code == 422, response.text
    assert not (tmp_path / "escaped").exists()
    assert not root.exists() or list(root.iterdir()) == []


@pytest.mark.parametrize(
    "endpoint",
    ["/api/train", "/api/backtest", "/api/evaluate", "/api/strategy/calibrate"],
)
def test_every_request_model_rejects_symbol_traversal(tmp_path, endpoint) -> None:
    """The five POST models share one constrained type, so all reject the same value."""
    root = tmp_path / "data"
    body = {"symbol": "../escaped", "run_id": "does-not-matter"}
    with TestClient(create_app(data_root=root)) as client:
        response = client.post(endpoint, json=body)

    assert response.status_code == 422, response.text
    assert not (tmp_path / "escaped").exists()
    assert not root.exists() or list(root.iterdir()) == []


@pytest.mark.parametrize("symbol", ["..", ".hidden", "XAU/USD", "a\\b", "XAUUSD\x00"])
def test_symbol_shape_is_enforced_for_query_parameters(tmp_path, symbol) -> None:
    """GET ids use the same rule as the request models; a bad one is a 422."""
    with TestClient(create_app(data_root=tmp_path)) as client:
        response = client.get("/api/data/status", params={"symbol": symbol, "timeframe": "M1"})

    assert response.status_code == 422, response.text


def test_csv_import_rejects_a_symlinked_directory_that_escapes(tmp_path) -> None:
    """``data/link -> /tmp`` must not pass a lexical prefix check."""
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "xauusd.csv").write_bytes(FIXTURE.read_bytes())
    (root / "link").symlink_to(outside, target_is_directory=True)

    with TestClient(create_app(data_root=root)) as client:
        response = client.post(
            "/api/data/download",
            json={**BODY, "provider": "csv", "path": "link/xauusd.csv", "tz_offset": 3},
        )

    assert response.status_code == 422, response.text
    assert "data root" in response.json()["detail"]


def test_csv_import_rejects_a_symlinked_file_that_escapes(tmp_path) -> None:
    """A symlinked file inside the root is resolved too, not just its parent."""
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "xauusd.csv"
    target.write_bytes(FIXTURE.read_bytes())
    (root / "escape.csv").symlink_to(target)

    with TestClient(create_app(data_root=root)) as client:
        response = client.post(
            "/api/data/download",
            json={**BODY, "provider": "csv", "path": str(root / "escape.csv"), "tz_offset": 3},
        )

    assert response.status_code == 422, response.text


def test_valid_symbols_and_in_root_csv_paths_still_work(tmp_path) -> None:
    """The constraint admits registry symbols and dotted ids, and a relative path."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "xauusd.csv").write_bytes(FIXTURE.read_bytes())

    with TestClient(create_app(data_root=root)) as client:
        synthetic = client.post("/api/data/download", json=BODY)
        status = client.get("/api/data/status", params={"symbol": "XAUUSD", "timeframe": "M1"})
        dotted = client.get("/api/data/status", params={"symbol": "XAUUSD.1", "timeframe": "M1"})
        imported = client.post(
            "/api/data/download",
            json={
                **BODY,
                "symbol": "XAGUSD",
                "provider": "csv",
                "path": "xauusd.csv",
                "tz_offset": 3,
            },
        )

    assert synthetic.status_code == 200, synthetic.text
    assert synthetic.json()["rows"] > 0
    assert status.status_code == 200
    assert status.json()["rows"] == synthetic.json()["rows"]
    assert dotted.status_code == 200
    assert imported.status_code == 200, imported.text
    assert imported.json()["rows"] > 0
