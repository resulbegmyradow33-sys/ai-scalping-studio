"""Symbol registry, provenance and validation endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scalping.api.app import create_app

BODY = {
    "provider": "synthetic",
    "symbol": "XAUUSD",
    "seed": 7,
    "start": "2026-01-05T00:00:00Z",
    "end": "2026-01-07T00:00:00Z",
}


def test_symbol_registry_lists_all_required_instruments() -> None:
    with TestClient(create_app()) as client:
        body = client.get("/api/symbols").json()
    names = {row["symbol"] for row in body}
    assert {
        "XAUUSD",
        "XAGUSD",
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDUSD",
        "NAS100",
        "US30",
        "USOIL",
        "BTCUSD",
    } <= names
    gold = next(row for row in body if row["symbol"] == "XAUUSD")
    assert gold["contract_size"] == 100.0
    assert gold["price_precision"] == 2


def test_timeframe_ladder_matches_the_registry() -> None:
    with TestClient(create_app()) as client:
        body = client.get("/api/timeframes").json()
    rules = {row["timeframe"]: row["rule"] for row in body}
    assert list(rules) == ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    assert rules["M30"] == "30m"
    assert rules["H4"] == "4h"


def test_download_writes_provenance_and_status_returns_it(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path)) as client:
        response = client.post("/api/data/download", json=BODY)
        assert response.status_code == 200
        status = client.get("/api/data/status", params={"symbol": "XAUUSD"}).json()
    assert status["rows"] > 0
    provenance = status["provenance"]
    assert provenance["synthetic"] is True
    assert provenance["source"] == "provider:synthetic"
    assert provenance["checksum"] != "empty"
    assert provenance["dataset_id"].startswith("XAUUSD:M1:")


def test_validate_reports_a_clean_synthetic_slice(tmp_path) -> None:
    with TestClient(create_app(data_root=tmp_path)) as client:
        assert client.post("/api/data/download", json=BODY).status_code == 200
        report = client.get("/api/data/validate", params={"symbol": "XAUUSD"}).json()
    assert report["rows"] > 0
    assert report["ok"] is True
    assert report["first_ts"] is not None


def test_ids_are_validated_before_touching_the_filesystem(tmp_path) -> None:
    """Query parameters become path segments, so traversal is a 422, not a read."""
    with TestClient(create_app(data_root=tmp_path)) as client:
        for path in ("/api/data/status", "/api/data/validate", "/api/regime"):
            bad_symbol = client.get(path, params={"symbol": "../../etc", "timeframe": "M1"})
            assert bad_symbol.status_code == 422, path
            bad_timeframe = client.get(path, params={"symbol": "XAUUSD", "timeframe": "../x"})
            assert bad_timeframe.status_code == 422, path


def test_csv_import_is_confined_to_the_imports_directory(tmp_path) -> None:
    """The request names a file; it never names a path the server can open."""
    fixture = Path(__file__).parent / "fixtures" / "xauusd_gmt3.csv"
    imports = tmp_path / "imports"
    imports.mkdir()
    (imports / "xauusd.csv").write_bytes(fixture.read_bytes())

    with TestClient(create_app(data_root=tmp_path)) as client:
        accepted = client.post(
            "/api/data/download",
            json={**BODY, "provider": "csv", "path": "xauusd.csv", "tz_offset": 3},
        )
        absolute = client.post(
            "/api/data/download",
            json={**BODY, "provider": "csv", "path": str(imports / "xauusd.csv")},
        )
        traversal = client.post(
            "/api/data/download", json={**BODY, "provider": "csv", "path": "../xauusd.csv"}
        )
        missing = client.post(
            "/api/data/download", json={**BODY, "provider": "csv", "path": "absent.csv"}
        )

    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["rows"] > 0
    assert absolute.status_code == 422
    assert traversal.status_code == 422
    assert missing.status_code == 422
    assert "imports" in missing.json()["detail"]


def test_validate_passes_the_outlier_opt_out(tmp_path, monkeypatch) -> None:
    from scalping.api import routes

    calls = []
    original = routes.validate_bars

    def spy(frame, **kwargs):
        calls.append(kwargs)
        return original(frame, **kwargs)

    monkeypatch.setattr(routes, "validate_bars", spy)
    with TestClient(create_app(data_root=tmp_path)) as client:
        assert client.post("/api/data/download", json=BODY).status_code == 200
        report = client.get(
            "/api/data/validate", params={"symbol": "XAUUSD", "include_outliers": "false"}
        )

    assert report.status_code == 200
    assert calls[-1]["include_outliers"] is False
