"""Market-regime endpoint.

The endpoint is descriptive: it returns the rule labels already proven causal in
``test_regime_rules.py``. These tests cover the HTTP contract (slicing, ordering,
limit validation and the empty-store state), not the rules themselves.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app


@pytest.fixture
def client(tmp_path, seed_data):
    return TestClient(create_app(data_root=tmp_path))


def test_regime_returns_current_counts_and_timeline(client):
    response = client.get("/api/regime", params={"symbol": "XAUUSD", "timeframe": "M1"})

    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "XAUUSD"
    assert body["timeframe"] == "M1"
    assert body["rows"] > 0

    current = body["current"]
    assert set(current) == {
        "ts",
        "trend",
        "trend_strength",
        "volatility",
        "expansion",
        "breakout",
        "unusual",
        "regime",
    }
    assert isinstance(current["unusual"], bool)

    counts = body["counts"]
    assert sum(row["bars"] for row in counts) == body["rows"]
    assert counts == sorted(counts, key=lambda row: (-row["bars"], row["regime"]))
    assert all(0.0 < row["share"] <= 1.0 for row in counts)
    assert current["regime"] in {row["regime"] for row in counts}

    series = body["series"]
    assert len(series) == min(500, body["rows"])
    assert series[-1]["regime"] == current["regime"]
    assert series[-1]["ts"] == current["ts"]
    assert [point["ts"] for point in series] == sorted(point["ts"] for point in series)


def test_limit_only_caps_the_timeline(client):
    full = client.get("/api/regime", params={"symbol": "XAUUSD"}).json()
    small = client.get(
        "/api/regime", params={"symbol": "XAUUSD", "limit": 5}
    ).json()

    assert len(small["series"]) == 5
    assert small["rows"] == full["rows"]
    assert small["counts"] == full["counts"]
    assert small["series"] == full["series"][-5:]


@pytest.mark.parametrize("limit", [0, 5001])
def test_limit_is_validated(client, limit):
    response = client.get("/api/regime", params={"symbol": "XAUUSD", "limit": limit})
    assert response.status_code == 422


def test_empty_store_is_a_normal_empty_response(tmp_path):
    with TestClient(create_app(data_root=tmp_path)) as client:
        response = client.get("/api/regime", params={"symbol": "XAUUSD"})

    assert response.status_code == 200
    assert response.json() == {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "rows": 0,
        "current": None,
        "counts": [],
        "series": [],
    }
