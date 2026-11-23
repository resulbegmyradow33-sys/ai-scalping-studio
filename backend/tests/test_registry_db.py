"""SQLite registry: schema, round trips, champion promotion and rollback."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scalping.registry import ModelRegistry, RegistryError
from scalping.registry.db import SCHEMA_VERSION

DATASET_ID = "XAUUSD:M1:h1:abc123def456"
OTHER_DATASET_ID = "XAUUSD:M5:h1:0123456789ab"


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry.sqlite3")


def _seed_dataset(registry: ModelRegistry, dataset_id: str = DATASET_ID) -> None:
    registry.register_dataset(
        dataset_id,
        symbol="XAUUSD",
        timeframe="M1",
        digest=dataset_id.rsplit(":", 1)[-1],
        path=f"/data/{dataset_id.replace(':', '_')}.parquet",
        rows=5_900_000,
        first_ts="2008-01-01T00:00:00+00:00",
        last_ts="2025-01-01T00:00:00+00:00",
        synthetic=False,
    )


def _seed_model(
    registry: ModelRegistry,
    model_id: str,
    *,
    dataset_id: str = DATASET_ID,
    metrics: dict[str, float] | None = None,
) -> None:
    registry.register_model(
        model_id,
        dataset_id=dataset_id,
        name="lightgbm",
        path=f"/models/{model_id}.pkl",
        metrics={"accuracy": 0.51} if metrics is None else metrics,
    )


def _seed_two_models(registry: ModelRegistry) -> None:
    _seed_dataset(registry)
    _seed_model(registry, "model-a")
    _seed_model(registry, "model-b")


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.schema_version == SCHEMA_VERSION

    registry.init_schema()
    registry.init_schema()

    assert registry.schema_version == SCHEMA_VERSION
    assert registry.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert registry.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    versions = registry.connection.execute("SELECT version FROM schema_version").fetchall()
    assert [row[0] for row in versions] == [SCHEMA_VERSION]
    registry.close()


def test_dataset_round_trip_survives_reopen(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    record = registry.dataset(DATASET_ID)
    assert record is not None
    assert record.symbol == "XAUUSD"
    assert record.rows == 5_900_000
    assert record.synthetic is False
    assert record.to_dict()["digest"] == "abc123def456"
    registry.close()

    reopened = _registry(tmp_path)
    again = reopened.dataset(DATASET_ID)
    assert again is not None
    assert again == record
    assert reopened.dataset("missing") is None
    reopened.close()


def test_model_round_trip_and_metrics(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    _seed_model(registry, "model-a", metrics={"accuracy": 0.52, "log_loss": 1.02})

    models = registry.list_models()
    assert [model.model_id for model in models] == ["model-a"]
    assert models[0].dataset_id == DATASET_ID
    assert models[0].metrics == {"accuracy": 0.52, "log_loss": 1.02}
    assert registry.list_models(dataset_id=DATASET_ID)[0].model_id == "model-a"
    assert registry.list_models(dataset_id=OTHER_DATASET_ID) == []
    assert registry.champion() is None
    registry.close()


def test_promote_twice_moves_champion_and_records_previous(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)

    first = registry.promote("model-a", "beats the prior baseline")
    assert first.previous_model_id is None
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-a"

    second = registry.promote("model-b", "lower out-of-sample log loss")
    assert second.previous_model_id == "model-a"
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-b"

    history = registry.list_promotions()
    assert [(item.model_id, item.previous_model_id) for item in history] == [
        ("model-a", None),
        ("model-b", "model-a"),
    ]
    registry.close()


def test_rollback_restores_the_previous_model(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "first candidate")
    registry.promote("model-b", "better validation")

    record = registry.rollback("model-a", "test-set regression")
    assert record.model_id == "model-a"
    assert record.previous_model_id == "model-b"
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-a"
    assert [item.model_id for item in registry.list_promotions()] == [
        "model-a",
        "model-b",
        "model-a",
    ]
    registry.close()


def test_promote_current_champion_is_rejected_without_a_new_row(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")

    with pytest.raises(RegistryError, match="already the champion"):
        registry.promote("model-a", "again")

    assert len(registry.list_promotions()) == 1
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-a"
    registry.close()


def test_unknown_models_raise_and_leave_state_untouched(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")

    with pytest.raises(RegistryError, match="ghost"):
        registry.promote("ghost", "does not exist")
    with pytest.raises(RegistryError, match="ghost"):
        registry.rollback("ghost", "does not exist")

    assert len(registry.list_promotions()) == 1
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-a"
    registry.close()


def test_rollback_to_a_model_that_never_won_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")

    with pytest.raises(RegistryError, match="never the champion"):
        registry.rollback("model-b", "not a previous champion")

    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-a"
    registry.close()


def test_rollback_without_a_champion_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)

    with pytest.raises(RegistryError, match="no champion"):
        registry.rollback("model-a", "nothing to roll back")

    assert registry.champion() is None
    registry.close()


def test_duplicate_ids_raise_a_clear_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    _seed_model(registry, "model-a")

    with pytest.raises(RegistryError, match=DATASET_ID):
        _seed_dataset(registry)
    with pytest.raises(RegistryError, match="model-a"):
        _seed_model(registry, "model-a")

    assert len(registry.list_models()) == 1
    registry.close()


def test_foreign_keys_reject_unknown_dataset(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(RegistryError, match=OTHER_DATASET_ID):
        _seed_model(registry, "model-a", dataset_id=OTHER_DATASET_ID)

    with pytest.raises(sqlite3.IntegrityError):
        registry.connection.execute(
            """
            INSERT INTO models (model_id, dataset_id, name, path, metrics_json, created_at)
            VALUES ('raw', ?, 'lightgbm', '/models/raw.pkl', '{}', '2026-01-01T00:00:00+00:00')
            """,
            (OTHER_DATASET_ID,),
        )
    assert registry.list_models() == []
    registry.close()


def test_nan_metrics_are_rejected_before_the_write(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    with pytest.raises(ValueError):
        _seed_model(registry, "model-nan", metrics={"accuracy": float("nan")})
    assert registry.list_models() == []
    registry.close()


def test_reopening_preserves_models_promotions_and_jobs(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")
    registry.promote("model-b", "better")
    job = registry.record_job(
        "train",
        "succeeded",
        started_at="2026-01-05T10:00:00+00:00",
        finished_at="2026-01-05T10:04:00+00:00",
        detail='{"model_id": "model-b"}',
    )
    registry.close()

    reopened = _registry(tmp_path)
    champion = reopened.champion()
    assert champion is not None and champion.model_id == "model-b"
    assert [model.model_id for model in reopened.list_models()] == ["model-a", "model-b"]
    assert [item.model_id for item in reopened.list_promotions()] == ["model-a", "model-b"]
    row = reopened.connection.execute(
        "SELECT kind, status, started_at, finished_at, detail FROM jobs WHERE id = ?",
        (job.id,),
    ).fetchone()
    assert tuple(row) == (
        "train",
        "succeeded",
        "2026-01-05T10:00:00+00:00",
        "2026-01-05T10:04:00+00:00",
        '{"model_id": "model-b"}',
    )
    reopened.close()


def test_record_job_returns_increasing_ids(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = registry.record_job("train", "running")
    second = registry.record_job("backtest", "failed", detail="out of memory")

    assert first.id == 1
    assert second.id == first.id + 1
    assert first.finished_at is None
    assert second.detail == "out of memory"
    registry.close()


def test_closed_registry_refuses_work(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.close()
    registry.close()

    with pytest.raises(RegistryError, match="closed"):
        registry.list_models()
    with pytest.raises(RegistryError, match="closed"):
        registry.promote("model-a", "first")


def test_unsupported_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    registry = ModelRegistry(path)
    registry.connection.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION + 1, "2026-01-01T00:00:00+00:00"),
    )
    registry.close()

    with pytest.raises(RegistryError, match="newer than supported"):
        ModelRegistry(path)
