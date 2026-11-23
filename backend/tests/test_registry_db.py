"""SQLite registry: schema, round trips, champion promotion and rollback."""

from __future__ import annotations

import itertools
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scalping.registry import ModelRecord, ModelRegistry, RegistryError
from scalping.registry.db import _MIGRATIONS, SCHEMA_VERSION

DATASET_ID = "XAUUSD:M1:h1:abc123def456"
OTHER_DATASET_ID = "XAUUSD:M5:h1:0123456789ab"

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

_PROMOTE_SCRIPT = """
import sys

from scalping.registry import ModelRegistry

path, model_id = sys.argv[1], sys.argv[2]
registry = ModelRegistry(path)
registry.promote(model_id, f"worker {model_id}")
registry.close()
"""

_OPEN_SCRIPT = """
import sys

from scalping.registry import ModelRegistry

registry = ModelRegistry(sys.argv[1])
registry.close()
"""

_CRASH_SCRIPT = """
import os
import signal
import sys

from scalping.registry import ModelRegistry

path, model_id = sys.argv[1], sys.argv[2]
registry = ModelRegistry(path)
conn = registry.connection
conn.execute("BEGIN IMMEDIATE")
conn.execute(
    "INSERT INTO promotions (model_id, previous_model_id, reason, created_at, scope)"
    " VALUES (?, NULL, 'uncommitted', '2026-01-01T00:00:00+00:00', 'global')",
    (model_id,),
)
os.kill(os.getpid(), signal.SIGKILL)
"""


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
    created_at: str | None = None,
) -> None:
    registry.register_model(
        model_id,
        dataset_id=dataset_id,
        name="lightgbm",
        path=f"/models/{model_id}.pkl",
        metrics={"accuracy": 0.51} if metrics is None else metrics,
        created_at=created_at,
    )


def _seed_two_models(registry: ModelRegistry) -> None:
    _seed_dataset(registry)
    _seed_model(registry, "model-a")
    _seed_model(registry, "model-b")


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing else f"{SRC_DIR}{os.pathsep}{existing}"
    return env


def _run(script: str, *args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", script, *args],
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.schema_version == SCHEMA_VERSION

    registry.init_schema()
    registry.init_schema()

    assert registry.schema_version == SCHEMA_VERSION
    assert registry.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert registry.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    versions = registry.connection.execute("SELECT version FROM schema_version").fetchall()
    assert [row[0] for row in versions] == list(range(1, SCHEMA_VERSION + 1))
    registry.close()


def test_init_schema_does_not_take_the_write_lock(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    ModelRegistry(path).close()

    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        reopened = ModelRegistry(path)
        assert reopened.schema_version == SCHEMA_VERSION
        reopened.close()
    finally:
        holder.rollback()
        holder.close()


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


def test_champion_promotion_returns_model_and_audit_fields(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    promotion = registry.promote("model-a", "beats baseline")

    champion = registry.champion_promotion()
    assert champion is not None
    assert champion.model.model_id == "model-a"
    assert champion.promotion_id == promotion.id
    assert champion.reason == "beats baseline"
    assert champion.promoted_at == promotion.created_at
    assert champion.scope == "global"
    assert champion.previous_model_id is None
    assert champion.to_dict()["model"]["model_id"] == "model-a"
    assert registry.champion_promotion(scope="other") is None
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


def test_rollback_after_rollback_keeps_the_chain_contiguous(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "a first")
    registry.promote("model-b", "b better")
    registry.rollback("model-a", "b regressed")
    registry.rollback("model-b", "a regressed too")

    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-b"
    history = registry.list_promotions()
    assert [item.model_id for item in history] == [
        "model-a",
        "model-b",
        "model-a",
        "model-b",
    ]
    assert [item.previous_model_id for item in history] == [
        None,
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


def test_ids_are_stripped_on_write(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    record = registry.register_dataset(
        "  spaced  ",
        symbol=" XAUUSD ",
        timeframe="M1",
        digest="abc",
        path="/data/x.parquet",
        rows=1,
    )
    assert record.dataset_id == "spaced"
    assert record.symbol == "XAUUSD"
    assert registry.dataset("spaced") is not None
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


def test_deleting_a_promoted_model_is_blocked_by_foreign_keys(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")

    with pytest.raises(sqlite3.IntegrityError):
        registry.connection.execute("DELETE FROM models WHERE model_id = 'model-a'")

    assert registry.champion() is not None
    registry.close()


def test_missing_champion_model_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    registry = ModelRegistry(path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")

    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA foreign_keys = OFF")
    raw.execute("DELETE FROM models WHERE model_id = 'model-a'")
    raw.close()

    with pytest.raises(RegistryError, match="champion model 'model-a' is missing"):
        registry.champion()
    with pytest.raises(RegistryError, match="champion model 'model-a' is missing"):
        registry.promote("model-b", "next")

    assert len(registry.list_promotions()) == 1
    registry.close()


def test_nan_metrics_are_rejected_before_the_write(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    with pytest.raises(ValueError):
        _seed_model(registry, "model-nan", metrics={"accuracy": float("nan")})
    assert registry.list_models() == []
    registry.close()


def test_invalid_stored_metrics_raise_registry_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    _seed_model(registry, "model-a")

    with pytest.raises(sqlite3.IntegrityError, match="valid JSON"):
        registry.connection.execute(
            "UPDATE models SET metrics_json = 'not json' WHERE model_id = 'model-a'"
        )

    invalid = ModelRecord(
        model_id="bad",
        dataset_id=DATASET_ID,
        name="lightgbm",
        path="/models/bad.pkl",
        metrics_json="{not json",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(RegistryError, match="invalid metrics JSON"):
        _ = invalid.metrics

    non_object = ModelRecord(
        model_id="list",
        dataset_id=DATASET_ID,
        name="lightgbm",
        path="/models/list.pkl",
        metrics_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(RegistryError, match="non-object"):
        _ = non_object.metrics
    registry.close()


def test_model_ordering_normalises_offsets_to_utc(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_dataset(registry)
    _seed_model(registry, "model-first", created_at="2026-01-01T10:00:00+03:00")
    _seed_model(registry, "model-second", created_at="2026-01-01T08:00:00+00:00")

    models = registry.list_models()
    assert [model.model_id for model in models] == ["model-first", "model-second"]
    assert models[0].created_at == "2026-01-01T07:00:00+00:00"
    registry.close()


def test_two_scopes_have_independent_champions(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    _seed_two_models(registry)

    registry.promote("model-a", "gold champion", scope="xauusd")
    registry.promote("model-b", "silver champion", scope="xagusd")

    xauusd = registry.champion(scope="xauusd")
    xagusd = registry.champion(scope="xagusd")
    assert xauusd is not None and xauusd.model_id == "model-a"
    assert xagusd is not None and xagusd.model_id == "model-b"
    assert registry.champion() is None

    registry.promote("model-b", "gold upgrade", scope="xauusd")
    xauusd = registry.champion(scope="xauusd")
    assert xauusd is not None and xauusd.model_id == "model-b"
    assert [item.model_id for item in registry.list_promotions(scope="xauusd")] == [
        "model-a",
        "model-b",
    ]

    registry.rollback("model-a", "gold regression", scope="xauusd")
    xauusd = registry.champion(scope="xauusd")
    assert xauusd is not None and xauusd.model_id == "model-a"
    xagusd = registry.champion(scope="xagusd")
    assert xagusd is not None and xagusd.model_id == "model-b"
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


def test_list_jobs_filters_and_update_job(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = registry.record_job("train", "running", detail="epoch 1")
    second = registry.record_job("backtest", "pending")

    assert [job.id for job in registry.list_jobs()] == [second.id, first.id]
    assert [job.id for job in registry.list_jobs(kind="train")] == [first.id]
    assert [job.id for job in registry.list_jobs(status="pending")] == [second.id]
    assert [job.id for job in registry.list_jobs(limit=1)] == [second.id]
    assert registry.list_jobs(kind="drift") == []

    updated = registry.update_job(
        first.id,
        "succeeded",
        finished_at="2026-01-05T10:04:00+00:00",
        detail='{"epochs": 3}',
    )
    assert updated.status == "succeeded"
    assert updated.finished_at == "2026-01-05T10:04:00+00:00"
    assert updated.detail == '{"epochs": 3}'
    assert updated.kind == "train"
    assert updated.started_at is None

    with pytest.raises(RegistryError, match="not registered"):
        registry.update_job(999, "succeeded")
    with pytest.raises(ValueError, match="status must be one of"):
        registry.update_job(second.id, "done")
    with pytest.raises(ValueError, match="limit"):
        registry.list_jobs(limit=0)
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


def test_version_one_database_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite3"
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    raw.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (1, '2026-01-01T00:00:00+00:00')"
    )
    for statement in _MIGRATIONS[1]:
        raw.execute(statement)
    raw.execute(
        """
        INSERT INTO datasets (dataset_id, symbol, timeframe, digest, path, rows, created_at)
        VALUES (?, 'XAUUSD', 'M1', 'abc', '/data/x.parquet', 10, '2026-01-01T00:00:00+00:00')
        """,
        (DATASET_ID,),
    )
    for model_id in ("model-a", "model-b"):
        raw.execute(
            """
            INSERT INTO models (model_id, dataset_id, name, path, metrics_json, created_at)
            VALUES (?, ?, 'lightgbm', ?, '{}', '2026-01-01T00:00:00+00:00')
            """,
            (model_id, DATASET_ID, f"/models/{model_id}.pkl"),
        )
    raw.execute(
        """
        INSERT INTO promotions (model_id, previous_model_id, reason, created_at)
        VALUES ('model-a', NULL, 'v1 champion', '2026-01-01T00:00:00+00:00')
        """
    )
    raw.close()

    registry = ModelRegistry(path)
    assert registry.schema_version == SCHEMA_VERSION
    history = registry.list_promotions()
    assert [item.scope for item in history] == ["global"]
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-a"

    registry.promote("model-b", "v2 champion")
    champion = registry.champion()
    assert champion is not None and champion.model_id == "model-b"
    registry.close()


def test_concurrent_first_open_of_a_fresh_database_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "fresh.sqlite3"
    processes = [_run(_OPEN_SCRIPT, str(path)) for _ in range(4)]
    results = [process.communicate(timeout=120) for process in processes]

    assert [process.returncode for process in processes] == [0, 0, 0, 0], [
        stderr.decode(errors="replace") for _, stderr in results
    ]
    registry = ModelRegistry(path)
    assert registry.schema_version == SCHEMA_VERSION
    registry.close()


def test_concurrent_promotions_keep_one_champion_and_a_contiguous_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.sqlite3"
    registry = ModelRegistry(path)
    _seed_dataset(registry)
    for index in range(4):
        _seed_model(registry, f"model-{index}")
    registry.close()

    processes = [_run(_PROMOTE_SCRIPT, str(path), f"model-{index}") for index in range(4)]
    results = [process.communicate(timeout=120) for process in processes]
    assert [process.returncode for process in processes] == [0, 0, 0, 0], [
        stderr.decode(errors="replace") for _, stderr in results
    ]

    reopened = ModelRegistry(path)
    history = reopened.list_promotions()
    assert len(history) == 4
    assert history[0].previous_model_id is None
    assert {item.model_id for item in history} == {f"model-{index}" for index in range(4)}
    for earlier, later in itertools.pairwise(history):
        assert later.previous_model_id == earlier.model_id
    champion = reopened.champion()
    assert champion is not None and champion.model_id == history[-1].model_id
    reopened.close()


def test_write_lock_timeout_raises_registry_error(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    registry = ModelRegistry(path)
    _seed_two_models(registry)
    registry.connection.execute("PRAGMA busy_timeout = 100")

    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RegistryError, match="write lock"):
            registry.promote("model-a", "blocked by another writer")
    finally:
        holder.rollback()
        holder.close()

    assert registry.list_promotions() == []
    registry.close()


def test_killed_process_leaves_no_uncommitted_promotion(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    registry = ModelRegistry(path)
    _seed_two_models(registry)
    registry.promote("model-a", "first")
    registry.close()

    process = _run(_CRASH_SCRIPT, str(path), "model-b")
    _, stderr = process.communicate(timeout=120)
    assert process.returncode == -9, stderr.decode(errors="replace")

    reopened = ModelRegistry(path)
    assert [item.model_id for item in reopened.list_promotions()] == ["model-a"]
    champion = reopened.champion()
    assert champion is not None and champion.model_id == "model-a"
    assert reopened.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    reopened.close()
