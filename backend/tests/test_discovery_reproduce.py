"""Reproducing a stored evidence bundle: identity, data, mismatch and CLI.

The fixture is the same constant-price tape the evidence tests use: an
always-long DSL spec, one purged chronological dataset and a real
``build_evidence`` call. Nothing here searches, promotes or evaluates the test
split; the tests only replay and compare.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.artifact import save_dataset_arrays
from scalping.datasets.builder import DatasetBundle, build_dataset
from scalping.discovery import reproduce as reproduce_module
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceConfig,
    EvidenceError,
    LockedOOSEvidence,
    StressEvidence,
    build_evidence,
    evidence_checksum,
    locked_oos_checksum,
)
from scalping.discovery.reproduce import (
    ReproduceError,
    ReproduceMismatch,
    dataset_artifact_dir,
    reproduce_bundle,
    spec_sidecar_path,
)
from scalping.discovery.run import code_version as current_code_version
from scalping.features.build import FeatureFrame, build_features
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)
from scalping.types import CostConfig

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)
ROWS = 1600
CODE_VERSION = current_code_version()

BACKEND = Path(__file__).resolve().parents[1]

ALWAYS = Compare(Const(0.0), "<", Const(1.0))
NEVER = Compare(Const(0.0), ">", Const(1.0))


def constant_bars(rows: int = ROWS) -> pl.DataFrame:
    """Flat M1 tape with a 0.20 spread; every round turn loses its costs."""
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=index) for index in range(rows)],
            "open": [1000.0] * rows,
            "high": [1000.001] * rows,
            "low": [999.999] * rows,
            "close": [1000.0] * rows,
            "bid_close": [1000.0] * rows,
            "ask_close": [1000.2] * rows,
            "spread": [0.2] * rows,
            "volume": [10] * rows,
        }
    )


def long_spec() -> StrategySpec:
    return StrategySpec(
        entries=(EntryRule("long", ALWAYS),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(1),
        max_lookback=5,
    )


def config(**overrides: Any) -> EvidenceConfig:
    values: dict[str, Any] = {
        "code_version": CODE_VERSION,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "folds": 3,
        "purge_bars": 0,
        "embargo_bars": 0,
        "min_trades_per_fold": 2,
        "cost_stress": (1.5, 2.0),
        "regime_coverage_min": 0.0,
        "bootstrap_samples": 100,
        "bootstrap_confidence": 0.95,
        "seed": 7,
        "search_budget": 50,
        "trial_count": 4,
        "window": 60,
    }
    values.update(overrides)
    return EvidenceConfig(**values)


def stored_bundle(
    tmp_path: Path,
    *,
    bars: pl.DataFrame | None = None,
    settings: Settings | None = None,
) -> SimpleNamespace:
    """Write one complete, reproducible experiment under ``tmp_path``."""
    store_root = tmp_path / "data"
    frame = constant_bars() if bars is None else bars
    ParquetStore(store_root).write(SYMBOL, TIMEFRAME, frame)

    features: FeatureFrame = build_features(frame)
    bundle: DatasetBundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
    )
    save_dataset_arrays(dataset_artifact_dir(store_root, bundle.manifest.checksum), bundle)
    evidence = build_evidence(
        bundle,
        long_spec(),
        config(),
        features=features,
        settings=settings,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    path = tmp_path / "candidate.evidence.json"
    path.write_text(json.dumps(evidence.to_dict(), indent=2) + "\n", encoding="utf-8")
    spec_path = spec_sidecar_path(path)
    spec_path.write_text(json.dumps(long_spec().canonical_dict()), encoding="utf-8")
    return SimpleNamespace(
        store_root=store_root,
        features=features,
        bundle=bundle,
        evidence=evidence,
        path=path,
        spec_path=spec_path,
        settings=Settings() if settings is None else settings,
    )


def reproduce(stored: SimpleNamespace, **kwargs: Any) -> CandidateEvidence | ReproduceMismatch:
    kwargs.setdefault("code_version", CODE_VERSION)
    return reproduce_bundle(stored.path, store_root=stored.store_root, **kwargs)


# --- exact replay -------------------------------------------------------------


def test_exact_round_trip_reproduces_the_checksum(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)

    result = reproduce(stored, spec=long_spec())

    assert isinstance(result, CandidateEvidence)
    assert result.checksum == stored.evidence.checksum
    assert result.folds == stored.evidence.folds
    assert result.aggregate == stored.evidence.aggregate
    assert result.splits == stored.evidence.splits
    assert result.created_at == stored.evidence.created_at
    assert result.splits.test_start_row >= result.splits.validation_end_row
    assert all(fold.test_end_row <= result.splits.validation_end_row for fold in result.folds)


def test_sidecar_spec_is_used_when_none_is_passed(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)

    result = reproduce(stored)

    assert isinstance(result, CandidateEvidence)
    assert result.checksum == stored.evidence.checksum


def test_inline_json_spec_is_accepted(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)

    result = reproduce(stored, spec=json.dumps(long_spec().canonical_dict()))

    assert isinstance(result, CandidateEvidence)
    assert result.checksum == stored.evidence.checksum


def test_missing_spec_is_refused(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    stored.spec_path.unlink()

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored)

    assert exc.value.code == "spec_missing"
    assert exc.value.invalid is True


# --- invalid input ------------------------------------------------------------


def _locked_section(evidence: CandidateEvidence) -> LockedOOSEvidence:
    splits = evidence.splits
    section = LockedOOSEvidence(
        checksum="0" * 64,
        evidence_checksum=evidence.checksum,
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        symbol=evidence.symbol,
        timeframe=evidence.timeframe,
        spec_hash=evidence.spec_hash,
        spec_version=evidence.spec_version,
        complexity=evidence.complexity,
        code_version=evidence.code_version,
        settings_digest=evidence.settings_digest,
        policy_id="test-policy",
        policy_version=1,
        test_start_row=splits.test_start_row,
        test_end_row=splits.test_end_row,
        test_rows=splits.test_end_row - splits.test_start_row,
        test_start_ts="2026-02-01T00:00:00+00:00",
        test_end_ts="2026-02-02T00:00:00+00:00",
        trades=10,
        net_profit=1.0,
        profit_factor=1.5,
        win_rate=0.5,
        expectancy_r=0.2,
        total_r=2.0,
        max_drawdown=0.1,
        expectancy_r_ci95_low=0.05,
        expectancy_r_ci95_high=0.35,
        total_r_ci95_low=0.5,
        total_r_ci95_high=4.0,
        bootstrap_samples=100,
        bootstrap_confidence=0.95,
        bootstrap_method="iid-bootstrap",
        regime_coverage=0.9,
        stress=tuple(
            StressEvidence(
                multiplier=multiplier,
                trades=10,
                net_profit=0.8,
                profit_factor=1.4,
                max_drawdown=0.1,
                expectancy_r=0.15,
            )
            for multiplier in (1.5, 2.0)
        ),
        created_at="2026-02-02T00:00:00+00:00",
    )
    return replace(section, checksum=locked_oos_checksum(section))


def _write_locked_bundle(stored: SimpleNamespace) -> LockedOOSEvidence:
    section = _locked_section(stored.evidence)
    bundle = replace(stored.evidence, locked_oos=section)
    bundle = replace(bundle, checksum=evidence_checksum(bundle))
    stored.path.write_text(json.dumps(bundle.to_dict(), indent=2) + "\n", encoding="utf-8")
    return section


def test_a_locked_bundle_round_trips_but_is_not_copied(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    section = _write_locked_bundle(stored)

    loaded = reproduce_module.evidence_from_dict(
        json.loads(stored.path.read_text(encoding="utf-8"))
    )
    assert loaded.locked_oos == section

    result = reproduce(stored, spec=long_spec())

    assert isinstance(result, CandidateEvidence)
    assert result.locked_oos is None
    assert reproduce_module._base_checksum(
        result, drop_locked=True
    ) == reproduce_module._base_checksum(stored.evidence, drop_locked=True)


def test_a_tampered_locked_bundle_fails_its_checksum(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    _write_locked_bundle(stored)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["locked_oos"]["expectancy_r"] = 0.9
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError) as exc:
        reproduce(stored)

    assert exc.value.code == "checksum_mismatch"


def test_an_unknown_field_is_invalid_payload(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["future_field"] = {"not": "supported"}
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError) as exc:
        reproduce(stored)

    assert exc.value.code == "invalid_payload"


def test_tampered_bundle_fails_its_checksum(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["aggregate"]["expectancy_r"] = payload["aggregate"]["expectancy_r"] + 0.001
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError) as exc:
        reproduce(stored)

    assert exc.value.code == "checksum_mismatch"


def test_old_schema_is_refused_before_a_rebuild(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceError) as exc:
        reproduce(stored)

    assert exc.value.code == "schema_mismatch"


def test_a_different_spec_is_refused_by_hash(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    other = replace(long_spec(), holding=HoldingLimit(2))

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored, spec=other)

    assert exc.value.code == "spec_mismatch"
    assert exc.value.invalid is False
    assert exc.value.expected == stored.evidence.checksum


# --- identity refusals --------------------------------------------------------


def test_changed_settings_are_detected(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored, settings=Settings(costs=CostConfig(spread_floor=0.31)))

    assert exc.value.code == "settings_mismatch"
    assert exc.value.invalid is False


def test_changed_code_version_is_detected(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored, code_version="another-tree")

    assert exc.value.code == "code_version_mismatch"
    assert exc.value.invalid is False


def test_missing_dataset_artifact_is_detected(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    artifact = dataset_artifact_dir(stored.store_root, stored.bundle.manifest.checksum)
    for child in artifact.iterdir():
        child.unlink()
    artifact.rmdir()

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored)

    assert exc.value.code == "dataset_missing"
    assert exc.value.invalid is False


def test_corrupt_dataset_artifact_is_detected(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    artifact = dataset_artifact_dir(stored.store_root, stored.bundle.manifest.checksum)
    (artifact / "X.npy").unlink()

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored)

    assert exc.value.code == "dataset_corrupt"


def test_changed_bars_break_the_dataset_identity(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    rows = constant_bars().to_dicts()
    rows[500]["close"] = 1001.5
    rows[500]["high"] = 1001.6
    rows[500]["bid_close"] = 1001.5
    rows[500]["ask_close"] = 1001.7
    ParquetStore(stored.store_root).write(SYMBOL, TIMEFRAME, pl.DataFrame(rows))

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored)

    assert exc.value.code == "dataset_checksum_mismatch"
    assert exc.value.invalid is False


# --- scope and side effects ---------------------------------------------------


def test_test_split_rows_are_identity_not_evidence(tmp_path: Path) -> None:
    """The stored checksum covers the test rows too, so editing them is a refusal.

    Reproduction has to rebuild the whole dataset array to prove its identity,
    but no fold, metric or gate ever slices those rows: the evidence build stops
    at the validation end. A changed holdout is therefore a dataset mismatch
    before an evidence section exists, never a silently different score.
    """
    stored = stored_bundle(tmp_path)
    cut = int(stored.bundle.manifest.test.start)
    rows = constant_bars().to_dicts()
    for row in rows[cut:]:
        row["open"] = 5000.0
        row["close"] = 5000.0
        row["high"] = 5000.2
        row["low"] = 4999.8
        row["bid_close"] = 5000.0
        row["ask_close"] = 5000.2
    ParquetStore(stored.store_root).write(SYMBOL, TIMEFRAME, pl.DataFrame(rows))

    with pytest.raises(ReproduceError) as exc:
        reproduce(stored)

    assert exc.value.code == "dataset_checksum_mismatch"


def test_reproduce_writes_nothing(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    before = stored.path.read_bytes()
    tree_before = sorted(path for path in stored.store_root.rglob("*"))

    reproduce(stored)

    assert stored.path.read_bytes() == before
    assert sorted(path for path in stored.store_root.rglob("*")) == tree_before


def test_reproduce_uses_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stored = stored_bundle(tmp_path)

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reproduction must not open a socket")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    result = reproduce(stored)

    assert isinstance(result, CandidateEvidence)
    assert result.checksum == stored.evidence.checksum


def test_a_rebuilt_difference_returns_a_typed_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stored = stored_bundle(tmp_path)
    before = stored.path.read_bytes()
    real = reproduce_module.build_evidence

    def drifted(*args: Any, **kwargs: Any) -> CandidateEvidence:
        evidence = real(*args, **kwargs)
        return replace(
            evidence,
            aggregate=replace(
                evidence.aggregate,
                expectancy_r=evidence.aggregate.expectancy_r + 1.0,
            ),
            checksum="0" * 64,
        )

    monkeypatch.setattr(reproduce_module, "build_evidence", drifted)

    outcome = reproduce(stored)

    assert isinstance(outcome, ReproduceMismatch)
    assert outcome.expected == stored.evidence.checksum
    assert outcome.reproduced == reproduce_module._base_checksum(
        outcome.evidence, drop_locked=False
    )
    assert outcome.sections == ("aggregate",)
    assert outcome.differences[0].path == "aggregate.expectancy_r"
    assert stored.path.read_bytes() == before


# --- CLI ----------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def cli_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_cli_reproduces_one_json_line_and_writes_the_report(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    out = tmp_path / "report.json"

    proc = run_cli(
        "discovery",
        "reproduce",
        "--bundle",
        str(stored.path),
        "--store-root",
        str(stored.store_root),
        "--out",
        str(out),
    )

    assert proc.returncode == 0, proc.stderr
    lines = cli_lines(proc)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload == {
        "reproduced": True,
        "checksum": stored.evidence.checksum,
        "expected": stored.evidence.checksum,
        "mismatches": [],
    }
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["reproduced"] is True
    assert report["evidence"]["checksum"] == stored.evidence.checksum


def test_cli_exits_one_when_settings_do_not_match(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    settings = tmp_path / "settings.yaml"
    settings.write_text("costs:\n  spread_floor: 0.31\n", encoding="utf-8")

    proc = run_cli(
        "discovery",
        "reproduce",
        "--bundle",
        str(stored.path),
        "--store-root",
        str(stored.store_root),
        "--settings",
        str(settings),
    )

    assert proc.returncode == 1, proc.stderr
    payload = json.loads(cli_lines(proc)[0])
    assert payload["reproduced"] is False
    assert payload["expected"] == stored.evidence.checksum
    assert payload["mismatches"] == ["settings_mismatch"]


def test_cli_exits_two_on_a_missing_bundle(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    proc = run_cli(
        "discovery",
        "reproduce",
        "--bundle",
        str(missing),
        "--store-root",
        str(tmp_path),
    )

    assert proc.returncode == 2
    assert cli_lines(proc) == []
    error = json.loads(next(line for line in proc.stderr.splitlines() if line.strip()))
    assert "bundle_missing" in error["error"]


def test_cli_exits_two_on_a_tampered_bundle(tmp_path: Path) -> None:
    stored = stored_bundle(tmp_path)
    payload = json.loads(stored.path.read_text(encoding="utf-8"))
    payload["aggregate"]["total_r"] = payload["aggregate"]["total_r"] + 5.0
    stored.path.write_text(json.dumps(payload), encoding="utf-8")

    proc = run_cli(
        "discovery",
        "reproduce",
        "--bundle",
        str(stored.path),
        "--store-root",
        str(stored.store_root),
    )

    assert proc.returncode == 2
    error = json.loads(next(line for line in proc.stderr.splitlines() if line.strip()))
    assert "checksum_mismatch" in error["error"]
