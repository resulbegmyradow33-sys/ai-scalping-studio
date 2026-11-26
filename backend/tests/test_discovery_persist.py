"""Writing the dataset artifact and evidence bundle that reproduce replays.

Issue #136: before this module, only a test fixture could produce the artifact
and bundle ``discovery reproduce`` expects. These tests cover the writer round
trip, the end-to-end CLI, overwrite refusal, atomic temp cleanup, the spec
sidecar, corrupt artifacts and the rule that evidence never reads a test-row
metric.

The tape is the constant-price M1 series the evidence and reproduce tests use:
one always-long DSL spec, a purged chronological dataset and a real
``build_evidence`` call. Nothing searches, promotes or scores the test split.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from scalping.config import Settings
from scalping.data.provenance import build_provenance, save_provenance
from scalping.data.store import ParquetStore
from scalping.datasets.artifact import ARTIFACT_FILE, save_dataset_arrays
from scalping.datasets.builder import build_dataset
from scalping.discovery import persist as persist_module
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceConfig,
    build_evidence,
)
from scalping.discovery.persist import (
    PersistError,
    load_dataset_artifact,
    load_evidence_bundle,
    load_strategy_spec,
    persist_evidence_run,
    save_dataset_artifact,
    save_evidence_bundle,
)
from scalping.discovery.promotion import PROMOTION_V1, evaluate_promotion
from scalping.discovery.reproduce import (
    ReproduceError,
    ReproduceMismatch,
    dataset_artifact_dir,
    reproduce_bundle,
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


def other_spec() -> StrategySpec:
    """A different, valid spec; its hash must not match the long one."""
    return StrategySpec(
        entries=(EntryRule("long", ALWAYS),),
        exits=(ExitRule(NEVER),),
        holding=HoldingLimit(2),
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


def stored_context(tmp_path: Path, *, rows: int = ROWS) -> dict[str, Any]:
    """Bars, feature frame and bundle for one experiment under ``tmp_path``."""
    store_root = tmp_path / "data"
    frame = constant_bars(rows)
    ParquetStore(store_root).write(SYMBOL, TIMEFRAME, frame)
    features = build_features(frame)
    bundle = build_dataset(
        features,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        horizon=1,
        train_fraction=0.6,
        validation_fraction=0.2,
        purge_bars=3,
        embargo_bars=2,
    )
    return {
        "store_root": store_root,
        "frame": frame,
        "features": features,
        "bundle": bundle,
        "settings": Settings(),
    }


def persisted_bundle(tmp_path: Path) -> dict[str, Any]:
    """Write one complete artifact + bundle + sidecar and return the paths."""
    context = stored_context(tmp_path)
    evidence = build_evidence(
        context["bundle"],
        long_spec(),
        config(),
        features=context["features"],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    artifact_dir = save_dataset_artifact(
        context["store_root"], context["bundle"], context["features"]
    )
    path = tmp_path / "candidate.evidence.json"
    save_evidence_bundle(path, evidence, long_spec())
    context.update(
        {
            "evidence": evidence,
            "artifact_dir": artifact_dir,
            "path": path,
        }
    )
    return context


# --- round trip ---------------------------------------------------------------


def test_artifact_lands_at_the_checksum_directory_and_verifies(tmp_path: Path) -> None:
    context = stored_context(tmp_path)

    artifact_dir = save_dataset_artifact(
        context["store_root"], context["bundle"], context["features"]
    )

    assert artifact_dir == dataset_artifact_dir(
        context["store_root"], context["bundle"].manifest.checksum
    )
    assert (artifact_dir / ARTIFACT_FILE).is_file()
    for name in ("X", "y_direction", "y_return", "ts", "split"):
        assert (artifact_dir / f"{name}.npy").is_file()
    manifest = json.loads((artifact_dir / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == context["bundle"].manifest.dataset_id
    assert manifest["manifest"]["checksum"] == context["bundle"].manifest.checksum
    assert set(manifest["arrays"]) == {"X", "y_direction", "y_return", "ts", "split"}
    for entry in manifest["arrays"].values():
        assert isinstance(entry["dtype"], str)
        assert isinstance(entry["shape"], list)
        assert isinstance(entry["checksum"], str) and entry["checksum"]

    loaded = load_dataset_artifact(context["store_root"], context["bundle"].manifest.checksum)
    assert loaded.rows("test") == context["bundle"].rows("test") > 0
    assert np.array_equal(np.asarray(loaded.X), np.asarray(context["bundle"].X))


def test_bundle_and_sidecar_reproduce_the_checksum(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)

    sidecar = context["path"].with_suffix(".spec.json")
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == long_spec().canonical_dict()

    outcome = reproduce_bundle(context["path"], store_root=context["store_root"])

    assert not isinstance(outcome, ReproduceMismatch)
    assert isinstance(outcome, CandidateEvidence)
    assert outcome.checksum == context["evidence"].checksum


def test_load_evidence_bundle_re_derives_the_checksum(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)

    loaded = load_evidence_bundle(context["path"])

    assert loaded.checksum == context["evidence"].checksum
    assert loaded.folds == context["evidence"].folds


def test_tampered_bundle_is_refused_on_read(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)
    payload = json.loads(context["path"].read_text(encoding="utf-8"))
    payload["aggregate"]["expectancy_r"] += 0.001
    context["path"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PersistError) as exc:
        load_evidence_bundle(context["path"])

    assert exc.value.code == "checksum_mismatch"


def test_sidecar_spec_must_match_the_bundle(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)
    path = tmp_path / "mismatch.evidence.json"

    with pytest.raises(PersistError) as exc:
        save_evidence_bundle(path, context["evidence"], other_spec())

    assert exc.value.code == "spec_mismatch"
    assert not path.exists()
    assert not path.with_suffix(".spec.json").exists()


def test_load_strategy_spec_accepts_inline_json_and_paths(tmp_path: Path) -> None:
    payload = json.dumps(long_spec().canonical_dict())
    path = tmp_path / "long.spec.json"
    path.write_text(payload, encoding="utf-8")
    expected = long_spec().canonical_hash()

    assert load_strategy_spec(payload).canonical_hash() == expected
    assert load_strategy_spec(path).canonical_hash() == expected

    with pytest.raises(PersistError) as exc:
        load_strategy_spec("not json")
    assert exc.value.code == "spec_missing"


# --- end-to-end producer ------------------------------------------------------


def test_persist_evidence_run_writes_and_reproduces(tmp_path: Path) -> None:
    context = stored_context(tmp_path)
    out = tmp_path / "runs" / "candidate.evidence.json"

    result = persist_evidence_run(
        context["store_root"],
        out,
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    assert result.artifact_dir.is_dir()
    assert (result.artifact_dir / ARTIFACT_FILE).is_file()
    assert result.bundle_path == out and out.is_file()
    assert result.spec_path == out.with_suffix(".spec.json") and result.spec_path.is_file()
    assert result.dataset_checksum == result.evidence.dataset_checksum
    assert result.checksum == result.evidence.checksum

    payload = result.to_dict()
    assert payload["persisted"] is True
    assert payload["bundle"] == str(out)
    assert payload["artifact"] == str(result.artifact_dir)
    assert payload["evidence_checksum"] == result.checksum
    assert payload["folds"] == 3

    loaded = load_dataset_artifact(context["store_root"], result.dataset_checksum)
    assert loaded.dataset_id == result.dataset_id
    assert load_evidence_bundle(out).checksum == result.checksum

    outcome = reproduce_bundle(out, store_root=context["store_root"])
    assert not isinstance(outcome, ReproduceMismatch)
    assert outcome.checksum == result.checksum


def test_persist_evidence_run_uses_the_default_config(tmp_path: Path) -> None:
    context = stored_context(tmp_path)
    out = tmp_path / "candidate.evidence.json"

    result = persist_evidence_run(
        context["store_root"],
        out,
        spec=long_spec().canonical_dict(),
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        settings=context["settings"],
        use_cache=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.evidence.code_version == CODE_VERSION
    assert result.evidence.complexity == long_spec().complexity()
    assert result.evidence.policy.folds == 3
    assert isinstance(reproduce_bundle(out, store_root=context["store_root"]), CandidateEvidence)


def test_persist_evidence_run_refuses_a_config_for_another_symbol(tmp_path: Path) -> None:
    context = stored_context(tmp_path)

    with pytest.raises(PersistError) as exc:
        persist_evidence_run(
            context["store_root"],
            tmp_path / "candidate.evidence.json",
            spec=long_spec(),
            config=config(symbol="EURUSD"),
            settings=context["settings"],
        )

    assert exc.value.code == "config_mismatch"


def test_persist_evidence_run_reports_missing_bars(tmp_path: Path) -> None:
    with pytest.raises(PersistError) as exc:
        persist_evidence_run(
            tmp_path / "empty-store",
            tmp_path / "candidate.evidence.json",
            spec=long_spec(),
            settings=Settings(),
        )

    assert exc.value.code == "bars_missing"


# --- store provenance --------------------------------------------------------


def test_evidence_copies_verified_store_provenance(tmp_path: Path) -> None:
    """A matching sidecar is the only source of a real provenance claim."""
    context = stored_context(tmp_path)
    save_provenance(
        context["store_root"],
        build_provenance(
            context["frame"],
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="csv:provenance-test",
            synthetic=False,
        ),
    )
    out = tmp_path / "candidate.evidence.json"

    result = persist_evidence_run(
        context["store_root"],
        out,
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    assert result.evidence.source == "csv:provenance-test"
    assert result.evidence.synthetic is False
    assert result.evidence.provenance_verified is True
    assert result.provenance_verified is True
    assert result.warnings == ()
    payload = result.to_dict()
    assert payload["source"] == "csv:provenance-test"
    assert payload["synthetic"] is False
    assert payload["provenance_verified"] is True
    assert payload["warnings"] == []


def test_evidence_leaves_provenance_unknown_without_a_sidecar(tmp_path: Path) -> None:
    """No sidecar means unknown; a missing file is never read as real bars."""
    context = stored_context(tmp_path)

    result = persist_evidence_run(
        context["store_root"],
        tmp_path / "candidate.evidence.json",
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    assert result.evidence.source is None
    assert result.evidence.synthetic is None
    assert result.evidence.provenance_verified is None
    assert result.provenance_verified is None
    assert result.warnings == ()


def test_evidence_rejects_a_stale_sidecar_and_records_a_warning(tmp_path: Path) -> None:
    """A sidecar that no longer hashes the slice cannot label the dataset."""
    context = stored_context(tmp_path)
    store = ParquetStore(context["store_root"])
    save_provenance(
        context["store_root"],
        build_provenance(
            context["frame"],
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="csv:stale",
            synthetic=False,
        ),
    )
    # Same row count and timestamp span, one changed close: only the checksum
    # can catch that the sidecar no longer describes the stored bars.
    changed = context["frame"].clone()
    closes = changed["close"].to_list()
    closes[10] = closes[10] + 1.0
    changed = changed.with_columns(pl.Series("close", closes))
    store.write(SYMBOL, TIMEFRAME, changed)

    result = persist_evidence_run(
        context["store_root"],
        tmp_path / "candidate.evidence.json",
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    assert result.evidence.source is None
    assert result.evidence.synthetic is None
    assert result.evidence.provenance_verified is False
    assert result.provenance_verified is False
    assert len(result.warnings) == 1
    assert "checksum" in result.warnings[0]


def test_reproduce_keeps_verified_provenance(tmp_path: Path) -> None:
    context = stored_context(tmp_path)
    save_provenance(
        context["store_root"],
        build_provenance(
            context["frame"],
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="csv:round-trip",
            synthetic=False,
        ),
    )
    out = tmp_path / "candidate.evidence.json"
    persisted = persist_evidence_run(
        context["store_root"],
        out,
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    rebuilt = reproduce_bundle(out, store_root=context["store_root"])

    assert isinstance(rebuilt, CandidateEvidence)
    assert rebuilt.source == persisted.evidence.source == "csv:round-trip"
    assert rebuilt.synthetic is persisted.evidence.synthetic is False
    assert rebuilt.provenance_verified is True
    assert rebuilt.checksum == persisted.checksum


def test_verified_provenance_bundle_loads_and_reproduces(tmp_path: Path) -> None:
    """A real persisted bundle survives both the strict loader and a rebuild."""
    context = stored_context(tmp_path)
    save_provenance(
        context["store_root"],
        build_provenance(
            context["frame"],
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="csv:load-reproduce",
            synthetic=False,
        ),
    )
    out = tmp_path / "candidate.evidence.json"
    persisted = persist_evidence_run(
        context["store_root"],
        out,
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    loaded = load_evidence_bundle(out)
    assert loaded.provenance_verified is True
    assert loaded.checksum == persisted.checksum

    rebuilt = reproduce_bundle(out, store_root=context["store_root"])

    assert isinstance(rebuilt, CandidateEvidence)
    assert rebuilt.provenance_verified is True
    assert (rebuilt.source, rebuilt.synthetic) == ("csv:load-reproduce", False)
    assert rebuilt.checksum == persisted.checksum


def test_stale_sidecar_bundle_loads_and_reproduces_as_unknown(tmp_path: Path) -> None:
    """A discarded sidecar stays unknown through load and reproduction."""
    context = stored_context(tmp_path)
    store = ParquetStore(context["store_root"])
    save_provenance(
        context["store_root"],
        build_provenance(
            context["frame"],
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="csv:stale-reproduce",
            synthetic=False,
        ),
    )
    changed = context["frame"].clone()
    closes = changed["close"].to_list()
    closes[10] = closes[10] + 1.0
    changed = changed.with_columns(pl.Series("close", closes))
    store.write(SYMBOL, TIMEFRAME, changed)
    out = tmp_path / "candidate.evidence.json"
    persisted = persist_evidence_run(
        context["store_root"],
        out,
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    assert persisted.provenance_verified is False
    assert persisted.warnings and "checksum" in persisted.warnings[0]

    loaded = load_evidence_bundle(out)
    assert (loaded.source, loaded.synthetic) == (None, None)
    assert loaded.provenance_verified is False
    assert loaded.checksum == persisted.checksum

    rebuilt = reproduce_bundle(out, store_root=context["store_root"])

    assert isinstance(rebuilt, CandidateEvidence)
    assert (rebuilt.source, rebuilt.synthetic) == (None, None)
    assert rebuilt.provenance_verified is False
    assert rebuilt.checksum == persisted.checksum


def test_a_synthetic_sidecar_is_carried_and_promotion_refuses_it(tmp_path: Path) -> None:
    context = stored_context(tmp_path)
    save_provenance(
        context["store_root"],
        build_provenance(
            context["frame"],
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="synthetic:fixture",
            synthetic=True,
        ),
    )

    result = persist_evidence_run(
        context["store_root"],
        tmp_path / "candidate.evidence.json",
        spec=long_spec(),
        config=config(),
        settings=context["settings"],
        use_cache=False,
    )

    assert result.evidence.source == "synthetic:fixture"
    assert result.evidence.synthetic is True
    assert result.evidence.provenance_verified is True

    decision = evaluate_promotion(result.evidence, PROMOTION_V1)
    assert decision.accepted is False
    assert "synthetic_data" in {rejection.code for rejection in decision.rejections}


# --- overwrite refusal and atomicity ----------------------------------------


def test_resaving_the_same_bundle_is_idempotent(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)
    manifest_path = context["artifact_dir"] / ARTIFACT_FILE
    before = manifest_path.read_bytes()

    again = save_dataset_artifact(context["store_root"], context["bundle"], context["features"])

    assert again == context["artifact_dir"]
    assert manifest_path.read_bytes() == before


def test_a_different_dataset_at_the_same_checksum_is_refused(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)

    # Write a genuinely different bundle into the target directory, as a buggy
    # or hand-made writer would: the directory is content-addressed, so the
    # manifest now records another checksum than the path.
    foreign = stored_context(tmp_path / "foreign", rows=900)
    save_dataset_arrays(context["artifact_dir"], foreign["bundle"])
    assert foreign["bundle"].manifest.checksum != context["bundle"].manifest.checksum

    with pytest.raises(PersistError) as exc:
        save_dataset_artifact(context["store_root"], context["bundle"], context["features"])

    assert exc.value.code == "artifact_conflict"
    manifest = json.loads((context["artifact_dir"] / ARTIFACT_FILE).read_text(encoding="utf-8"))
    assert manifest["manifest"]["checksum"] == foreign["bundle"].manifest.checksum


def test_a_manifest_that_lies_about_its_arrays_is_refused(tmp_path: Path) -> None:
    context = stored_context(tmp_path)
    forged = replace(
        context["bundle"],
        manifest=replace(context["bundle"].manifest, checksum="0" * 32),
    )

    with pytest.raises(PersistError) as exc:
        save_dataset_artifact(context["store_root"], forged, context["features"])

    assert exc.value.code == "dataset_checksum_mismatch"
    datasets = context["store_root"] / "datasets"
    assert not (datasets / ("0" * 32)).exists()
    assert not datasets.exists()


def test_a_failed_stage_leaves_no_temp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = stored_context(tmp_path)

    def boom(*args: Any, **kwargs: Any) -> Path:
        raise PersistError("artifact_write_failed", "simulated disk failure")

    monkeypatch.setattr(persist_module, "save_dataset_arrays", boom)

    with pytest.raises(PersistError) as exc:
        save_dataset_artifact(context["store_root"], context["bundle"], context["features"])

    assert exc.value.code == "artifact_write_failed"
    assert list((context["store_root"] / "datasets").iterdir()) == []


def test_a_successful_save_leaves_no_temp_files(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)

    entries = sorted((context["store_root"] / "datasets").iterdir())

    assert entries == [context["artifact_dir"]]
    assert not any(entry.name.startswith(".") for entry in entries)


# --- corruption ---------------------------------------------------------------


def test_a_corrupt_artifact_is_refused_by_the_loader_and_reproduce(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)
    array_path = context["artifact_dir"] / "X.npy"
    data = bytearray(array_path.read_bytes())
    data[-1] ^= 0xFF
    array_path.write_bytes(bytes(data))

    with pytest.raises(PersistError) as exc:
        load_dataset_artifact(context["store_root"], context["bundle"].manifest.checksum)
    assert exc.value.code == "artifact_corrupt"

    with pytest.raises(ReproduceError) as exc:
        reproduce_bundle(context["path"], store_root=context["store_root"])
    assert exc.value.code == "dataset_corrupt"

    with pytest.raises(PersistError) as exc:
        save_dataset_artifact(context["store_root"], context["bundle"], context["features"])
    assert exc.value.code == "artifact_corrupt"


def test_a_missing_artifact_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PersistError) as exc:
        load_dataset_artifact(tmp_path, "a" * 32)

    assert exc.value.code == "artifact_missing"


def test_a_non_hex_checksum_cannot_escape_the_store(tmp_path: Path) -> None:
    with pytest.raises(PersistError) as exc:
        load_dataset_artifact(tmp_path, "../outside")

    assert exc.value.code == "checksum_invalid"


# --- test-split discipline ----------------------------------------------------


def test_persisted_evidence_never_reads_a_test_row_metric(tmp_path: Path) -> None:
    context = persisted_bundle(tmp_path)
    evidence = context["evidence"]
    features = context["features"]

    assert evidence.splits.test_start_row >= evidence.splits.validation_end_row
    assert all(fold.test_end_row <= evidence.splits.validation_end_row for fold in evidence.folds)
    assert context["bundle"].rows("test") > 0

    # Poison every test-row price. Evidence stops at the validation end, so the
    # checksum must not move; if a fold or metric read a test row, it would.
    poisoned = FeatureFrame(
        X=np.array(features.X, copy=True),
        ts=np.array(features.ts, copy=True),
        ohlc=np.array(features.ohlc, copy=True),
        spread=np.array(features.spread, copy=True),
        atr=np.array(features.atr, copy=True),
        valid_from=features.valid_from,
        columns=list(features.columns),
    )
    source_start = features.valid_from + evidence.splits.test_start_row
    poisoned.ohlc[source_start:] = 1_000_000.0
    assert np.asarray(poisoned.ohlc)[source_start, 3] == 1_000_000.0

    rebuilt = build_evidence(
        context["bundle"],
        long_spec(),
        config(),
        features=poisoned,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert rebuilt.checksum == evidence.checksum
    assert rebuilt.folds == evidence.folds
    assert rebuilt.aggregate == evidence.aggregate


# --- CLI ----------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def cli_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_cli_writes_the_producer_output_and_reproduce_accepts_it(tmp_path: Path) -> None:
    context = stored_context(tmp_path)
    spec = tmp_path / "long.spec.json"
    spec.write_text(json.dumps(long_spec().canonical_dict()), encoding="utf-8")
    out = tmp_path / "runs" / "candidate.evidence.json"

    proc = run_cli(
        "discovery",
        "evidence",
        "--symbol",
        SYMBOL,
        "--timeframe",
        TIMEFRAME,
        "--spec",
        str(spec),
        "--out",
        str(out),
        "--root",
        str(context["store_root"]),
        "--folds",
        "3",
        "--regime-coverage-min",
        "0",
        "--bootstrap-samples",
        "64",
        "--min-trades-per-fold",
        "2",
        "--search-budget",
        "50",
        "--trials",
        "4",
        "--seed",
        "7",
        "--no-cache",
    )

    assert proc.returncode == 0, proc.stderr
    lines = cli_lines(proc)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["persisted"] is True
    assert payload["dataset_checksum"]
    assert out.is_file()
    assert out.with_suffix(".spec.json").is_file()
    assert (Path(payload["artifact"]) / ARTIFACT_FILE).is_file()

    replay = run_cli(
        "discovery",
        "reproduce",
        "--bundle",
        str(out),
        "--store-root",
        str(context["store_root"]),
    )

    assert replay.returncode == 0, replay.stderr
    report = json.loads(cli_lines(replay)[0])
    assert report["reproduced"] is True
    assert report["checksum"] == payload["evidence_checksum"]
