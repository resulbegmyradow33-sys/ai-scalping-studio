"""End-to-end acceptance on deterministic synthetic XAUUSD data.

The harness exists to answer one question with evidence instead of prose: **does
the real chain still run end to end?** It builds a tiny synthetic store, runs the
real validator, dataset builder, bounded discovery search, candidate-evidence
builder, promotion policy, an optional locked holdout when that module exists,
the autonomous paper runner and the matured-outcome reader. Every stage records
its own status, artifacts, checksums and counts.

It deliberately does not prove anything about the market. Synthetic bars are not
XAUUSD, a three-fold evidence bundle is not a locked out-of-sample result, and a
paper session over synthetic bars is not a forward paper-trading trial. The
report's ``claims`` block says so in machine-readable form:

* ``implemented`` / ``tested`` — the code path and its tests exist;
* ``validated`` / ``paper_traded`` / ``profitable`` — **false**, because this run
  cannot truthfully claim any of them.

Failures are first-class. A stage that raises aborts the run, writes a report
whose failed stage is explicit, and raises :class:`AcceptanceError`; the CLI maps
that to a non-zero exit. No metric is invented to keep a later stage alive.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.data.validate import validate_bars
from scalping.datasets.builder import DatasetBundle, build_dataset, save_manifest
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceConfig,
    EvidenceError,
    build_evidence,
)
from scalping.discovery.promotion import (
    PROMOTION_V1,
    ExpectedIdentity,
    PromotionDecision,
    PromotionPolicy,
    evaluate_promotion,
)
from scalping.discovery.run import (
    DiscoveryRunConfig,
    DiscoveryRunResult,
    run_discovery,
)
from scalping.features.build import FeatureFrame
from scalping.features.cache import FeatureCache, cached_build_features
from scalping.paper.autonomous import AutonomousPaperRunner
from scalping.paper.continuous import OutcomeBatch, matured_outcomes, outcome_fingerprint
from scalping.paper.ledger import PaperLedger
from scalping.registry.gate import GateCriteria
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

__all__ = [
    "ACCEPTANCE_FIXTURE_SOURCE",
    "DISCOVERY_CANDIDATE_SOURCE",
    "STAGE_NAMES",
    "AcceptanceConfig",
    "AcceptanceError",
    "AcceptanceResult",
    "AcceptanceStage",
    "acceptance_spec",
    "run_acceptance",
]

#: The deterministic candidate used when discovery records NO TRADE or the
#: discovered candidate cannot produce per-fold evidence. It enters long when
#: flat and lets the holding limit close the trade, so the paper path and the
#: matured-outcome reader are exercised even on a bad search sample.
ACCEPTANCE_FIXTURE_SOURCE: Final[str] = "acceptance_fixture"

#: The candidate the bounded discovery run itself selected.
DISCOVERY_CANDIDATE_SOURCE: Final[str] = "discovery_best"

#: Stable stage order; a report with a missing or reordered stage is malformed.
STAGE_NAMES: Final[tuple[str, ...]] = (
    "import_validate",
    "dataset",
    "discovery",
    "evidence",
    "promotion",
    "holdout",
    "paper_advance",
    "matured_outcomes",
)

#: Default synthetic start: a Monday, so a one-day run always produces bars.
_DEFAULT_START: Final[date] = date(2026, 1, 5)

#: Evidence errors that mean this candidate is not eligible on the tiny sample,
#: not that the harness is broken. The evidence stage falls back to the
#: deterministic acceptance fixture and records the reason.
_INELIGIBLE_EVIDENCE_CODES: Final[frozenset[str]] = frozenset(
    {
        "fold_trades_below_minimum",
        "regime_coverage_below_minimum",
        "insufficient_folds",
    }
)


class AcceptanceError(Exception):
    """A stage failed; the typed result and the failed stage travel with it."""

    def __init__(self, *, stage: str, code: str, message: str, result: AcceptanceResult) -> None:
        super().__init__(f"{stage}/{code}: {message}")
        self.stage = stage
        self.code = code
        self.result = result


class AcceptanceStageError(RuntimeError):
    """A stage knows why it refused; the runner records ``code`` verbatim."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def acceptance_spec() -> StrategySpec:
    """The deterministic long/holding-limit candidate used by the harness."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(0.0), "<", Const(1.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(5),
        max_lookback=2,
    )


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    """Inputs for one deterministic acceptance run.

    ``out`` is the directory that receives ``<acceptance-id>/report.json``. The
    synthetic store is created under the system temp directory and removed
    unless ``keep`` is true. Inputs that affect the execution are part of
    ``config_digest``; output paths are not, so the same experiment has the same
    identity wherever it is written.
    """

    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    start: date = _DEFAULT_START
    days: int = 1
    seed: int = 7
    start_price: float = 2350.0
    window: int = 20
    max_trials: int = 6
    population: int = 3
    generations: int = 2
    horizon: int = 1
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    purge_bars: int = 3
    embargo_bars: int = 2
    folds: int = 3
    min_trades_per_fold: int = 1
    cost_stress: tuple[float, ...] = (1.5,)
    bootstrap_samples: int = 500
    paper_bars: int = 120
    out: Path = Path("runs/acceptance")
    keep: bool = False

    def __post_init__(self) -> None:
        for name in ("symbol", "timeframe"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.start, date):
            raise TypeError("start must be a date")
        for name, minimum in (
            ("days", 1),
            ("window", 1),
            ("max_trials", 1),
            ("population", 2),
            ("generations", 1),
            ("horizon", 1),
            ("folds", 3),
            ("min_trades_per_fold", 1),
            ("bootstrap_samples", 500),
            ("paper_bars", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < minimum:
                raise ValueError(f"{name} must be >= {minimum}, got {value}")
        for name in ("purge_bars", "embargo_bars", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not np.isfinite(float(self.start_price)) or float(self.start_price) <= 0.0:
            raise ValueError("start_price must be a finite positive number")
        if not 0.0 < float(self.train_fraction) < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 <= float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if float(self.train_fraction) + float(self.validation_fraction) >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        multipliers: list[float] = []
        for raw in self.cost_stress:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("cost_stress multipliers must be numbers")
            number = float(raw)
            if not np.isfinite(number) or number <= 1.0:
                raise ValueError("cost_stress multipliers must be finite and > 1")
            multipliers.append(number)
        if not multipliers:
            raise ValueError("cost_stress must contain at least one multiplier")
        if len({f"{value:g}" for value in multipliers}) != len(multipliers):
            raise ValueError("cost_stress multipliers must be distinct")
        object.__setattr__(self, "cost_stress", tuple(multipliers))
        object.__setattr__(self, "out", Path(self.out))
        if not isinstance(self.keep, bool):
            raise TypeError("keep must be a bool")

    @property
    def end_day(self) -> date:
        return self.start + timedelta(days=self.days - 1)

    @property
    def start_at(self) -> datetime:
        return datetime.combine(self.start, time.min, tzinfo=UTC)

    @property
    def end_at(self) -> datetime:
        return datetime.combine(self.end_day, time(23, 59), tzinfo=UTC)

    def identity_payload(self) -> dict[str, Any]:
        """Everything that changes execution, excluding output/retention paths."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start": self.start.isoformat(),
            "days": self.days,
            "seed": self.seed,
            "start_price": float(self.start_price),
            "window": self.window,
            "max_trials": self.max_trials,
            "population": self.population,
            "generations": self.generations,
            "horizon": self.horizon,
            "train_fraction": float(self.train_fraction),
            "validation_fraction": float(self.validation_fraction),
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "folds": self.folds,
            "min_trades_per_fold": self.min_trades_per_fold,
            "cost_stress": list(self.cost_stress),
            "bootstrap_samples": self.bootstrap_samples,
            "paper_bars": self.paper_bars,
        }

    @property
    def config_digest(self) -> str:
        return _digest_json(self.identity_payload())

    @property
    def acceptance_id(self) -> str:
        return f"acc-{self.config_digest[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "out": str(self.out),
            "keep": self.keep,
            "config_digest": self.config_digest,
            "acceptance_id": self.acceptance_id,
        }


@dataclass(frozen=True, slots=True)
class AcceptanceStage:
    """One stage's outcome and the evidence it left behind."""

    name: str
    status: str
    detail: str
    artifacts: Mapping[str, str]
    checksums: Mapping[str, str]
    counts: Mapping[str, Any]
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "error_code": self.error_code,
            "artifacts": dict(self.artifacts),
            "checksums": dict(self.checksums),
            "counts": dict(self.counts),
        }


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """The whole run: stage statuses, artifacts, claims and where it was written."""

    acceptance_id: str
    status: str
    synthetic: bool
    created_at: str
    config: Mapping[str, Any]
    config_digest: str
    report_path: str
    store_path: str
    store_retained: bool
    stages: tuple[AcceptanceStage, ...]
    claims: Mapping[str, bool]
    claim_basis: Mapping[str, str]
    scope: Mapping[str, bool]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_id": self.acceptance_id,
            "status": self.status,
            "synthetic": self.synthetic,
            "created_at": self.created_at,
            "config": dict(self.config),
            "config_digest": self.config_digest,
            "report_path": self.report_path,
            "store_path": self.store_path,
            "store_retained": self.store_retained,
            "stages": [stage.to_dict() for stage in self.stages],
            "claims": dict(self.claims),
            "claim_basis": dict(self.claim_basis),
            "scope": dict(self.scope),
            "notes": list(self.notes),
        }

    def cli_dict(self) -> dict[str, Any]:
        """One-line CLI projection; the full report stays on disk."""
        return {
            "status": self.status,
            "acceptance_id": self.acceptance_id,
            "report": self.report_path,
            "synthetic": self.synthetic,
            "claims": dict(self.claims),
            "stages": [
                {
                    "name": stage.name,
                    "status": stage.status,
                    "error_code": stage.error_code,
                }
                for stage in self.stages
            ],
            "store_path": self.store_path,
            "store_retained": self.store_retained,
        }


class _AcceptanceRun:
    """Mutable state for one run; stages run in the fixed ``STAGE_NAMES`` order."""

    def __init__(self, config: AcceptanceConfig) -> None:
        self.config = config
        self.run_dir = config.out / config.acceptance_id
        self.report_path = self.run_dir / "report.json"
        self.store_dir = Path(tempfile.mkdtemp(prefix="scalping-acceptance-"))
        self.settings = Settings()
        self.features: FeatureFrame | None = None
        self.bundle: DatasetBundle | None = None
        self.discovery: DiscoveryRunResult | None = None
        self.spec: StrategySpec | None = None
        self.evidence: CandidateEvidence | None = None
        self.candidate_source: str | None = None
        self.identity: ExpectedIdentity | None = None
        self.pre_decision: PromotionDecision | None = None
        self.acceptance_policy: PromotionPolicy | None = None
        self.paper_session_id: str | None = None
        self.paper_ledger_path = self.run_dir / "paper.sqlite"
        self.outcomes: OutcomeBatch | None = None

    # ------------------------------------------------------------------ driver
    def execute(self) -> AcceptanceResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stages: list[AcceptanceStage] = []
        for name, method in (
            ("import_validate", self._stage_import_validate),
            ("dataset", self._stage_dataset),
            ("discovery", self._stage_discovery),
            ("evidence", self._stage_evidence),
            ("promotion", self._stage_promotion),
            ("holdout", self._stage_holdout),
            ("paper_advance", self._stage_paper),
            ("matured_outcomes", self._stage_outcomes),
        ):
            try:
                stage = method()
            except Exception as exc:
                code = getattr(exc, "code", type(exc).__name__)
                stage = AcceptanceStage(
                    name=name,
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    artifacts={},
                    checksums={},
                    counts={},
                    error_code=str(code),
                )
                stages.append(stage)
                result = self._result("failed", stages)
                self._write_report(result)
                raise AcceptanceError(
                    stage=name,
                    code=str(code),
                    message=str(exc),
                    result=result,
                ) from exc
            stages.append(stage)
        result = self._result("passed", stages)
        self._write_report(result)
        return result

    # ------------------------------------------------------------------- stages
    def _stage_import_validate(self) -> AcceptanceStage:
        provider = SyntheticProvider(seed=self.config.seed, start_price=self.config.start_price)
        bars = provider.fetch(self.config.symbol, self.config.start_at, self.config.end_at)
        if bars.is_empty():
            raise AcceptanceStageError(
                "empty_synthetic_store",
                (
                    f"the {self.config.start}..{self.config.end_day} synthetic range produced "
                    "no weekday bars; pick a range that contains a weekday"
                ),
            )
        store = ParquetStore(self.store_dir)
        added = store.write(self.config.symbol, self.config.timeframe, bars)
        stored = store.read(self.config.symbol, self.config.timeframe)
        report = validate_bars(stored, symbol=self.config.symbol, timeframe=self.config.timeframe)
        if not report.ok:
            codes = ", ".join(issue.code for issue in report.issues if issue.severity == "error")
            raise AcceptanceStageError("invalid_bars", f"stored synthetic bars failed: {codes}")
        coverage = store.coverage(self.config.symbol, self.config.timeframe)
        self._write_json(
            self.run_dir / "validation.json",
            {
                "rows": report.rows,
                "ok": report.ok,
                "issues": [issue.to_dict() for issue in report.issues],
            },
        )
        return AcceptanceStage(
            name="import_validate",
            status="passed",
            detail="deterministic synthetic bars written and validated by the real validator",
            artifacts={
                "store": str(self.store_dir),
                "validation": str(self.run_dir / "validation.json"),
            },
            checksums={"store_tree": _tree_digest(self.store_dir)},
            counts={
                "rows": coverage.rows,
                "added": added,
                "first_ts": None if coverage.first_ts is None else coverage.first_ts.isoformat(),
                "last_ts": None if coverage.last_ts is None else coverage.last_ts.isoformat(),
                "issues": len(report.issues),
            },
        )

    def _stage_dataset(self) -> AcceptanceStage:
        bars = ParquetStore(self.store_dir).read(self.config.symbol, self.config.timeframe)
        cache = FeatureCache(self.store_dir / ".feature-cache")
        features = cached_build_features(
            cache,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            frame=bars,
            settings=self.settings,
        )
        bundle = build_dataset(
            features,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            horizon=self.config.horizon,
            train_fraction=self.config.train_fraction,
            validation_fraction=self.config.validation_fraction,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )
        manifest_path = save_manifest(self.run_dir / "dataset.json", bundle)
        self.features = features
        self.bundle = bundle
        return AcceptanceStage(
            name="dataset",
            status="passed",
            detail="chronological purged dataset built from the stored synthetic bars",
            artifacts={"manifest": str(manifest_path)},
            checksums={
                "dataset": bundle.manifest.checksum,
                "manifest_file": _file_digest(manifest_path),
            },
            counts={
                "rows": bundle.manifest.rows,
                "train": bundle.rows("train"),
                "validation": bundle.rows("validation"),
                "test": bundle.rows("test"),
                "purged": int((bundle.split == -1).sum()),
                "features": int(bundle.X.shape[1]),
                "dataset_id": bundle.manifest.dataset_id,
            },
        )

    def _stage_discovery(self) -> AcceptanceStage:
        archive = self.run_dir / "discovery.sqlite"
        config = DiscoveryRunConfig(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            max_trials=self.config.max_trials,
            population=self.config.population,
            generations=self.config.generations,
            seed=self.config.seed,
            archive=archive,
            use_cache=True,
            stress=self.config.cost_stress,
            gate=GateCriteria(min_trades=self.config.min_trades_per_fold),
            baselines=("hold", "ret_5"),
            horizon=self.config.horizon,
            train_fraction=self.config.train_fraction,
            validation_fraction=self.config.validation_fraction,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            window=self.config.window,
            max_lookback=10,
            max_depth=2,
        )
        result = run_discovery(config, root=self.store_dir)
        self.discovery = result
        result_path = self.run_dir / "discovery.json"
        self._write_json(result_path, result.to_dict())
        _checkpoint_sqlite(archive)
        return AcceptanceStage(
            name="discovery",
            status="passed",
            detail=(
                "bounded evolutionary search completed; "
                f"decision={result.decision!r} ({result.reason})"
            ),
            artifacts={"result": str(result_path), "archive": str(archive)},
            checksums={
                "result": _file_digest(result_path),
                "archive": _file_digest(archive),
                "code_version": result.code_version,
            },
            counts={
                "trials": result.trials,
                "accepted": result.accepted,
                "skipped": result.skipped,
                "deduped": result.deduped,
                "decision": result.decision,
                "best_candidate": result.best_id,
                "dataset_id": result.dataset_id,
            },
        )

    def _stage_evidence(self) -> AcceptanceStage:
        self._require_dataset()
        if self.discovery is None:
            raise AcceptanceStageError("missing_discovery", "evidence needs the discovery stage")
        fallback_reason: str | None = None
        if self.discovery.best_canonical is not None:
            spec = StrategySpec.from_dict(self.discovery.best_canonical)
            try:
                evidence = self._build_evidence(spec)
                source = DISCOVERY_CANDIDATE_SOURCE
            except EvidenceError as exc:
                if exc.code not in _INELIGIBLE_EVIDENCE_CODES:
                    raise
                fallback_reason = f"{exc.code}: {exc}"
                spec = acceptance_spec()
                evidence = self._build_evidence(spec)
                source = ACCEPTANCE_FIXTURE_SOURCE
        else:
            spec = acceptance_spec()
            evidence = self._build_evidence(spec)
            source = ACCEPTANCE_FIXTURE_SOURCE
        self.spec = spec
        self.evidence = evidence
        self.candidate_source = source
        path = self.run_dir / "evidence.json"
        self._write_json(path, evidence.to_dict())
        detail = (
            "candidate evidence built on train+validation folds only"
            if source == DISCOVERY_CANDIDATE_SOURCE
            else (
                "discovery produced no evidence-eligible candidate; "
                "evidence built for the deterministic acceptance fixture"
            )
        )
        if fallback_reason is not None:
            detail = f"{detail} ({fallback_reason})"
        return AcceptanceStage(
            name="evidence",
            status="passed",
            detail=detail,
            artifacts={"evidence": str(path)},
            checksums={
                "evidence": evidence.checksum,
                "evidence_file": _file_digest(path),
                "spec": evidence.spec_hash,
            },
            counts={
                "candidate_source": source,
                "fold_count": len(evidence.folds),
                "trades": evidence.aggregate.total_trades,
                "expectancy_r": evidence.aggregate.expectancy_r,
                "complexity": evidence.complexity,
                "dataset_id": evidence.dataset_id,
            },
        )

    def _stage_promotion(self) -> AcceptanceStage:
        if self.evidence is None or self.bundle is None:
            raise AcceptanceStageError("missing_evidence", "promotion needs an evidence bundle")
        if self.discovery is None:
            raise AcceptanceStageError("missing_discovery", "promotion needs a code version")
        policy = _acceptance_policy(require_locked_oos=False, max_trials=self.config.max_trials)
        identity = ExpectedIdentity.current(
            dataset_checksum=self.bundle.manifest.checksum,
            settings=self.settings,
            code_version=self.discovery.code_version,
        )
        decision = evaluate_promotion(self.evidence, policy, expected_identity=identity)
        self.acceptance_policy = policy
        self.identity = identity
        self.pre_decision = decision
        path = self.run_dir / "promotion.json"
        self._write_json(
            path,
            {
                "policy": {
                    "id": policy.policy_id,
                    "version": policy.version,
                    "digest": policy.digest,
                    "require_locked_oos": policy.require_locked_oos,
                },
                "decision": decision.to_dict(),
                "expected_identity": {
                    "code_version": identity.code_version,
                    "dataset_checksum": identity.dataset_checksum,
                    "settings_digest": identity.settings_digest,
                },
            },
        )
        return AcceptanceStage(
            name="promotion",
            status="passed",
            detail=(
                f"explicit acceptance test policy evaluated: accepted={decision.accepted} "
                f"(policy={policy.policy_id})"
            ),
            artifacts={"decision": str(path)},
            checksums={
                "policy_digest": policy.digest,
                "decision": _decision_digest(decision),
                "evidence": self.evidence.checksum,
            },
            counts={
                "accepted": decision.accepted,
                "rejections": list(decision.codes),
                "policy_id": policy.policy_id,
                "policy_version": policy.version,
                "trial_count": self.evidence.trial_count,
                "search_budget": self.evidence.search_budget,
            },
        )

    def _stage_holdout(self) -> AcceptanceStage:
        module = _holdout_module()
        if module is None:
            return AcceptanceStage(
                name="holdout",
                status="skipped",
                detail="scalping.discovery.holdout is not installed in this tree",
                artifacts={},
                checksums={},
                counts={"attached": False},
            )
        if self.pre_decision is None or not self.pre_decision.accepted:
            return AcceptanceStage(
                name="holdout",
                status="skipped",
                detail="the pre-holdout acceptance policy rejected; the holdout was not spent",
                artifacts={},
                checksums={},
                counts={"attached": False, "spent": False},
            )
        if self.evidence is None or self.spec is None or self.bundle is None:
            raise AcceptanceStageError("missing_evidence", "holdout needs candidate evidence")
        if self.identity is None:
            raise AcceptanceStageError("missing_identity", "holdout needs the expected identity")

        ledger_path = self.run_dir / "holdout.sqlite"
        ledger = module.HoldoutLedger(str(ledger_path))
        try:
            holdout_spec = module.HoldoutSpec(
                dataset_id=self.bundle.manifest.dataset_id,
                dataset_checksum=self.bundle.manifest.checksum,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                spec_hash=self.evidence.spec_hash,
                code_version=self.evidence.code_version,
                settings_digest=self.evidence.settings_digest,
            )
            ledger.seal(holdout_spec, actor="acceptance", reason="synthetic acceptance run")
            token = ledger.unlock(
                holdout_spec,
                decision=self.pre_decision,
                policy=PROMOTION_V1,
                actor="acceptance",
                reason="pre-holdout acceptance decision passed",
            )
            holdout_config = module.HoldoutConfig(
                code_version=self.evidence.code_version,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                window=self.config.window,
                cost_stress=self.config.cost_stress,
                bootstrap_samples=self.config.bootstrap_samples,
                bootstrap_confidence=0.95,
                seed=self.config.seed,
                settings=self.settings,
                features=self.features,
                dataset_id=self.bundle.manifest.dataset_id,
                dataset_checksum=self.bundle.manifest.checksum,
            )
            result = module.evaluate_holdout(
                self.bundle,
                self.spec,
                holdout_config,
                ledger=ledger,
                unlock_token=token,
            )
            attached = module.attach_holdout(self.evidence, result)
            production = evaluate_promotion(attached, PROMOTION_V1, expected_identity=self.identity)
        finally:
            ledger.close()
        _checkpoint_sqlite(ledger_path)
        path = self.run_dir / "holdout.json"
        self._write_json(
            path,
            {
                "result": result.to_dict(),
                "production_decision": production.to_dict(),
            },
        )
        return AcceptanceStage(
            name="holdout",
            status="passed",
            detail="the optional locked holdout was attached and production policy evaluated",
            artifacts={"result": str(path), "ledger": str(ledger_path)},
            checksums={
                "result": _digest_json(result.to_dict()),
                "ledger": _file_digest(ledger_path),
                "evidence": attached.checksum,
                "production_decision": _decision_digest(production),
            },
            counts={
                "attached": True,
                "spent": True,
                "trades": getattr(result, "trades", None),
                "production_accepted": production.accepted,
                "production_rejections": list(production.codes),
            },
        )

    def _stage_paper(self) -> AcceptanceStage:
        if self.spec is None:
            raise AcceptanceStageError("missing_spec", "paper advance needs a strategy spec")
        session_id: str | None = None
        with PaperLedger(self.paper_ledger_path) as ledger:
            runner = AutonomousPaperRunner(
                ledger,
                self.spec,
                self.config.symbol,
                self.config.timeframe,
                self.settings,
                window=self.config.window,
                cache=FeatureCache(self.store_dir / ".feature-cache"),
                store=ParquetStore(self.store_dir),
                started_at=self.config.start_at,
                seed=self.config.seed,
            )
            try:
                run_result = runner.run(bars=self.config.paper_bars)
            finally:
                runner.close()
            summary = runner.summary()
            session_id = runner.session_id
        self.paper_session_id = session_id
        _checkpoint_sqlite(self.paper_ledger_path)
        path = self.run_dir / "paper.json"
        self._write_json(
            path,
            {
                "session_id": session_id,
                "steps": run_result.steps,
                "balance": run_result.balance,
                "equity": run_result.equity,
                "last_bar_ts": None
                if run_result.last_bar_ts is None
                else run_result.last_bar_ts.isoformat(),
                "done": run_result.done,
                "reason": run_result.reason,
                "summary": {
                    "status": summary.status,
                    "bars_processed": summary.bars_processed,
                    "trades": summary.trades,
                    "realized_pnl": summary.realized_pnl,
                },
            },
        )
        return AcceptanceStage(
            name="paper_advance",
            status="passed",
            detail="autonomous DSL paper runner advanced on the synthetic store (no real money)",
            artifacts={"ledger": str(self.paper_ledger_path), "summary": str(path)},
            checksums={"ledger": _file_digest(self.paper_ledger_path)},
            counts={
                "session_id": session_id,
                "bars_processed": summary.bars_processed,
                "trades": summary.trades,
                "status": summary.status,
                "real_money": False,
            },
        )

    def _stage_outcomes(self) -> AcceptanceStage:
        if self.paper_session_id is None:
            raise AcceptanceStageError("missing_session", "outcomes need a paper session")
        with PaperLedger.open_reader(self.paper_ledger_path) as ledger:
            batch = matured_outcomes(ledger, self.paper_session_id)
        self.outcomes = batch
        path = self.run_dir / "outcomes.json"
        self._write_json(path, batch.to_dict())
        fingerprint = outcome_fingerprint(batch)
        return AcceptanceStage(
            name="matured_outcomes",
            status="passed",
            detail="only closed paper trades were read; open positions are counted, not claimed",
            artifacts={"outcomes": str(path)},
            checksums={"fingerprint": fingerprint, "outcomes_file": _file_digest(path)},
            counts={
                "closed_trades": batch.closed_trades,
                "open_trades": batch.open_trades,
                "strategy_hash": batch.strategy_hash,
                "dataset_id": batch.dataset_id,
            },
        )

    # ------------------------------------------------------------------ helpers
    def _require_dataset(self) -> tuple[DatasetBundle, FeatureFrame]:
        if self.bundle is None or self.features is None:
            raise AcceptanceStageError("missing_dataset", "this stage needs the built dataset")
        return self.bundle, self.features

    def _build_evidence(self, spec: StrategySpec) -> CandidateEvidence:
        bundle, features = self._require_dataset()
        if self.discovery is None:
            raise AcceptanceStageError("missing_discovery", "evidence needs a code version")
        config = EvidenceConfig(
            code_version=self.discovery.code_version,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            folds=self.config.folds,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
            min_trades_per_fold=self.config.min_trades_per_fold,
            cost_stress=self.config.cost_stress,
            regime_coverage_min=0.0,
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_confidence=0.95,
            seed=self.config.seed,
            search_budget=self.config.max_trials,
            trial_count=max(1, self.discovery.trials),
            window=self.config.window,
        )
        return build_evidence(bundle, spec, config, features=features, settings=self.settings)

    def _result(self, status: str, stages: Sequence[AcceptanceStage]) -> AcceptanceResult:
        claims = {
            "implemented": True,
            "tested": True,
            "validated": False,
            "paper_traded": False,
            "profitable": False,
        }
        claim_basis = {
            "implemented": "the real store/dataset/discovery/evidence/promotion/paper/outcome "
            "modules ran in this process",
            "tested": "backend/tests/test_acceptance.py exercises a passing run, a typed stage "
            "failure and the CLI contract",
            "validated": "no independent chronological out-of-sample market evidence was produced",
            "paper_traded": "the paper engine ran on synthetic bars, not a forward market trial",
            "profitable": "synthetic acceptance cannot establish profitability",
        }
        scope = {
            "synthetic_data": True,
            "out_of_sample": False,
            "market_evidence": False,
            "real_money": False,
        }
        notes = (
            (
                "Synthetic acceptance only: this report proves the chain executes, not that a "
                "strategy works."
            ),
            (
                "The evidence bundle excludes the test split; without a locked holdout this is "
                "not out-of-sample validation."
            ),
            (
                "Paper rows come from synthetic bars and a local SQLite ledger; no broker, network "
                "or real money is involved."
            ),
            (
                "NO TRADE is a valid discovery and promotion outcome; a rejected decision is "
                "reported, not hidden."
            ),
        )
        return AcceptanceResult(
            acceptance_id=self.config.acceptance_id,
            status=status,
            synthetic=True,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            config=self.config.to_dict(),
            config_digest=self.config.config_digest,
            report_path=str(self.report_path),
            store_path=str(self.store_dir),
            store_retained=self.config.keep,
            stages=tuple(stages),
            claims=claims,
            claim_basis=claim_basis,
            scope=scope,
            notes=notes,
        )

    def _write_report(self, result: AcceptanceResult) -> None:
        self._write_json(self.report_path, result.to_dict())

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def run_acceptance(config: AcceptanceConfig | None = None) -> AcceptanceResult:
    """Run every stage in order; any stage failure is a typed, non-zero result.

    The temporary synthetic store is removed on exit unless ``config.keep`` is
    true. Reports and the small artifacts (manifest, evidence, decisions, paper
    ledger and outcomes) always stay under ``config.out/<acceptance-id>/``.
    """
    cfg = AcceptanceConfig() if config is None else config
    if not isinstance(cfg, AcceptanceConfig):
        raise TypeError(f"config must be an AcceptanceConfig, got {type(cfg).__name__}")
    runner = _AcceptanceRun(cfg)
    try:
        return runner.execute()
    finally:
        if not cfg.keep:
            shutil.rmtree(runner.store_dir, ignore_errors=True)


# --- shared helpers -----------------------------------------------------------


def _acceptance_policy(*, require_locked_oos: bool, max_trials: int) -> PromotionPolicy:
    """A named, non-production policy with every field explicit.

    It is a test policy, not a relaxed production gate: the fields below are
    thresholds for exercising the evaluator on a tiny synthetic sample. The
    production ``PROMOTION_V1`` is used only after a real locked holdout is
    attached.
    """
    return replace(
        PROMOTION_V1,
        policy_id="acceptance-test-policy",
        version=1,
        min_folds=3,
        min_trades=1,
        min_expectancy_r=0.0,
        min_profit_factor=1.01,
        max_drawdown=1.0,
        min_positive_fold_fraction=1.0 / 3.0,
        min_regime_coverage=1.0 / 3.0,
        min_ci_lower_r=-10.0,
        require_cost_stress=(1.5,),
        max_complexity=200,
        max_trials=max(1, int(max_trials)),
        min_improvement_r=0.01,
        min_bootstrap_confidence=0.95,
        min_bootstrap_samples=500,
        max_stress_degradation=10.0,
        require_locked_oos=require_locked_oos,
        max_selection_fold_share=0.0,
    )


def _holdout_module() -> Any | None:
    """Import ``scalping.discovery.holdout`` only when it exists and is compatible."""
    if importlib.util.find_spec("scalping.discovery.holdout") is None:
        return None
    module = importlib.import_module("scalping.discovery.holdout")
    required = (
        "HoldoutLedger",
        "HoldoutSpec",
        "HoldoutConfig",
        "evaluate_holdout",
        "attach_holdout",
    )
    if not all(hasattr(module, name) for name in required):
        return None
    return module


def _json_default(value: object) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _digest_json(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decision_digest(decision: PromotionDecision) -> str:
    """Deterministic identity of a decision; the wall-clock stamp is excluded."""
    payload = decision.to_dict()
    payload.pop("created_at", None)
    return _digest_json(payload)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_sqlite(path: Path) -> None:
    """Merge a WAL into the main file before hashing it as an artifact."""
    if not path.exists():
        return
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
