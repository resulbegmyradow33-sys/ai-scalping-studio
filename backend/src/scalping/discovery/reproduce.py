"""Re-run a stored discovery evidence bundle from its identity and compare.

Master prompt §36.3 asks the system to reproduce a candidate experiment from
its saved data hash, code version, configuration and random seed. This module
is that replay: it reads one :class:`~scalping.discovery.evidence.CandidateEvidence`
JSON file, rebuilds the dataset from the stored bars, rebuilds the evidence with
the stored policy and returns either an identical bundle or a typed
:class:`ReproduceMismatch` naming the sections that drifted.

What a bundle must carry
------------------------
The evidence payload stores the candidate **identity** (``spec_hash``,
``complexity``, ``spec_version``) but not the strategy itself, because a hash
cannot be inverted. Reproduction therefore needs the spec as input:

* ``spec`` argument: a :class:`~scalping.strategy.dsl.nodes.StrategySpec`, its
  canonical dict, a path to that JSON, or the JSON text itself; or
* a sidecar file next to the bundle, at :func:`spec_sidecar_path`
  (``candidate.evidence.json`` -> ``candidate.evidence.spec.json``).

Either way the loaded spec must hash to the stored ``spec_hash``, so the
sidecar is a convenience, not a trusted shortcut.

Dataset layout
--------------
``store_root`` is the same root every other command uses:

* ``<store_root>/<symbol>/<timeframe>/<yyyy>/<mm>.parquet`` — the bar store
  (:class:`~scalping.data.store.ParquetStore`);
* ``<store_root>/datasets/<dataset_checksum>/artifact.json`` — the dataset
  artifact written by
  :func:`scalping.datasets.artifact.save_dataset_arrays`, looked up by the
  checksum the bundle recorded (see :func:`dataset_artifact_dir`).

The artifact is the stored manifest and the tamper check. The arrays that enter
the evidence are **rebuilt from the bars**, not copied from the artifact, and
their checksum must equal the stored one. Changing the bars therefore fails
before an evidence metric is compared.

The stored dataset checksum covers every array row, including the test split.
Reproduction rebuilds those rows too, because that is what proves the dataset
identity, but it never slices or scores them: evidence is built from the
discovery timeline only, and the test end is checked to stay outside every
fold.

What reproduction refuses
-------------------------
Identity differences are refusals with a stable ``code``, not mismatches:

``settings_mismatch``
    The supplied settings do not canonicalise to the stored ``settings_digest``.
``code_version_mismatch``
    The running code version differs from the stored one. The version digest
    covers the modules that define the arithmetic, so a different tree is a
    different experiment, not a replay.
``dataset_missing`` / ``dataset_corrupt``
    No artifact for the stored checksum, or its files no longer verify.
``dataset_checksum_mismatch`` / ``dataset_mismatch``
    The stored bars no longer rebuild the stored dataset, or the artifact's
    manifest describes a different dataset id, symbol or timeframe.
``spec_mismatch`` and friends
    The supplied spec is not the candidate the bundle measured.

A rebuild that succeeds but produces a different evidence checksum returns
:class:`ReproduceMismatch` with the first differing leaf paths. The original
file is never rewritten: the CLI's ``--out`` is a separate report path.

Limits
------
Reproduction is deterministic only inside the same environment: the same bars,
the same dataset artifact, the same code version digest, the same settings, the
same seed and the same libraries. Floating point arithmetic and BLAS threading
are not pinned here, so a different machine, NumPy build or CPU can move a
sixteenth decimal and change a checksum. That is a reported mismatch, not a
silent pass. The final test split is never sliced, no gate is run and no
promotion is recorded; the module has no network path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.artifact import ARTIFACT_FILE, ArtifactError, load_dataset_arrays
from scalping.datasets.builder import SPLIT_NAMES, DatasetBundle, DatasetManifest, SplitRange
from scalping.discovery.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    AggregateEvidence,
    CandidateEvidence,
    EvidenceConfig,
    EvidenceError,
    EvidencePolicy,
    FoldEvidence,
    RegimeEvidence,
    SplitBoundaries,
    StressEvidence,
    build_evidence,
    settings_digest,
    verify_evidence,
)
from scalping.discovery.run import code_version as current_code_version
from scalping.features.build import FeatureFrame, build_features
from scalping.features.registry import columns_for, indices_for
from scalping.strategy.dsl.nodes import DslValidationError, StrategySpec

__all__ = [
    "DATASET_ARTIFACT_DIRNAME",
    "ReproduceError",
    "ReproduceMismatch",
    "SectionDifference",
    "dataset_artifact_dir",
    "evidence_from_dict",
    "reproduce_bundle",
    "spec_sidecar_path",
]

#: Subdirectory of ``store_root`` that holds one artifact per dataset checksum.
DATASET_ARTIFACT_DIRNAME: Final[str] = "datasets"

#: Suffix appended to the bundle path to find the sidecar spec when none is given.
_SPEC_SIDECAR_SUFFIX: Final[str] = ".spec.json"

#: Fields outside the checksum; they cannot make a replay differ.
_UNCHECKED_FIELDS: Final[frozenset[str]] = frozenset({"checksum", "created_at"})

#: A checksum is a hex digest, and it is used as a directory name.
_CHECKSUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{16,128}")

_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "spec_hash",
    "spec_version",
    "complexity",
    "dataset_id",
    "dataset_checksum",
    "symbol",
    "timeframe",
    "code_version",
    "settings_digest",
    "seed",
    "trial_count",
    "search_budget",
    "splits",
    "policy",
    "folds",
    "aggregate",
    "checksum",
    "created_at",
    "schema_version",
)
_POLICY_KEYS: Final[tuple[str, ...]] = (
    "folds",
    "purge_bars",
    "embargo_bars",
    "min_trades_per_fold",
    "cost_stress",
    "regime_coverage_min",
    "bootstrap_samples",
    "bootstrap_confidence",
    "window",
)
_SPLIT_KEYS: Final[tuple[str, ...]] = (
    "train_start_row",
    "train_end_row",
    "validation_start_row",
    "validation_end_row",
    "test_start_row",
    "test_end_row",
)
_FOLD_KEYS: Final[tuple[str, ...]] = (
    "fold",
    "train_start_row",
    "train_end_row",
    "test_start_row",
    "test_end_row",
    "train_rows",
    "test_rows",
    "train_start_ts",
    "train_end_ts",
    "test_start_ts",
    "test_end_ts",
    "purge_bars",
    "embargo_bars",
    "trades",
    "net_profit",
    "profit_factor",
    "win_rate",
    "expectancy_r",
    "max_drawdown",
    "regimes",
    "stress",
)
_REGIME_KEYS: Final[tuple[str, ...]] = (
    "regime",
    "trades",
    "net_profit",
    "win_rate",
    "expectancy_r",
    "total_r",
)
_STRESS_KEYS: Final[tuple[str, ...]] = (
    "multiplier",
    "trades",
    "net_profit",
    "profit_factor",
    "max_drawdown",
    "expectancy_r",
)
_AGGREGATE_KEYS: Final[tuple[str, ...]] = (
    "total_trades",
    "total_r",
    "expectancy_r",
    "profit_factor",
    "max_drawdown",
    "positive_folds",
    "positive_fold_fraction",
    "expectancy_r_ci95_low",
    "expectancy_r_ci95_high",
    "total_r_ci95_low",
    "total_r_ci95_high",
    "bootstrap_samples",
    "bootstrap_confidence",
    "bootstrap_method",
    "regime_coverage",
    "regime_trades",
)
_MANIFEST_SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")


class ReproduceError(ValueError):
    """A bundle cannot be reproduced; ``code`` is the machine-readable reason.

    ``invalid`` is true when the caller or the files are unusable (missing
    bundle, malformed JSON, absent dataset) and false when the stored identity
    is valid but does not match this environment (settings, code version, data
    checksum). The CLI maps the two to exit codes 2 and 1.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        invalid: bool = True,
        expected: str | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.invalid = invalid
        self.expected = expected


# --- JSON helpers -------------------------------------------------------------


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError("invalid_payload", f"{where} must be a JSON object")
    return value


def _only(data: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    extra = sorted(set(data) - set(keys), key=repr)
    missing = [key for key in keys if key not in data]
    if extra or missing:
        raise EvidenceError(
            "invalid_payload", f"{where} fields mismatch (unknown={extra}, missing={missing})"
        )


def _sequence(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise EvidenceError("invalid_payload", f"{where} must be a JSON array")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("invalid_payload", f"{where} must be a non-empty string")
    return value


def _count(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError("invalid_payload", f"{where} must be an integer")
    return value


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError("invalid_payload", f"{where} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError("invalid_payload", f"{where} must be finite")
    return number


# --- payload -> CandidateEvidence --------------------------------------------


def _policy(payload: object, where: str) -> EvidencePolicy:
    data = _mapping(payload, where)
    _only(data, _POLICY_KEYS, where)
    return EvidencePolicy(
        folds=_count(data["folds"], f"{where}.folds"),
        purge_bars=_count(data["purge_bars"], f"{where}.purge_bars"),
        embargo_bars=_count(data["embargo_bars"], f"{where}.embargo_bars"),
        min_trades_per_fold=_count(data["min_trades_per_fold"], f"{where}.min_trades_per_fold"),
        cost_stress=tuple(
            _number(value, f"{where}.cost_stress[{index}]")
            for index, value in enumerate(_sequence(data["cost_stress"], f"{where}.cost_stress"))
        ),
        regime_coverage_min=_number(data["regime_coverage_min"], f"{where}.regime_coverage_min"),
        bootstrap_samples=_count(data["bootstrap_samples"], f"{where}.bootstrap_samples"),
        bootstrap_confidence=_number(data["bootstrap_confidence"], f"{where}.bootstrap_confidence"),
        window=_count(data["window"], f"{where}.window"),
    )


def _splits(payload: object, where: str) -> SplitBoundaries:
    data = _mapping(payload, where)
    _only(data, _SPLIT_KEYS, where)
    return SplitBoundaries(
        train_start_row=_count(data["train_start_row"], f"{where}.train_start_row"),
        train_end_row=_count(data["train_end_row"], f"{where}.train_end_row"),
        validation_start_row=_count(data["validation_start_row"], f"{where}.validation_start_row"),
        validation_end_row=_count(data["validation_end_row"], f"{where}.validation_end_row"),
        test_start_row=_count(data["test_start_row"], f"{where}.test_start_row"),
        test_end_row=_count(data["test_end_row"], f"{where}.test_end_row"),
    )


def _regime(payload: object, where: str) -> RegimeEvidence:
    data = _mapping(payload, where)
    _only(data, _REGIME_KEYS, where)
    return RegimeEvidence(
        regime=_text(data["regime"], f"{where}.regime"),
        trades=_count(data["trades"], f"{where}.trades"),
        net_profit=_number(data["net_profit"], f"{where}.net_profit"),
        win_rate=_number(data["win_rate"], f"{where}.win_rate"),
        expectancy_r=_number(data["expectancy_r"], f"{where}.expectancy_r"),
        total_r=_number(data["total_r"], f"{where}.total_r"),
    )


def _stress(payload: object, where: str) -> StressEvidence:
    data = _mapping(payload, where)
    _only(data, _STRESS_KEYS, where)
    return StressEvidence(
        multiplier=_number(data["multiplier"], f"{where}.multiplier"),
        trades=_count(data["trades"], f"{where}.trades"),
        net_profit=_number(data["net_profit"], f"{where}.net_profit"),
        profit_factor=_number(data["profit_factor"], f"{where}.profit_factor"),
        max_drawdown=_number(data["max_drawdown"], f"{where}.max_drawdown"),
        expectancy_r=_number(data["expectancy_r"], f"{where}.expectancy_r"),
    )


def _fold(payload: object, where: str) -> FoldEvidence:
    data = _mapping(payload, where)
    _only(data, _FOLD_KEYS, where)
    return FoldEvidence(
        fold=_count(data["fold"], f"{where}.fold"),
        train_start_row=_count(data["train_start_row"], f"{where}.train_start_row"),
        train_end_row=_count(data["train_end_row"], f"{where}.train_end_row"),
        test_start_row=_count(data["test_start_row"], f"{where}.test_start_row"),
        test_end_row=_count(data["test_end_row"], f"{where}.test_end_row"),
        train_rows=_count(data["train_rows"], f"{where}.train_rows"),
        test_rows=_count(data["test_rows"], f"{where}.test_rows"),
        train_start_ts=_text(data["train_start_ts"], f"{where}.train_start_ts"),
        train_end_ts=_text(data["train_end_ts"], f"{where}.train_end_ts"),
        test_start_ts=_text(data["test_start_ts"], f"{where}.test_start_ts"),
        test_end_ts=_text(data["test_end_ts"], f"{where}.test_end_ts"),
        purge_bars=_count(data["purge_bars"], f"{where}.purge_bars"),
        embargo_bars=_count(data["embargo_bars"], f"{where}.embargo_bars"),
        trades=_count(data["trades"], f"{where}.trades"),
        net_profit=_number(data["net_profit"], f"{where}.net_profit"),
        profit_factor=_number(data["profit_factor"], f"{where}.profit_factor"),
        win_rate=_number(data["win_rate"], f"{where}.win_rate"),
        expectancy_r=_number(data["expectancy_r"], f"{where}.expectancy_r"),
        max_drawdown=_number(data["max_drawdown"], f"{where}.max_drawdown"),
        regimes=tuple(
            _regime(row, f"{where}.regimes[{index}]")
            for index, row in enumerate(_sequence(data["regimes"], f"{where}.regimes"))
        ),
        stress=tuple(
            _stress(row, f"{where}.stress[{index}]")
            for index, row in enumerate(_sequence(data["stress"], f"{where}.stress"))
        ),
    )


def _aggregate(payload: object, where: str) -> AggregateEvidence:
    data = _mapping(payload, where)
    _only(data, _AGGREGATE_KEYS, where)
    return AggregateEvidence(
        total_trades=_count(data["total_trades"], f"{where}.total_trades"),
        total_r=_number(data["total_r"], f"{where}.total_r"),
        expectancy_r=_number(data["expectancy_r"], f"{where}.expectancy_r"),
        profit_factor=_number(data["profit_factor"], f"{where}.profit_factor"),
        max_drawdown=_number(data["max_drawdown"], f"{where}.max_drawdown"),
        positive_folds=_count(data["positive_folds"], f"{where}.positive_folds"),
        positive_fold_fraction=_number(
            data["positive_fold_fraction"], f"{where}.positive_fold_fraction"
        ),
        expectancy_r_ci95_low=_number(
            data["expectancy_r_ci95_low"], f"{where}.expectancy_r_ci95_low"
        ),
        expectancy_r_ci95_high=_number(
            data["expectancy_r_ci95_high"], f"{where}.expectancy_r_ci95_high"
        ),
        total_r_ci95_low=_number(data["total_r_ci95_low"], f"{where}.total_r_ci95_low"),
        total_r_ci95_high=_number(data["total_r_ci95_high"], f"{where}.total_r_ci95_high"),
        bootstrap_samples=_count(data["bootstrap_samples"], f"{where}.bootstrap_samples"),
        bootstrap_confidence=_number(data["bootstrap_confidence"], f"{where}.bootstrap_confidence"),
        bootstrap_method=_text(data["bootstrap_method"], f"{where}.bootstrap_method"),
        regime_coverage=_number(data["regime_coverage"], f"{where}.regime_coverage"),
        regime_trades=tuple(
            _regime(row, f"{where}.regime_trades[{index}]")
            for index, row in enumerate(_sequence(data["regime_trades"], f"{where}.regime_trades"))
        ),
    )


def evidence_from_dict(payload: object) -> CandidateEvidence:
    """Strict loader for :meth:`CandidateEvidence.to_dict`.

    Unknown or missing fields, a wrong schema version or a non-numeric metric
    raise :class:`~scalping.discovery.evidence.EvidenceError` with code
    ``invalid_payload`` or ``schema_mismatch``. The checksum is **not** checked
    here; :func:`reproduce_bundle` calls
    :func:`~scalping.discovery.evidence.verify_evidence` right after loading.
    """
    data = _mapping(payload, "evidence")
    _only(data, _EVIDENCE_KEYS, "evidence")
    schema = _count(data["schema_version"], "schema_version")
    if schema != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError(
            "schema_mismatch",
            f"bundle schema {schema} is not the supported {EVIDENCE_SCHEMA_VERSION}",
        )
    return CandidateEvidence(
        spec_hash=_text(data["spec_hash"], "spec_hash"),
        spec_version=_count(data["spec_version"], "spec_version"),
        complexity=_count(data["complexity"], "complexity"),
        dataset_id=_text(data["dataset_id"], "dataset_id"),
        dataset_checksum=_text(data["dataset_checksum"], "dataset_checksum"),
        symbol=_text(data["symbol"], "symbol"),
        timeframe=_text(data["timeframe"], "timeframe"),
        code_version=_text(data["code_version"], "code_version"),
        settings_digest=_text(data["settings_digest"], "settings_digest"),
        seed=_count(data["seed"], "seed"),
        trial_count=_count(data["trial_count"], "trial_count"),
        search_budget=_count(data["search_budget"], "search_budget"),
        splits=_splits(data["splits"], "splits"),
        policy=_policy(data["policy"], "policy"),
        folds=tuple(
            _fold(row, f"folds[{index}]")
            for index, row in enumerate(_sequence(data["folds"], "folds"))
        ),
        aggregate=_aggregate(data["aggregate"], "aggregate"),
        checksum=_text(data["checksum"], "checksum"),
        created_at=_text(data["created_at"], "created_at"),
        schema_version=schema,
    )


# --- mismatch reporting -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionDifference:
    """One leaf path whose stored and reproduced values differ."""

    path: str
    expected: Any
    reproduced: Any

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "expected": self.expected, "reproduced": self.reproduced}


@dataclass(frozen=True, slots=True)
class ReproduceMismatch:
    """The rebuild succeeded but its checksum differs from the stored one.

    ``differences`` holds the first leaf paths that differ (bounded by the
    caller), and ``sections`` is their ordered top-level section list. The
    rebuilt bundle is carried for inspection; it is never written over the
    original.
    """

    expected: str
    reproduced: str
    differences: tuple[SectionDifference, ...]
    evidence: CandidateEvidence

    @property
    def sections(self) -> tuple[str, ...]:
        """Unique top-level section names of the differing paths, in order."""
        seen: list[str] = []
        for difference in self.differences:
            section = difference.path.split(".", 1)[0].split("[", 1)[0]
            if section and section not in seen:
                seen.append(section)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reproduced": False,
            "checksum": self.reproduced,
            "expected": self.expected,
            "mismatches": list(self.sections),
            "differences": [difference.to_dict() for difference in self.differences],
            "dataset_id": self.evidence.dataset_id,
            "spec_hash": self.evidence.spec_hash,
        }


def _differences(
    expected: object,
    reproduced: object,
    *,
    path: str = "",
    limit: int = 10,
    found: list[SectionDifference] | None = None,
) -> tuple[SectionDifference, ...]:
    """First ``limit`` leaf paths where two JSON-safe payloads disagree."""
    result = [] if found is None else found
    if len(result) >= limit:
        return tuple(result)

    if isinstance(expected, Mapping) and isinstance(reproduced, Mapping):
        for key in expected:
            child = f"{path}.{key}" if path else str(key)
            if key not in reproduced:
                result.append(SectionDifference(child, expected[key], None))
            else:
                _differences(expected[key], reproduced[key], path=child, limit=limit, found=result)
            if len(result) >= limit:
                return tuple(result)
        for key in reproduced:
            if key not in expected:
                child = f"{path}.{key}" if path else str(key)
                result.append(SectionDifference(child, None, reproduced[key]))
            if len(result) >= limit:
                return tuple(result)
        return tuple(result)

    if (
        isinstance(expected, (list, tuple))
        and isinstance(reproduced, (list, tuple))
        and not isinstance(expected, str)
        and not isinstance(reproduced, str)
    ):
        if len(expected) != len(reproduced):
            result.append(SectionDifference(path, len(expected), len(reproduced)))
            return tuple(result)
        for index, (left, right) in enumerate(zip(expected, reproduced, strict=True)):
            _differences(left, right, path=f"{path}[{index}]", limit=limit, found=result)
            if len(result) >= limit:
                return tuple(result)
        return tuple(result)

    if expected != reproduced:
        result.append(SectionDifference(path, expected, reproduced))
    return tuple(result)


# --- inputs -------------------------------------------------------------------


def spec_sidecar_path(bundle_path: str | Path) -> Path:
    """Where a spec is looked up when the caller passes none.

    ``candidate.evidence.json`` -> ``candidate.evidence.spec.json``. The sidecar
    is only a location convention; its content is still validated against the
    stored ``spec_hash``.
    """
    return Path(bundle_path).with_suffix(_SPEC_SIDECAR_SUFFIX)


def _is_file(value: str) -> bool:
    """``Path(value).is_file()`` without raising on an over-long inline JSON."""
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _load_spec(
    spec: StrategySpec | Mapping[str, Any] | str | Path | None,
    bundle_path: Path,
) -> StrategySpec:
    source: object = spec
    origin = "spec"
    if spec is None:
        sidecar = spec_sidecar_path(bundle_path)
        if not sidecar.is_file():
            raise ReproduceError(
                "spec_missing",
                (
                    f"the bundle does not carry the strategy; pass spec= or write {sidecar.name} "
                    "next to it"
                ),
            )
        origin = str(sidecar)
        try:
            source = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproduceError("spec_invalid", f"cannot read {sidecar}: {exc}") from exc
    elif isinstance(spec, Path) or (isinstance(spec, str) and _is_file(spec)):
        candidate_path = Path(spec)
        origin = str(candidate_path)
        if not candidate_path.is_file():
            raise ReproduceError("spec_missing", f"no spec file at {candidate_path}")
        try:
            source = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReproduceError("spec_invalid", f"cannot read {candidate_path}: {exc}") from exc
    elif isinstance(spec, str):
        origin = "inline spec"
        try:
            source = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ReproduceError(
                "spec_missing",
                f"spec {spec!r} is neither an existing file nor inline JSON: {exc}",
            ) from exc

    if isinstance(source, StrategySpec):
        return source
    try:
        return StrategySpec.from_dict(source)
    except (DslValidationError, TypeError, ValueError) as exc:
        raise ReproduceError("spec_invalid", f"{origin} is not a valid strategy: {exc}") from exc


def _load_settings(settings: Settings | None) -> Settings:
    if settings is None:
        return Settings()
    if not isinstance(settings, Settings):
        raise TypeError(f"settings must be a Settings, got {type(settings).__name__}")
    return settings


def _load_code_version(code_version: str | None) -> str:
    if code_version is None:
        return current_code_version()
    if not isinstance(code_version, str) or not code_version.strip():
        raise TypeError("code_version must be a non-empty str")
    return code_version


def dataset_artifact_dir(store_root: str | Path, dataset_checksum: str) -> Path:
    """Directory of the saved dataset artifact for one checksum.

    The checksum is validated as a hex digest before it becomes a path segment,
    so a crafted evidence file cannot point the lookup outside ``store_root``.
    """
    if (
        not isinstance(dataset_checksum, str)
        or _CHECKSUM_PATTERN.fullmatch(dataset_checksum) is None
    ):
        raise ReproduceError(
            "dataset_checksum_invalid",
            f"{dataset_checksum!r} is not a hex dataset checksum",
        )
    if isinstance(store_root, str) and not store_root.strip():
        raise ReproduceError("store_invalid", "store_root must not be empty")
    return Path(store_root) / DATASET_ARTIFACT_DIRNAME / dataset_checksum


def _manifest(payload: Mapping[str, Any]) -> DatasetManifest:
    """Parse the dataset manifest stored inside an artifact."""
    try:
        splits = _mapping(payload.get("splits"), "manifest.splits")
        ranges: dict[str, SplitRange] = {}
        for name in _MANIFEST_SPLITS:
            span = _mapping(splits.get(name), f"manifest.splits.{name}")
            ranges[name] = SplitRange(
                start=_int(span.get("start"), f"manifest.splits.{name}.start"),
                end=_int(span.get("end"), f"manifest.splits.{name}.end"),
            )
        groups = tuple(str(group) for group in _list(payload.get("groups"), "manifest.groups"))
        columns = tuple(str(column) for column in _list(payload.get("columns"), "manifest.columns"))
        notes = payload.get("notes", "")
        if not isinstance(notes, str):
            raise ReproduceError("dataset_manifest_invalid", "manifest.notes must be a string")
        return DatasetManifest(
            dataset_id=_string(payload.get("dataset_id"), "manifest.dataset_id"),
            symbol=_string(payload.get("symbol"), "manifest.symbol"),
            timeframe=_string(payload.get("timeframe"), "manifest.timeframe"),
            horizon=_int(payload.get("horizon"), "manifest.horizon"),
            groups=groups,
            columns=columns,
            rows=_int(payload.get("rows"), "manifest.rows"),
            train=ranges["train"],
            validation=ranges["validation"],
            test=ranges["test"],
            purge_bars=_int(payload.get("purge_bars"), "manifest.purge_bars"),
            embargo_bars=_int(payload.get("embargo_bars"), "manifest.embargo_bars"),
            checksum=_string(payload.get("checksum"), "manifest.checksum"),
            created_at=_string(payload.get("created_at"), "manifest.created_at"),
            notes=notes,
        )
    except ReproduceError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReproduceError("dataset_manifest_invalid", str(exc)) from exc


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{where} must be a non-empty string")
    return value


def _int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} must be an integer")
    return value


def _list(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{where} must be an array")
    return value


# --- dataset rebuild ----------------------------------------------------------


def _array_checksum(arrays: Sequence[np.ndarray]) -> str:
    """Blake2b-128 over the dataset arrays, in the builder's order.

    This repeats :func:`scalping.datasets.builder._checksum` rather than
    importing a private name. The builder is part of the code-version digest, so
    a change to either spelling is a different experiment and cannot silently
    compare equal.
    """
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(f"{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _rebuild_dataset(features: FeatureFrame, manifest: DatasetManifest) -> DatasetBundle:
    """Rebuild the exact arrays of ``manifest`` from the current feature frame.

    Same slicing, dtype casts and label formula as
    :func:`scalping.datasets.builder.build_dataset`, but the split edges come
    from the stored manifest instead of fractions, which are not stored. The
    recomputed checksum is compared with the manifest, so a changed bar fails
    here before any evidence is built.
    """
    start = int(features.valid_from)
    end = int(features.X.shape[0]) - int(manifest.horizon)
    if start < 0 or end <= start:
        raise ReproduceError(
            "dataset_geometry",
            f"a {features.X.shape[0]}-row frame with valid_from={start} and "
            f"horizon={manifest.horizon} has no usable rows",
        )
    rows = end - start
    if rows != int(manifest.rows):
        raise ReproduceError(
            "dataset_geometry",
            f"the manifest records {manifest.rows} rows but the bars rebuild {rows}",
            invalid=False,
            expected=manifest.checksum,
        )
    if columns_for(manifest.groups) != manifest.columns:
        raise ReproduceError(
            "dataset_columns_mismatch",
            (
                "the stored groups select different columns than the manifest recorded; "
                "the feature registry has changed"
            ),
            invalid=False,
            expected=manifest.checksum,
        )

    indices = indices_for(manifest.groups)
    X = np.ascontiguousarray(features.X[start:end][:, indices], dtype=np.float32)
    close = np.asarray(features.ohlc[start:end, 3], dtype=np.float64)
    future = np.asarray(
        features.ohlc[start + int(manifest.horizon) : int(features.X.shape[0]), 3],
        dtype=np.float64,
    )
    if close.shape != future.shape:
        raise ReproduceError("dataset_geometry", "horizon alignment failed")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(close > 0, future / close - 1.0, np.nan)
    direction = np.sign(returns)
    direction = np.nan_to_num(direction, nan=0.0).astype(np.int8)
    returns = returns.astype(np.float32)

    split = np.full(X.shape[0], -1, dtype=np.int8)
    for name, span in (
        ("train", manifest.train),
        ("validation", manifest.validation),
        ("test", manifest.test),
    ):
        low = int(span.start) - start
        high = int(span.end) - start
        if low < 0 or high > X.shape[0] or high < low:
            raise ReproduceError(
                "dataset_geometry",
                f"the stored {name} range [{span.start}, {span.end}) does not fit the rebuild",
                invalid=False,
                expected=manifest.checksum,
            )
        split[low:high] = SPLIT_NAMES.index(name)

    ts = np.asarray(features.ts[start:end])
    bundle = DatasetBundle(
        X=X,
        y_direction=direction,
        y_return=returns,
        ts=ts,
        split=split,
        manifest=manifest,
    )
    computed = _array_checksum((X, direction, returns, ts, split))
    if computed != manifest.checksum:
        raise ReproduceError(
            "dataset_checksum_mismatch",
            (
                f"the stored bars rebuild dataset checksum {computed!r}, "
                f"not the recorded {manifest.checksum!r}"
            ),
            invalid=False,
            expected=manifest.checksum,
        )
    return bundle


# --- reproduce ----------------------------------------------------------------


def _evidence_config(evidence: CandidateEvidence) -> EvidenceConfig:
    """Rebuild the fold configuration the bundle recorded."""
    policy = evidence.policy
    return EvidenceConfig(
        code_version=evidence.code_version,
        symbol=evidence.symbol,
        timeframe=evidence.timeframe,
        folds=policy.folds,
        purge_bars=policy.purge_bars,
        embargo_bars=policy.embargo_bars,
        min_trades_per_fold=policy.min_trades_per_fold,
        cost_stress=policy.cost_stress,
        regime_coverage_min=policy.regime_coverage_min,
        bootstrap_samples=policy.bootstrap_samples,
        bootstrap_confidence=policy.bootstrap_confidence,
        seed=evidence.seed,
        search_budget=evidence.search_budget,
        trial_count=evidence.trial_count,
        complexity=evidence.complexity,
        window=policy.window,
    )


def _identity_body(evidence: CandidateEvidence) -> dict[str, Any]:
    """``to_dict()`` without ``checksum``/``created_at``, the fields outside identity."""
    return {key: value for key, value in evidence.to_dict().items() if key not in _UNCHECKED_FIELDS}


def _stamp(created_at: str) -> datetime | None:
    """The bundle's creation stamp, when it parses; the checksum ignores it."""
    try:
        return datetime.fromisoformat(created_at)
    except ValueError:
        return None


def reproduce_bundle(
    bundle_path: str | Path,
    *,
    store_root: str | Path,
    spec: StrategySpec | Mapping[str, Any] | str | Path | None = None,
    settings: Settings | None = None,
    code_version: str | None = None,
    mismatch_limit: int = 10,
) -> CandidateEvidence | ReproduceMismatch:
    """Rebuild one stored evidence bundle and compare its checksum.

    Returns the rebuilt :class:`CandidateEvidence` when the checksum is equal,
    or a :class:`ReproduceMismatch` with the first ``mismatch_limit`` differing
    leaf paths. Identity refusals (settings, code version, dataset) raise
    :class:`ReproduceError`; an unverifiable bundle raises
    :class:`~scalping.discovery.evidence.EvidenceError`.

    ``spec`` is required because a hash cannot be inverted; when omitted, the
    sidecar at :func:`spec_sidecar_path` is used. ``settings`` defaults to
    :class:`~scalping.config.Settings` and ``code_version`` to
    :func:`scalping.discovery.run.code_version`; either mismatch is a refusal,
    not a quiet rerun under different inputs.
    """
    if isinstance(mismatch_limit, bool) or not isinstance(mismatch_limit, int):
        raise TypeError("mismatch_limit must be an int")
    if mismatch_limit < 1:
        raise ValueError("mismatch_limit must be >= 1")

    path = Path(bundle_path)
    if not path.is_file():
        raise ReproduceError("bundle_missing", f"no evidence bundle at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproduceError("bundle_invalid", f"cannot read {path}: {exc}") from exc

    evidence = evidence_from_dict(payload)
    verify_evidence(
        evidence,
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        spec_hash=evidence.spec_hash,
        code_version=evidence.code_version,
        settings_digest=evidence.settings_digest,
    )

    candidate = _load_spec(spec, path)
    try:
        spec_hash = candidate.canonical_hash()
        complexity = int(candidate.complexity())
        spec_version = int(candidate.spec_version)
    except DslValidationError as exc:
        raise ReproduceError("spec_invalid", str(exc)) from exc
    if spec_hash != evidence.spec_hash:
        raise ReproduceError(
            "spec_mismatch",
            f"the supplied spec hashes to {spec_hash!r}, not {evidence.spec_hash!r}",
            invalid=False,
            expected=evidence.checksum,
        )
    if complexity != evidence.complexity:
        raise ReproduceError(
            "spec_complexity_mismatch",
            f"the supplied spec costs {complexity}, not {evidence.complexity}",
            invalid=False,
            expected=evidence.checksum,
        )
    if spec_version != evidence.spec_version:
        raise ReproduceError(
            "spec_version_mismatch",
            f"the supplied spec is version {spec_version}, not {evidence.spec_version}",
            invalid=False,
            expected=evidence.checksum,
        )

    run_settings = _load_settings(settings)
    try:
        digest = settings_digest(run_settings)
    except (TypeError, ValueError) as exc:
        raise ReproduceError("settings_invalid", str(exc)) from exc
    if digest != evidence.settings_digest:
        raise ReproduceError(
            "settings_mismatch",
            f"the supplied settings digest to {digest!r}, not {evidence.settings_digest!r}",
            invalid=False,
            expected=evidence.checksum,
        )

    version = _load_code_version(code_version)
    if version != evidence.code_version:
        raise ReproduceError(
            "code_version_mismatch",
            f"the running code version is {version!r}, not {evidence.code_version!r}",
            invalid=False,
            expected=evidence.checksum,
        )

    artifact = dataset_artifact_dir(store_root, evidence.dataset_checksum)
    if not (artifact / ARTIFACT_FILE).is_file():
        raise ReproduceError(
            "dataset_missing",
            f"no dataset artifact at {artifact} for checksum {evidence.dataset_checksum}",
            invalid=False,
            expected=evidence.checksum,
        )
    try:
        stored = load_dataset_arrays(artifact)
    except ArtifactError as exc:
        raise ReproduceError("dataset_corrupt", f"{artifact}: {exc}") from exc

    manifest = _manifest(stored.manifest)
    if manifest.symbol != evidence.symbol or manifest.timeframe != evidence.timeframe:
        raise ReproduceError(
            "dataset_mismatch",
            (
                f"the artifact is for {manifest.symbol} {manifest.timeframe}, "
                f"not {evidence.symbol} {evidence.timeframe}"
            ),
            invalid=False,
            expected=evidence.checksum,
        )
    if manifest.dataset_id != evidence.dataset_id:
        raise ReproduceError(
            "dataset_mismatch",
            f"the artifact is dataset {manifest.dataset_id!r}, not {evidence.dataset_id!r}",
            invalid=False,
            expected=evidence.checksum,
        )
    if manifest.checksum != evidence.dataset_checksum:
        raise ReproduceError(
            "dataset_checksum_mismatch",
            (
                f"the artifact records checksum {manifest.checksum!r}, "
                f"not the bundle's {evidence.dataset_checksum!r}"
            ),
            invalid=False,
            expected=evidence.checksum,
        )

    bars = ParquetStore(store_root).read(evidence.symbol, evidence.timeframe)
    if bars.is_empty():
        raise ReproduceError(
            "bars_missing",
            f"no {evidence.symbol} {evidence.timeframe} bars under {store_root}",
        )
    try:
        features = build_features(bars, costs=run_settings.costs)
    except (TypeError, ValueError) as exc:
        raise ReproduceError("bars_invalid", str(exc)) from exc

    try:
        bundle = _rebuild_dataset(features, manifest)
    except ReproduceError as exc:
        # The dataset refusal is raised deep in the rebuild; a caller reporting
        # the identity wants the evidence checksum, not the dataset checksum.
        if not exc.invalid:
            exc.expected = evidence.checksum
        raise
    try:
        config = _evidence_config(evidence)
    except (TypeError, ValueError) as exc:
        raise ReproduceError("bundle_invalid", f"the stored policy is unusable: {exc}") from exc
    rebuilt = build_evidence(
        bundle,
        candidate,
        config,
        features=features,
        settings=run_settings,
        created_at=_stamp(evidence.created_at),
    )
    if rebuilt.checksum == evidence.checksum:
        return rebuilt

    differences = _differences(
        _identity_body(evidence),
        _identity_body(rebuilt),
        limit=mismatch_limit,
    )
    if not differences:
        # The two payloads agree field by field yet hash differently: the
        # checksum algorithm or canonical JSON changed without the version
        # digest noticing. Say so instead of reporting an empty section list.
        differences = (SectionDifference("checksum", evidence.checksum, rebuilt.checksum),)
    return ReproduceMismatch(
        expected=evidence.checksum,
        reproduced=rebuilt.checksum,
        differences=differences,
        evidence=rebuilt,
    )
