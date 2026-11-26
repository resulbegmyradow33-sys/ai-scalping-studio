"""The production writer for a dataset artifact and an evidence bundle.

``discovery reproduce`` replays a stored experiment from five things: the
evidence JSON, the strategy sidecar, the bar store, the dataset artifact under
``<store_root>/datasets/<dataset_checksum>/`` and a code/settings identity.
Until this module existed nothing in the production path wrote the artifact or
the bundle, so a bundle could only be created by a test fixture. Issue #136 is
that gap.

Two writers, one layout
-----------------------
:func:`save_dataset_artifact`
    Stage the arrays and manifest written by
    :func:`scalping.datasets.artifact.save_dataset_arrays` in a temporary
    directory beside the final one, re-read them with digest verification, and
    only then move the whole directory onto
    ``<store_root>/datasets/<dataset_checksum>/``. The checksum is the identity,
    so a directory that already holds a **different** dataset is a refusal, not
    an overwrite: a stale or hand-written artifact must not be silently
    replaced. Re-saving the same bundle is idempotent.
:func:`save_evidence_bundle`
    Write the evidence JSON from :meth:`CandidateEvidence.to_dict` in canonical
    form, plus the strategy sidecar at the exact location
    :func:`scalping.discovery.reproduce.spec_sidecar_path` looks up. The sidecar
    is written first and the bundle second, so a crash can leave an orphan
    sidecar but never a bundle without its strategy. Both files go through a
    unique sibling temp file and ``os.replace``.

The dataset artifact is content-addressed. The evidence bundle is not: its path
is the caller's ``--out``, and a rerun is expected to replace the previous
report. Nothing here promotes, evaluates the test split or opens a socket.

Read-side counterparts
----------------------
:func:`load_dataset_artifact` re-verifies the artifact's file digests **and**
recomputes the dataset checksum from the loaded arrays, so a manifest that
claims a checksum its arrays do not produce is caught. ``load_dataset_arrays``
alone checks the per-file digests, not the array-derived dataset id.
:func:`load_evidence_bundle` parses with the strict field loader from
:mod:`scalping.discovery.reproduce` and re-derives the bundle checksum before
returning it.

Convenience
-----------
:func:`persist_evidence_run` is the CLI path: read the stored bars, build the
causal feature frame and a chronological dataset, persist the artifact, build
the evidence for one DSL spec and persist the bundle plus sidecar. It never
reads the test split for a metric (``build_evidence`` stops at the validation
end) and never promotes anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.config import Settings
from scalping.data.provenance import (
    DatasetProvenance,
    load_provenance,
)
from scalping.data.provenance import (
    checksum as provenance_checksum,
)
from scalping.data.store import ParquetStore
from scalping.datasets.artifact import (
    ARTIFACT_FILE,
    ArtifactError,
    LoadedArrays,
    load_dataset_arrays,
    save_dataset_arrays,
)
from scalping.datasets.builder import DatasetBundle, DatasetManifest, build_dataset
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceConfig,
    EvidenceError,
    build_evidence,
    verify_evidence,
)
from scalping.discovery.protocols import canonical_json
from scalping.discovery.reproduce import (
    DATASET_ARTIFACT_DIRNAME,
    evidence_from_dict,
    spec_sidecar_path,
)
from scalping.discovery.run import code_version as current_code_version
from scalping.features.build import FeatureFrame
from scalping.features.cache import FEATURE_CACHE_DIRNAME, FeatureCache, cached_build_features
from scalping.strategy.dsl.nodes import DslValidationError, StrategySpec

__all__ = [
    "PersistError",
    "PersistedRun",
    "StoreProvenance",
    "load_dataset_artifact",
    "load_evidence_bundle",
    "load_store_provenance",
    "load_strategy_spec",
    "persist_evidence_run",
    "save_dataset_artifact",
    "save_evidence_bundle",
]

#: A dataset checksum is a hex digest and becomes a directory name.
_CHECKSUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{16,128}")

#: Array order the dataset checksum and the artifact module both use.
_ARRAY_ORDER: Final[tuple[str, ...]] = ("X", "y_direction", "y_return", "ts", "split")

#: Elements per digest update; keeps verification memory bounded on a large X.
_CHECKSUM_CHUNK: Final[int] = 1 << 18


class PersistError(ValueError):
    """The artifact or bundle cannot be written or read safely.

    ``code`` is the machine-readable reason and follows the same vocabulary as
    :class:`~scalping.discovery.reproduce.ReproduceError` where the meaning is
    the same (``dataset_checksum_mismatch``, ``spec_mismatch`` and so on), so a
    caller can report a writer refusal and a replay refusal the same way.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class PersistedRun:
    """What :func:`persist_evidence_run` wrote, for the CLI's one JSON line."""

    symbol: str
    timeframe: str
    dataset_id: str
    dataset_checksum: str
    artifact_dir: Path
    bundle_path: Path
    spec_path: Path
    evidence: CandidateEvidence
    provenance_verified: bool | None = None
    warnings: tuple[str, ...] = ()

    @property
    def checksum(self) -> str:
        return self.evidence.checksum

    def to_dict(self) -> dict[str, Any]:
        return {
            "persisted": True,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            "artifact": str(self.artifact_dir),
            "bundle": str(self.bundle_path),
            "spec": str(self.spec_path),
            "evidence_checksum": self.checksum,
            "schema_version": self.evidence.schema_version,
            "source": self.evidence.source,
            "synthetic": self.evidence.synthetic,
            "provenance_verified": self.provenance_verified,
            "warnings": list(self.warnings),
            "folds": len(self.evidence.folds),
            "validation_end_row": self.evidence.splits.validation_end_row,
            "test_start_row": self.evidence.splits.test_start_row,
        }


# --- paths and arrays ---------------------------------------------------------


def _artifact_dir(store_root: str | Path, dataset_checksum: str) -> Path:
    """``<store_root>/datasets/<checksum>``, with the checksum validated first."""
    if (
        not isinstance(dataset_checksum, str)
        or _CHECKSUM_PATTERN.fullmatch(dataset_checksum) is None
    ):
        raise PersistError(
            "checksum_invalid", f"{dataset_checksum!r} is not a hex dataset checksum"
        )
    if isinstance(store_root, str) and not store_root.strip():
        raise PersistError("store_invalid", "store_root must not be empty")
    return Path(store_root) / DATASET_ARTIFACT_DIRNAME / dataset_checksum


def _arrays(bundle: DatasetBundle) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(getattr(bundle, name)) for name in _ARRAY_ORDER)


def _arrays_checksum(arrays: Sequence[np.ndarray]) -> str:
    """Blake2b-128 over the arrays, in the builder's and reproduce's order.

    The builder computes the dataset id this way and reproduction repeats the
    formula instead of trusting the stored manifest. This writer is the third
    user, and it must agree with both or a saved artifact fails its own replay.
    The byte stream is identical to ``array.tobytes()``; it is fed in element
    chunks so a memory-mapped ``X`` is not copied whole.
    """
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(f"{values.dtype}:{values.shape}".encode())
        flat = values.reshape(-1)
        for start in range(0, int(flat.size), _CHECKSUM_CHUNK):
            digest.update(flat[start : start + _CHECKSUM_CHUNK].tobytes())
    return digest.hexdigest()


def _check_bundle(bundle: DatasetBundle) -> None:
    if not isinstance(bundle, DatasetBundle):
        raise TypeError(f"bundle must be a DatasetBundle, got {type(bundle).__name__}")
    if not isinstance(bundle.manifest, DatasetManifest):
        raise TypeError("bundle.manifest must be a DatasetManifest")
    rows = int(np.asarray(bundle.X).shape[0])
    for name in _ARRAY_ORDER[1:]:
        array = np.asarray(getattr(bundle, name))
        if array.ndim != 1 or int(array.shape[0]) != rows:
            raise PersistError(
                "bundle_geometry",
                f"{name} has shape {array.shape}; expected ({rows},) to match X",
            )
    if rows != int(bundle.manifest.rows):
        raise PersistError(
            "bundle_geometry",
            f"the manifest records {bundle.manifest.rows} rows but X has {rows}",
        )
    computed = _arrays_checksum(_arrays(bundle))
    if computed != bundle.manifest.checksum:
        raise PersistError(
            "dataset_checksum_mismatch",
            (
                f"the bundle arrays hash to {computed!r}, not the manifest's "
                f"{bundle.manifest.checksum!r}"
            ),
        )


def _check_features(bundle: DatasetBundle, features: FeatureFrame) -> None:
    """Refuse a frame that is not the one the bundle was cut from.

    Timestamps and row counts are the same check ``build_evidence`` makes, so a
    mismatched pair cannot be persisted and then fail only at evidence time.
    """
    if not isinstance(features, FeatureFrame):
        raise TypeError(f"features must be a FeatureFrame, got {type(features).__name__}")
    offset = int(features.valid_from)
    horizon = int(bundle.manifest.horizon)
    expected = int(features.X.shape[0]) - offset - horizon
    rows = int(bundle.ts.shape[0])
    if expected != rows:
        raise PersistError(
            "frame_mismatch",
            (
                f"features has {features.X.shape[0]} rows from valid_from={offset} and "
                f"horizon={horizon}; expected {expected} bundle rows, got {rows}"
            ),
        )
    if not np.array_equal(np.asarray(bundle.ts), np.asarray(features.ts[offset : offset + rows])):
        raise PersistError("frame_mismatch", "the bundle timestamps do not match the feature frame")


# --- atomic text --------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write through a unique sibling temp file and ``os.replace`` onto ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


# --- dataset artifact ---------------------------------------------------------


def _verify_manifest_of(loaded: LoadedArrays, bundle: DatasetBundle) -> None:
    """Reject an artifact whose manifest or arrays disagree with ``bundle``."""
    manifest = loaded.manifest
    stored = manifest.get("checksum")
    if not isinstance(stored, str) or not stored.strip():
        raise PersistError("dataset_manifest_invalid", "the artifact manifest has no checksum")
    if stored != bundle.manifest.checksum:
        raise PersistError(
            "artifact_conflict",
            (
                f"the artifact at {loaded.directory} records dataset checksum {stored!r}; "
                f"refusing to replace it with {bundle.manifest.checksum!r}"
            ),
        )
    if manifest.get("dataset_id") != bundle.manifest.dataset_id:
        raise PersistError(
            "artifact_conflict",
            (
                f"the artifact at {loaded.directory} is dataset "
                f"{manifest.get('dataset_id')!r}, not {bundle.manifest.dataset_id!r}"
            ),
        )
    computed = _arrays_checksum(
        (loaded.X, loaded.y_direction, loaded.y_return, loaded.ts, loaded.split)
    )
    if computed != stored:
        raise PersistError(
            "dataset_checksum_mismatch",
            f"the artifact arrays hash to {computed!r}, not the recorded {stored!r}",
        )


def _verify_existing_artifact(target: Path, bundle: DatasetBundle) -> None:
    """Accept only an identical, verifiable artifact already at ``target``."""
    if not target.is_dir():
        raise PersistError("artifact_conflict", f"{target} exists and is not a directory")
    if not (target / ARTIFACT_FILE).is_file():
        raise PersistError(
            "artifact_incomplete",
            f"{target} has no {ARTIFACT_FILE}; a partial write is not a dataset",
        )
    try:
        loaded = load_dataset_arrays(target, mmap=True)
    except ArtifactError as exc:
        raise PersistError("artifact_corrupt", f"{target}: {exc}") from exc
    _verify_manifest_of(loaded, bundle)


def _write_canonical_manifest(root: Path) -> None:
    """Rewrite the staged manifest as canonical, sorted-key JSON.

    The array table and dataset manifest are data, not a template; canonical
    spelling makes two identical saves byte-identical. It is safe to rewrite the
    commit marker here because the directory is still private to this writer.
    """
    path = root / ARTIFACT_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistError(
            "artifact_write_failed", f"staged manifest is unreadable: {exc}"
        ) from exc
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def save_dataset_artifact(
    store_root: str | Path,
    bundle: DatasetBundle,
    features: FeatureFrame,
) -> Path:
    """Persist ``bundle`` under ``<store_root>/datasets/<dataset_checksum>/``.

    The arrays and manifest come from
    :func:`scalping.datasets.artifact.save_dataset_arrays`, but they are written
    into a unique staging directory first, re-read with digest verification,
    canonicalised and only then moved into place. If the final directory already
    exists it must hold the same dataset; a different checksum or a corrupt
    manifest is a :class:`PersistError`, never an overwrite. Returns the
    artifact directory, which is the checksum's on-disk identity.
    """
    _check_bundle(bundle)
    _check_features(bundle, features)

    target = _artifact_dir(store_root, bundle.manifest.checksum)
    datasets_root = target.parent
    datasets_root.mkdir(parents=True, exist_ok=True)

    if target.exists():
        _verify_existing_artifact(target, bundle)
        return target

    staging = Path(tempfile.mkdtemp(dir=datasets_root, prefix=f".{target.name}.", suffix=".tmp"))
    try:
        save_dataset_arrays(staging, bundle)
        _write_canonical_manifest(staging)
        loaded = load_dataset_arrays(staging, mmap=True)
        _verify_manifest_of(loaded, bundle)
        try:
            os.replace(staging, target)
        except OSError as exc:
            # A concurrent writer may have created the target after our check.
            # Accept only a matching artifact; never delete a winner's files.
            if target.exists():
                _verify_existing_artifact(target, bundle)
                return target
            raise PersistError(
                "artifact_write_failed", f"cannot move {staging} to {target}: {exc}"
            ) from exc
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def load_dataset_artifact(store_root: str | Path, dataset_checksum: str) -> LoadedArrays:
    """Load and fully verify one saved artifact.

    Per-file digests are checked by :func:`load_dataset_arrays`; this wrapper
    additionally recomputes the dataset checksum from the arrays and compares it
    with the requested checksum and the stored manifest. It never rebuilds from
    bars; that is reproduction's job.
    """
    target = _artifact_dir(store_root, dataset_checksum)
    if not (target / ARTIFACT_FILE).is_file():
        raise PersistError("artifact_missing", f"no dataset artifact at {target}")
    try:
        loaded = load_dataset_arrays(target, mmap=True)
    except ArtifactError as exc:
        raise PersistError("artifact_corrupt", f"{target}: {exc}") from exc
    stored = loaded.manifest.get("checksum")
    if stored != dataset_checksum:
        raise PersistError(
            "dataset_checksum_mismatch",
            f"the artifact records {stored!r}, not the requested {dataset_checksum!r}",
        )
    computed = _arrays_checksum(
        (loaded.X, loaded.y_direction, loaded.y_return, loaded.ts, loaded.split)
    )
    if computed != dataset_checksum:
        raise PersistError(
            "dataset_checksum_mismatch",
            f"the artifact arrays hash to {computed!r}, not {dataset_checksum!r}",
        )
    return loaded


# --- strategy spec ------------------------------------------------------------


def _is_file(value: str) -> bool:
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _read_json_file(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistError("spec_invalid", f"cannot read {path}: {exc}") from exc


def load_strategy_spec(
    spec: StrategySpec | Mapping[str, Any] | str | Path,
) -> StrategySpec:
    """Load a spec from a typed object, a mapping, a path or inline JSON.

    Unknown or invalid payloads raise :class:`PersistError` with
    ``spec_invalid``; nothing is executed or evaluated.
    """
    source: object = spec
    origin = "spec"
    if isinstance(spec, StrategySpec):
        return spec
    if isinstance(spec, Path):
        origin = str(spec)
        if not spec.is_file():
            raise PersistError("spec_missing", f"no spec file at {spec}")
        source = _read_json_file(spec)
    elif isinstance(spec, str):
        if _is_file(spec):
            origin = spec
            source = _read_json_file(Path(spec))
        else:
            origin = "inline spec"
            try:
                source = json.loads(spec)
            except json.JSONDecodeError as exc:
                raise PersistError(
                    "spec_missing",
                    f"spec {spec!r} is neither an existing file nor inline JSON: {exc}",
                ) from exc
    elif not isinstance(spec, Mapping):
        raise TypeError(
            f"spec must be a StrategySpec, mapping, path or JSON string, got {type(spec).__name__}"
        )

    try:
        return StrategySpec.from_dict(source)
    except (DslValidationError, TypeError, ValueError) as exc:
        raise PersistError("spec_invalid", f"{origin} is not a valid strategy: {exc}") from exc


def _spec_identity(spec: StrategySpec) -> tuple[str, int, int]:
    try:
        return spec.canonical_hash(), int(spec.complexity()), int(spec.spec_version)
    except DslValidationError as exc:
        raise PersistError("spec_invalid", str(exc)) from exc


# --- evidence bundle ----------------------------------------------------------


def save_evidence_bundle(
    path: str | Path,
    evidence: CandidateEvidence,
    spec: StrategySpec | Mapping[str, Any] | str | Path,
) -> Path:
    """Write ``evidence`` and its strategy sidecar; return the bundle path.

    The bundle checksum is re-derived before writing, and the spec must hash to
    the recorded ``spec_hash``, ``complexity`` and ``spec_version``; a bundle
    and a strategy that do not belong together is a refusal. The sidecar goes to
    :func:`scalping.discovery.reproduce.spec_sidecar_path` first, the bundle
    second, each atomically. An existing file at ``path`` is replaced: that path
    is the caller's report location, not an identity.
    """
    if type(evidence) is not CandidateEvidence:
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    candidate = load_strategy_spec(spec)
    spec_hash, complexity, spec_version = _spec_identity(candidate)
    if spec_hash != evidence.spec_hash:
        raise PersistError(
            "spec_mismatch",
            f"the supplied spec hashes to {spec_hash!r}, not {evidence.spec_hash!r}",
        )
    if complexity != evidence.complexity:
        raise PersistError(
            "spec_complexity_mismatch",
            f"the supplied spec costs {complexity}, not {evidence.complexity}",
        )
    if spec_version != evidence.spec_version:
        raise PersistError(
            "spec_version_mismatch",
            f"the supplied spec is version {spec_version}, not {evidence.spec_version}",
        )
    try:
        verify_evidence(
            evidence,
            dataset_id=evidence.dataset_id,
            dataset_checksum=evidence.dataset_checksum,
            spec_hash=spec_hash,
            code_version=evidence.code_version,
            settings_digest=evidence.settings_digest,
        )
    except EvidenceError as exc:
        raise PersistError(exc.code, str(exc)) from exc

    bundle_path = Path(path)
    sidecar = spec_sidecar_path(bundle_path)
    _atomic_write_text(sidecar, canonical_json(candidate.canonical_dict()) + "\n")
    _atomic_write_text(bundle_path, canonical_json(evidence.to_dict()) + "\n")
    return bundle_path


def load_evidence_bundle(path: str | Path) -> CandidateEvidence:
    """Read one bundle, reject unknown fields and re-derive its checksum.

    The strict loader from :mod:`scalping.discovery.reproduce` owns the nested
    shape; :func:`verify_evidence` owns the checksum and required sections. A
    file that does not pass both is not a usable bundle.
    """
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise PersistError("bundle_missing", f"no evidence bundle at {bundle_path}")
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PersistError("bundle_invalid", f"cannot read {bundle_path}: {exc}") from exc
    try:
        evidence = evidence_from_dict(payload)
        verify_evidence(
            evidence,
            dataset_id=evidence.dataset_id,
            dataset_checksum=evidence.dataset_checksum,
            spec_hash=evidence.spec_hash,
            code_version=evidence.code_version,
            settings_digest=evidence.settings_digest,
        )
    except EvidenceError as exc:
        raise PersistError(exc.code, str(exc)) from exc
    return evidence


# --- store provenance ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoreProvenance:
    """What the store sidecar proved about the bars an evidence run read.

    ``verified`` is ``True`` only after the sidecar's symbol, timeframe, row
    count, first/last timestamp and content checksum matched the frame.
    ``False`` means a sidecar exists but was not about this slice (the values
    are discarded) and ``None`` means no sidecar was available. ``source`` and
    ``synthetic`` stay ``None`` unless verification succeeded; a declarative
    sidecar is never copied blindly.
    """

    source: str | None = None
    synthetic: bool | None = None
    verified: bool | None = None
    warnings: tuple[str, ...] = ()


def _same_instant(declared: str | None, actual: object) -> bool:
    """Compare an ISO sidecar timestamp with a frame timestamp as instants."""
    if actual is None:
        return declared is None
    if declared is None or not isinstance(actual, datetime):
        return False
    try:
        parsed = datetime.fromisoformat(declared)
    except ValueError:
        return False
    if parsed.tzinfo is None and actual.tzinfo is not None:
        parsed = parsed.replace(tzinfo=UTC)
    if actual.tzinfo is None and parsed.tzinfo is not None:
        actual = actual.replace(tzinfo=UTC)
    return parsed == actual


def _sidecar_mismatch(
    declared: DatasetProvenance,
    bars: pl.DataFrame,
    symbol: str,
    timeframe: str,
) -> str | None:
    """Why ``declared`` is not the provenance of ``bars``, or ``None``."""
    if (declared.symbol, declared.timeframe) != (symbol, timeframe):
        return (
            f"the sidecar describes {declared.symbol} {declared.timeframe}, "
            f"not {symbol} {timeframe}"
        )
    rows = int(bars.height)
    if declared.rows != rows:
        return f"the sidecar records {declared.rows} rows but the store has {rows}"
    first = bars["ts"].min() if rows else None
    last = bars["ts"].max() if rows else None
    if not _same_instant(declared.first_ts, first) or not _same_instant(declared.last_ts, last):
        return (
            "the sidecar timestamp span does not match the stored bars "
            f"({declared.first_ts!r}..{declared.last_ts!r})"
        )
    computed = provenance_checksum(bars)
    if declared.checksum != computed:
        return (
            f"the sidecar checksum {declared.checksum!r} does not hash the stored bars "
            f"({computed!r})"
        )
    return None


def load_store_provenance(
    store_root: str | Path,
    symbol: str,
    timeframe: str,
    bars: pl.DataFrame,
) -> StoreProvenance:
    """Verify the sidecar against ``bars`` and return usable provenance.

    A missing sidecar is unknown, not real. An unreadable sidecar, or one whose
    row span, timestamps or checksum do not describe this frame, is recorded as
    a warning and contributes no ``source``/``synthetic`` value: a dataset must
    never inherit an identity that belongs to another slice.
    """
    try:
        declared = load_provenance(store_root, symbol, timeframe)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return StoreProvenance(
            verified=False,
            warnings=(f"the store provenance sidecar is unreadable: {exc}",),
        )
    if declared is None:
        return StoreProvenance()
    problem = _sidecar_mismatch(declared, bars, symbol, timeframe)
    if problem is not None:
        return StoreProvenance(
            verified=False,
            warnings=(f"ignoring store provenance: {problem}",),
        )
    return StoreProvenance(
        source=declared.source,
        synthetic=declared.synthetic,
        verified=True,
    )


# --- end-to-end convenience ---------------------------------------------------


def persist_evidence_run(
    store_root: str | Path,
    out: str | Path,
    *,
    spec: StrategySpec | Mapping[str, Any] | str | Path,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    config: EvidenceConfig | None = None,
    settings: Settings | None = None,
    groups: tuple[str, ...] | None = None,
    horizon: int = 1,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    dataset_purge_bars: int = 0,
    dataset_embargo_bars: int = 0,
    use_cache: bool = True,
    created_at: datetime | None = None,
    source: str | None = None,
    synthetic: bool | None = None,
) -> PersistedRun:
    """Build one experiment from stored bars and persist the artifact and bundle.

    Steps, in order: read the bar store, build the causal feature frame, cut a
    chronological dataset with the same builder every other command uses,
    persist the dataset artifact, build the evidence bundle with ``config`` and
    persist it with the strategy sidecar. ``config`` defaults to an
    :class:`~scalping.discovery.evidence.EvidenceConfig` naming the current code
    version and the spec's structural cost; a supplied config must agree on
    symbol and timeframe.

    ``source``/``synthetic`` are a fallback for a caller that knows the data and
    has no sidecar; they are **not** a way to override one. The store sidecar is
    the authority: when it verifies against the bars it is used, when it exists
    but does not match the frame its claim is discarded (provenance stays
    unknown, ``provenance_verified`` is ``False`` and ``PersistedRun.warnings``
    records why) and only a complete absence of a sidecar lets these parameters
    through, still marked unverified. Values are written into the dataset
    manifest and the evidence identity; omitted values stay unknown instead of
    becoming real.

    The evidence stops at the dataset's validation end; the artifact keeps the
    test rows because the dataset checksum covers them, but no fold, metric or
    gate reads them.
    """
    candidate = load_strategy_spec(spec)
    run_settings = Settings() if settings is None else settings
    if not isinstance(run_settings, Settings):
        raise TypeError(f"settings must be a Settings, got {type(run_settings).__name__}")
    if config is not None:
        if not isinstance(config, EvidenceConfig):
            raise TypeError(f"config must be an EvidenceConfig, got {type(config).__name__}")
        if config.symbol != symbol or config.timeframe != timeframe:
            raise PersistError(
                "config_mismatch",
                (f"config is for {config.symbol} {config.timeframe}, not {symbol} {timeframe}"),
            )

    bars = ParquetStore(store_root).read(symbol, timeframe)
    if bars.is_empty():
        raise PersistError(
            "bars_missing", f"no {symbol} {timeframe} bars under {store_root}; ingest bars first"
        )
    store_provenance = load_store_provenance(store_root, symbol, timeframe, bars)
    if store_provenance.verified:
        if source is not None and source != store_provenance.source:
            raise PersistError(
                "provenance_mismatch",
                (
                    f"source={source!r} contradicts the verified store sidecar "
                    f"{store_provenance.source!r}"
                ),
            )
        if synthetic is not None and synthetic != store_provenance.synthetic:
            raise PersistError(
                "provenance_mismatch",
                (
                    f"synthetic={synthetic!r} contradicts the verified store sidecar "
                    f"{store_provenance.synthetic!r}"
                ),
            )
        run_source = store_provenance.source
        run_synthetic = store_provenance.synthetic
        run_verified: bool | None = True
    elif store_provenance.verified is False:
        run_source = None
        run_synthetic = None
        run_verified = False
    else:
        run_source = source
        run_synthetic = synthetic
        run_verified = None if source is None and synthetic is None else False
    run_warnings = list(store_provenance.warnings)
    if store_provenance.verified is None and run_verified is False:
        run_warnings.append(
            "provenance was supplied by the caller without a store sidecar and is unverified"
        )
    cache = FeatureCache(Path(store_root) / FEATURE_CACHE_DIRNAME) if use_cache else None
    features = cached_build_features(
        cache,
        symbol=symbol,
        timeframe=timeframe,
        frame=bars,
        settings=run_settings,
        groups=groups,
    )

    try:
        bundle = build_dataset(
            features,
            symbol=symbol,
            timeframe=timeframe,
            groups=groups,
            horizon=horizon,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            purge_bars=dataset_purge_bars,
            embargo_bars=dataset_embargo_bars,
            source=run_source,
            synthetic=run_synthetic,
        )
    except ValueError as exc:
        raise PersistError(
            "dataset_geometry", f"cannot build a chronological dataset: {exc}"
        ) from exc

    if config is None:
        complexity = _spec_identity(candidate)[1]
        config = EvidenceConfig(
            code_version=current_code_version(),
            symbol=symbol,
            timeframe=timeframe,
            complexity=complexity,
            source=run_source,
            synthetic=run_synthetic,
            provenance_verified=run_verified,
        )
    else:
        config = replace(config, provenance_verified=run_verified)

    artifact_dir = save_dataset_artifact(store_root, bundle, features)
    try:
        evidence = build_evidence(
            bundle,
            candidate,
            config,
            features=features,
            settings=run_settings,
            created_at=created_at,
        )
    except EvidenceError as exc:
        raise PersistError(exc.code, str(exc)) from exc

    bundle_path = save_evidence_bundle(out, evidence, candidate)
    return PersistedRun(
        symbol=evidence.symbol,
        timeframe=evidence.timeframe,
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        artifact_dir=artifact_dir,
        bundle_path=bundle_path,
        spec_path=spec_sidecar_path(bundle_path),
        evidence=evidence,
        provenance_verified=run_verified,
        warnings=tuple(run_warnings),
    )
