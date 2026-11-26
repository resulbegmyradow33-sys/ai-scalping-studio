"""Runtime promotion: turn one verified evidence bundle into a governed champion.

This module is the missing call site between an evidence bundle and the archive.
It does not add policy rules and it does not relax any gate. Every decision is
made by :func:`scalping.discovery.promotion.evaluate_promotion` and every write
goes through :func:`scalping.discovery.promotion.promote_with_evidence`, which
re-evaluates the policy inside the archive.

The locked final holdout is the expensive, one-shot resource. It is spent only
when all of the following are true:

* the evidence passes the production policy on every check **except** the
  missing ``locked_oos`` section;
* the caller explicitly sets ``spend_holdout=True``;
* a non-empty owner approval token is supplied (from a file, an environment
  variable or an in-process value);
* the candidate exists in the archive with an accepted trial on the same
  dataset checksum, settings digest and code version;
* the data artifact and bars behind the evidence still rebuild to the recorded
  dataset checksum.

If any precondition fails, the function returns a refusal and the holdout ledger
is untouched. If the unlock succeeds but the evaluation fails, the holdout stays
spent with no result: losing one evaluation is safer than silently repeating it.

Every path returns a :class:`PromoteResult` carrying the decision record: the
policy id/version/digest, the stable code list and the evidence identity. The
approval token is never stored, logged or echoed in that record.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.artifact import LoadedArrays
from scalping.datasets.builder import DatasetBundle
from scalping.discovery.archive import ArchiveError, DiscoveryArchive
from scalping.discovery.evidence import (
    CandidateEvidence,
    EvidenceError,
    evidence_checksum,
    evidence_from_dict,
    settings_digest,
    verify_evidence,
)
from scalping.discovery.holdout import (
    HoldoutConfig,
    HoldoutError,
    HoldoutLedger,
    HoldoutRecord,
    HoldoutSpec,
    attach_holdout,
    evaluate_holdout,
)
from scalping.discovery.ids import canonical_dataset_id
from scalping.discovery.persist import (
    PersistError,
    load_dataset_artifact,
    load_evidence_bundle,
    load_strategy_spec,
)
from scalping.discovery.promotion import (
    PRODUCTION_POLICIES,
    ExpectedIdentity,
    PromotionDecision,
    PromotionPolicy,
    PromotionRefused,
    evaluate_promotion,
    promote_with_evidence,
)
from scalping.discovery.reproduce import (
    ReproduceError,
    _manifest,
    _rebuild_dataset,
    spec_sidecar_path,
)
from scalping.features.build import FeatureFrame, build_features

__all__ = [
    "PROMOTE_STATUSES",
    "PromoteRequest",
    "PromoteResult",
    "promote_evidence",
]

#: The three outcomes a promotion attempt can have.
PROMOTE_STATUSES: Final[tuple[str, ...]] = ("accepted", "rejected", "refused")

#: Name of the eligibility policy derived from the target production policy.
#: It differs only in ``require_locked_oos`` and never promotes anything.
_ELIGIBILITY_SUFFIX: Final[str] = "-eligibility"

#: Ledger file kept next to the discovery archive.
_HOLDOUT_FILENAME: Final[str] = "holdout.sqlite"

Status = Literal["accepted", "rejected", "refused"]


@dataclass(frozen=True, slots=True)
class PromoteRequest:
    """Everything one promotion attempt must state explicitly.

    ``expected_identity`` is the **current** universe the evidence must belong
    to. Build it with :meth:`ExpectedIdentity.current` from the live process and
    the dataset manifest; do not copy the values out of the evidence file, or
    the freshness check compares the bundle with itself.

    ``approval_token`` may be a token string, a path to a token file, or
    ``env:NAME`` to read ``NAME`` from the environment. An empty or missing
    token is a refusal, never an implicit approval. The token is only checked
    for non-emptiness and is never written to the result or an exception.

    ``data_root`` is the bar store root that also contains ``datasets/<checksum>``
    (the same root ``discovery evidence`` writes). It is read only when the
    locked holdout has to be evaluated. ``settings`` must be the execution
    settings the evidence was built with; ``None`` means the built-in defaults,
    and the digest is compared with ``expected_identity`` before any feature is
    rebuilt.
    """

    archive_path: str | Path
    candidate_id: str
    evidence_path: str | Path
    scope: str
    expected_identity: ExpectedIdentity
    data_root: str | Path
    reason: str
    policy_id: str = "promotion-v1"
    spend_holdout: bool = False
    approval_token: str | Path | None = field(default=None, repr=False)
    dry_run: bool = False
    settings: Settings | None = None

    def __post_init__(self) -> None:
        for name in ("archive_path", "evidence_path", "data_root"):
            value = getattr(self, name)
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} must not be empty")
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{name} must be a path, got {type(value).__name__}")
        for name in ("candidate_id", "scope", "reason", "policy_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.expected_identity, ExpectedIdentity):
            raise TypeError(
                "expected_identity must be an ExpectedIdentity, "
                f"got {type(self.expected_identity).__name__}"
            )
        for name in ("spend_holdout", "dry_run"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        token = self.approval_token
        if token is not None and not isinstance(token, (str, Path)):
            raise TypeError("approval_token must be a str, a Path or None")
        if self.settings is not None and not isinstance(self.settings, Settings):
            raise TypeError(
                f"settings must be a Settings or None, got {type(self.settings).__name__}"
            )


@dataclass(frozen=True, slots=True)
class PromoteResult:
    """The decision record for one attempt; it is always produced.

    ``status == "rejected"`` means the policy ran and did not accept the
    evidence. ``status == "refused"`` means a governance or structural
    precondition failed before the policy could accept anything. ``codes`` is
    the stable machine-readable list in both cases.
    """

    status: Status
    scope: str
    candidate_id: str
    reason: str
    codes: tuple[str, ...]
    dry_run: bool
    policy_id: str | None
    policy_version: int | None
    policy_digest: str | None
    evidence_checksum: str | None = None
    dataset_id: str | None = None
    dataset_checksum: str | None = None
    decision: Mapping[str, Any] | None = None
    holdout: Mapping[str, Any] | None = None
    champion: Mapping[str, Any] | None = None
    approval_source: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "dry_run": self.dry_run,
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "codes": list(self.codes),
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
            "evidence_checksum": self.evidence_checksum,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            "decision": None if self.decision is None else dict(self.decision),
            "holdout": None if self.holdout is None else dict(self.holdout),
            "champion": None if self.champion is None else dict(self.champion),
            "approval_source": self.approval_source,
        }


class _Stop(Exception):
    """Internal control flow: stop with a written refusal."""

    def __init__(self, codes: tuple[str, ...], reason: str) -> None:
        super().__init__(reason)
        self.codes = codes
        self.reason = reason


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or type(exc).__name__


def _resolve_policy(policy_id: str) -> PromotionPolicy | None:
    """Return the registered production policy named ``policy_id``, or None."""
    wanted = policy_id.strip()
    for policy in PRODUCTION_POLICIES.values():
        if policy.policy_id == wanted:
            return policy
    return None


def _read_token(value: str | Path | None) -> tuple[str | None, str | None]:
    """Read the owner token and its source; never include the value in an error."""
    if value is None:
        return None, None
    if isinstance(value, Path):
        if not value.is_file():
            return None, "file"
        try:
            return value.read_text(encoding="utf-8").strip() or None, "file"
        except OSError:
            return None, "file"
    text = value.strip()
    if text.startswith("env:"):
        name = text[4:].strip()
        if not name:
            return None, "env"
        return os.environ.get(name, "").strip() or None, "env"
    try:
        candidate = Path(text)
        if len(text) < 4096 and candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8").strip() or None, "file"
            except OSError:
                return None, "file"
    except OSError:
        pass
    return text or None, "inline"


def _identity_checks(evidence: CandidateEvidence, request: PromoteRequest) -> None:
    """Refuse a bundle that is not the current candidate/dataset/settings."""
    if evidence.spec_hash != request.candidate_id:
        raise _Stop(
            ("candidate_mismatch",),
            f"evidence is for spec {evidence.spec_hash!r}, not candidate {request.candidate_id!r}",
        )
    identity = request.expected_identity
    try:
        verify_evidence(
            evidence,
            dataset_id=evidence.dataset_id,
            dataset_checksum=identity.dataset_checksum,
            spec_hash=request.candidate_id,
            code_version=identity.code_version,
            settings_digest=identity.settings_digest,
        )
    except EvidenceError as exc:
        raise _Stop(("identity_mismatch", exc.code), _short_error(exc)) from exc


def _champion_context(
    archive: DiscoveryArchive, scope: str
) -> tuple[Any, CandidateEvidence | None, str | None]:
    """Mirror the comparison inputs ``promote_with_evidence`` will use.

    An evidence champion contributes its stored bundle. An exploratory or
    manual champion has no evidence bridge and is replaced as an explicitly
    audited first evidence promotion. A missing stored bundle is a refusal, not
    an empty baseline.
    """
    champion = archive.champion(scope)
    if champion is None:
        return None, None, None
    champion_id = champion.candidate.candidate_id
    if champion.mode != "evidence":
        return champion, None, None
    record = archive.champion_evidence(scope)
    if record is None:
        raise _Stop(
            ("champion_evidence_missing",),
            f"champion {champion_id!r} of {scope!r} is evidence-mode but its stored "
            "bundle is missing",
        )
    try:
        loaded = evidence_from_dict(record.payload)
        if evidence_checksum(loaded) != record.checksum:
            raise _Stop(
                ("champion_checksum_mismatch",),
                f"stored champion evidence {record.checksum[:12]} does not re-derive",
            )
    except EvidenceError as exc:
        raise _Stop(("champion_invalid_evidence", exc.code), _short_error(exc)) from exc
    if loaded.spec_hash != champion_id:
        raise _Stop(
            ("champion_spec_mismatch",),
            f"stored champion evidence is for spec {loaded.spec_hash!r}, not {champion_id!r}",
        )
    return champion, loaded, champion_id


def _evaluate(
    evidence: CandidateEvidence,
    policy: PromotionPolicy,
    *,
    champion_evidence: CandidateEvidence | None,
    champion_spec_hash: str | None,
    identity: ExpectedIdentity,
) -> PromotionDecision:
    return evaluate_promotion(
        evidence,
        policy,
        champion_evidence=champion_evidence,
        champion_spec_hash=champion_spec_hash,
        expected_identity=identity,
    )


def _structural_precheck(
    archive: DiscoveryArchive,
    evidence: CandidateEvidence,
    candidate_id: str,
) -> None:
    """Refuse a promotion the archive cannot store before spending the holdout.

    This repeats the archive's own acceptance query so an attempt that would
    fail with ``no_accepted_trial`` or ``stale_trial_evidence`` never claims the
    one-shot evaluation slot. Dataset ids are compared canonically, so a search
    trial under ``validation:<manifest id>`` satisfies evidence that stores the
    manifest id; the checksum and settings digest are still compared exactly.
    """
    row = archive.connection.execute(
        "SELECT 1 FROM candidates WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    if row is None:
        raise _Stop(("unknown_candidate",), f"unknown candidate {candidate_id!r}")
    accepted = archive.accepted_trial(
        candidate_id, evidence.dataset_id, code_version=evidence.code_version
    )
    if accepted is None:
        raise _Stop(
            ("no_accepted_trial",),
            f"candidate {candidate_id!r} has no accepted trial on {evidence.dataset_id!r} "
            f"under code version {evidence.code_version!r}",
        )
    checksum = accepted.dataset_checksum
    digest = accepted.settings_digest
    if (
        checksum is None
        or digest is None
        or str(checksum) != evidence.dataset_checksum
        or str(digest) != evidence.settings_digest
    ):
        raise _Stop(
            ("stale_trial_evidence",),
            "the accepted trial does not carry the same dataset checksum and settings "
            "digest as the evidence bundle",
        )


def _holdout_record(ledger: HoldoutLedger, evidence: CandidateEvidence) -> HoldoutRecord | None:
    return ledger.get(evidence.dataset_id, evidence.dataset_checksum)


def _spec_for_holdout(evidence_path: Path) -> Any:
    sidecar = spec_sidecar_path(evidence_path)
    if not sidecar.is_file():
        raise _Stop(
            ("spec_missing",),
            f"the evidence bundle has no spec sidecar at {sidecar}; the holdout cannot "
            "be evaluated without the exact strategy",
        )
    try:
        return load_strategy_spec(sidecar)
    except PersistError as exc:
        raise _Stop((exc.code,), _short_error(exc)) from exc


def _holdout_data(
    evidence: CandidateEvidence,
    data_root: Path,
    settings: Settings,
) -> tuple[DatasetBundle, FeatureFrame]:
    """Rebuild the exact frame and dataset the evidence identity names.

    The artifact is verified twice: its per-array digests and its dataset
    checksum on load, then the bars are rebuilt into the recorded geometry and
    the checksum is compared again. A changed store therefore fails before the
    holdout is claimed.
    """
    try:
        loaded: LoadedArrays = load_dataset_artifact(data_root, evidence.dataset_checksum)
    except PersistError as exc:
        raise _Stop((exc.code,), _short_error(exc)) from exc
    try:
        manifest = _manifest(loaded.manifest)
    except ReproduceError as exc:
        raise _Stop(("dataset_manifest_invalid", exc.code), _short_error(exc)) from exc
    if canonical_dataset_id(manifest.dataset_id) != canonical_dataset_id(evidence.dataset_id):
        raise _Stop(
            ("dataset_mismatch",),
            f"artifact is dataset {manifest.dataset_id!r}, evidence names {evidence.dataset_id!r}",
        )
    if manifest.checksum != evidence.dataset_checksum:
        raise _Stop(
            ("dataset_checksum_mismatch",),
            f"artifact records {manifest.checksum!r}, evidence names {evidence.dataset_checksum!r}",
        )
    if (manifest.source, manifest.synthetic) != (evidence.source, evidence.synthetic):
        raise _Stop(
            ("provenance_mismatch",),
            "the artifact manifest and the evidence bundle disagree on source/synthetic",
        )
    bars = ParquetStore(data_root).read(evidence.symbol, evidence.timeframe)
    if bars.is_empty():
        raise _Stop(
            ("bars_missing",),
            f"no {evidence.symbol} {evidence.timeframe} bars under {data_root}",
        )
    try:
        features = build_features(bars, costs=settings.costs, groups=manifest.groups)
        bundle = _rebuild_dataset(features, manifest)
    except (ReproduceError, TypeError, ValueError) as exc:
        raise _Stop(("dataset_mismatch",), _short_error(exc)) from exc
    return bundle, features


def _promotion_payload(champion: Any) -> dict[str, Any] | None:
    if champion is None:
        return None
    payload = champion.to_dict()
    return {
        "candidate_id": payload["candidate"]["candidate_id"],
        "promotion_id": payload["promotion_id"],
        "mode": payload["mode"],
        "promoted_at": payload["promoted_at"],
        "evidence_checksum": payload["evidence_checksum"],
        "policy_id": payload["policy_id"],
        "policy_version": payload["policy_version"],
        "policy_digest": payload["policy_digest"],
    }


def _make_result(
    request: PromoteRequest,
    *,
    status: Status,
    codes: tuple[str, ...],
    reason: str,
    policy: PromotionPolicy | None,
    evidence: CandidateEvidence | None = None,
    decision: PromotionDecision | None = None,
    holdout: Mapping[str, Any] | None = None,
    champion: Any = None,
    approval_source: str | None = None,
) -> PromoteResult:
    decision_payload = None if decision is None else decision.to_dict()
    return PromoteResult(
        status=status,
        scope=request.scope,
        candidate_id=request.candidate_id,
        reason=reason,
        codes=codes,
        dry_run=request.dry_run,
        policy_id=None if policy is None else policy.policy_id,
        policy_version=None if policy is None else policy.version,
        policy_digest=None if policy is None else policy.digest,
        evidence_checksum=(
            decision.evidence_checksum
            if decision is not None and decision.evidence_checksum is not None
            else (None if evidence is None else evidence.checksum)
        ),
        dataset_id=None if evidence is None else evidence.dataset_id,
        dataset_checksum=None if evidence is None else evidence.dataset_checksum,
        decision=decision_payload,
        holdout=holdout,
        champion=_promotion_payload(champion),
        approval_source=approval_source,
    )


def _decision_codes(decision: PromotionDecision) -> tuple[str, ...]:
    return decision.codes if decision.codes else ("promotion_rejected",)


def promote_evidence(request: PromoteRequest) -> PromoteResult:
    """Evaluate one bundle and, only if every gate passes, promote it.

    The function never raises for a governed outcome: a bad request shape is a
    ``ValueError`` (programmer error), while a stale bundle, a policy rejection,
    a missing approval or a spent holdout is a :class:`PromoteResult` with
    ``status`` ``"rejected"`` or ``"refused"``. That keeps the CLI's decision
    record complete on every path.
    """
    if not isinstance(request, PromoteRequest):
        raise TypeError(f"request must be a PromoteRequest, got {type(request).__name__}")

    policy = _resolve_policy(request.policy_id)
    if policy is None:
        return _make_result(
            request,
            status="refused",
            codes=("non_production_policy",),
            reason=(
                f"policy {request.policy_id!r} is not a registered production policy; "
                "a real promotion refuses every non-production policy"
            ),
            policy=None,
        )

    try:
        evidence = load_evidence_bundle(request.evidence_path)
    except PersistError as exc:
        return _make_result(
            request,
            status="refused",
            codes=(exc.code,),
            reason=_short_error(exc),
            policy=policy,
        )

    try:
        _identity_checks(evidence, request)
    except _Stop as stop:
        return _make_result(
            request,
            status="refused",
            codes=stop.codes,
            reason=stop.reason,
            policy=policy,
            evidence=evidence,
        )

    archive_path = Path(request.archive_path)
    archive: DiscoveryArchive | None = None
    champion_evidence: CandidateEvidence | None = None
    champion_spec_hash: str | None = None
    pre_decision: PromotionDecision | None = None
    try:
        if request.dry_run and not archive_path.is_file():
            archive = None
        else:
            archive = DiscoveryArchive(archive_path)

        try:
            if archive is not None:
                _, champion_evidence, champion_spec_hash = _champion_context(archive, request.scope)
        except _Stop as stop:
            return _make_result(
                request,
                status="refused",
                codes=stop.codes,
                reason=stop.reason,
                policy=policy,
                evidence=evidence,
            )
        except ArchiveError as exc:
            return _make_result(
                request,
                status="refused",
                codes=("champion_invalid_evidence",),
                reason=_short_error(exc),
                policy=policy,
                evidence=evidence,
            )

        decision = _evaluate(
            evidence,
            policy,
            champion_evidence=champion_evidence,
            champion_spec_hash=champion_spec_hash,
            identity=request.expected_identity,
        )

        if request.dry_run:
            if decision.accepted:
                return _make_result(
                    request,
                    status="accepted",
                    codes=(),
                    reason="dry run: the production policy accepted the evidence; nothing "
                    "was written",
                    policy=policy,
                    evidence=evidence,
                    decision=decision,
                )
            return _make_result(
                request,
                status="rejected",
                codes=_decision_codes(decision),
                reason=decision.summary,
                policy=policy,
                evidence=evidence,
                decision=decision,
            )

        if decision.accepted:
            assert archive is not None
            return _finish_promotion(
                request,
                archive=archive,
                policy=policy,
                evidence=evidence,
                decision=decision,
                holdout=None,
            )

        # The only rejection the holdout can resolve is the missing section.
        if not policy.require_locked_oos or "missing_locked_oos" not in decision.codes:
            return _make_result(
                request,
                status="rejected",
                codes=_decision_codes(decision),
                reason=decision.summary,
                policy=policy,
                evidence=evidence,
                decision=decision,
            )

        # Reading the ledger is cheap and read-only. A spent holdout is refused
        # before any policy work, so a second attempt cannot even look like it
        # might re-evaluate the candidate, whether or not the caller remembered
        # to pass --spend-holdout.
        ledger_path = _ledger_path(archive_path)
        spent_record: HoldoutRecord | None = None
        if ledger_path.is_file():
            with HoldoutLedger(ledger_path) as existing_ledger:
                spent_record = _holdout_record(existing_ledger, evidence)
        if spent_record is not None and spent_record.status == "spent":
            return _make_result(
                request,
                status="refused",
                codes=("holdout_spent",),
                reason=(
                    f"the locked holdout for dataset {evidence.dataset_id!r} was already "
                    f"spent at {spent_record.spent_at}; one dataset build gets one evaluation"
                ),
                policy=policy,
                evidence=evidence,
                decision=decision,
                holdout=_holdout_payload(spent_record, ledger_path),
            )

        if not request.spend_holdout:
            return _make_result(
                request,
                status="refused",
                codes=("holdout_not_spent",),
                reason=(
                    "the production policy requires the locked holdout; pass "
                    "spend_holdout=True only with explicit owner approval"
                ),
                policy=policy,
                evidence=evidence,
                decision=decision,
            )

        token, source = _read_token(request.approval_token)
        if token is None:
            return _make_result(
                request,
                status="refused",
                codes=("approval_missing",),
                reason=(
                    "spending the one-shot locked holdout requires a non-empty owner approval token"
                ),
                policy=policy,
                evidence=evidence,
                decision=decision,
                approval_source=source,
            )

        # The archive must be able to store this promotion before we spend the
        # single evaluation. Otherwise a structural failure would destroy it.
        assert archive is not None
        try:
            _structural_precheck(archive, evidence, request.candidate_id)
        except _Stop as stop:
            return _make_result(
                request,
                status="refused",
                codes=stop.codes,
                reason=stop.reason,
                policy=policy,
                evidence=evidence,
                decision=decision,
                approval_source=source,
            )

        eligibility = replace(
            policy,
            policy_id=policy.policy_id + _ELIGIBILITY_SUFFIX,
            require_locked_oos=False,
        )
        pre_decision = _evaluate(
            evidence,
            eligibility,
            champion_evidence=champion_evidence,
            champion_spec_hash=champion_spec_hash,
            identity=request.expected_identity,
        )
        if not pre_decision.accepted:
            return _make_result(
                request,
                status="rejected",
                codes=_decision_codes(pre_decision),
                reason=(
                    "the candidate does not pass the non-holdout checks, so the locked "
                    f"holdout was not spent: {pre_decision.summary}"
                ),
                policy=policy,
                evidence=evidence,
                decision=pre_decision,
                approval_source=source,
            )
    finally:
        if archive is not None:
            archive.close()

    # Everything below can still refuse, but the archive is closed and the
    # holdout may be spent. Re-open the archive only through the promotion API.
    assert pre_decision is not None
    return _spend_and_promote(
        request,
        policy=policy,
        evidence=evidence,
        eligibility_decision=pre_decision,
        champion_evidence=champion_evidence,
        champion_spec_hash=champion_spec_hash,
        approval_source=source,
    )


def _ledger_path(archive_path: Path) -> Path:
    return archive_path.parent / _HOLDOUT_FILENAME


def _spend_and_promote(
    request: PromoteRequest,
    *,
    policy: PromotionPolicy,
    evidence: CandidateEvidence,
    eligibility_decision: PromotionDecision,
    champion_evidence: CandidateEvidence | None,
    champion_spec_hash: str | None,
    approval_source: str | None,
) -> PromoteResult:
    """Spend the sealed holdout once, attach it, then promote through the API."""
    archive_path = Path(request.archive_path)
    ledger_path = _ledger_path(archive_path)
    data_root = Path(request.data_root)
    settings = Settings() if request.settings is None else request.settings
    if settings_digest(settings) != request.expected_identity.settings_digest:
        return _make_result(
            request,
            status="refused",
            codes=("settings_mismatch",),
            reason=(
                "this runtime builds features with the default Settings(), whose digest "
                "is not the expected identity; refusing to evaluate another execution "
                "configuration"
            ),
            policy=policy,
            evidence=evidence,
            decision=eligibility_decision,
            approval_source=approval_source,
        )

    try:
        spec = _spec_for_holdout(Path(request.evidence_path))
    except _Stop as stop:
        return _make_result(
            request,
            status="refused",
            codes=stop.codes,
            reason=stop.reason,
            policy=policy,
            evidence=evidence,
            decision=eligibility_decision,
            approval_source=approval_source,
        )
    try:
        bundle, features = _holdout_data(evidence, data_root, settings)
    except _Stop as stop:
        return _make_result(
            request,
            status="refused",
            codes=stop.codes,
            reason=stop.reason,
            policy=policy,
            evidence=evidence,
            decision=eligibility_decision,
            approval_source=approval_source,
        )

    holdout_spec = HoldoutSpec(
        dataset_id=evidence.dataset_id,
        dataset_checksum=evidence.dataset_checksum,
        symbol=evidence.symbol,
        timeframe=evidence.timeframe,
        spec_hash=evidence.spec_hash,
        code_version=evidence.code_version,
        settings_digest=evidence.settings_digest,
        source=evidence.source,
        synthetic=evidence.synthetic,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = HoldoutLedger(ledger_path)
    try:
        record = _holdout_record(ledger, evidence)
        if record is not None and record.status == "spent":
            return _make_result(
                request,
                status="refused",
                codes=("holdout_spent",),
                reason=(
                    f"the locked holdout for dataset {evidence.dataset_id!r} was already "
                    f"spent at {record.spent_at}; one dataset build gets one evaluation"
                ),
                policy=policy,
                evidence=evidence,
                decision=eligibility_decision,
                holdout=_holdout_payload(record, ledger_path),
                approval_source=approval_source,
            )
        try:
            ledger.seal(
                holdout_spec,
                actor="promote-cli",
                reason="sealed for the explicit production promotion",
            )
            token = ledger.unlock(
                holdout_spec,
                decision=eligibility_decision,
                policy=policy,
                actor="owner-approved",
                reason=(
                    "explicit owner approval to spend the one-shot locked holdout; "
                    f"promotion reason: {request.reason}"
                ),
            )
        except HoldoutError as exc:
            current = _holdout_record(ledger, evidence)
            return _make_result(
                request,
                status="refused",
                codes=(exc.code,),
                reason=_short_error(exc),
                policy=policy,
                evidence=evidence,
                decision=eligibility_decision,
                holdout=None if current is None else _holdout_payload(current, ledger_path),
                approval_source=approval_source,
            )

        config = HoldoutConfig(
            code_version=evidence.code_version,
            symbol=evidence.symbol,
            timeframe=evidence.timeframe,
            window=evidence.policy.window,
            cost_stress=evidence.policy.cost_stress,
            bootstrap_samples=evidence.policy.bootstrap_samples,
            bootstrap_confidence=evidence.policy.bootstrap_confidence,
            seed=evidence.seed,
            settings=settings,
            features=features,
            dataset_id=evidence.dataset_id,
            dataset_checksum=evidence.dataset_checksum,
            source=evidence.source,
            synthetic=evidence.synthetic,
        )
        try:
            result = evaluate_holdout(
                bundle,
                spec,
                config,
                ledger=ledger,
                unlock_token=token,
            )
            attached = attach_holdout(evidence, result)
        except HoldoutError as exc:
            spent = _holdout_record(ledger, evidence)
            return _make_result(
                request,
                status="refused",
                codes=("holdout_evaluation_failed", exc.code),
                reason=(
                    f"the holdout was claimed and the evaluation failed ({_short_error(exc)}); "
                    "it stays spent with no result and is never retried"
                ),
                policy=policy,
                evidence=evidence,
                decision=eligibility_decision,
                holdout=None if spent is None else _holdout_payload(spent, ledger_path),
                approval_source=approval_source,
            )
        final_decision = _evaluate(
            attached,
            policy,
            champion_evidence=champion_evidence,
            champion_spec_hash=champion_spec_hash,
            identity=request.expected_identity,
        )
        holdout_payload = {
            "spent": True,
            "ledger": str(ledger_path),
            "result_checksum": result.checksum,
            "evidence_checksum": attached.checksum,
            "trades": result.trades,
            "test_rows": result.test_rows,
            "expectancy_r": result.expectancy_r,
        }
        if not final_decision.accepted:
            return _make_result(
                request,
                status="rejected",
                codes=_decision_codes(final_decision),
                reason=(
                    "the locked holdout is spent, but the attached bundle is rejected: "
                    f"{final_decision.summary}"
                ),
                policy=policy,
                evidence=attached,
                decision=final_decision,
                holdout=holdout_payload,
                approval_source=approval_source,
            )

        archive = DiscoveryArchive(archive_path)
        try:
            return _finish_promotion(
                request,
                archive=archive,
                policy=policy,
                evidence=attached,
                decision=final_decision,
                holdout=holdout_payload,
                approval_source=approval_source,
            )
        finally:
            archive.close()
    finally:
        ledger.close()


def _holdout_payload(record: HoldoutRecord, ledger_path: Path) -> dict[str, Any]:
    return {
        "spent": record.status == "spent",
        "status": record.status,
        "ledger": str(ledger_path),
        "dataset_id": record.dataset_id,
        "dataset_checksum": record.dataset_checksum,
        "sealed_at": record.sealed_at,
        "spent_at": record.spent_at,
        "policy_id": record.policy_id,
        "policy_version": record.policy_version,
    }


def _finish_promotion(
    request: PromoteRequest,
    *,
    archive: DiscoveryArchive,
    policy: PromotionPolicy,
    evidence: CandidateEvidence,
    decision: PromotionDecision,
    holdout: Mapping[str, Any] | None,
    approval_source: str | None = None,
) -> PromoteResult:
    """Call the one production promotion API and classify its outcome."""
    try:
        champion = promote_with_evidence(
            archive,
            request.candidate_id,
            evidence,
            policy,
            scope=request.scope,
            reason=request.reason,
            expected_identity=request.expected_identity,
        )
    except PromotionRefused as exc:
        text = _short_error(exc)
        if " was rejected by policy " in text:
            return _make_result(
                request,
                status="rejected",
                codes=_decision_codes(decision),
                reason=text,
                policy=policy,
                evidence=evidence,
                decision=decision,
                holdout=holdout,
                approval_source=approval_source,
            )
        code = text.split(":", 1)[0].strip() or "promotion_refused"
        referred = code in {
            "non_production_policy",
            "no_accepted_trial",
            "stale_trial_evidence",
            "unknown_candidate",
        }
        return _make_result(
            request,
            status="refused" if referred else "rejected",
            codes=(code,),
            reason=text,
            policy=policy,
            evidence=evidence,
            decision=decision,
            holdout=holdout,
            approval_source=approval_source,
        )
    except ArchiveError as exc:
        text = _short_error(exc)
        code = text.split(":", 1)[0].strip() or "archive_error"
        return _make_result(
            request,
            status="refused",
            codes=(code,),
            reason=text,
            policy=policy,
            evidence=evidence,
            decision=decision,
            holdout=holdout,
            approval_source=approval_source,
        )
    return _make_result(
        request,
        status="accepted",
        codes=(),
        reason=(
            f"policy {policy.policy_id} v{policy.version} accepted evidence "
            f"{evidence.checksum[:12]} and promoted candidate {request.candidate_id}"
        ),
        policy=policy,
        evidence=evidence,
        decision=decision,
        holdout=holdout,
        champion=champion,
        approval_source=approval_source,
    )
