"""Authoritative search exposure derived from the archive, not from a claim.

A candidate evidence bundle carries ``trial_count``: how many evaluations the
search spent before it kept this winner. That field is written by the caller,
and a caller can understate it. ``CandidateEvidence`` is checksummed, but the
checksum only proves the bundle was not edited after it was built; it says
nothing about whether the number was true when it was built.

This module closes that gap for the one part the archive can answer: given a
spec, a dataset and a code version, how many trials were actually recorded? The
archive is the evidence log, so its row count is the denominator. A
self-declared count below the archive count is refused with
``trial_count_understated``; a count at or above it is replaced with the
archive's number so the bundle carries the observed exposure.

Honest limits
-------------
* The archive only sees what was written to it. Trials run in another process,
  deleted rows or a search that never used an archive are invisible. The
  derived number is a floor on the true exposure, never a ceiling.
* Search-wide counts follow ``(dataset_id, code_version)``. Two unrelated runs
  that share a dataset and code version are counted together. That can only
  make exposure look larger, which is the safe direction for a
  multiple-testing guard. Dataset ids are compared through
  :func:`scalping.discovery.ids.canonical_dataset_id`, so the search's
  ``validation:<id>`` rows and evidence's ``<id>`` rows are one series while
  the stored ``per_dataset`` keys keep their original spelling.
* No policy decision is made here. This module returns exposure and rewrites
  ``trial_count``; ``scalping.discovery.promotion`` still owns every promotion
  threshold and refusal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from scalping.discovery.archive import DiscoveryArchive, TrialStats
from scalping.discovery.evidence import CandidateEvidence, evidence_checksum
from scalping.discovery.ids import canonical_dataset_id

__all__ = [
    "ExposureError",
    "SearchExposure",
    "apply_to_evidence",
    "exposure_for",
]


class ExposureError(ValueError):
    """Exposure cannot be derived or applied; ``code`` is machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True, slots=True)
class SearchExposure:
    """The archive's answer for one candidate on one dataset build.

    ``trial_count`` is the number of recorded trial rows for the whole search
    series (the anchor's dataset and code version), not just the anchor
    candidate. ``search_budget`` is the observed floor on the run's cap: the
    archive stores no declared budget, so the only defensible number is the
    attempts it can prove. ``accepted``/``rejected`` split those rows, and
    ``distinct_candidates`` says how many different hypotheses produced them.
    ``per_dataset`` is the same total broken down by dataset id.
    """

    spec_hash: str
    dataset_id: str
    code_version: str
    trial_count: int
    search_budget: int
    distinct_candidates: int
    accepted: int
    rejected: int
    per_dataset: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in ("spec_hash", "dataset_id", "code_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty str")
        for name in (
            "trial_count",
            "search_budget",
            "distinct_candidates",
            "accepted",
            "rejected",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.trial_count < 1:
            raise ValueError("trial_count must be >= 1; an empty archive is not exposure")
        if self.search_budget < self.trial_count:
            raise ValueError("search_budget cannot be below trial_count")
        if self.accepted + self.rejected != self.trial_count:
            raise ValueError("accepted + rejected must equal trial_count")
        if not 1 <= self.distinct_candidates <= self.trial_count:
            raise ValueError("distinct_candidates must be in [1, trial_count]")
        if isinstance(self.per_dataset, Mapping):
            object.__setattr__(
                self, "per_dataset", MappingProxyType(dict(sorted(self.per_dataset.items())))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "dataset_id": self.dataset_id,
            "code_version": self.code_version,
            "trial_count": self.trial_count,
            "search_budget": self.search_budget,
            "distinct_candidates": self.distinct_candidates,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "per_dataset": dict(sorted(self.per_dataset.items())),
        }


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty str")
    return value


def exposure_for(
    archive: DiscoveryArchive,
    spec_hash: str,
    dataset_id: str,
    code_version: str,
) -> SearchExposure:
    """Derive the authoritative exposure for one candidate's evidence claim.

    The candidate must exist and must have been evaluated on this dataset with
    this code version; otherwise there is no archive evidence for the claim and
    :class:`ExposureError` ``missing_trial_evidence`` is raised. The returned
    ``trial_count`` is the number of trial rows in the archive for the same
    dataset and code version, including every other candidate the search tried.
    """
    if not isinstance(archive, DiscoveryArchive):
        raise TypeError(f"archive must be a DiscoveryArchive, got {type(archive).__name__}")
    spec_hash = _text(spec_hash, "spec_hash")
    dataset_id = _text(dataset_id, "dataset_id")
    code_version = _text(code_version, "code_version")

    stats: TrialStats = archive.trial_stats(
        spec_hash, dataset_id=dataset_id, code_version=code_version
    )
    if stats.total_attempts < 1 or stats.spec_attempts < 1:
        raise ExposureError(
            "missing_trial_evidence",
            f"no recorded trial for spec {spec_hash!r} on dataset {dataset_id!r} "
            f"with code version {code_version!r}",
        )
    return SearchExposure(
        spec_hash=spec_hash,
        dataset_id=dataset_id,
        code_version=code_version,
        trial_count=stats.total_attempts,
        search_budget=stats.total_attempts,
        distinct_candidates=stats.distinct_candidates,
        accepted=stats.accepted,
        rejected=stats.rejected,
        per_dataset=stats.per_dataset,
    )


def apply_to_evidence(
    evidence: CandidateEvidence,
    exposure: SearchExposure,
) -> CandidateEvidence:
    """Return a new bundle carrying ``exposure.trial_count``.

    The caller's claim is accepted only when it is **not lower** than the
    archive count. A lower number is a refusal (``trial_count_understated``):
    replacing it quietly would hide a lie, and accepting it would leave the
    multiple-testing guard reading a number the archive disproves. A claim at or
    above the archive count is reduced to the archive count, because the archive
    is the evidence.

    The bundle is frozen, so the input is never mutated. The returned bundle is
    a new object with a recomputed checksum; ``created_at`` and every other
    field are preserved. A bundle whose stored checksum does not match its
    contents is refused before it can be re-signed.
    """
    if type(evidence) is not CandidateEvidence:
        raise TypeError(f"evidence must be a CandidateEvidence, got {type(evidence).__name__}")
    if not isinstance(exposure, SearchExposure):
        raise TypeError(f"exposure must be a SearchExposure, got {type(exposure).__name__}")
    for name in ("spec_hash", "dataset_id", "code_version"):
        claimed = getattr(evidence, name)
        archived = getattr(exposure, name)
        if name == "dataset_id":
            same = canonical_dataset_id(claimed) == canonical_dataset_id(archived)
        else:
            same = claimed == archived
        if not same:
            raise ExposureError(
                "exposure_mismatch",
                f"bundle {name} {claimed!r} does not match exposure {archived!r}",
            )
    if evidence.checksum != evidence_checksum(evidence):
        raise ExposureError(
            "checksum_mismatch",
            f"bundle checksum {evidence.checksum!r} does not match its contents",
        )
    if evidence.trial_count < exposure.trial_count:
        raise ExposureError(
            "trial_count_understated",
            f"bundle claims {evidence.trial_count} trials but the archive records "
            f"{exposure.trial_count}",
        )

    updated = replace(
        evidence,
        trial_count=exposure.trial_count,
        search_budget=max(evidence.search_budget, exposure.trial_count),
        checksum="",
    )
    return replace(updated, checksum=evidence_checksum(updated))
