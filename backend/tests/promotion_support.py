"""Test-only promotion helpers.

The production module intentionally has no test seed path and its production
registry is a read-only mapping. These helpers call the private promotion core
with ``require_production=False`` and provide a scoped registry injection; they
are never imported by application code.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from scalping.discovery import promotion as promotion_module
from scalping.discovery.archive import ChampionRecord, DiscoveryArchive
from scalping.discovery.evidence import CandidateEvidence
from scalping.discovery.promotion import ExpectedIdentity, PromotionPolicy


def promote_for_tests(
    archive: DiscoveryArchive,
    candidate_id: str,
    evidence: CandidateEvidence,
    policy: PromotionPolicy,
    *,
    scope: str,
    reason: str,
    expected_identity: ExpectedIdentity,
) -> ChampionRecord:
    return promotion_module._promote_with_evidence(
        archive,
        candidate_id,
        evidence,
        policy,
        scope=scope,
        reason=reason,
        expected_identity=expected_identity,
        require_production=False,
    )


@contextmanager
def registered_test_policies(*policies: PromotionPolicy) -> Iterator[None]:
    with promotion_module._test_policies(*policies):
        yield


__all__: list[str] = ["promote_for_tests", "registered_test_policies"]
