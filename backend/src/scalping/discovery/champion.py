"""Read the paper-trading champion strategy out of the discovery archive.

The archive stores a canonical spec payload and records which candidate is the
champion of a scope. This module is the narrow read seam between them and the
paper path: it loads that payload, validates it as a
:class:`~scalping.strategy.dsl.nodes.StrategySpec` and hands the spec to a caller
that already knows how to run it (``scalping.paper.autonomous``).

What this module deliberately does not do
-----------------------------------------
* It does not decide who is champion. Promotion, replacement and rollback are
  :class:`~scalping.discovery.archive.DiscoveryArchive` operations with their own
  audit trail; this module only reads the result.
* It does not fall back. No champion, an unreadable payload, an invalid spec, a
  hash that disagrees with the stored id, or a champion promoted in exploratory
  or manual mode all raise. A hand-written strategy hidden behind a "default"
  would make an autonomous run silently stop testing what the search produced,
  which is the failure this seam exists to prevent.
* It does not import :mod:`scalping.paper`. The dependency stays one-way: paper
  depends on the DSL, discovery depends on the DSL and the archive, and nothing
  in discovery depends on paper.

Identity
--------
The candidate id is the sha256 of the stored canonical payload. The provider
recomputes the hash of the loaded spec and compares it with the archived id, so a
row whose payload was changed after registration is refused instead of traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scalping.discovery.archive import (
    DEFAULT_SCOPE,
    ChampionRecord,
    DiscoveryArchive,
)
from scalping.discovery.promotion import is_production_policy_digest
from scalping.strategy.dsl import StrategySpec
from scalping.strategy.dsl.nodes import DslValidationError

__all__ = [
    "ChampionError",
    "ChampionProvider",
    "ChampionSpec",
    "InvalidChampionSpec",
    "NoChampionError",
    "NoEvidenceChampion",
    "scope_for",
]


class ChampionError(Exception):
    """The archive cannot supply an executable champion strategy."""


class NoChampionError(ChampionError):
    """The scope has no champion; nothing has been promoted yet."""


class NoEvidenceChampion(ChampionError):
    """The champion exists but was not promoted through the evidence policy.

    An exploratory or manual champion is a research record, not a governed
    strategy. Handing it to the paper runner would trade a candidate that never
    passed the evidence gate.
    """


class InvalidChampionSpec(ChampionError):
    """The stored champion payload is not a valid, self-consistent strategy."""


@dataclass(frozen=True, slots=True)
class ChampionSpec:
    """One validated champion spec plus the provenance a ledger can record."""

    scope: str
    spec: StrategySpec
    candidate_id: str
    promotion_id: int
    promoted_at: str
    reason: str
    metrics: dict[str, float]
    mode: str = "evidence"
    evidence_checksum: str | None = None
    policy_id: str | None = None
    policy_version: int | None = None
    policy_digest: str | None = None

    @property
    def spec_hash(self) -> str:
        """Canonical sha256 of the spec; the strategy version for paper rows."""
        return self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "candidate_id": self.candidate_id,
            "spec_hash": self.spec_hash,
            "promotion_id": self.promotion_id,
            "promoted_at": self.promoted_at,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "mode": self.mode,
            "evidence_checksum": self.evidence_checksum,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_digest": self.policy_digest,
            "spec": self.spec.canonical_dict(),
        }


def scope_for(symbol: str, timeframe: str) -> str:
    """The conventional scope key for one market: ``<symbol>:<timeframe>``.

    Scopes keep independent champions, so promoting a strategy for ``XAUUSD:M5``
    never replaces the champion of ``EURUSD:M5``.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(timeframe, str) or not timeframe.strip():
        raise ValueError("timeframe must be a non-empty string")
    return f"{symbol.strip()}:{timeframe.strip()}"


def _require_scope(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"scope must be a str, got {type(value).__name__}")
    scope = value.strip()
    if not scope:
        raise ValueError("scope must not be empty")
    return scope


class ChampionProvider:
    """Turn ``archive.champion(scope)`` into a validated :class:`StrategySpec`.

    ``scope`` in the constructor is the default for both read methods; an
    explicit argument to :meth:`champion` or :meth:`spec` wins. Without either,
    the archive's :data:`~scalping.discovery.archive.DEFAULT_SCOPE` is used.
    """

    def __init__(self, archive: DiscoveryArchive, *, scope: str | None = None) -> None:
        if not isinstance(archive, DiscoveryArchive):
            raise TypeError(f"archive must be a DiscoveryArchive, got {type(archive).__name__}")
        self._archive = archive
        self._scope = None if scope is None else _require_scope(scope)

    @property
    def archive(self) -> DiscoveryArchive:
        """The archive this provider reads; it never writes to it."""
        return self._archive

    @property
    def scope(self) -> str:
        """The default scope, resolved."""
        return self._resolve(None)

    def champion(self, scope: str | None = None) -> ChampionSpec:
        """The champion spec of ``scope`` with its promotion provenance.

        Raises :class:`NoChampionError` when nothing was promoted,
        :class:`NoEvidenceChampion` when the champion came from the exploratory
        or manual path, and :class:`InvalidChampionSpec` when the stored payload
        does not load, fails DSL validation or does not hash to its archived id.
        """
        resolved = self._resolve(scope)
        record = self._archive.champion(resolved)
        if record is None:
            raise NoChampionError(
                f"no champion in scope {resolved!r}; promote a validated candidate first"
            )
        if record.mode != "evidence":
            raise NoEvidenceChampion(
                f"champion {record.candidate.candidate_id!r} in scope {resolved!r} was "
                f"promoted in {record.mode!r} mode; only an evidence-mode champion may "
                "trade paper"
            )
        if (
            not is_production_policy_digest(record.policy_digest)
            or record.policy_id is None
            or record.policy_version is None
        ):
            raise NoEvidenceChampion(
                f"champion {record.candidate.candidate_id!r} in scope {resolved!r} was "
                "promoted by a policy that is not in the production allow-list"
            )
        spec = self._load(record)
        return ChampionSpec(
            scope=record.scope,
            spec=spec,
            candidate_id=record.candidate.candidate_id,
            promotion_id=record.promotion_id,
            promoted_at=record.promoted_at,
            reason=record.reason,
            metrics=dict(record.metrics),
            mode=record.mode,
            evidence_checksum=record.evidence_checksum,
            policy_id=record.policy_id,
            policy_version=record.policy_version,
            policy_digest=record.policy_digest,
        )

    def spec(self, scope: str | None = None) -> StrategySpec:
        """Just the validated spec, for a caller that only needs to run it."""
        return self.champion(scope).spec

    def _resolve(self, scope: str | None) -> str:
        if scope is not None:
            return _require_scope(scope)
        if self._scope is not None:
            return self._scope
        return DEFAULT_SCOPE

    @staticmethod
    def _load(record: ChampionRecord) -> StrategySpec:
        candidate_id = record.candidate.candidate_id
        try:
            spec = StrategySpec.from_dict(record.canonical)
        except (DslValidationError, ValueError, TypeError) as exc:
            raise InvalidChampionSpec(
                f"champion {candidate_id!r} in scope {record.scope!r} does not load as a "
                f"strategy: {exc}"
            ) from exc
        digest = spec.canonical_hash()
        if digest != candidate_id:
            raise InvalidChampionSpec(
                f"champion {candidate_id!r} in scope {record.scope!r} hashes to {digest!r}; "
                "the stored payload does not match its archived identity"
            )
        return spec
