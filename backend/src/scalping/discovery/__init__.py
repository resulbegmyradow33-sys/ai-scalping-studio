"""Autonomous strategy discovery: candidate protocols, archive and search loop.

The strategy representation is injected. This package owns the evidence chain
(canonical identity, lineage, trial history, decisions) and a bounded search
loop; a DSL implementation supplies generation, mutation and validation, and an
evaluator supplies simulation results. Nothing here executes generated code.
"""

from scalping.discovery.archive import (
    DECISIONS,
    DEFAULT_SCOPE,
    SCHEMA_VERSION,
    ArchiveError,
    CandidateRecord,
    ChampionComparison,
    ChampionPromotion,
    ChampionRecord,
    DecisionRecord,
    DiscoveryArchive,
    TrialRecord,
    UnknownCandidateError,
)
from scalping.discovery.protocols import (
    Candidate,
    Evaluator,
    Factory,
    TrialResult,
    canonical_digest,
    canonical_json,
)
from scalping.discovery.search import STOP_REASONS, SearchConfig, SearchResult, evolve

__all__ = [
    "DECISIONS",
    "DEFAULT_SCOPE",
    "SCHEMA_VERSION",
    "STOP_REASONS",
    "ArchiveError",
    "Candidate",
    "CandidateRecord",
    "ChampionComparison",
    "ChampionPromotion",
    "ChampionRecord",
    "DecisionRecord",
    "DiscoveryArchive",
    "Evaluator",
    "Factory",
    "SearchConfig",
    "SearchResult",
    "TrialRecord",
    "TrialResult",
    "UnknownCandidateError",
    "canonical_digest",
    "canonical_json",
    "evolve",
]
