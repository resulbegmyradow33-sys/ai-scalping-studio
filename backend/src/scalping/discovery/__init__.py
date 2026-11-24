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
from scalping.discovery.evidence import (
    AggregateEvidence,
    CandidateEvidence,
    EvidenceConfig,
    EvidenceError,
    EvidencePolicy,
    FoldEvidence,
    RegimeEvidence,
    StressEvidence,
    bootstrap_ci,
    build_evidence,
    verify_evidence,
)
from scalping.discovery.protocols import (
    Candidate,
    Evaluator,
    Factory,
    TrialResult,
    canonical_digest,
    canonical_json,
)
from scalping.discovery.run import (
    BASELINES,
    DiscoveryRunConfig,
    DiscoveryRunError,
    DiscoveryRunResult,
    decide,
    run_discovery,
)
from scalping.discovery.search import STOP_REASONS, SearchConfig, SearchResult, evolve

__all__ = [
    "BASELINES",
    "DECISIONS",
    "DEFAULT_SCOPE",
    "SCHEMA_VERSION",
    "STOP_REASONS",
    "AggregateEvidence",
    "ArchiveError",
    "Candidate",
    "CandidateEvidence",
    "CandidateRecord",
    "ChampionComparison",
    "ChampionPromotion",
    "ChampionRecord",
    "DecisionRecord",
    "DiscoveryArchive",
    "DiscoveryRunConfig",
    "DiscoveryRunError",
    "DiscoveryRunResult",
    "Evaluator",
    "EvidenceConfig",
    "EvidenceError",
    "EvidencePolicy",
    "Factory",
    "FoldEvidence",
    "RegimeEvidence",
    "SearchConfig",
    "SearchResult",
    "StressEvidence",
    "TrialRecord",
    "TrialResult",
    "UnknownCandidateError",
    "bootstrap_ci",
    "build_evidence",
    "canonical_digest",
    "canonical_json",
    "decide",
    "evolve",
    "run_discovery",
    "verify_evidence",
]
