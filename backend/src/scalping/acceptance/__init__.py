"""End-to-end acceptance harness over deterministic synthetic data.

This package wires the real modules together — store/validate, dataset build,
bounded discovery, evidence, promotion policy, optional holdout, paper advance
and matured outcomes — and reports what actually happened. It is an engineering
acceptance check, not market evidence: the bars are synthetic, the holdout is
usually unavailable, and the report's ``claims`` block keeps the distinction
explicit.
"""

from scalping.acceptance.run import (
    ACCEPTANCE_FIXTURE_SOURCE,
    DISCOVERY_CANDIDATE_SOURCE,
    STAGE_NAMES,
    AcceptanceConfig,
    AcceptanceError,
    AcceptanceResult,
    AcceptanceStage,
    acceptance_spec,
    run_acceptance,
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
