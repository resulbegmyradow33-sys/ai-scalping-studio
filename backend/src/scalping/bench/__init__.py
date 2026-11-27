"""Repeatable timing and resource measurements for the autonomous pipeline.

This package is a measurement harness, not an acceptance gate. It runs named
stages on synthetic data by default, records wall time, CPU time, high-water
RSS growth and optional temp-directory disk growth, and writes one JSON report.
The report carries an explicit ``claims`` block so a reader cannot mistake a
local timing for validation or profitability evidence.

Modules
-------
``budget``
    :class:`RssBudget`, a reusable process-RSS stop signal.
``pipeline``
    :func:`measure_stage` and :class:`StageMeasurement`.
``suite``
    :func:`run_suite`, the JSON report, and the built-in stage factories.
"""

from scalping.bench.budget import MemoryBudgetError, RssBudget, process_rss_mb
from scalping.bench.pipeline import StageMeasurement, measure_stage
from scalping.bench.suite import (
    CLAIMS_SCHEMA,
    REPORT_SCHEMA,
    Stage,
    StageError,
    dataset_build_stage,
    discovery_smoke_stage,
    evidence_build_stage,
    features_build_stage,
    paper_advance_stage,
    run_suite,
    walk_forward_stage,
)

__all__ = [
    "CLAIMS_SCHEMA",
    "REPORT_SCHEMA",
    "MemoryBudgetError",
    "RssBudget",
    "Stage",
    "StageError",
    "StageMeasurement",
    "dataset_build_stage",
    "discovery_smoke_stage",
    "evidence_build_stage",
    "features_build_stage",
    "measure_stage",
    "paper_advance_stage",
    "process_rss_mb",
    "run_suite",
    "walk_forward_stage",
]
