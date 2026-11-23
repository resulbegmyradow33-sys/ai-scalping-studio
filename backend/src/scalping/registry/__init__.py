"""Persistent local registry for datasets, models, promotions and jobs.

Metadata lives in one SQLite file; Parquet datasets and model checkpoints stay
on disk and the registry stores their paths and digests. The champion is the
newest promotion in a scope, so promotion and rollback leave a queryable audit
trail and each scope keeps an independent champion.
"""

from scalping.registry.db import (
    JOB_STATUSES,
    SCHEMA_VERSION,
    Champion,
    DatasetRecord,
    DuplicateError,
    JobRecord,
    ModelRecord,
    ModelRegistry,
    PromotionRecord,
    RegistryError,
    UnknownDatasetError,
    UnknownModelError,
)
from scalping.registry.drift import (
    DEFAULT_PSI_THRESHOLD,
    DriftReport,
    FeatureDrift,
    PredictionDrift,
    drift_from_datasets,
    ks_statistic,
    population_stability_index,
    prediction_drift,
)
from scalping.registry.gate import (
    GateCriteria,
    GateDecision,
    PromotionRefused,
    evaluate_candidate,
    promote_candidate,
)

__all__ = [
    "DEFAULT_PSI_THRESHOLD",
    "JOB_STATUSES",
    "SCHEMA_VERSION",
    "Champion",
    "DatasetRecord",
    "DriftReport",
    "DuplicateError",
    "FeatureDrift",
    "GateCriteria",
    "GateDecision",
    "JobRecord",
    "ModelRecord",
    "ModelRegistry",
    "PredictionDrift",
    "PromotionRecord",
    "PromotionRefused",
    "RegistryError",
    "UnknownDatasetError",
    "UnknownModelError",
    "drift_from_datasets",
    "evaluate_candidate",
    "ks_statistic",
    "population_stability_index",
    "prediction_drift",
    "promote_candidate",
]
