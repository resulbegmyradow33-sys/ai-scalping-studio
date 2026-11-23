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

__all__ = [
    "JOB_STATUSES",
    "SCHEMA_VERSION",
    "Champion",
    "DatasetRecord",
    "DuplicateError",
    "JobRecord",
    "ModelRecord",
    "ModelRegistry",
    "PromotionRecord",
    "RegistryError",
    "UnknownDatasetError",
    "UnknownModelError",
]
