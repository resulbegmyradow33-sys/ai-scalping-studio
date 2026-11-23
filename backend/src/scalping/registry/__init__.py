"""Persistent local registry for datasets, models, promotions and jobs.

Metadata lives in one SQLite file; Parquet datasets and model checkpoints stay
on disk and the registry stores their paths and digests. The champion is the
newest promotion, so promotion and rollback leave a queryable audit trail.
"""

from scalping.registry.db import (
    SCHEMA_VERSION,
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
    "SCHEMA_VERSION",
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
