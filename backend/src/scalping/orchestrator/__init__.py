"""Persistent learning orchestrator: job queue, controls and trigger sources.

Phase 18 slice one. The queue and its audit log are real code with tests; the
handlers that do the learning are injected by the caller, and the ML, DSL and
paper-trading pipelines are wired in later slices once they merge. See
``docs/orchestrator.md`` for the lifecycle, the controls and the honest limits.
"""

from scalping.orchestrator.core import (
    CONTROL_EVENTS,
    AllowAllGuard,
    Handler,
    HandlerResult,
    JobContext,
    Orchestrator,
    OrchestratorStatus,
    PermanentJobError,
    ResourceGuard,
    RetryPolicy,
    RunOutcome,
)
from scalping.orchestrator.resources import (
    EXPLORATORY_PROFILE,
    M2_8GB_PROFILE,
    MeasuredResourceGuard,
    Measurement,
    Probe,
    ResourceLimits,
    ResourceProbeError,
    ResourceSample,
    ResourceSnapshot,
    SystemProbe,
    rss_mb_from_ru_maxrss,
)
from scalping.orchestrator.store import (
    BLOCKED_DEPENDENCY_STATUSES,
    JOB_STATUSES,
    SCHEMA_VERSION,
    TERMINAL_STATUSES,
    BlockedJob,
    Event,
    Job,
    JobStateError,
    JobStats,
    JobStore,
    Recovery,
    StoreError,
    UnknownJobError,
)
from scalping.orchestrator.triggers import (
    FileTriggerSource,
    ScanOutcome,
    Trigger,
    TriggerSource,
    scan_once,
)

__all__ = [
    "BLOCKED_DEPENDENCY_STATUSES",
    "CONTROL_EVENTS",
    "EXPLORATORY_PROFILE",
    "JOB_STATUSES",
    "M2_8GB_PROFILE",
    "SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "AllowAllGuard",
    "BlockedJob",
    "Event",
    "FileTriggerSource",
    "Handler",
    "HandlerResult",
    "Job",
    "JobContext",
    "JobStateError",
    "JobStats",
    "JobStore",
    "MeasuredResourceGuard",
    "Measurement",
    "Orchestrator",
    "OrchestratorStatus",
    "PermanentJobError",
    "Probe",
    "Recovery",
    "ResourceGuard",
    "ResourceLimits",
    "ResourceProbeError",
    "ResourceSample",
    "ResourceSnapshot",
    "RetryPolicy",
    "RunOutcome",
    "ScanOutcome",
    "StoreError",
    "SystemProbe",
    "Trigger",
    "TriggerSource",
    "UnknownJobError",
    "rss_mb_from_ru_maxrss",
    "scan_once",
]
