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
    "JOB_STATUSES",
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
    "Orchestrator",
    "OrchestratorStatus",
    "PermanentJobError",
    "Recovery",
    "ResourceGuard",
    "RetryPolicy",
    "RunOutcome",
    "ScanOutcome",
    "StoreError",
    "Trigger",
    "TriggerSource",
    "UnknownJobError",
    "scan_once",
]
