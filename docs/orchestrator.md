# Learning orchestrator

Phase 18 slice one: a persistent job queue, the loop that runs it, owner
controls and trigger sources. The queue, its audit log and the controls are real
code with tests. Slice two (P3 slice 1) registers five production handlers:
document ingestion, dataset build, discovery, paper advance and outcome
analysis. There is still no scheduler thread and no measured resource guard.

This slice answers one question honestly: *can the system remember, order,
refuse, retry and explain work across a restart?* The answer is tested. The
handlers answer *what* work exists for the first research cycle; scheduling and
resource measurement remain later changes.

## Files

| File | Role |
|---|---|
| `backend/src/scalping/orchestrator/store.py` | SQLite queue + append-only events + persisted state |
| `backend/src/scalping/orchestrator/core.py` | Handler registry, controls, claim/run/fail/retry loop |
| `backend/src/scalping/orchestrator/triggers.py` | `TriggerSource` protocol, `FileTriggerSource`, `scan_once` |
| `backend/src/scalping/orchestrator/handlers.py` | `HandlerContext`, payload validators, the five pipeline handlers |
| `backend/tests/test_orchestrator_store.py` | Queue semantics, transitions, recovery |
| `backend/tests/test_orchestrator_core.py` | Loop, controls, heavy job, guard, triggers |
| `backend/tests/test_orchestrator_handlers.py` | Real handlers on synthetic stores and temporary roots |

## What is not in this slice

- **No scheduler thread.** `run_once()` processes one unit of work;
  `run_until_idle()` drains for a CLI or a test. A daemon loop is a later change
  and would not alter these semantics. Nothing calls the handlers on a timer.
- **No measured resource guard.** `ResourceGuard` is a veto hook, not a monitor.
  The default `AllowAllGuard` never refuses. `HEAVY_HANDLER_KINDS` names the two
  heavy kinds so a caller can enforce the M2 one-heavy-job rule, but peak RSS,
  CPU and disk are not measured yet.
- **No promotion handler.** Discovery only records `no_trade`, `no_change` or
  `rejected`. `paper.advance` reads a champion that a reviewed path promoted;
  neither handler can promote, replace or roll back one.
- **No training handler.** Training on locked OOS is out of scope for this
  slice, and no handler here fits a model.
- **No retention policy.** Jobs and events are never deleted, which is correct
  for an experiment history and will eventually need an archive step.
- **No API/UI.** State is queryable through `JobStore` and
  `Orchestrator.status()`; exposing it is future work.
- **One writer at a time.** One connection per thread, mirroring the registry and
  the Mac mini's "one heavy job" rule. Two processes are safe through
  `BEGIN IMMEDIATE`, not fast.

## Job lifecycle

```
enqueue ──► queued ──claim──► running ──complete──► succeeded
              │                 │
              │                 ├──fail────────────► failed
              │                 ├──retry───────────► queued (later scheduled_at)
              │                 ├──defer (guard)───► queued (attempt not spent)
              │                 ├──cancel──────────► cancelled
              │                 └──process restart─► interrupted
              │                                        │ attempts remain
              │                                        └──► queued
              ├──cancel────────────────────────────────► cancelled
              └──dependency failed──────────────────────► skipped
```

| Status | Meaning | Terminal |
|---|---|---|
| `queued` | Waiting for its schedule, its dependencies or a free worker | no |
| `running` | Claimed; exactly one process owns it | no |
| `succeeded` | Handler reported success; `result_json` holds its document | yes |
| `failed` | Handler failed and no retry is left, or it failed permanently | yes |
| `cancelled` | Owner cancelled it, queued or running | yes |
| `interrupted` | The process died while it ran and no attempt was left | yes |
| `skipped` | A dependency can never succeed; `error` says which | yes |

`attempts` counts starts, not failures: a job claimed once has `attempts == 1`.
With `max_retries = 2` a job may start three times.

Every transition is written to `events` in the same transaction as the row
change, so the audit trail cannot disagree with the state. If an event insert
fails, the transition rolls back too.

## Enqueue and dedupe

`enqueue` returns `(job, created)`. `dedupe_key` is `UNIQUE`; a second call with
an existing key returns the existing row untouched and records a
`deduplicated` event. That is what lets a scanner run every minute without
opening the same work twice.

Dependencies are validated at enqueue time: an unknown dependency id is
rejected now instead of leaving a job that can never run. Enqueue does not
require the parent to be *finished*, only to exist.

## Claim order

One `BEGIN IMMEDIATE` transaction selects and updates the next job:

1. `status = 'queued'` and `scheduled_at <= now`;
2. kind not in `exclude_kinds` (research disabled);
3. every dependency `succeeded` — otherwise the job waits;
4. if its kind is heavy, no other heavy kind is `running`;
5. order: `priority DESC`, `scheduled_at ASC`, `id ASC`.

Steps 1–4 are filters, step 5 picks. The update and the `started` event happen
in the same transaction, so a second worker sees the row as running or still
queued, never both. A claim clears `error`: the previous attempt's reason
survives in the event log, where history belongs.

## Dependencies and skip semantics

A dependency in `failed`, `cancelled`, `interrupted` or `skipped` can never
succeed, so its dependents are resolved rather than starved. `run_once` checks
for blocked jobs **before** claiming and skips at most one per call:

```
dependency 7 is failed → job 8 skipped with error "skipped: dependency 7 is failed"
```

The skip is a transition with an event, not a retry, so it appears once and
stays. A chain unwinds one link per iteration: the grandchild sees its parent
`skipped` on the next pass. `run_until_idle` finishes the chain.

Skips ignore `scheduled_at` on purpose: a failed dependency is terminal, so
waiting for the child's nominal start time would only delay the explanation.

With research disabled, research jobs are not skipped either. The owner asked
for that class of work to pause, so it stays `queued` with its history intact.

## Retries and backoff

`RetryPolicy(base_seconds=30, factor=2, max_seconds=900)` computes
`base * factor ** (attempt - 1)`, capped. `retry` puts the job back to `queued`
with `scheduled_at = now + delay` and keeps `attempts`. A job is retried only
while `attempts <= max_retries`; the store enforces that bound so a bug in a
caller cannot loop work forever.

Failures that a retry cannot fix are `permanent`:

- a handler raises `PermanentJobError`;
- a handler returns `HandlerResult.failure(..., permanent=True)`;
- no handler is registered for the kind;
- the handler's return is not a `HandlerResult`, mapping or `None`;
- the success result is not JSON-serialisable.

The last three are deployment or contract errors, not transient conditions, so
they fail on the first attempt. Any other exception is a retryable failure whose
detail is `TypeName: message`.

## One heavy job and the resource guard

A "heavy" kind is named by the caller (`heavy_kinds`). The store refuses to
claim a heavy job while another heavy job is `running`; due light jobs are
claimed in the meantime. This is the M2 rule expressed at the queue, not in the
handlers.

After a claim and before the handler, `guard.can_start(kind)` gets a veto. A
refusal does not fail the job and does not spend an attempt: the job is
deferred after `guard_retry_seconds` (60 by default), `attempts` is decremented
back, and the guard's reason is stored in `jobs.error` with `resource_refused`
and `deferred` events. Waiting for RAM cannot exhaust a job's retry budget,
which a retry-based refusal would. The refusal is visible, not a silent drop.

## Owner controls

| Method | Effect | Persisted key |
|---|---|---|
| `pause(reason)` / `resume()` | Claim nothing while paused | `orchestrator.paused` |
| `safe_mode(True/False)` | Claim nothing at all, even light jobs | `orchestrator.safe_mode` |
| `disable_research(True/False)` | Research kinds stay queued; other kinds run | `orchestrator.research_disabled` |
| `cancel(job_id, reason)` | Queued job cancelled; running job's result discarded | job row |

Controls are written to the `state` table with their reason, and read on every
iteration, so they survive a restart and two orchestrators over one database
agree. A refused claim writes a `claim_refused` event naming the control and its
reason — the answer to "why did nothing run?".

`safe_mode` is checked before `pause`; both before any claim. A running handler
is not interrupted by a control switch: the orchestrator compares the row after
the handler returns and discards a result for a cancelled job
(`cancelled_while_running`). That is cooperative cancellation, not a kill.

## Crash recovery

`Orchestrator.start()` calls `recover_interrupted()` before any work:

- every `running` row becomes `interrupted` with an `interrupted` event;
- if `attempts <= max_retries`, it is set back to `queued` (due immediately)
  with a `requeued` event, so the spent attempt is not lost;
- if no attempt remains it stays `interrupted` — the history shows a job that
  started and never finished instead of pretending it failed.

When anything is recovered, one `orchestrator_started` event summarises the
counts. A clean start writes nothing.

## Audit log

`events` is append-only: this module has no update or delete path for it. A job
event carries `job_id`; a control event carries `NULL`. `detail` is compact JSON
so a reader can parse the reason and a human can still read it.

| Event | Written by | Meaning |
|---|---|---|
| `enqueued` | store | New job with priority, deps and schedule |
| `deduplicated` | store | Same `dedupe_key` seen again; existing job kept |
| `started` | store | Claim, with attempt number and heavy flag |
| `succeeded` | store | Result document stored |
| `failed` | store | Terminal failure with the error |
| `retry` | store | Requeued with delay and reason |
| `deferred` | store | Guard refusal; delay recorded, attempt refunded |
| `cancelled` | store | Cancelled with previous status and reason |
| `skipped` | store | Dependency reason stored |
| `interrupted` / `requeued` | store | Restart recovery |
| `handler_error` | core | Handler failed; the attempt and permanence |
| `resource_refused` | core | Guard veto and its reason |
| `no_handler` | core | Kind not wired yet |
| `cancelled_while_running` | core | Result discarded after a cancel |
| `claim_refused` | core | Paused, safe mode, research disabled or heavy busy |
| `paused` / `resumed` / `safe_mode_on` / `safe_mode_off` / `research_disabled` / `research_enabled` | core | Control changes |
| `orchestrator_started` | core | Recovery summary |
| `trigger_scan` | triggers | What a scan enqueued and deduplicated |

## Handler contract

```python
@dataclass(frozen=True)
class HandlerResult:
    ok: bool = True
    permanent: bool = False
    detail: str = ""              # required when ok is False
    data: Mapping[str, Any] = {}  # stored as jobs.result_json["data"]
```

A handler is `(job, context) -> HandlerResult | Mapping | None`. `None` is a
success with an empty result; a plain mapping is a success whose content is the
result document. Stored results look like:

```json
{"ok": true, "permanent": false, "detail": "", "data": {"rows": 1234}}
```

`JobContext` gives the handler the store, the job, the current control snapshot
and two deliberate side effects: `log(event, detail)` to explain a decision, and
`cancel(reason)` for cooperative cancellation. It cannot enqueue or claim jobs;
planning stays in the orchestrator, which keeps the loop the only writer of job
state.

Example:

```python
def prepare_dataset(job, context):
    upstream = context.store.get(int(job.payload["upstream_job"]))
    rows = (upstream.result or {})["data"]["rows"]
    context.log("dataset_ready", f"upstream_rows={rows}")
    return {"path": f"/data/{job.payload['symbol']}_M1.parquet", "rows": rows}
```

## Production handlers (P3 slice 1)

`register_default_handlers(registry, context)` binds five namespaced kinds. The
context is the only source of paths and settings; there is no module-level
state:

```python
context = HandlerContext(
    data_root=Path("data"),
    runs_root=Path("runs"),
    archive_root=Path("archive"),
    settings=Settings(),
    use_feature_cache=True,
    discovery_runner=None,       # None means scalping.discovery.run.run_discovery
)
register_default_handlers(orchestrator, context)
```

| Field | Holds | Loss impact |
|---|---|---|
| `data_root` | Parquet bars (read only) and `.feature-cache` | Re-ingestable source data |
| `runs_root` | Dataset manifests and `discovery/discovery.sqlite` | Re-computable run products |
| `archive_root` | `knowledge/index.sqlite`, preserved sources, `paper.sqlite` | Durable evidence; back it up |

| Kind | Handler | Payload (all fields validated; unknown fields are rejected) | Success result |
|---|---|---|---|
| `knowledge.ingest` | document → index → drafts | `document_path` (required), `market_scope`, `timeframe_scope`, `max_hypotheses` | `doc_id`, `checksum`, `status`, `sections`, `drafts`, `skipped`, `scanned_sentences`, `duplicate`, `stored_path`, `index` |
| `dataset.build` | stored bars → cached features → manifest | `symbol`, `timeframe` (required), `horizon`, `train_fraction`, `validation_fraction`, `purge_bars`, `embargo_bars`, `notes` | Manifest document plus `bars_rows`, `purged_rows`, `manifest_path` |
| `discovery.run` | bounded search → recorded decision | `symbol`, `timeframe`, `max_trials`, `seed`, `gate` (all required), `stress`, `population`, `generations`, `patience`, `horizon` | `decision`, `reason`, `promoted: false`, `trials`, `accepted`, `best_candidate_id`, `dataset_id`, `decision_id`, `stop_reason`, `seed`, `code_version`, `archive` |
| `paper.advance` | champion → autonomous DSL paper run | `symbol`, `timeframe`, `bars` (required) | `action: "advanced"` or `"no_trade"`, `spec_hash`, `session_id`, bar/trade counts, ledger path |
| `paper.analyse` | ledger → closed-trade summary | `symbol`, `timeframe`, `max_sessions` (all optional) | Counts, P&L, R, win rate, `open_positions_excluded`, per-session attribution |

### Payload contract

A request is a frozen dataclass with `from_payload`. Validation is strict:
missing required fields, unknown fields, wrong types, booleans used as integers,
non-finite numbers, path traversal (`..`, NUL) and out-of-range bounds all
reject. Bounds exist so one job cannot run for hours: `max_trials <= 2000`,
`population <= 256`, `generations <= 100`, `bars <= 100_000`,
`max_sessions <= 1000`, `max_hypotheses <= 500`. `symbol`/`timeframe` must match
`^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$`, which also keeps a store path from
escaping its root.

`discovery.run` requires the `gate` block even when every threshold is
disabled. An empty gate is exploratory search policy, not promotion evidence,
and writing it down makes that visible in the job payload. `stress` defaults to
`[1.5, 2.0]` but is always passed to `DiscoveryRunConfig` explicitly.

### Failure classification

| Class | Example | Queue behaviour |
|---|---|---|
| Permanent (`PermanentJobError`) | unknown field, missing field, wrong type, nonexistent document, unsupported PDF, unknown symbol/timeframe, missing gate, gate value out of range | `failed` on the first attempt; `handler_error` event has `"permanent": true` |
| Retryable (any other exception) | SQLite busy, I/O error, unexpected bug | bounded retry with exponential backoff |
| No handler | kind not registered | `failed`, permanent, `no_handler` event |
| NO TRADE | no champion, invalid champion payload | `succeeded`; the result document says `action: "no_trade"` and why |

NO TRADE is never a failure. A retry would not change a missing champion, so it
must not consume attempts.

### Idempotency

- `knowledge.ingest` hashes the file before extraction. A checksum already in the
  index reuses the stored copy and re-runs extraction, whose `(statement,
  section)` dedupe creates no second draft; a retry after a crash between
  indexing and extraction therefore converges on the intended end state.
- `dataset.build` derives the dataset id and manifest filename from the content
  checksum, so a rerun overwrites the same path with the same id and checksum;
  `created_at` records the latest build, not the first.
- `discovery.run` deduplicates candidates by canonical hash in the archive; a
  rerun on the same dataset/seed evaluates no new trials and records
  `no_change`.
- `paper.advance` is idempotent at the queue, not at the handler: the dedupe key
  is what prevents a second advance. Advancing is a step, so calling the same
  handler twice with different keys intentionally advances twice. Use
  `paper.advance:{symbol}:{timeframe}:{bars}` (or a bar-range key) as the dedupe
  key.
- `paper.analyse` only reads; a rerun returns the same summary unless the ledger
  changed.

### No-promotion rule

`discovery.run` cannot write a `promoted` decision: `run_discovery` refuses it
and `DiscoveryRunResult` raises if asked. The handler also checks the returned
decision before storing its result. Promotion and rollback stay in
`DiscoveryArchive.promote/rollback`, reached by a separate reviewed path.
`paper.advance` selects the champion of `symbol:timeframe` through
`ChampionProvider`. If the archive is absent, has no champion, or the stored
payload fails DSL validation or hashes inconsistently, the job records NO TRADE
and the handler opens no ledger, writes no order and never falls back to the
legacy `flow_momentum` rule.

### Outcome attribution

`paper.analyse` counts only `trades` rows, which the ledger writes when a trade
closes. Open `positions` are reported as `open_positions_excluded` and are never
summed into P&L or R. Each session row carries `spec_hash` from
`model_version`, `config_spec_hash` from the stored autonomous config, and a
`data_fingerprint` from `features.cache.frame_meta` over the stored bars the
session's signals span. The window ends at the last signal, so appending new
bars later does not change a past attribution. When the bars are unavailable the
fingerprint is `null` and `data_attribution` says `"unavailable"`; a session
spanning more than 31 days is not read at all and reports
`"window_too_large"`. No hash is invented.

### Honest limits

- No scheduler thread, daemon or timer. A caller must invoke `run_once` or
  `run_until_idle`.
- The resource guard still does not measure anything. `HEAVY_HANDLER_KINDS`
  names `dataset.build` and `discovery.run` for the caller's `heavy_kinds`, but
  peak RAM/CPU/disk on the M2 are unproven here.
- `paper.advance` is dedupe-key idempotent, not replay-idempotent.
- Outcome analysis is descriptive: counts, P&L and R. It does not test a
  hypothesis, enqueue the next discovery cycle or train anything.
- `discovery.run` still evaluates on one validation slice. It is not
  promotion-grade walk-forward evidence; see the audit's P1.

## Trigger sources

`TriggerSource` has two methods: `scan()` returns `Trigger(kind, dedupe_key,
payload, priority, max_retries, depends_on)`, and `seen(triggers)` is called
after the orchestrator enqueued them. The two phases exist because neither order
is safe alone: marking a file seen before the job exists loses the job on a
crash, and enqueuing without a cursor re-hashes the directory every tick. A crash
between the phases re-enqueues the same `dedupe_key`, which is idempotent.

`FileTriggerSource` hashes new or changed files under a directory (SHA-256,
streamed) and stores one cursor document per source in the `state` table. A file
whose size and `mtime_ns` match the cursor is skipped without hashing. The
dedupe key is `"{kind}:file:{sha256}"`, so byte-identical copies under a second
name produce one job, while an edit produces a new one. Removed files drop out
of the cursor on the next scan.

`scan_once(orchestrator, sources)` enqueues, acknowledges each source and writes
one `trigger_scan` event. There is no filesystem watcher; a caller decides how
often to scan.

## Verification

```
cd backend
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/test_orchestrator_store.py tests/test_orchestrator_core.py \
  tests/test_orchestrator_handlers.py -q
uv run pytest
```

The slice is accepted on those commands passing, not on the design above. The
queue tests cover idempotent enqueue, priority and insertion order, schedule
gating, dependency order, dependency failure to `skipped` (including a cascade),
retry backoff bounds and exhaustion, permanent failures, cancel while queued and
while running, pause/resume/safe-mode/disable-research (including persistence
across a reopen), crash recovery with and without attempts left, one heavy job,
guard refusal without spending an attempt, event-log completeness, and
file-trigger dedupe across a store restart. The handler tests cover a real
ingest/index/extract round trip, checksum idempotency, dataset manifest
reproducibility, an unknown symbol refusal, a real bounded discovery run that
records no promotion, a no-champion NO TRADE that writes nothing, a governed
champion paper advance with queue dedupe, an invalid champion payload, closed-
only outcome analysis with an open position excluded, and permanent failure for
invalid payloads with the retry budget untouched.
