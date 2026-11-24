# Learning orchestrator

Phase 18 slice one: a persistent job queue, the loop that runs it, owner
controls and trigger sources. The queue, its audit log and the controls are real
code with tests. Slice two (P3 slice 1) registers five production handlers:
document ingestion, dataset build, discovery, paper advance and outcome
analysis. Slice three (P3 slice 2) adds a supervised long-running daemon that
scans triggers, drains the queue and installs the measured guard by default.

This slice answers one question honestly: *can the system remember, order,
refuse, retry and explain work across a restart?* The answer is tested. The
handlers answer *what* work exists for the first research cycle; the daemon
answers *when* it runs and under which measured limits. It is still one local
process, not an operating-system service.

## Files

| File | Role |
|---|---|
| `backend/src/scalping/orchestrator/store.py` | SQLite queue + append-only events + persisted state |
| `backend/src/scalping/orchestrator/core.py` | Handler registry, controls, claim/run/fail/retry loop |
| `backend/src/scalping/orchestrator/triggers.py` | `TriggerSource` protocol, `FileTriggerSource`, `scan_once` |
| `backend/src/scalping/orchestrator/handlers.py` | `HandlerContext`, payload validators, the five pipeline handlers |
| `backend/src/scalping/orchestrator/resources.py` | Measured `ResourceGuard`: stdlib probe, limits, samples |
| `backend/src/scalping/orchestrator/daemon.py` | `DaemonConfig`, `OrchestratorDaemon`, tick order, background loop, status |
| `backend/tests/test_orchestrator_store.py` | Queue semantics, transitions, recovery |
| `backend/tests/test_orchestrator_core.py` | Loop, controls, heavy job, guard, triggers |
| `backend/tests/test_orchestrator_handlers.py` | Real handlers on synthetic stores and temporary roots |
| `backend/tests/test_orchestrator_resources.py` | Limits, caching, fail-closed probes, samples |
| `backend/tests/test_orchestrator_daemon.py` | Tick order, dedupe, controls, guard defer, isolation, start/stop, CLI |
| `backend/tests/test_orchestrator_integration.py` | Real handlers, triggers and daemon on temp roots: chain, isolation, restart, boundaries |

## What is not in this slice

- **One process, not an operating-system service.** `OrchestratorDaemon` runs a
  daemon thread inside whatever process calls `start()`. There is no launchd,
  systemd, supervisor, multi-worker scheduler or persisted timer. A process
  restart relies on `recover_interrupted` and each job's own `scheduled_at`.
- **Paper execution is still caller/thread driven.** The daemon calls
  `paper.advance` when a job is due; it does not own a market-data feed, a bar
  clock or a trading loop. Running it against live/appended bars is the caller's
  responsibility.
- **The measured guard fences start, not growth.** It measures RAM/load/disk
  before a job starts. A job that grows after the check is not interrupted;
  `max_process_rss_mb` only refuses the next job.
- **No promotion handler.** Discovery only records `no_trade`, `no_change` or
  `rejected`. `paper.advance` reads a champion that a reviewed path promoted;
  neither handler can promote, replace or roll back one.
- **No training handler.** Training on locked OOS is out of scope for this
  slice, and no handler here fits a model.
- **No retention policy.** Jobs and events are never deleted, which is correct
  for an experiment history and will eventually need an archive step.
- **No write API/UI.** The only HTTP surface is the read-only autonomy viewer
  described below; there is no endpoint to enqueue, cancel, pause, promote or
  start a daemon. State is queryable in-process through `JobStore`,
  `Orchestrator.status()` and `OrchestratorDaemon.status()`.
- **One daemon per database.** SQLite `BEGIN IMMEDIATE` makes claims unique, but
  it does not provide ownership: a second daemon's `start()` recovery sees the
  first daemon's `running` rows as interrupted, requeues them and runs them
  twice. Run one daemon per database; an ownership/lease protocol is future
  work.

## Daemon and scheduler (P3 slice 2)

`OrchestratorDaemon(store, registry, config, *, guard=None, triggers=(),
clock=time.monotonic, sleeper=time.sleep)` wires the existing queue, handler
registry and measured guard into one supervised process. `registry` is either a
`{kind: handler}` mapping or anything exposing `.handlers` — the CLI builds an
`Orchestrator`, calls `register_default_handlers`, and passes it in, so there is
no second registration table.

### Configuration

| Field | Default | Meaning |
|---|---|---|
| `poll_seconds` | `5.0` | Wait between ticks. Bounded to 1–60 so a typo cannot busy-spin or hide a stop. |
| `guard_profile` | `m2_8gb` | `m2_8gb` and `exploratory` build `MeasuredResourceGuard` with the matching profile; `none` installs `AllowAllGuard` for tests/dev only. |
| `scan_triggers` | `True` | Call `scan_once` before the drain. With no sources the scan is skipped entirely. |
| `max_jobs_per_tick` | `10` | Cap passed to `run_until_idle`; `0` scans and enqueues but claims nothing. |
| `stop_grace_seconds` | `5.0` | How long `stop()` joins the loop before returning; bounded to 0–300. |
| `heavy_kinds` | `()` | Passed to the core's one-heavy-job rule. The CLI passes `HEAVY_HANDLER_KINDS`. |
| `research_kinds` | `()` | Kinds the persisted `disable research` control excludes. The CLI passes dataset, discovery, paper advance and paper analyse. |

An explicit `guard=` argument overrides the profile; status reports both the
configured `guard_profile` and the actual guard `type`, plus
`unmeasured: true` when `AllowAllGuard` is in use. `none` is documented as a
test/development profile and must not be used for an unattended run.

### Tick order

1. **Crash recovery, once.** The first `tick()` runs
   `store.recover_interrupted()` on the caller's connection. `start()` runs it on
   the worker thread's own connection, before the first tick, and blocks until it
   finishes. The flag is per daemon instance; a second `start()` never repeats
   it.
2. **Scan triggers.** `scan_once(orchestrator, triggers)` asks each source, goes
   through idempotent `enqueue`, acknowledges each source and writes one
   `trigger_scan` event. Dedupe is the trigger's `dedupe_key`, so two ticks that
   see the same content produce one job. The daemon passes its stop flag as
   `should_stop` to the drain, so a stop requested between jobs does not start
   the remaining ones.
3. **Drain.** `run_until_idle(max_jobs_per_tick, should_stop=...)` claims and
   runs due work.

Scanning and enqueueing remain allowed while the queue is paused, in safe mode
or with research disabled. Only claims are refused, with the core's existing
`claim_refused` / `research_disabled` event. That is deliberate: the owner
stopped spending work, not the recording of new work, and a hidden scanner would
lose the event that created the job.

A scan or store failure is recorded in `last_error` and the drain still runs, so
a broken trigger source cannot stop paper jobs. Handler exceptions are already
per-job failures in the core: one failed heavy training-like job does not stop a
paper job that is due later in the same tick or on the next one.

### Guard refusals

`MeasuredResourceGuard.can_start` runs after the claim and before the handler.
A refusal writes `resource_refused` with the measured reason and calls
`store.defer`, so the job returns to `queued` with `attempts` unchanged and a
`scheduled_at` at least `guard_retry_seconds` (60 s in the daemon) in the future.
The reason is in the event log, the job row's `error` and the next status
snapshot's guard block.

Deferrals are bounded: each job carries `defer_count` and `max_deferrals`
(default 1440 = 24 h at 60 s). A successful defer increments the count; the
refusal that would exceed the cap writes `defer_exhausted`, leaves the job
`failed` with a reason naming the cap and leaves the claimed attempt spent. That
makes a permanently starved job visible instead of queued forever. `JobStats`
reports the summed `deferrals`.

### Start, stop and status

`start()` is idempotent: while the thread is alive a second call returns `[]`.
It clears the stop flag *before* recovery (under the lifecycle lock), starts a
named daemon thread and waits for that thread's recovery to finish. If `stop()`
arrived during recovery, `start()` returns `[]` and leaves no running thread.
Recovery exceptions are re-raised by `start()`.

The loop opens its own `JobStore` connection because SQLite connections are
thread-bound; recovery and every tick use that connection. WAL lets `status()`
read while the worker writes. `tick()` still uses the connection that created
`store` and must be called on that thread; `start()` and `status()` do not.
`:memory:` stores cannot cross threads, so only their creating thread may call
`status()`, and `start()` rejects them — use `tick()`.

`stop()` sets the flag before reading the thread reference and does not take the
lifecycle lock while reading it, so a stop can win while `start()` is blocked in
recovery. It joins for at most `stop_grace_seconds`; it never raises, and calling
it twice or before `start()` is safe. The wait between ticks is sliced at 0.25 s
so a 60 s poll does not delay shutdown by a minute. A handler already running
cannot be interrupted from another thread; the loop stops after that `tick()`
returns. Stop does not cancel the running job.

`status()` returns a JSON-safe `DaemonStatus` from a short-lived
`query_only` connection for a file-backed store: `running`, `worker_ident`,
`paused`, `safe_mode`, `disabled_research`, queue counts from `JobStats` (with
`deferrals`), tick count, **terminal** job count, last tick time and duration,
last error, recovered-job count, and the guard profile/type/unmeasured flag with
its last snapshot (never a fresh probe).

### CLI

```
# one deterministic pass; exactly one JSON line on stdout
uv run scalping.cli orchestrator run --once --profile m2_8gb

# supervised loop; SIGINT/SIGTERM stop it and print a final snapshot
uv run scalping.cli orchestrator run --forever --poll-seconds 5

# watch a directory: each new/changed file enqueues knowledge.ingest
uv run scalping.cli orchestrator run --forever --watch-dir docs/research

# tests/dev only: no measured guard, status says unmeasured
uv run scalping.cli orchestrator run --once --profile none
```

`--db` defaults to `runs/orchestrator.sqlite3`; `--root`, `--runs-root` and
`--archive-root` feed the existing `HandlerContext` (`--root data`,
`--runs-root runs`, `--archive-root data`, with the paper ledger at
`<archive-root>/paper.sqlite`, matching the API default).

There is no trigger by default: without `--watch-dir` the daemon scans nothing
and the queue is filled by another process. `--watch-dir` binds one
`FileTriggerSource`; `--watch-kind` defaults to `knowledge.ingest` and the CLI
adapts the file payload to that handler's `document_path`. `--no-scan` is
meaningful only in that default mode; combining it with `--watch-dir` is a CLI
error.

`--once` exits 2 with a JSON error when the tick itself failed (scan, recovery or
store error). A job-level handler failure is a recorded job outcome and still
exits 0. `--forever` detects a worker thread that died without a stop request,
prints `{"status": "failed", ...}` and exits 1. A requested stop prints
`{"status": "stopped", ...}` and exits 0. Other CLI commands keep the single JSON
line contract; `--forever` prints `started` and then `stopped`/`failed`.

### Read-only status API

`scalping.api.routes_autonomy` mounts under `/api/autonomy`. Every route is a
`GET`; nothing here enqueues, cancels, pauses, promotes or starts a daemon.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/autonomy/status` | Aggregate paper, orchestrator, champion, discovery and data sections |
| `GET` | `/api/autonomy/champions` | Champion per scope with `mode`, `policy_id` and `policy_digest` when the payload records them |
| `GET` | `/api/autonomy/jobs` | Newest jobs first: status, kind, attempts and defer count |
| `GET` | `/api/autonomy/events` | Newest audit events first; the free-form `detail` column is not exposed |

The database paths are app configuration (`create_app(orchestrator_db=...,
discovery_archive=...)`), never request fields. Both files default under
`runs_root` to the CLI locations. Every store is opened with a `mode=ro` URI
and `PRAGMA query_only = ON`: SQLite cannot create a missing database, cannot
write through the connection and takes no write lock. A `GET` therefore never
runs `init_schema`, never changes the database file and keeps working while the
daemon holds `BEGIN IMMEDIATE`; WAL readers see the last committed snapshot. A
read-only connection to a WAL database may still create or update its `-wal` and
`-shm` sidecars when the directory is writable. The test pins that tree exactly:
only the database and those two sidecars may exist after a GET.

Each section reports both flags. `configured: false` means the file is absent.
`configured: true, readable: false` means it exists but could not be opened, has
no schema yet, or lacks a column this reader needs; items and counts are then
omitted instead of answering as if the queue or archive were empty. Only
`readable: true` produces counts, and a genuinely empty store returns zero.

`/jobs` defaults to 50 and accepts `1..500`; `/events` defaults to 100 and
accepts `1..500`; anything outside is a `422`. A v1 jobs table without
`defer_count` is unreadable in both `/status` and `/jobs`, so the two views
never disagree. The discovery archive is read with plain SQL, so a v1 archive
without `champions` returns an empty list instead of migrating. `runs` in
`/status` is the archived trial count and `decisions` is the gate-decision
count. `/status` opens the archive once and derives both sections from that
snapshot. `events.detail` is deliberately withheld: it can contain local paths
or handler errors that are outside the status contract.

### Autonomy page (`/autonomy`)

`frontend/src/app/autonomy/page.tsx` is the operator view of those three reads
(`/status`, `/jobs`, `/events`). It is read-only: the refresh button re-runs the
GETs, and nothing on the page enqueues, cancels, pauses, promotes or starts a
daemon. `autonomy/layout.tsx` gives the route its own document title
(“Otonomi — AI Scalping Studio”).

- **Sections.** Paper ledger counts (sessions, open positions, closed trades),
  orchestrator queue and persisted controls (queued/running/failed, paused, safe
  mode, research disabled, last event), champions per scope, discovery
  run/decision counts and data symbols with a month partition.
- **Honest states.** `configured: false` renders “yapılandırılmadı” and an
  unreadable store renders “okunamıyor”; both replace the numbers and table
  rows rather than showing zero. A count field absent from a readable response
  renders `—`. A readable store that really answered zero shows zero, and an
  empty champion list says NO TRADE is a valid outcome.
- **Champion modes.** The badge separates `evidence` from `exploratory` and
  `manual` (and an unrecognised or missing mode is labelled “kanıtsız”). The
  section states that only an `evidence` champion may trade paper; every other
  row is marked “paper dışı”.
- **Bounded windows.** Jobs and events are newest-first windows of 25, 50 or 100
  rows. The API has no offset, so the page exposes a window size and never
  offers a “previous page” that does not exist.
- **Accessibility.** Section headings, a paper-only notice, captions with
  `scope` on every header cell, focusable scroll regions named by their
  captions, labelled window selectors, `role="alert"` for failures and a single
  polite live region announcing the last successful read.
- **Tests.** `frontend/src/app/autonomy/page.test.tsx` covers loaded, missing,
  unreadable, empty, error/retry, absent counts, champion modes and the
  accessible table region; `frontend/src/components/nav.test.tsx` covers the
  navigation link.

### Honest limits

- **One daemon per database.** `BEGIN IMMEDIATE` keeps claims unique, but there
  is no ownership or lease. A second daemon's recovery treats the first's
  `running` rows as interrupted, requeues them and can run them twice. Start two
  only for an experiment you can reconcile; a lease is a future slice.
- No OS service, no process supervision, no restart-on-crash. A supervisor or
  the operator restarts it, and recovery handles the interrupted rows.
- Paper jobs are driven by the queue, not by a market clock; the daemon cannot
  create bars or guarantee a cadence for `paper.advance`.
- Guard numbers describe this machine at this instant. They do not predict the
  next allocation and, on macOS, cannot see memory held by other processes.
- Trigger sources are scanned sequentially in the calling thread; a slow hash of
  a large document delays that tick.
- `status()` opens a connection per call. It is cheap for an operator poll, not
  intended as a high-frequency metrics endpoint.


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
back, `defer_count` is incremented, and the guard's reason is stored in
`jobs.error` with `resource_refused` and `deferred` events. Waiting for RAM
cannot exhaust a job's retry budget, which a retry-based refusal would. The
refusal is visible, not a silent drop.

A deferral is not unlimited: `max_deferrals` (default 1440) caps the wait. The
refusal past the cap is terminal — `failed`, `defer_exhausted` event, the reason
names the cap — so a permanently starved job cannot sit queued for weeks.

## Measured resource guard (P6 slice one)

`orchestrator/resources.py` implements the veto with real readings:
`MeasuredResourceGuard(limits, probe=SystemProbe(), clock=time.monotonic,
cache_seconds=1.0)`. It is **not** the orchestrator default —
`AllowAllGuard` still is — so a deployment opts in explicitly:

```python
from scalping.orchestrator.resources import M2_8GB_PROFILE, MeasuredResourceGuard

guard = MeasuredResourceGuard(M2_8GB_PROFILE)
orchestrator = Orchestrator(store, handlers=..., heavy_kinds=("train",), guard=guard)
```

| Limit | Refuses when | 0 means |
|---|---|---|
| `max_rss_mb` | system RAM in use (`total - available`) is at or above it | disabled |
| `max_process_rss_mb` | this process's high-water RSS (`ru_maxrss`) is at or above it | disabled |
| `max_load_1m` | the 1-minute load average is at or above it | disabled |
| `min_free_disk_mb` | free space on the probed volume is at or below it | disabled |

A reading exactly at a ceiling, or exactly at the disk floor, is already a
refusal: a limit is a hard edge, not a warning level. The first breached limit
in the table order is the one reported, so an audit log is reproducible.

| Profile | `max_rss_mb` | `max_process_rss_mb` | `max_load_1m` | `min_free_disk_mb` | Use |
|---|---|---|---|---|---|
| `M2_8GB_PROFILE` | 6000 | 4500 | 6.0 | 2048 | safe default for the M2 8 GB target |
| `EXPLORATORY_PROFILE` | 7500 | 7000 | 7.5 | 512 | owner-supervised exploration only |

Both profiles are frozen constants; `ResourceLimits()` with no arguments
disables every limit and is not a promotion policy. A refusal reason names the
limit and the measurement, for example:

```
max_load_1m=6 exceeded: load1=7.42
```

That string reaches `jobs.error` and the `resource_refused` event, so "why did
nothing run?" is answered with a number instead of "resources".

### Probe, cache and fail-closed behaviour

`SystemProbe` uses `resource.getrusage` for its own high-water RSS,
`shutil.disk_usage` for the disk floor, `os.getloadavg` for load and
`os.sysconf` for physical memory. No subprocess and no network call happens on
any path: a probe that shelled out would consume the resources it polices.

- **RSS units.** `ru_maxrss` is bytes on macOS/BSD and KiB on Linux; one helper
  owns the branch and is tested for both.
- **Total RAM fallback.** When `os.sysconf` cannot report `SC_PHYS_PAGES`, the
  probe assumes 8192 MiB (the target machine class). On a larger host that
  under-states capacity, so limits refuse earlier rather than later.
- **Available RAM fallback.** `SC_AVPHYS_PAGES` exists on Linux; macOS raises
  `ValueError`, so the probe falls back to `total - own RSS`. That ignores other
  processes and is an optimistic upper bound. The process and disk limits exist
  partly because this number is weak.
- **Cache.** One snapshot serves every claim inside `cache_seconds` (1 s), so a
  queue drain measures once. The age comes from the injected monotonic clock; a
  clock that moves backwards forces a fresh probe. `cache_seconds=0` measures
  every call and is what the tests use when they want exact counts.
- **Fail closed.** A probe that raises, returns the wrong type or produces a
  non-finite value yields `(False, "probe_failed: TypeError: ...")`. The guard
  never raises into the orchestrator, and the failed probe is cached for the
  same short window so a broken probe is not hammered by a drain.

### Measuring a job

`guard.measure(kind)` is a context manager that records wall time, CPU time and
high-water RSS growth into a JSON-safe `ResourceSample`:

```python
with guard.measure("features") as measurement:
    build_features(...)
measurement.sample.to_dict()   # kind, elapsed_seconds, cpu_seconds, rss_delta_mb, ok, ...
```

A block that raises still produces a sample with `ok=False` and the exception
text; the exception itself propagates unchanged. `samples` keeps the newest 64
(with `sample_limit` to change or disable it) for a future API/dashboard view.
`rss_delta_mb` is the growth of a high-water mark, not an integral: a peak that
is allocated and freed inside the block is visible, a flat resident set is zero.

### What these numbers are not

They are local measurements of this machine at this instant. They fence the
*start* of a job and cannot promise the job will stay inside its budget: a later
allocation can still swap, and the next `can_start` is the only place that shows
it. On macOS "available RAM" is an estimate for the reason above, and
`ru_maxrss` never decreases, so it reports the largest the process has ever
been rather than its current residency. Treat a refusal as evidence and a pass
as permission to try, not as a memory guarantee.

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
| `deferred` | store | Guard refusal; delay and `defer_count` recorded, attempt refunded |
| `defer_exhausted` | store | Deferral cap reached; job failed with the reason |
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

`JobContext` gives the handler a read-only `JobStoreReader` (jobs, events and
state can be read, an audit line can be appended), the job and the current
control snapshot, plus two deliberate side effects: `log(event, detail)` to
explain a decision, and `cancel(reason)` for cooperative cancellation of its own
job. It cannot enqueue, claim or cancel another job; planning stays in the
orchestrator, which keeps the loop the only writer of job state. The facade is
accidental-misuse protection, not a sandbox: the private delegate is reachable
through introspection by in-process code.

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
decision before storing its result. Promotion stays an explicit, reviewed call:
`promote_with_evidence` for production, `promote_exploratory` for research, and
`rollback` to restore a previous champion. `paper.advance` selects the champion of
`symbol:timeframe` through `ChampionProvider`. If the archive is absent, has no
champion, the champion is exploratory/manual (`NoEvidenceChampion`), or the
stored payload fails DSL validation or hashes inconsistently, the job records NO
TRADE and the handler opens no ledger, writes no order and never falls back to
the legacy `flow_momentum` rule.

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

### Matured paper outcomes (P5 slice 2)

`PaperOutcomeTriggerSource` closes part of the loop: a paper session's completed
trades become one research trigger. It reads the ledger with a read-only
connection, calls `matured_outcomes(ledger, session_id)` for each session and
emits one trigger per batch whose `outcome_fingerprint` the cursor has not
recorded.

| Field | Value |
|---|---|
| `kind` | `paper.outcomes` (the only accepted kind) |
| `dedupe_key` | `paper.outcomes:{fingerprint}` |
| payload | `session_id`, `fingerprint`, `trades`, `pnl`, `r_total`, `strategy_hash`, `dataset_id`, `source`, `synthetic` |
| `max_retries` | `1` by default; a transient handler failure cannot burn the batch |

Only closed trades contribute. `matured_outcomes` counts open positions in
`open_trades` and never turns them into rows, so an unrealised trade cannot
start research. A `running` session is eligible: its closed trades are already
matured, and waiting for session close would delay feedback for no correctness
gain.

#### Scan order, coverage and backlog

Sessions are ordered by last trade activity — `MAX(trades.exit_ts)`, falling
back to `started_at` for a session with no closed trade. Activity order keeps a
long-running session that keeps closing trades from being starved by newer,
quiet sessions.

A scan reads at most `max_sessions` (default 1000, maximum 10 000) sessions,
starting at a persisted rotation offset. When the ledger holds more, the next
scan resumes at the offset where this one stopped and wraps, so every session is
polled even if a single scan cannot reach all of them. Sessions not reached are
recorded with `reason: truncated` (up to 25 individual `paper.outcome_not_queued`
lines) and one `paper.outcome_truncated` summary event carrying `deferred`,
`scanned`, `total` and `resume_offset`. A session whose trigger already exists is
not logged as truncated: outside this scan's window is not an actionable refusal.
They are delayed, never silently dropped.

Two bounds keep a first scan over a long ledger cheap:

| Option | Default | Limit | Effect |
|---|---|---|---|
| `since` | `None` | — | Only sessions with `started_at >= since` are in the window. Accepts a timezone-aware `datetime` or an ISO-8601 string with an offset; a naive timestamp is rejected. |
| `max_proposals` | `100` | `1000` | At most this many triggers per scan, so a historical ledger cannot propose an unbounded research backlog at once. The rotation cursor resumes at the first unprocessed session. |

`since` is a session-start bound, not a trade-time bound: a session that started
before the bound is out of scope entirely, and it emits no per-session refusal
line (the window is a deliberate configuration, not a data problem). A session
whose trigger was already emitted before `since` was set stays acknowledged; the
window trims the read, it does not retroactively cancel proposals.

#### Eligibility and refusal reasons

A session emits only when the batch has at least one closed trade, a strategy
hash and a dataset id. Otherwise the source writes one `paper.outcome_not_queued`
event with a machine-readable `reason` and a human `detail`. The reason is
recorded once per session and rewritten only when it changes, so a periodic scan
does not flood the audit log.

| `reason` | Meaning | Emits? |
|---|---|---|
| `empty_batch` | No trade has closed yet; the P&L is not final | No |
| `no_strategy_hash` | `model_version` and signals carry no attribution | No |
| `no_dataset_id` | Session config records no dataset id to scope research | No |
| `outcome_error` | Ambiguous attribution or unreadable session config | No |
| `no_ledger` | The configured ledger path does not exist | No |
| `ledger_unavailable` | Unreadable file or no sessions table | No |
| `truncated` | Deferred by `max_sessions` or `max_proposals`; the next scan resumes at it | No |

#### Provenance in the payload

`source` and `synthetic` are copied from explicit keys in the session config:
`source`, `provenance.source`, `data.source` or `extra.source`, and the
`synthetic` counterpart. Missing or non-boolean values become `null`, which
means **unknown** — never an invented `false`. A session whose config predates
these fields must not be presented as real, non-synthetic evidence, so
downstream code can refuse `synthetic is not false` without guessing. The
runner's `origin` field is deliberately not used as a data source: it names the
process, not the data.

#### Dedupe

The cursor is one JSON document in the job store's `state` table under
`trigger.paper.outcomes:<absolute ledger path>`:

```json
{"version": 3, "fingerprints": {"<sha256>": "<session_id>"},
 "not_queued": {"<session_id>": "<reason>"},
 "pending_ack": {"<sha256>": "<session_id>"}, "offset": 0}
```

`scan()` computes it and `seen()` persists it, matching the file-trigger
protocol. `seen(triggers)` records only the fingerprints the caller actually
acknowledged. Acknowledging a subset moves the rest into `pending_ack`, and the
next scan re-emits them before it resumes the rotation — they do not wait for the
offset to wrap. A late acknowledgement of an older trigger promotes its
fingerprint from `pending_ack` to the durable set. `scan_once` acknowledges every
trigger it enqueued, so the converged case stores them all. A crash between
enqueue and `seen` re-emits the same trigger on the next scan; the job
`dedupe_key` makes that a duplicate instead of a second job. Deduplication
therefore has two independent layers: the durable fingerprint set and the store
key. Any newly closed trade changes the fingerprint and emits one new trigger.

The source emits with `max_retries=1`, so one transient handler failure is
retried instead of consuming the only trigger for a matured batch. Each attempt
logs its own `paper.outcome_received` line, so a retried job has two: the audit
log records attempts, not logical triggers. Once both attempts fail the job is
terminal and the fingerprint stays acknowledged; an operator requeue is then the
explicit recovery, and the cursor is not silently rewound to re-propose the same
work.

#### Proposal only, no auto-training

`make_paper_outcome_handler(should_queue=None)` and
`register_paper_outcome_handler(registry)` bind `paper.outcomes`. The handler:

1. requires the seven core payload fields and accepts the optional `source` and
   `synthetic` pair (unknown, missing-required, mistyped or non-finite input is
   a permanent failure; an old job without provenance is `null`, not real);
2. logs `paper.outcome_received` for the job, including provenance;
3. returns `{"action": "queue_discovery" | "no_action", "reason": ...,
   "queued": false, "executed": false}` plus the attribution it received.

It imports no discovery, training, simulator or champion module, reads no bars,
features or holdout split, and cannot enqueue: a handler receives
`JobContext.store` as a `JobStoreReader`, a read-only facade over the queue. It
can read jobs, events and state and append an audit line, but `enqueue`,
`claim_next`, `retry`, `defer`, `cancel` and `set_state` are not on it.
`JobContext.cancel()` is scoped to the handler's own job through a private
callable, so the reader cannot cancel another job. The facade is
accidental-misuse protection, not a security boundary: the delegate lives in a
private slot and in-process code can still reach it through introspection. It
stops a handler from growing or rewriting the queue by mistake, which is the
contract the docs promise.
`should_queue(payload) -> bool` is the explicit policy hook. Returning `True`
proposes the next research cycle; it still does not start it. Without a callback
the proposal is `queue_discovery` with `executed: false`, which is a recorded
recommendation, not work.

#### Honest limits

- **Single process, no scheduler and no market clock.** `scan()` is a call.
  Nothing polls the ledger, sleeps or decides that a bar has arrived; a daemon
  or CLI is future work.
- **Read cost grows with the ledger.** Each scan reads every session's signals,
  trades and positions through `matured_outcomes`. The `max_sessions` cap bounds
  one scan; a larger ledger is covered over several scans by the rotation
  offset, so a full cycle costs proportionally more wall time.
- **Rotation is offset-based.** Inserting a session or changing its activity
  between scans can shift the offset, so a session may be visited twice (the
  fingerprint makes that a no-op) or skipped until the next wrap. It cannot
  disappear permanently.
- **Unacknowledged triggers are the caller's debt.** `pending_ack` is retried at
  the start of every scan until acknowledged; a caller that scans but never
  calls `seen()` keeps re-reading them, which is a protocol violation, not a
  safe steady state.
- **No backfill.** A session without a dataset id stays refused; the trigger
  does not modify the ledger to manufacture attribution.
- **The sets are unbounded.** Fingerprints and quiet refusal reasons accumulate;
  no pruning or compaction exists yet. A `since` window trims the sessions read,
  not the cursor document.
- **A trigger is not evidence.** It says a paper batch completed, not that the
  strategy has an edge. The queue step and any later evaluation remain separate
  reviewed decisions, and the final OOS/holdout split is never touched here.


## Verification

```
cd backend
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/test_orchestrator_store.py tests/test_orchestrator_core.py \
  tests/test_orchestrator_cancel.py tests/test_orchestrator_handlers.py \
  tests/test_orchestrator_resources.py tests/test_orchestrator_daemon.py \
  tests/test_orchestrator_integration.py -q
uv run pytest
```

The slice is accepted on those commands passing, not on the design above. The
queue tests cover idempotent enqueue, priority and insertion order, schedule
gating, dependency order, dependency failure to `skipped` (including a cascade),
retry backoff bounds and exhaustion, permanent failures, cancel while queued and
while running, pause/resume/safe-mode/disable-research (including persistence
across a reopen), crash recovery with and without attempts left, one heavy job,
guard refusal without spending an attempt, the deferral cap turning a starved
job terminal, the v1→v2 deferral migration, event-log completeness,
`run_until_idle(should_stop=...)`, and file-trigger dedupe across a store
restart. The handler tests cover a real
ingest/index/extract round trip, checksum idempotency, dataset manifest
reproducibility, an unknown symbol refusal, a real bounded discovery run that
records no promotion, a no-champion NO TRADE that writes nothing, a governed
champion paper advance with queue dedupe, an invalid champion payload, closed-
only outcome analysis with an open position excluded, and permanent failure for
invalid payloads with the retry budget untouched.

### Cancellation, retry and deferral contract

`tests/test_orchestrator_cancel.py` pins the semantics that decide whether work
may be banked or retried, separately from the store's single-transition tests:

- **Cancel queued.** The row is terminal before any worker sees it. A second
  cancel is a no-op with no second event, and a sibling job still runs.
- **Cancel while running.** The handler is not stopped. After it returns, the loop
  compares the row, writes one `cancelled_while_running` event and discards both
  a successful and a failed result. Nothing is requeued.
- **Cancel terminal or unknown.** A succeeded or failed row refuses cancellation
  with `JobStateError`; an unknown id raises `UnknownJobError` and writes no
  audit line.
- **Retry.** Backoff is `base_seconds * factor ** (attempt - 1)`, capped at
  `max_seconds`; the job is idle before its delay, spends one attempt per run and
  fails after `max_retries + 1` attempts. `PermanentJobError` and
  `HandlerResult.failure(..., permanent=True)` fail on the first attempt without a
  `retry` event. The store's own `retry` guard refuses a spent budget
  independently of the orchestrator.
- **Defer ceiling.** A guard refusal restores the claimed attempt. After
  `max_deferrals` refusals the next one turns the job `failed` with
  `deferral cap reached ...` and a `defer_exhausted` event; the terminal claim is
  not restored and the handler never ran.
- **Drains.** `should_stop` is a boundary between jobs, never inside one. A pause
  raised from a handler stops the current drain, and `resume()` lets the next one
  finish the remainder. A cancelled dependency skips its dependents, cascading to
  grandchildren, and a dedupe key still returns the failed job instead of
  reviving it.

Mutation check (each patch applied alone, run against this file, then reverted;
the committed tree is green):

| Mutation | Test that fails |
|---|---|
| delete the `current.status != "running"` guard in `core._execute` | `test_cancel_while_running_discards_the_result`, `..._a_failure_too` |
| reduce `_fail`'s condition to `permanent` only (drop `attempts > max_retries`) | `test_retry_backoff_is_capped_then_terminal` (plus `test_cancel_terminal_job_is_refused` and the idempotent-enqueue test, which share the failure path) |
| delete the `attempts > max_retries` guard in `store.retry` | `test_store_retry_refuses_a_spent_budget` |
| delete the `defer_count >= max_deferrals` branch in `store.defer` | `test_defer_ceiling_turns_a_starved_job_terminal` |

What remains unproven: cancellation is cooperative. There is no preemption, no
signal, thread kill or cancellation token. A running handler keeps its CPU and
I/O until it returns; only its result is discarded, and a handler that never
returns holds the claimed row. Cancellation and pause are observed at loop
boundaries, and no test drives two processes over one database.

The measured-guard tests cover: disabled limits allow; each limit refuses at its
boundary and allows just inside it; reasons carry the limit and the measured
value; the documented limit order; probe exceptions and wrong return types fail
closed with `probe_failed`; non-finite snapshots fail closed; cache reuse, zero
cache and a backwards clock; the macOS byte / Linux KiB `ru_maxrss` branch; the
real `SystemProbe` on the host; `measure` recording finite wall/CPU/RSS values
and a failed block without masking it; a bounded sample ring; JSON-safe
payloads; frozen profiles; and one end-to-end check that `Orchestrator` defers a
job with the measured reason in its event log.

The paper-outcome trigger tests (`tests/test_paper_outcome_triggers.py`) cover: a
new fingerprint enqueues one job; the store key dedupes a crash before
acknowledgement; the cursor dedupes a repeat scan and a store restart; an open
position is counted but never in the payload; missing strategy hash and dataset
id are refused with a logged reason that is not repeated; an empty batch and a
missing ledger are no-ops; the handler returns a proposal and creates no second
job; the policy callback can decline; and one more closed trade changes the
fingerprint. They also cover the review fixes: activity order keeps a busy old
session ahead of a newer quiet one; the rotation cursor visits every session
beyond `max_sessions` and each deferred session is recorded with `truncated`,
without a stale note for a session whose trigger already exists; a partial
acknowledgement is re-emitted at the start of the very next scan; a transient
handler failure is retried before the batch is burned; `source`/`synthetic`
provenance is forwarded and absent provenance stays `null`; the handler store is
a read-only facade with no `enqueue` or cross-job `cancel`; an injected `kind`
and an out-of-range scan bound are rejected; and `since` plus `max_proposals`
keep a large historical ledger from emitting an unbounded backlog.

The daemon tests cover: config bounds and profile/type selection; tick scans
before it drains; trigger dedupe across ticks; pause and safe mode still scan and
enqueue but claim nothing; disabled research leaves research queued while
ingestion runs; a guard refusal defers without spending an attempt and records
its reason; the deferral cap fails the job after the configured refusals; a
raised heavy handler does not stop a later paper job on the next tick; recovery
happens once, `start()` is idempotent and keeps one worker thread; a stop during
blocked recovery is not lost and leaves no thread; `stop()` joins the loop from
the caller thread and slices a 60 s poll into ≤0.25 s waits; the drain stops
between jobs; `:memory:` is rejected by `start()` and stays single-threaded for
`status()`; `status()` is readable from another thread for a file store and is
JSON-safe; the CLI `--once` exits 2 on a tick error but 0 on a job failure;
`--forever` reports a dead worker with exit 1; and `--watch-dir` enqueues a
document once and dedupes the second pass.

### Integration coverage (Phase 18)

`tests/test_orchestrator_integration.py` runs the real store, loop, handlers,
trigger sources and daemon together on `tmp_path` roots and one day of seeded
synthetic M1. It asserts what the per-seam tests cannot: that the seams agree
with each other.

Proven by that file:

- **Dependency chain.** One `Orchestrator` runs `knowledge.ingest` and then
  `dataset.build`; the child is not claimable before the parent succeeds, both
  results are stored (document checksum and index rows, dataset manifest on
  disk) and the event log names every transition in order.
- **Failure isolation with a budget.** A `discovery.run` handler that raises
  spends its one retry, leaves a light `paper.advance` job free to succeed in
  the same drain, then fails terminally; its dependent `paper.analyse` is
  `skipped` with the failed dependency named, and no third attempt exists an
  hour later.
- **Trigger → proposal.** `PaperOutcomeTriggerSource` + `scan_once` enqueue one
  `paper.outcomes` job from a matured ledger batch; the cursor dedupes after a
  store restart, the store dedupe key dedupes after the cursor is lost, and the
  handler records a proposal with `queued: false` / `executed: false`, creating
  no discovery or paper job and no archive.
- **Daemon tick.** One tick recovers a `running` row (once per instance), scans,
  and drains both the recovered paper job and the scanned outcome job; pause and
  safe mode refuse claims with their reason in `claim_refused`; a measured guard
  backed by a fake starved probe defers a job (`defer_count = 1`,
  `attempts = 0`, `resource_refused` and `deferred` events, 60 s delay).
- **Restart.** A reopen keeps queued rows, `recover_interrupted` requeues the
  running row with its spent attempt intact, a deduplicated enqueue returns the
  succeeded job, and each handler runs exactly once.
- **Boundary.** No module in the flow imports a holdout or real-money module
  (static import scan plus a `sys.modules` diff around a live run). During a
  real, bounded discovery run the only slice built is
  `[validation.start, validation.end)`, every scored frame lies inside it, the
  recorded dataset id starts with `validation:` and trips no production
  holdout token.

Not proven by it:

- **`paper.advance` still uses the slice runner.** The handler calls
  `AutonomousPaperRunner` (`paper/autonomous.py`), not
  `ContinuousPaperRunner` (`paper/continuous.py`). The queue path is proven for
  the slice runner only; continuous cadence, gap and resume semantics are not
  reachable from `paper.advance` yet.
- **`tick(now=...)` does not reach crash recovery.** `Orchestrator.start()` has
  no `now` parameter, so `recover_interrupted` stamps a requeued row with the
  wall clock. The integration test therefore drives the daemon with the wall
  clock; a purely simulated clock would leave the recovered job scheduled in the
  future. Production calls `start()` with `now=None`, so this is a test-seam
  limitation, not a runtime defect.
- **A proposal is still not executed.** The paper-outcome handler records
  `queue_discovery`; nothing turns that proposal into a queued discovery job.
- **Synthetic inputs only.** One day of seeded M1 proves wiring, not market
  behaviour, data quality or edge. No real bar, ledger or document is read.
- **One process.** Nothing here exercises a second daemon, a lease, or two
  processes over one database; the ownership limit above still stands.


