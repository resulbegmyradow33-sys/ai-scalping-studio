# Operator runbook

Practical, copy-paste instructions for running the current system on one machine.
Every command and flag was checked against the parser (top-level `--help` and
`backend/src/scalping/cli.py`) and executed where the text says "verified".
Anything not executed is marked **[UNVERIFIED]**.

Snapshot: branch `feat/finaldocs`, base `main` at `54eef5e` (2026-11-25,
#220). Sections 2.6, 2.7 and 3.2 cover the runtime promotion command and the
paper scheduler merged in #218/#220. The system is paper-only. Real-money
execution is disabled in code.

## 1. Prerequisites and install

Target machine is a Mac mini M2 with 8 GB RAM. One heavy job at a time.

| Requirement | Version | Check |
|---|---|---|
| macOS | any current | `sw_vers` |
| Python | `>=3.12,<3.13` (`backend/pyproject.toml`) | `python3.12 --version` |
| uv | 0.11+ | `uv --version` |
| Node.js | 20+ (Next 15) | `node --version` |
| npm | 10+ | `npm --version` |
| libomp | Homebrew | `brew list libomp` |

Install:

```bash
# from the repository root
brew install libomp          # XGBoost/LightGBM OpenMP runtime on macOS
uv sync --project backend    # creates backend/.venv
npm ci --prefix frontend     # frontend dependencies
```

`uv sync --project backend` verified. `npm ci` **[UNVERIFIED]** in this session
(network install not run); `frontend/package.json` is the source of truth.

If `import xgboost` or `import lightgbm` fails with a missing `libomp.dylib`,
install libomp and rerun the same Python. Both libraries import cleanly on the
machine used for this runbook; the failure itself was not reproduced.

### Where state lives

Default roots, relative to the repository root:

```
data/                              # Parquet bar store (CLI --root)
  XAUUSD/M1/2026/01.parquet        # <symbol>/<timeframe>/<YYYY>/<MM>.parquet
  _meta/XAUUSD/M1.json             # provenance: source, synthetic flag, checksum
  .feature-cache/<xx>/<hash>.npz   # rebuilt when bars/settings change
  datasets/<dataset_checksum>/     # artifact.json + X.npy, y_*.npy, ts.npy, split.npy
  paper.sqlite                     # paper ledger (CLI archive-root / API default)

runs/                              # run products (CLI --runs-root)
  orchestrator.sqlite3             # job queue, events, persisted controls
  discovery/discovery.sqlite       # candidates, trials, decisions, champions
  datasets/<manifest>.json         # dataset.build manifests
  acceptance/<acceptance-id>/report.json

runs/evidence.json                 # only where --out points; there is no default
runs/evidence.spec.json            # sidecar written next to every evidence bundle
```

`data/` and `runs/` are gitignored. Never commit either. All paths default
relative to the current directory, so running from `backend/` without `--root`
writes `backend/data/` and `backend/runs/`. Pass explicit absolute paths, or
always run from the same directory.

All stored bar timestamps are UTC. A CSV import converts from the broker offset
(`--tz-offset`, assumed `+3` hours when omitted).

Notes verified from code and files on disk:

- the store is month-partitioned Parquet, one file per `<symbol>/<timeframe>/<year>/<month>`;
- `_meta/<symbol>/<timeframe>.json` carries `synthetic: true|false`, a checksum
  and a `dataset_id`;
- the dataset artifact is written last and is the commit marker for
  `<root>/datasets/<checksum>/`;
- the paper ledger and orchestrator queue are separate SQLite files; the
  discovery archive is a third.

## 2. Daily and once-only flows

All commands below run from the repository root. The equivalent from `backend/`
is `cd backend && uv run python -m scalping.cli ...` (paths then need `../`).
Every command except `orchestrator run --forever` prints exactly one JSON line
on stdout. Diagnostics go to stderr; an error is one JSON line there. Exit codes:
`2` for input/data problems, `1` for unexpected failure, `0` for success.

### 2.1 Import and validate

Synthetic data (safe, deterministic, `synthetic: true` in provenance):

```bash
uv run --project backend python -m scalping.cli data synth \
  --root data --symbol XAUUSD --start 2026-01-05 --end 2026-02-03 --seed 7
```

Broker CSV (defaults to a `+3` hour broker offset; pass `--tz-offset` to override):

```bash
uv run --project backend python -m scalping.cli data csv \
  --path export.csv --symbol XAUUSD --tz-offset 3 --root data
```

JSON array of bars, and timeframe derivation:

```bash
uv run --project backend python -m scalping.cli data json --path bars.json --root data
uv run --project backend python -m scalping.cli data derive \
  --root data --symbol XAUUSD --source M1 --timeframes M5,M15,M30,H1,H4,D1
```

Validate the stored slice:

```bash
uv run --project backend python -m scalping.cli data validate \
  --root data --symbol XAUUSD --timeframe M1
```

Expected keys: `symbol`, `timeframe`, `rows`, `ok`, `first_ts`, `last_ts`,
`issues[]` where each issue has `code`, `severity`, `count`, `detail`, `sample`.
`ok: true` with warnings is normal (weekend gaps, a few outliers).

### 2.2 Build a dataset manifest

```bash
uv run --project backend python -m scalping.cli dataset build \
  --root data --symbol XAUUSD --timeframe M1 \
  --out runs/datasets/xauusd-m1.json
```

Expected keys: `manifest`, `dataset_id`, `rows`, `columns`, `train`,
`validation`, `test`, `purged`.

Default `--timeframe` is **H1**, not M1. If your store only has M1, omitting the
flag fails with `no XAUUSD H1 bars under ...`. Pass `--timeframe M1` explicitly.

### 2.3 Discovery run (validation slice only)

```bash
uv run --project backend python -m scalping.cli discovery run \
  --root data --symbol XAUUSD --timeframe M1 \
  --max-trials 24 --population 8 --generations 4 --seed 0 \
  --archive runs/discovery/discovery.sqlite \
  --out runs/discovery/last.json
```

Expected keys: `symbol`, `timeframe`, `dataset_id`, `validation` (`start`,
`end`, `rows`, `first_ts`, `last_ts`), `seed`, `code_version`, `trials`,
`skipped`, `deduped`, `accepted`, `stop_reason`, `rejections`, `best`
(`candidate_id`, `canonical`, `metrics`), `baseline[]`, `comparison`,
`decision`, `reason`, `decision_id`, `decision_candidate`, `archive`, `out`.

`decision` is `no_trade`, `no_change` or `rejected`. None of them is a
promotion. `--max-trials 24` is a wiring smoke budget, not a search.

The archive path must match what the orchestrator reads:
`<runs-root>/discovery/discovery.sqlite`.

### 2.4 Evidence build (persist)

```bash
uv run --project backend python -m scalping.cli discovery evidence \
  --root data --symbol XAUUSD --timeframe M1 \
  --spec runs/candidate.spec.json \
  --out runs/evidence.json \
  --folds 3 --bootstrap-samples 1000 \
  --min-trades-per-fold 5 \
  --search-budget 100 --trials 1
```

The spec must already exist: save the candidate's `best.canonical` from
`runs/discovery/last.json` to `runs/candidate.spec.json`, or pass inline JSON.
The command does not invent a strategy. A `StrategySpec` payload looks like
`{"kind":"strategy","spec_version":1,...}`. The command writes the dataset
artifact under `<root>/datasets/<dataset_checksum>/` and copies the spec to the
sidecar next to the bundle (`<bundle stem>.spec.json`).

Expected keys: `persisted`, `symbol`, `timeframe`, `dataset_id`,
`dataset_checksum`, `artifact`, `bundle`, `spec`, `evidence_checksum`,
`schema_version`, `source`, `synthetic`, `folds`, `validation_end_row`,
`test_start_row`.

The bundle itself contains `aggregate`, `folds[]`, `splits`, `policy`,
`checksum`, `locked_oos`, `spec_hash`, `trial_count`, `search_budget`,
`source`, `synthetic` and identity fields. `locked_oos` is `null` until a real
holdout is attached. `source`/`synthetic` come from the dataset sidecar; a
config/manifest disagreement is a `provenance_mismatch` refusal, and an unknown
dataset is never silently recorded as real. From this point on, promotion
refuses a bundle whose provenance is missing (`provenance_unknown`) or
synthetic (`synthetic_data`). Evidence built before #216 is schema v3 and must
be rebuilt; the strict reader refuses it rather than guessing provenance.

`--stress ""` is refused: `missing_cost_stress: missing_cost_stress: at least
one cost-stress multiplier must be greater than 1.0` (the prefix is duplicated
by the library).

### 2.5 Reproduce

```bash
uv run --project backend python -m scalping.cli discovery reproduce \
  --bundle runs/evidence.json --store-root data \
  --out runs/reproduce.json
```

`--store-root` is the bar store root that also contains `datasets/<checksum>/`.
The sidecar spec is used automatically; pass `--spec` only to override it.

Success keys: `reproduced: true`, `checksum`, `expected`, `mismatches: []`.
A tampered bundle is refused before recomputation with
`{"error": "checksum_mismatch: bundle checksum '...' does not match '...'"}`,
exit code `2`.

### 2.6 Promotion: the owner-approved runtime sequence

`discovery promote` is the runtime call site for DSL evidence promotion. It is
the only CLI path, and only a registered production policy may run it.
`PROMOTION_V1` is the only entry in `PRODUCTION_POLICIES`. The legacy
`/api/strategy/calibrate` path promotes the old card; it is not this path.

Run the four steps in order. Step 3 is the only irreversible one.

**Step 1 — persist evidence.** Complete §2.4 first and confirm the bundle
records real provenance (`source` names the data, `synthetic` is `false`).
`locked_oos: null` is expected here: the holdout is still sealed.

```bash
uv run --project backend python -m scalping.cli discovery evidence \
  --root data --symbol XAUUSD --timeframe M1 \
  --spec runs/candidate.spec.json \
  --out runs/evidence.json
```

**Step 2 — dry run.** Evaluates `PROMOTION_V1` and writes neither the archive
nor the holdout ledger. Always run this first.

```bash
uv run --project backend python -m scalping.cli discovery promote \
  --archive runs/discovery/discovery.sqlite \
  --candidate "<candidate_id from discovery run>" \
  --evidence runs/evidence.json \
  --data-root data \
  --reason "promotion-v1 dry run" \
  --dry-run --out runs/promotion.dryrun.json
```

A dry run that is rejected, or that reports `missing_locked_oos`, is the
expected outcome until step 3. **Do not run step 3 to "see what happens".**

**Step 3 — the one-shot holdout spend.** This is a governance act, not a test.
The evidence must pass every non-holdout check first; the holdout is spent only
when `--spend-holdout` and a non-empty owner token are both present. The token
is read from the environment or a `0600` file, checked only for
non-emptiness, and never printed, stored or echoed:

```bash
# owner supplies the token out of band; one line, chmod 600, never committed
export SCALPING_APPROVAL_TOKEN="$(cat ~/.secrets/xauusd-approval.token)"

uv run --project backend python -m scalping.cli discovery promote \
  --archive runs/discovery/discovery.sqlite \
  --candidate "<candidate_id from discovery run>" \
  --evidence runs/evidence.json \
  --data-root data \
  --scope XAUUSD:M1 \
  --reason "owner approved the one-shot holdout spend" \
  --spend-holdout \
  --approval-token-env SCALPING_APPROVAL_TOKEN \
  --out runs/promotion.json
```

`--approval-token-file ~/.secrets/xauusd-approval.token` is the equivalent
file form; the two flags are mutually exclusive. The ledger is
`runs/discovery/holdout.sqlite`, beside the archive.

Safety warnings for step 3:

- **One-shot.** One spend per `(dataset_id, dataset_checksum)`. A second
  attempt exits `2` with `holdout_spent`, even if the first evaluation failed.
  A failure after the unlock leaves the holdout spent with no result.
- **No token logging.** Never pass the token as an argument, never `echo` it,
  never commit it, and never add it to a log or the `--out` record. Prefer
  `--approval-token-env`; delete the exported variable after the command
  (`unset SCALPING_APPROVAL_TOKEN`) if the shell is shared.
- **No retries.** Do not delete or edit the ledger, rebuild the dataset to get a
  new checksum, or mint a second token to repeat a losing evaluation. That is
  tampering, not a retry.
- **No concurrency.** One promotion at a time; two processes over the same
  archive and ledger are unsupported.

**Step 4 — paper scheduler.** Only after step 3 accepted a champion. The
scheduler reads the discovery archive and stored bars and enqueues one
`paper.advance` per scope for the newest **decidable** bar. A repeat scan of the
same bar is deduplicated; the ledger cursor is the idempotency boundary, so a
restart cannot double-advance:

```bash
uv run --project backend python -m scalping.cli orchestrator run --forever \
  --schedule-paper \
  --db runs/orchestrator.sqlite3 --root data --runs-root runs --archive-root data
```

Without an evidence champion it records `paper.advance_refused` and enqueues
nothing. It never reads a live feed or invents a bar: if the newest stored bar
is the provisional tail, the scope waits for the next one. A gap still reaches
the continuous runner unchanged under the default `strict` policy.

Exit codes: `0` accepted, `1` rejected by the policy, `2` refused (non-production
policy, missing approval, spent holdout, identity mismatch, missing dataset or
spec) or invalid input. `--out` receives the decision record on every path,
including refusals. The record carries the code list, policy id/version/digest
and evidence/dataset identity; it never carries the token. If the evidence was
built with a settings YAML, pass the same `--settings`; a different digest is an
`identity_mismatch` refusal, not a silent re-run.

The library call remains available; it is what the command uses:

```python
import json
from pathlib import Path

from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.evidence import evidence_from_dict
from scalping.discovery.promotion import (
    PROMOTION_V1, ExpectedIdentity, promote_with_evidence,
)

bundle = evidence_from_dict(json.loads(Path("runs/evidence.json").read_text()))
archive = DiscoveryArchive(Path("runs/discovery/discovery.sqlite"))
try:
    champion = promote_with_evidence(
        archive,
        "<candidate_id from discovery run>",
        bundle,
        PROMOTION_V1,
        scope="XAUUSD:M1",
        reason="promotion-v1 accepted identical folds",
        expected_identity=ExpectedIdentity.current(
            dataset_checksum=bundle.dataset_checksum,
        ),
    )
finally:
    archive.close()
```

Rules enforced by the call, verified on the probe bundle:

- a modified policy copy is refused:
  `non_production_policy: only a registered production policy may promote; ... is not registered`;
- a bundle without a locked OOS section is rejected with the code
  `missing_locked_oos` and the message
  `the policy requires a locked out-of-sample section and the bundle has none`;
- the rejection list is returned in `PromotionRefused`, e.g.
  `min_positive_fold_fraction; min_expectancy_r; min_profit_factor;
  min_ci_lower_r; min_bootstrap_samples; cost_stress; stress_degradation;
  missing_locked_oos`;
- a bundle with no provenance is rejected with `provenance_unknown`, and a
  bundle measured on synthetic data with `synthetic_data`; a candidate whose
  provenance differs from the incumbent is `champion_provenance_mismatch` (all
  #216).

A test policy is registered only through a private, test-only context manager.
Operators cannot use one. "Test policy" in this repository means a weaker policy
used inside tests, never a promotion path.

**No successful promotion was executed here** **[UNVERIFIED]**: the probe bundle
had no locked OOS, so only rejection paths were observed. Do not claim a champion
exists until `promote_with_evidence` returns a record.

### 2.7 Paper advance

Paper advance is an orchestrator job kind, not a CLI subcommand. It reads the
governed champion for `(<symbol>, <timeframe>)` from
`<runs-root>/discovery/discovery.sqlite`; with no champion or no archive it
records NO TRADE and touches nothing.

Two ways to produce the job: the scheduler reads stored bars and enqueues one
advance per evidence champion when a new decidable bar exists (§2.6 step 4,
`--schedule-paper`), or a caller enqueues one explicitly. The scheduler is the
supported unattended path; the explicit enqueue is for a controlled probe.

```python
from pathlib import Path
from scalping.orchestrator.store import JobStore

store = JobStore(Path("runs/orchestrator.sqlite3"))
job, created = store.enqueue(
    "paper.advance",
    dedupe_key="paper:XAUUSD:M1:2026-11-24",
    payload={"symbol": "XAUUSD", "timeframe": "M1", "bars": 50},
)
store.close()
```

Then drain once:

```bash
uv run --project backend python -m scalping.cli orchestrator run --once \
  --db runs/orchestrator.sqlite3 --root data --runs-root runs --archive-root data
```

Observed NO TRADE result (verified):

```json
{"action":"no_trade","bars_advanced":0,"bars_requested":50,"champion":null,
 "ledger_created":false,"mode":"continuous",
 "reason":"no discovery archive at runs/discovery/discovery.sqlite; scope 'XAUUSD:M1' has no governed champion, so the autonomous path does not trade",
 "reason_class":"no_champion","touched":false}
```

With a real champion the same job returns `action: "advanced"`, `session_id`,
`spec_hash`, `candidate_id`, `promotion_id`, `bars_advanced`, `new_signals`,
`new_trades`, `realized_pnl`, `balance`, `equity`, `position`, `last_bar_ts`,
`done`. That path is **[UNVERIFIED]** here because no evidence champion exists.

The scheduler refuses a scope with an audit event instead of a job. The reason
is one of `no_archive`, `archive_unavailable`, `no_champion`,
`no_evidence_champion`, `invalid_champion`, `invalid_scope`, `no_bars`,
`bars_unavailable`, `ledger_unavailable`, `warming_up` or `awaiting_bar`.
Exploratory and manual champions are specifically refused; there is no legacy
fallback. The HTTP API under `/api/paper/sessions` records a session row; it
does not run or advance the engine. Creating a session is not paper trading.

### 2.8 Autonomy status

Read-only. Start the API, then query:

```bash
uv run --project backend python -m scalping.cli serve --root data --port 8000
curl -s http://127.0.0.1:8000/api/autonomy/status
```

Expected top-level keys: `paper`, `orchestrator`, `champions`,
`champions_readable`, `discoveries`, `data`. Each store section carries
`configured` and `readable`; counts are omitted when `readable` is false. This is
deliberate: a missing/corrupt store is not reported as zero.

Related reads: `/api/autonomy/jobs?limit=50`,
`/api/autonomy/events?limit=100`, `/api/autonomy/champions`.
All are GET-only. Nothing there enqueues, cancels, promotes or starts a daemon.

### 2.9 One-shot synthetic acceptance

The fastest honest check that the pipeline pieces connect:

```bash
uv run --project backend python -m scalping.cli acceptance run \
  --out runs/acceptance
```

Verified output keys: `status`, `acceptance_id`, `report`, `synthetic`,
`claims` (`implemented`, `tested`, `validated`, `paper_traded`, `profitable`),
`stages[]`, `store_path`, `store_retained`. On this tree the run passed, with
`holdout` reported `skipped`, `synthetic: true`, `validated: false`,
`profitable: false`. That is the correct claim level: wiring works, no market
edge is claimed.

## 3. Orchestrator

The orchestrator is one process with a SQLite queue. There is no launchd or
systemd unit; you start and stop it yourself.

### 3.1 One tick or a loop

```bash
# one deterministic pass, one JSON line
uv run --project backend python -m scalping.cli orchestrator run --once \
  --db runs/orchestrator.sqlite3 --root data --runs-root runs --archive-root data

# supervised loop; SIGINT/SIGTERM stops it cleanly
uv run --project backend python -m scalping.cli orchestrator run --forever \
  --poll-seconds 5 --db runs/orchestrator.sqlite3 \
  --root data --runs-root runs --archive-root data
```

`--once` keys: `status` (`running`, `worker_ident`, `paused`, `safe_mode`,
`disabled_research`, `queue`, `ticks`, `terminal_jobs`, `last_tick_at`,
`last_tick_seconds`, `last_error`, `recovered_jobs`, `guard`, `config`) and
`tick` (`started_at`, `finished_at`, `elapsed_seconds`, `scan`, `outcomes`,
`error`).

`--forever` prints `{"status":"started",...}`, then `{"status":"stopped",...}`
on a requested stop (exit 0), or `{"status":"failed",...}` if the worker died
without a stop request (exit 1). Verified: start, SIGTERM after a few seconds,
`stopped`, exit 0.

Defaults: `--profile m2_8gb` installs the measured guard; `--poll-seconds 5`;
`--max-jobs-per-tick 10`; `--stop-grace-seconds 5`; database
`runs/orchestrator.sqlite3`.

### 3.2 Watch directories

```bash
# documents -> knowledge.ingest
uv run --project backend python -m scalping.cli orchestrator run --forever \
  --watch-dir docs/research --watch-kind knowledge.ingest

# market-data root -> one dataset.build per <symbol>/<timeframe>
uv run --project backend python -m scalping.cli orchestrator run --forever \
  --watch-data-dir data

# evidence champions + stored bars -> one paper.advance per new decidable bar
uv run --project backend python -m scalping.cli orchestrator run --forever \
  --schedule-paper --db runs/orchestrator.sqlite3 \
  --root data --runs-root runs --archive-root data

# paper ledger -> matured-outcome proposal (no advance)
uv run --project backend python -m scalping.cli orchestrator run --forever \
  --watch-paper-ledger runs/paper.sqlite
```

`--watch-data-dir` accepts only `<symbol>/<timeframe>/.../<file>.parquet|.csv`.
A file outside that layout is recorded as `data.trigger_refused` with
`unknown_layout`, `invalid_symbol` or `unknown_timeframe`; it is not guessed.
`--watch-dir` and `--watch-data-dir` are mutually exclusive, and
`--watch-kind dataset.build` is rejected with a pointer to `--watch-data-dir`.
`--schedule-paper` can be combined with either directory watcher and with
`--watch-paper-ledger`; it is a CLI error with `--no-scan`.

Data triggers never emit `discovery.run`: a filename cannot supply a gate,
budget and seed. `--schedule-paper` never promotes and never calls a feed; it
only turns an already-stored, already-decidable bar into a queued advance.

### 3.3 Controls

Controls are persisted rows in the queue's `state` table and are read on every
iteration, so they survive a restart. Audited local owner controls are exposed
through `GET` and `POST /api/orchestrator/control` (#175, merged). These
requests change persisted flags only: they do not start/stop the daemon, enqueue
jobs, cancel jobs, or promote a strategy.

Start the API on loopback (the default):

```bash
uv run --project backend python -m scalping.cli serve --root data --host 127.0.0.1 --port 8000
```

Read the current state before changing it:

```bash
curl -s http://127.0.0.1:8000/api/orchestrator/control
```

Change one control at a time. A reason is required for `pause`,
`safe_mode_on` and `safe_mode_off`; it is optional for `resume`,
`disable_research` and `enable_research`:

```bash
# Stop claiming jobs; trigger scanning can continue.
curl -s -X POST http://127.0.0.1:8000/api/orchestrator/control \
  -H 'Content-Type: application/json' -d '{"action":"pause","reason":"maintenance"}'

# Hard stop: the daemon claims no jobs.
curl -s -X POST http://127.0.0.1:8000/api/orchestrator/control \
  -H 'Content-Type: application/json' -d '{"action":"safe_mode_on","reason":"operator stop"}'

# Resume claiming after checking the queue and evidence.
curl -s -X POST http://127.0.0.1:8000/api/orchestrator/control \
  -H 'Content-Type: application/json' -d '{"action":"resume"}'

# Block dataset/discovery/paper research jobs while leaving ingestion possible.
curl -s -X POST http://127.0.0.1:8000/api/orchestrator/control \
  -H 'Content-Type: application/json' -d '{"action":"disable_research"}'
```

Allowed actions: `pause`, `resume`, `safe_mode_on`, `safe_mode_off`,
`disable_research`, `enable_research`. The response reports
`configured`, `readable`, `paused`, `safe_mode`, `research_disabled`
and the latest `last_change` audit event. A missing database returns
`configured: false, readable: false`; an existing but unreadable or
uninitialised database returns `readable: false` rather than pretending all
flags are off. POST returns `409` when the database is absent/unavailable and
`422` for an unknown action, a missing required reason or an extra body field.
Each accepted transition is appended to the audit event log.

**Security boundary:** this API has no authentication/identity layer. Keep the
server bound to `127.0.0.1`; do not expose it on a public interface or forward
port 8000 to the internet. CORS is not authentication. The route uses the
database path configured by the app and rejects arbitrary paths in the request.

The Python API remains available for local scripts and tests:

```python
from pathlib import Path
from scalping.orchestrator.core import Orchestrator
from scalping.orchestrator.store import JobStore

store = JobStore(Path("runs/orchestrator.sqlite3"))
orchestrator = Orchestrator(store)
orchestrator.pause("maintenance")
orchestrator.resume()
orchestrator.safe_mode(True, "stop")
orchestrator.disable_research(True)
orchestrator.cancel(42, "superseded")  # job cancellation is Python-only
store.close()
```

`safe_mode` is the hard stop. `pause` stops claims but keeps scanning and
recording new work. `disable_research` keeps ingestion possible. A control
change takes effect when the daemon next reads persisted state; it does not
interrupt a handler already running.

### 3.4 Crash recovery

On start, every `running` row is marked `interrupted` with
`interrupted: process restarted`. If attempts remain (`attempts <= max_retries`)
it is requeued immediately; otherwise it stays `interrupted` as a terminal
record. Recovery runs once per daemon instance, before the first tick.

One daemon per database. Two daemons can treat each other's running rows as
interrupted and run the same job twice. There is no lease in this slice.

### 3.5 Logs and stopping safely

- stdout/stderr: one JSON line per CLI event. Redirect them if you run detached.
- audit trail: the `events` table in `runs/orchestrator.sqlite3`.
- read-only query:

```bash
sqlite3 "file:$PWD/runs/orchestrator.sqlite3?mode=ro" \
  "select id, ts, event, job_id from events order by id desc limit 20;"
```

To stop: send SIGINT or SIGTERM once and wait up to `--stop-grace-seconds`.
The loop stops after the current tick. A handler already running is not
cancelled; a `paper.advance` mid-write completes or fails on its own. Do not
`kill -9` unless the process is already wedged, and do not start a second
daemon while the first is alive.

## 4. Troubleshooting

| Symptom | Exact message / signal | Cause | Fix |
|---|---|---|---|
| XGBoost/LightGBM import fails | `Library not loaded: @rpath/libomp.dylib` | macOS has no OpenMP runtime | `brew install libomp`, rerun. **[UNVERIFIED]** here (imports worked) |
| Validation is empty | `{"error": "no XAUUSD M1 bars under ..."}` exit 2 | wrong `--root`, wrong symbol/timeframe, or nothing ingested | ingest first; check `data/<symbol>/<timeframe>/` |
| Dataset build says no H1 bars | `no XAUUSD H1 bars under ...` | `dataset build` defaults to `--timeframe H1` | pass `--timeframe M1` |
| Discovery has no validation rows | `the built dataset has no validation rows; discovery may not read the test split, so there is nothing to search on` | too little history for the split geometry | ingest more bars or lower `--purge`/`--embargo`; never shrink the test split |
| Validation slice too short | `the validation split has N rows; a W-bar observation window needs at least W+1 to make one decision` | `--window` too large for the slice | ingest more data or reduce `--window` |
| Promotion rejected | code `missing_locked_oos`; message `the policy requires a locked out-of-sample section and the bundle has none` | bundle built by `discovery evidence` has `locked_oos: null` | rerun `discovery promote` with `--spend-holdout` and an owner approval token; do not bypass the policy |
| Promotion refuses to spend | code `approval_missing` or `holdout_not_spent` | no token file/env, or `--spend-holdout` omitted | supply both; the holdout is one-shot and owner-approved |
| Promotion refused outright | `non_production_policy: only a registered production policy may promote; '<id>' v1 digest ... is not registered` | modified/copied policy, or a test policy | use `PROMOTION_V1` unchanged |
| Promotion refuses provenance | code `provenance_unknown` or `synthetic_data` | bundle predates schema v4, or the dataset is synthetic | rebuild evidence on real bars; do not add an `allow_synthetic` production policy |
| PDF import fails | `PDF text extraction needs the optional pypdf extra: run 'uv sync --extra pdf' ...` | extra not installed | `cd backend && uv sync --extra pdf`; the file is preserved either way |
| Paper advance job fails as retryable | job error prefix `bar_gap:`; message `session '...': missing bar at <ts> before <ts>` | a bar is missing at the timeframe cadence | ingest the missing bar, then let the retry run. `bar_order`, `session_conflict`, `continuous_refused` are permanent |
| Holdout cannot be spent twice | `holdout_spent: holdout was already spent at <ts>; a second spec cannot reuse it` | the one-shot ledger already spent this `(dataset_id, dataset_checksum)` | use a new dataset checksum; do not delete the ledger |
| Autonomy page/API shows no numbers | `configured: true, readable: false` | wrong path, corrupt SQLite, missing schema/column | check the configured paths and file permissions; a missing store must not be read as zero |
| Daemon died | CLI prints `{"status":"failed","detail":"worker exited without a stop request",...}`, exit 1 | unhandled crash in the worker thread | inspect `events` and `last_error`, restart; recovery requeues interrupted jobs |
| Port already used | `{"error": "cannot bind 127.0.0.1:8123: Address already in use", "type": "RuntimeError"}` | another `serve` (or process) owns the port | stop the other process or choose `--port` |
| Job stays queued with `resource_refused` | e.g. `max_load_1m=6 exceeded: load1=34.27` (observed example, not a target) | measured guard vetoed the start; the job is deferred, attempt not spent | let the machine idle; deferred jobs retry. `--profile none` is tests/dev only |

`--profile none` disables the measured guard. It is for tests and short probes;
do not leave an unattended run on it.

## 5. Safety rules

1. **Real money is disabled.** `paper/mt5.py` refuses every order/account
   method and is disabled by default. The paper ledger and simulator are the only
   execution model. Do not add a broker path.
2. **Paper only.** Promotion selects a DSL champion for paper. There is no
   order route, no API key and no live adapter.
3. **Synthetic provenance is enforced.** Imports write `synthetic: true|false`
   into `_meta/<symbol>/<timeframe>.json`. Since #216 the dataset
   manifest/artifact and the evidence bundle (schema v4) carry `source` and
   `synthetic`; `PROMOTION_V1` refuses `provenance_unknown` and
   `synthetic_data`, and a candidate cannot be compared with a champion of
   different provenance. Do not build evidence on synthetic bars and call it
   market evidence; the policy will refuse it, and a test policy that allows it
   is not a promotion path.
4. **Holdout is one-shot.** One spend per `(dataset_id, dataset_checksum)`, even
   if the evaluation fails. Deleting or editing the holdout ledger is
   tampering, not a retry.
5. **Do not tune on test.** Discovery reads train+validation only; the test rows
   are never copied into the search slice. Promotion requires a locked OOS
   section. If you find yourself evaluating variants on test, stop.
6. **Fail closed.** No champion, unreadable champion, invalid spec, missing
   evidence or missing archive means NO TRADE with a reason. Never add a fallback
   to the legacy handwritten rule in the autonomous path. The interactive API
   replay has a separate migration-only M1 momentum fallback. The explicit
   fail-closed switch (#177, merged) is:

   ```bash
   uv run --project backend python -m scalping.cli serve --root data --no-legacy-fallback
   ```

   With this switch, a missing or unusable legacy card produces HOLD/NO TRADE;
   it does not enable or disable autonomous paper trading.
7. **One heavy job.** The guard profile and the queue's heavy-kind rule exist for
   an 8 GB machine. Do not run discovery, evidence and training at once.
8. **Evidence vocabulary.** `implemented` means code and tests. `validated` means
   independent chronological OOS plus cost stress. `paper-traded` means the exact
   version ran in the paper ledger. Passing CI or a synthetic acceptance run is
   none of those.

## 6. Command and payload reference

| Command | Purpose | Main JSON keys |
|---|---|---|
| `data synth` | deterministic synthetic M1 | `rows`, `symbol`, `first_ts`, `last_ts`, `provenance` |
| `data csv` / `data json` | import bars | same as `synth`; `provenance.source` names the file |
| `data derive` | resample M1 | `symbol`, `source`, `written{M5..D1}` |
| `data validate` | structural report | `symbol`, `timeframe`, `rows`, `ok`, `first_ts`, `last_ts`, `issues[]` |
| `dataset build` | split manifest | `manifest`, `dataset_id`, `rows`, `columns`, `train`, `validation`, `test`, `purged` |
| `discovery run` | bounded search + decision | `dataset_id`, `validation`, `trials`, `accepted`, `decision`, `reason`, `best`, `baseline`, `comparison`, `archive` |
| `discovery evidence` | persist artifact + bundle | `persisted`, `dataset_checksum`, `artifact`, `bundle`, `spec`, `evidence_checksum`, `folds` |
| `discovery reproduce` | checksum re-derivation | `reproduced`, `checksum`, `expected`, `mismatches` |
| `discovery promote` | policy evaluation and governed champion | `status`, `accepted`, `dry_run`, `scope`, `codes`, `policy_digest`, `decision`, `holdout`, `champion`, `approval_source` |
| `acceptance run` | synthetic end-to-end | `status`, `claims`, `stages[]`, `report`, `synthetic` |
| `serve` | API | `status`, `host`, `port` |
| `orchestrator run --once` | one tick | `status`, `tick` |
| `orchestrator run --forever` | supervised loop | `started` / `stopped` / `failed` + `daemon` |
| `orchestrator run --schedule-paper` | enqueue advances for evidence champions | `started` / `stopped`; per-scan refusals in `events` |
| `GET /api/autonomy/status` | read-only dashboard | `paper`, `orchestrator`, `champions`, `discoveries`, `data` |
| `GET /api/discovery/summary` | archive totals + champions | `configured`, `readable`, candidate/trial/decision totals, `champions[]` |
| `GET /api/discovery/candidates` | bounded candidate list | `configured`, `readable`, `items[]`, `total` |
| `GET /api/discovery/candidates/{id}` | one candidate + trials + lineage | `candidate`, `trials[]`, `lineage`, `decisions[]`; `404` unknown, `409` unreadable |
| `GET /api/discovery/trials` | bounded trial window | `configured`, `readable`, `items[]`, `total` |
| `GET /api/orchestrator/control` (#175) | read persisted owner controls | `configured`, `readable`, `paused`, `safe_mode`, `research_disabled`, `last_change` |
| `POST /api/orchestrator/control` (#175) | audited local control transition | same state fields; `409` unavailable DB, `422` invalid request |
| `GET /api/autonomy/jobs` | queue window | `configured`, `readable`, `items[]`, `total` |
| `GET /api/paper/sessions` | ledger sessions | `items[]`, `total`, `limit`, `offset` |

Job kinds handled by the daemon: `knowledge.ingest`, `dataset.build`,
`discovery.run`, `paper.advance`, `paper.analyse`. There is no handler that
promotes, rolls back, trains or sends an order. Promotion stays an explicit CLI
act (`discovery promote`); the scheduler only enqueues `paper.advance` for an
already-governed evidence champion.

## 7. Do not do

- **Do not merge your own work or use admin merge.** `main` is protected:
  PR, one review, green `backend`/`frontend` checks, up-to-date branch, linear
  history, no force push.
- **Do not rewrite history.** No force push, no rebase of published `main`, no
  `filter-branch`, no deleting another contributor's branch or PR.
- **Do not delete legacy code without the inventory.** `flow_momentum`,
  `PolicyCard`, `nightly_calibrate`, the five fixed families, LinUCB, PPO/BC and
  `nightly_rl` have documented consumers (`docs/legacy-inventory.md`). Remove
  only after a reviewed migration with parity tests.
- **Do not commit data, stores, `runs/`, model weights or `.env`.**
- **Do not delete failed experiments.** Failed runs are evidence; record them.
- **Do not disable or weaken the promotion policy, the holdout ledger or the
  risk limits** to make a candidate pass.
- **Do not spend the locked holdout without explicit owner approval.** One
  `--spend-holdout` per `(dataset_id, dataset_checksum)`; a retry after a
  failure is not a retry, it is tampering.
- **Do not pass, echo, log or commit the approval token.** Use
  `--approval-token-env` or a `0600` file; the decision record never carries it.
- **Do not run two daemons on one queue database.**
- **Do not use `--profile none` for an unattended run.**
- **Do not treat synthetic acceptance as validation or profitability.**
- **Do not push, open PRs or edit the task board from an agent task unless the
  owner asked for it.**

## 8. Verification log

Executed on this branch, from the repository root unless noted:

The first block was captured against base `626409f` (2026-11-24). Commands added
later (#215–#220) are documented from the parser definitions and their test
modules; they were not executed end-to-end here and are marked accordingly.

| Check | Result |
|---|---|
| top-level `--help` plus the parser definitions | command and flag surface matches this document |
| `data synth` + `data validate` (31 680 M1 bars) | `ok: true`, 4 weekend gaps, 2 outliers |
| `dataset build --timeframe M1` | 31 619 rows, 59 columns, 22 133 / 4 742 / 4 742 splits |
| `dataset build` without `--timeframe` | fails with `no XAUUSD H1 bars` (documented gotcha) |
| `discovery run --max-trials 6` | `decision: no_trade`; all keys present |
| `discovery evidence --folds 3` | artifact + bundle + sidecar written |
| `discovery reproduce` | `reproduced: true` |
| tampered bundle reproduce | `checksum_mismatch`, exit 2 |
| `promote_with_evidence` with probe bundle | rejected; codes include `missing_locked_oos` |
| modified policy promotion | `non_production_policy` |
| `orchestrator run --once` | one JSON line; queue/guard/config sections present |
| `orchestrator run --forever` + SIGTERM | `started`, then `stopped`, exit 0 |
| `paper.advance` without champion | `no_trade` / `no_champion`, `touched: false` |
| `acceptance run` | `passed`; `holdout: skipped`; `validated: false` |
| `GET /api/autonomy/status` (TestClient) | 200, section keys as documented |
| `POST /api/paper/sessions` (TestClient) | 201, server-generated `session_id` |
| targeted tests: paper gap, holdout reuse, promotion production | 2 + 1 + 8 passed |
| `serve` on a busy port | `cannot bind ... Address already in use`, exit 1 |

Not executed or not reproducible here, marked in place: a real XAUUSD run, a
successful promotion, a holdout spend/attach outside tests, paper advance with a
real evidence champion, the `--schedule-paper` daemon on a real champion, the
libomp failure, `npm ci`, and any frontend build. No runtime, memory or
profitability number is claimed.
