# Paper trading engine

Phase 12 slices one and two: a persistent, restart-safe paper-trading ledger, a
deterministic session runner, a read/write HTTP API over the ledger, and a
disabled MT5 demo adapter skeleton. Phase 19 slice A adds an autonomous runner
that paper-trades a validated DSL strategy spec. Slice B adds incremental
operation on appended bars and matured-outcome analysis. There is still no
scheduler; a session row is a record, not a running process. Phase 13 adds the
read-only Paper Trading page at `/paper`, where closing a session is the only
write.

## No real-money execution

The project runs in **paper trading only** mode. The modules in
`backend/src/scalping/paper/` import no network client and hold no credentials.
`PaperLedger` can only write to a local SQLite file, and `PaperSession` can only
call `Simulator.step`. No code path in this slice reaches a broker, an exchange
or a terminal. `Mt5DemoAdapter` exists so the boundary is named, is disabled by
default, and refuses every operation; see “MT5 demo adapter” below.

## Scope

Implemented:

- `scalping.paper.PaperLedger` — SQLite persistence with WAL, foreign keys, a
  schema version and explicit `BEGIN IMMEDIATE` transactions.
- `scalping.paper.PaperSession` — drives the existing
  `scalping.env.simulator.Simulator` one bar at a time and records every step.
- `scalping.paper.autonomous.AutonomousPaperRunner` — loads stored bars through
  the feature cache and paper-trades a validated DSL `StrategySpec` without a
  hand-written policy; see “Autonomous DSL runner”.
- `scalping.paper.continuous.ContinuousPaperRunner` — advances the same session
  as new bars are stored, refuses gaps and ordering defects, and exposes
  `matured_outcomes` / `outcome_fingerprint` for the research loop. The
  orchestrator's `paper.advance` job is the wired entry point; see
  “Continuous operation”.
- Restart safety: reopening a ledger and resuming a session replays the
  recorded actions, verifies the account against the stored state and only then
  appends. Duplicate history is impossible by construction.
- `scalping.api.routes_paper` — session open/close and paginated reads over the
  ledger (table in “HTTP API”).
- `scalping.paper.mt5.Mt5DemoAdapter` — a disabled skeleton that refuses every
  operation.

Deliberately reused instead of re-implemented:

- fill prices, spread, slippage, commission, SL/TP and P&L come from
  `scalping.env.broker.PaperBroker` through the simulator;
- risk gates and position sizing come from `scalping.env.risk.RiskEngine`;
- the observation window and bar timeline come from the simulator.

The roadmap names `broker/adapter.py`, `broker/paper.py` and
`broker/mt5_demo.py`. The repository already has a broker *engine*
(`env/broker.py`) and these slices add persistence and an API, not a second fill
model, so they live in `scalping/paper/` and `scalping/api/`. The MT5 file is
`scalping/paper/mt5.py` and is a disabled skeleton, not an adapter interface the
paper path depends on.

## Schema

One SQLite file per ledger. Every row belongs to a session through a foreign key
with `ON DELETE CASCADE`.

| Table | Purpose | Key columns |
|---|---|---|
| `schema_version` | Applied migration versions | `version`, `applied_at` |
| `sessions` | One paper run | `session_id`, symbol, timeframe, strategy, model version, start/end, status, `config_json` |
| `signals` | Every policy decision | session, ts, action, reason, model version, `probabilities_json` |
| `orders` | Every order attempt | session, signal, ts, side, lot, requested price, status, `rejection_reason` |
| `fills` | Entry fills | order, ts, price, spread, slippage, commission, lot |
| `positions` | Position lifecycle | session, order, side, lot, entry ts/price, SL, TP, status, close ts/price |
| `trades` | Closed trades | session, side, entry ts/price, exit ts/price, lot, P&L, R multiple, exit reason |
| `equity_curve` | Account marks | session, ts, balance, equity |

Why `positions` exists although the brief lists the other six tables: fills
alone cannot reconstruct SL/TP, and `resume` must return the state needed to
continue instead of a guess. A partial unique index enforces at most one open
position per session, matching the broker's single-position model.

Constraints that encode the audit rules:

- `orders.status = 'rejected'` requires a non-null `rejection_reason`; a
  rejection cannot be stored silently.
- `trades.exit_ts >= trades.entry_ts`.
- `positions` allows one `open` row per session.
- Unknown session, signal or order ids are refused before the insert.

Timestamps are timezone-aware and stored as UTC ISO-8601 text, so they sort
lexicographically. Money and prices are `REAL`.

### Accounting invariant

The broker embeds spread and slippage in the fill prices and charges the
round-turn commission once. The ledger stores the broker's number and never
re-derives it. For a fixed case the tests assert:

```
pnl == (exit_price - entry_price) * sign(side) * lot * contract_size
       - commission_per_lot * lot
```

Spread and slippage must not be subtracted from `pnl` again; they are already
inside `entry_price` and `exit_price`.

## Resume semantics

`PaperLedger.resume(session_id)` reads, and only reads:

- the open position from `positions`, including SL and TP;
- `last_bar_ts`, the furthest bar seen over `signals` and `equity_curve`;
- the last `balance` and `equity` from `equity_curve` (`None` before the first
  mark);
- `pending_orders`, the rows with `status = 'pending'`.

`PaperSession.resume()` creates the session if it is new. If the session exists
it replays the recorded signals through a fresh `Simulator` **without writing**,
then checks the replayed balance and open position against the ledger. A
mismatch raises `ResumeError`; the runner never continues on a state it cannot
reproduce. After a successful check, `run(bars)` appends only the new steps.

Replay, not a serialised snapshot, is the deliberate choice: the simulator is a
pure function of `(features, settings, recorded actions, seed)`, so replay has
no private state to copy and a divergence fails loudly. The cost is linear in
the session length, which is acceptable for a paper session and not acceptable
for a training run (which does not use this package).

A closed session cannot be resumed. `close_session` is idempotent so a shutdown
path can call it twice.

## Determinism

`PaperSession.run(bars)` advances at most `bars` decision bars and stops early
when the episode terminates or truncates. Two sessions over the same feature
frame, settings, policy and fixed `started_at` produce identical databases row
for row; the test compares all seven tables. The policy's actions are recorded,
so replay does not depend on the policy being deterministic.

## Usage

```python
from scalping.paper import PaperLedger, PaperSession

with PaperLedger("var/paper/session-1.db") as ledger:
    session = PaperSession(
        ledger,
        features,          # FeatureFrame from scalping.features.build
        settings,          # scalping.config.Settings
        policy,            # observation -> 0 hold / 1 buy / 2 sell / 3 close
        session_id="2026-11-23-eurusd-flow",
        symbol="XAUUSD",
        timeframe="M1",
        strategy="flow_momentum",
        model_version="card-2026-11",
    )
    session.run(500)       # creates the session, or resumes and appends
    session.close()
```

`started_at` is injectable for byte-identical replays in tests; production
callers leave it out.

## Autonomous DSL runner

`scalping.paper.autonomous.AutonomousPaperRunner` removes the hand-written
policy from the paper path. It accepts a validated
`scalping.strategy.dsl.nodes.StrategySpec` — the same object the discovery
engine searches over — and drives the existing `PaperSession`, `Simulator` and
`RiskEngine`. The strategy is **discovered, not human-authored**: this slice
supplies the execution boundary, so a candidate selected by the autonomous
pipeline can be paper-traded without the owner writing entry or exit rules.

```python
from scalping.data.store import ParquetStore
from scalping.paper import PaperLedger
from scalping.paper.autonomous import AutonomousPaperRunner

with PaperLedger("var/paper/autonomous.db") as ledger:
    runner = AutonomousPaperRunner(
        ledger,
        spec,                       # StrategySpec from discovery
        "XAUUSD",
        "M1",
        settings,                   # scalping.config.Settings
        store=ParquetStore("data"), # stored bars
        session_id="auto-xauusd-m1",
    )
    runner.run()                    # every remaining stored bar
    print(runner.summary())         # session id, hash, bars, trades, P&L, position
    runner.close()
```

What the runner does:

1. Validates the spec with the DSL validator **before** reading bars or creating
   session state. An invalid spec raises `DslValidationError` and leaves the
   ledger untouched — no session row and no feature-cache entry.
2. Reads the stored slice through `ParquetStore` and builds the feature matrix
   through `FeatureCache` (enabled by default under
   `<store root>/.feature-cache`; pass `cache=False` or a shared
   `FeatureCache`). There is no second data path and no matrix rebuilt per bar.
3. Calls the compiled interpreter for every decision bar with the current
   feature row, the open position's side and its completed-bar count. It maps
   `BUY/SELL/HOLD/CLOSE/NO_TRADE` to the simulator's `1/2/0/3/0`; `NO_TRADE`
   suppresses entries rather than flattening a position, matching the
   interpreter's own precedence.
4. Writes every signal, order, fill, trade and equity mark through
   `PaperLedger`'s existing methods. The runner holds no SQL and opens no second
   connection, so the ledger's transaction and constraint rules still apply.

### Spec-hash attribution

The canonical spec hash (`StrategySpec.canonical_hash()`) is the strategy
version:

| Record | Attribution |
|---|---|
| `sessions.strategy` | `"dsl"` |
| `sessions.model_version` | canonical spec hash |
| `sessions.config_json.extra.dsl` | `spec_version`, `spec_hash` and the full canonical payload |
| `signals.model_version` | canonical spec hash, on every decision |
| `signals.action` | mapped action name (`hold`/`buy`/`sell`/`close`) |
| `signals.reason` | the DSL reason (`entry_long`, `exit`, `holding_limit`, `no_trade`, …); a gate that refused the action is appended as `; blocked=<reason>` |
| `signals.probabilities_json` | `NULL` — the DSL is deterministic and emits no probability |

The DSL `spec_version` is part of the canonical payload, so a signal carrying
the hash is pinned to the exact version and every behavioural field. The
explicit integer is also stored once per session in `config_json.extra.dsl`
because the `signals` table has no dedicated version column. The full canonical
payload in the session config means the strategy can be reconstructed and its
hash recomputed from the ledger alone.

### Resume and determinism

The default session id is `auto-<symbol>-<timeframe>-<hash prefix>`, so a
restart reopens the same record instead of creating a duplicate; pass
`session_id` to separate runs of the same spec. `resume()` and `run()` use
`PaperSession`'s replay semantics: recorded actions are replayed through a fresh
simulator and the balance and open position are checked against the ledger
before anything new is appended. The DSL history is reseeded from feature rows
strictly before the decision bar, so a resume cannot introduce look-ahead.

`run(bars=None, max_bars=None)` caps one call with `bars` and the session's
total decisions with `max_bars` (counted from the ledger, so the cap survives a
restart). With neither, every remaining stored bar is processed and the call
stops when the episode ends.

Determinism follows from the inputs: the same store slice, spec and seed produce
identical signals and trades, and two complete ledgers compare row for row in
the tests. A different `started_at` only changes the session metadata, not the
trading path.

### Risk and real money

The spec never sees `settings.risk`. Entries, stops, targets and position size
come from the simulator's `RiskEngine`; the DSL sizing nodes (`FixedFraction`,
`AtrMultiple`) are part of the candidate identity but do not reach the broker.
A generated strategy therefore cannot raise a limit, disable the risk engine or
grant itself a live path. Risk-gate refusals are written to the signal reason
and as rejected orders with their reason, and the runner lets them propagate
instead of catching or retrying around them.

The module imports no network client, holds no credentials and has no
order-routing code. Real-money execution remains disabled and out of scope; the
autonomous runner is a paper record, not a broker connection.

## Continuous operation

`scalping.paper.continuous.ContinuousPaperRunner` keeps one paper session
current as bars are appended to the store. It executes the same validated
`StrategySpec` and the same `PaperSession` / `Simulator` / `RiskEngine` path as
the autonomous runner; it adds only a durable cursor, a strict store check and
an `advance()` call.

```python
from scalping.paper import PaperLedger
from scalping.paper.continuous import ContinuousPaperRunner, matured_outcomes

with PaperLedger("var/paper/autonomous.db") as ledger:
    runner = ContinuousPaperRunner(
        ledger,
        spec,                       # StrategySpec from the champion layer
        "XAUUSD",
        "M1",
        settings,
        store_root="data",
        scope="XAUUSD:M1",
    )
    while new_bars_were_stored():
        summary = runner.advance()
        print(summary.new_bars, summary.new_signals, summary.new_trades)

    batch = matured_outcomes(ledger, runner.session_id)
```

`advance()` reads the whole stored slice, builds or loads the causal feature
matrix, resumes the session and appends only new decisions. The call is
caller-driven: there is no polling thread, daemon or scheduler in this slice.

The normal entry point is now the orchestrator's `paper.advance` job. The
handler resolves the evidence-mode champion through `ChampionProvider`, builds
one `ContinuousPaperRunner`, calls `advance()` once and records
`mode: "continuous"`, `bars_advanced`, `bars_processed`, `last_bar_ts` and the
session id in the job result. The payload's `bars` is validated and recorded,
but the continuous contract advances every newly closed stored bar, so it is a
budget label, not a per-call cap.

### Idempotency rule

The ledger is the cursor. The last processed timestamp is the newest signal in
the session, not an in-memory counter, so a process restart resumes from the
same bar. `PaperSession.resume()` replays the recorded actions through a fresh
simulator and verifies balance and open position before anything is appended.
`advance()` additionally checks that every recorded signal timestamp lands on
the stored bar it was recorded on; a rewritten history raises
`SessionMismatchError` instead of shifting decisions onto other bars. The result
is idempotent by bar timestamp: a second call without new bars writes no signal,
order, fill, trade or equity mark, and a recorded bar can never produce a second
signal.

The session id remains `auto-<symbol>-<timeframe>-<hash prefix>`. A session
created by the autonomous runner is resumed if symbol, timeframe, strategy hash,
window, seed and contract/risk/cost settings match; a mismatch raises
`SessionMismatchError` rather than silently replaying a different account, and a
ledger mark the replay cannot reproduce raises `ResumeError`. From the
orchestrator either refusal is a permanent `session_conflict` job failure. A
legacy slice session needs no cursor migration: its newest recorded signal is
the cursor, provided the signals still land on the stored decision bars.
`scope` (default `<symbol>:<timeframe>`) is recorded for audit and does not
change the id.

### The provisional last bar

The newest stored bar is priced but never used as a decision bar. The simulator
needs a following bar to evaluate stops and targets; taking the tail as a
decision bar would book a `data_end` close and make a later append
unreplayable. Appending `N` bars therefore advances exactly `N` decisions: the
previous tail becomes decidable and the new tail stays provisional. This is the
one-bar latency of continuous operation, not a skipped bar.

### Gap and ordering policy

Before anything is written:

- the whole stored slice must be strictly increasing by `ts` and free of
  duplicates, otherwise `BarOrderError`;
- the range from the ledger cursor to the newest bar must contain one bar per
  timeframe slot, otherwise `BarGapError`;
- a jump is allowed only when the missing instants fall inside the symbol's
  closed session window from the registry (a weekend or a daily close). A symbol
  that is not in the registry gets the strict cadence check; a timeframe with no
  known bar size is checked for ordering only.

A gap is refused rather than skipped: silently filling it would change the
bar-to-signal alignment and invent decisions on bars that were never stored.
Backfill the missing partition and call `advance()` again. History before the
cursor is not re-checked for cadence (it is already reflected in decisions), but
ordering is checked across the whole slice.

### Matured outcomes

`matured_outcomes(ledger, session_id)` returns an immutable `OutcomeBatch` of
**closed trades only**. Each row carries entry and exit timestamps, side, entry
and exit price, lot, P&L, R multiple, exit reason, the session's canonical
strategy hash, the stored dataset id when the session config has one, and the
session's `source`/`synthetic` provenance. Open positions are not rows; the
batch counts them in `open_trades`. `to_dict()` / `to_json()` are the JSON-safe
projections.

`outcome_fingerprint(batch)` is a sha256 over a canonical payload of the batch.
It excludes `trade_id` (a re-created ledger can renumber rows) and
`open_trades` (an open position is not matured), and sorts rows by their own
fields, so identical outcomes hash identically regardless of read order. The
fingerprint includes `source` and `synthetic`: the same trades claimed against
different evidence are different outcomes, so a provenance change changes the
hash. `PaperOutcomeTriggerSource` in `scalping.orchestrator.paper_triggers` uses
that fingerprint for dedupe; its contract, scan order, coverage cursor, refusal
reasons, `since` window, synthetic policy and proposal-only handler are in
`docs/orchestrator.md`.

### Provenance contract

A paper session must say which evidence produced its trades. The contract has
three rules:

1. **Written once, at creation.** `ContinuousPaperRunner` reads the dataset
   sidecar with `scalping.data.provenance.load_provenance` and stores
   `{"source": ..., "synthetic": ...}` in the session config under
   `extra.provenance`. A store with no sidecar stores nothing.
2. **Unknown is `None`, never an invented `False`.** `session_provenance(config)`
   in `scalping.paper.ledger` returns `(source, synthetic)` and reports
   `(None, None)` when the creator recorded nothing. A demo run must be
   distinguishable from real market evidence, so no code path may default
   `synthetic` to `False`.
3. **A change is a refusal.** When a session already exists and the store
   sidecar now reports a different known source or flag, `advance()` raises
   `SessionMismatchError`. An unreadable sidecar raises `ContinuousError`
   before anything is written; a session recorded before provenance existed
   (unknown) still resumes.

`matured_outcomes` copies the pair onto every row and the batch. The trigger
payload carries `source` and `synthetic` with `null` for unknown, and
`PaperOutcomeTriggerSource` refuses a `synthetic=True` batch by default
(`allow_synthetic=False`) with a logged `synthetic_outcome` reason. The
handler applies the same default, so a synthetic job that arrives from an
opted-in source or an older queue is answered `no_action` with one
`paper.outcome_refused` line. Opting in is explicit and auditable; it is never
the default, because synthetic data must not become real market evidence.

### Honest limits

- **Single process, no scheduler.** `advance()` is a synchronous call; nothing
  polls the store, sleeps or retries. A scheduler/daemon is the next slice.
- **Outcomes only after close.** A trade enters the batch when `trades` has a
  row. Until then its P&L is unrealized and excluded.
- **Session-level data id.** The `signals` table has no data column, so the
  dataset id and the provenance pair are recorded once in the session config
  when the session is created. A per-trade data hash would need a schema
  migration.
- **Replay cost grows with the session.** Each `advance()` rebuilds the feature
  matrix for the stored slice and replays the recorded actions. This is exact
  and cheap for a paper session of a few thousand bars, but it is linear in
  session length; snapshot resume or session rollover is future work.
- **No gap repair.** A gap is a refusal. The runner does not fetch, interpolate
  or invent the missing bar.
- **No champion selection.** `spec` is supplied by the caller; this module does
  not choose it and has no legacy fallback.
- **No real money.** No network client, credentials or order routing exists in
  the module; paper trading remains the only execution target.

## HTTP API

`scalping.api.routes_paper` mounts under `/api/paper`. It reads and writes the
ledger record; it does not start a runner, so an opened session stays `running`
until something drives it and calls close.

The shapes below are part of the guarded public contract: any change to a path,
field, required key or enum fails `tests/test_openapi_snapshot.py` until the
snapshot is reviewed and rewritten. See `docs/api-contracts.md`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/paper/sessions` | Open a session record; `201` with the generated id, status and start stamp |
| `GET` | `/api/paper/sessions` | Sessions newest first, each with its status; paginated |
| `GET` | `/api/paper/sessions/{session_id}` | Summary: balance, equity, open position, trade count, started/ended, last bar ts, stored config |
| `POST` | `/api/paper/sessions/{session_id}/close` | Close a running session; `404` unknown, `409` already closed |
| `GET` | `/api/paper/sessions/{session_id}/positions` | Positions oldest first, closed ones included |
| `GET` | `/api/paper/sessions/{session_id}/orders` | Order attempts oldest first, rejections with their reason |
| `GET` | `/api/paper/sessions/{session_id}/history` | Closed trades oldest first |
| `GET` | `/api/paper/sessions/{session_id}/equity` | The newest equity window in chronological order |

Every list answers `{items, total, limit, offset}`. `limit` defaults to `100`
and is bounded to `1..1000`; `offset` is `>= 0`; anything outside is a `422`.
`/equity` is the exception to "oldest first": `offset=0` is the tail of the
curve, which is what a live chart draws, and raising `offset` pages into older
marks. The items inside one page are always chronological.

Opened sessions carry `{"origin": "api", "extra": <request config>}` in
`config_json`. `close` is idempotent in effect — the ledger's `UPDATE` is guarded
by `status = 'running'`, so a retry cannot move `ended_at` — but the retry is
answered `409` so a caller can tell a retry from a first close.

### Ledger path and ids

The database path is app configuration: `create_app(paper_ledger=...)`, or
`<data_root>/paper.sqlite` by default. It is not a request field, so no body can
move it, and the file is created on the first write, not when the app starts.

A session id is generated by the server (`uuid4().hex`) and accepted back only as
an opaque token: `^[A-Za-z0-9][A-Za-z0-9_-]*$`, at most 64 characters. It is
passed to SQL as a bound parameter and is never joined to a path.

### Read-only autonomy view

`/api/autonomy/status` includes a `paper` section so a monitor can show ledger
activity without opening the write router:

| Field | Meaning |
|---|---|
| `configured` | Whether the ledger file exists |
| `readable` | Whether it could be opened and has the sessions schema |
| `sessions` | Session rows in the ledger |
| `open_positions` | Positions with `status = 'open'` |
| `closed_trades` | Rows in `trades` |

The section is read through `PaperLedger.open_reader` (`mode=ro`); the API then
sets `PRAGMA query_only = ON` on that connection. A `GET` never creates the
database or migrates it. When the ledger is missing the response is
`{"configured": false, "readable": false}` and the counts are absent, not zero.
When the file exists but cannot be opened or has no schema yet, it is
`{"configured": true, "readable": false}` with no counts. Only `readable: true`
carries counts. A read-only WAL connection may create `-wal`/`-shm` sidecars in
a writable directory, but the ledger file itself is unchanged. The endpoint
never exposes the ledger path and never starts a session or a runner. See
[`orchestrator.md`](orchestrator.md#read-only-status-api) for the full route
table and bounds.

### Reads during a write

Every GET opens a `mode=ro` connection (`PaperLedger.open_reader`), runs plain
queries in autocommit and closes. It runs no migration, takes no write
transaction and cannot create the file, so it keeps working while the session
runner holds `BEGIN IMMEDIATE`; WAL readers see the last committed snapshot. A
GET that lands while the first writer is still creating the schema sees an empty
page instead of a 500.

## Paper Trading page

`/paper` in the frontend is a view of this ledger and nothing else. It calls
only `/api/paper`, does not open sessions, does not start a runner and has no
broker path. The one write it performs is closing a running session.

- **Sessions** — newest first, with status, symbol/timeframe, strategy, model
  version and start/end. An absent ledger is an explicit empty state that says
  the page does not open sessions; it is not a fabricated list.
- **Session detail** — balance, equity, the open position, the closed-trade
  count and the last bar timestamp from `/sessions/{id}`. A `null` balance or
  equity renders as a dash: there is no placeholder account and no P&L total
  computed in the browser.
- **Tables** — positions, orders, closed trades and equity, one endpoint each,
  requested in 20-row pages. The pager is bounded by
  `offset + items.length >= total`, so it never asks for a page the ledger does
  not have. Equity pages *backwards*: `offset=0` is the newest window and each
  step moves into older marks.
- **Attribution** — strategy, `model_version` and, when the stored config has
  them, `extra.dsl.spec_hash`, `extra.dsl.spec_version` and
  `extra.dataset_id`. A session without a DSL hash says so instead of showing a
  blank hash.
- **Close** — a two-step inline confirmation, then
  `POST /sessions/{id}/close`. A `200` re-reads the summary and the session
  list. A `409` is presented as "already closed" and refreshes too, because a
  retry is not an error.
- **Polling** — only a `running` session is re-read, every 5 seconds. The timer
  is cleared on unmount, on selection, on pagination and as soon as the session
  stops running. A failed poll stops the loop and offers a manual retry rather
  than leaving stale values on screen looking current.

The page shows a visible paper-only notice: no real money, no broker and no
order entry. It is a ledger view, not a trading terminal — there is no live
price, no strategy selection and no way to open a session from it.

## MT5 demo adapter

`scalping/paper/mt5.py` holds `Mt5DemoAdapter`. It is a skeleton, not an
integration:

- constructing it without `enabled=True` raises `AdapterDisabled`;
- `enable()` lazily loads the `MetaTrader5` package and stops there — it opens
  no socket, starts no terminal and stores nothing; a machine without the
  package also gets `AdapterDisabled`;
- every operation (`connect`, `account_info`, `open_positions`, `place_order`,
  `modify_order`, `cancel_order`, `close_position`) raises
  `NotImplementedError`; there is no flag that unlocks one;
- there are no credentials in the module or in settings, and real-money
  execution stays out of scope.

## Not built yet

- A demo or live MT5 integration. `Mt5DemoAdapter` refuses every call.
- Session scheduler and a long-running process separate from training; the API
  records a session but nothing steps it.
- A direct HTTP/CLI entry point that constructs a runner. The orchestrator's
  `paper.advance` job reaches `ContinuousPaperRunner` with champion gating; there
  is no endpoint or command that selects a spec and runs a session in-process.
- A scheduler or daemon for continuous operation. `advance()` is caller-driven;
  nothing polls the store, sleeps or retries on new bars, and no endpoint starts
  a loop.
- Limit orders and partial fills. `orders.status` already has `pending`, and
  `resume` reports pending rows, but the runner only sends market orders.
- Exit fills in `fills`. The broker settles exits as `trades`; the exit price
  and reason live there.
- Probability vectors on signals. The column exists and is written when a
  caller supplies one; the current policy contract returns a discrete action,
  and the autonomous DSL runner deliberately stores `NULL`.

## Tests

`backend/tests/test_paper_ledger.py` covers schema idempotency, a round trip of
every record type across a reopen, rejection audit, resume after reopen,
duplicate session ids, foreign-key enforcement, the one-open-position rule, the
P&L invariant for long and short, deterministic replay of two complete
databases, restart without duplicated history, equity reconciliation, and a
source scan that forbids network imports in the package.

`backend/tests/test_api_paper.py` covers the session lifecycle, the summary
fields, 404/409/422 paths, pagination bounds and slicing, equity window
ordering, close idempotency in effect, session-id traversal rejection, an absent
ledger staying absent on reads, and a GET served while a second connection holds
`BEGIN IMMEDIATE`.

`backend/tests/test_paper_mt5.py` proves the adapter is disabled by default, that
importing the module does not load the terminal package, that no network client
or secret appears in the source, and that every operation refuses even after a
caller sets flags on the instance.

`backend/tests/test_paper_autonomous.py` covers the DSL runner: an invalid spec
is refused before a session row exists, two fresh ledgers over the same slice
are byte-identical, 60 bars plus a reopen plus 440 bars equals one 500-bar run,
every signal carries the canonical hash and a reason, a NO TRADE spec records
signals without orders, a daily-loss block is recorded and never bypassed, a
sizing node cannot change the broker's lot, `max_bars` survives across calls, a
reopened ledger keeps the attribution, and a source scan finds no network,
credential or real-money path.

`backend/tests/test_paper_continuous.py` covers incremental operation: an empty
store and a no-new-bars call write nothing, appending 20 bars advances exactly
20 decisions, a restart over a reopened ledger resumes without duplicate signals
or trades, an overlapping refetch adds no duplicate signal, a gap or an
out-of-order/duplicate bar is refused before any row is written, a session with
a different spec hash is refused, an open position is counted but excluded from
the matured batch, the outcome fingerprint is stable and changes when a trade
matures, appending future bars leaves earlier signals unchanged, and a source
scan finds no network or legacy strategy reference. Provenance is covered end to
end: a synthetic sidecar is flagged in the session, the matured batch and the
trigger (which refuses it by default and accepts it under `allow_synthetic`), a
real dataset records `synthetic=False`, a store without a sidecar stays unknown
(`None`, never `False`), a re-ingested provenance change refuses resume, an
unreadable sidecar refuses the advance, the batch keeps its flag across a ledger
reopen, and the outcome fingerprint changes with provenance.

`backend/tests/test_paper_ledger.py` also pins the provenance helpers:
`provenance_config` refuses a mistyped flag, `session_provenance` reads the
stored pair after a reopen and reports `(None, None)` when nothing was recorded,
and `open_session` refuses a mistyped provenance block before the insert.

`frontend/src/app/paper/page.test.tsx` covers the page: an empty ledger, a
loaded session with attribution, a `404` session, close `200` and `409`, bounded
session and equity pagination, list/detail error states with retry, polling only
while running with cleanup on unmount, and a ledger with no balance, hash or
position (dashes and explicit sentences, never invented numbers).

```bash
cd backend
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/test_api_paper.py tests/test_paper_mt5.py \
  tests/test_paper_autonomous.py tests/test_paper_continuous.py -q
uv run pytest

cd ../frontend
npm test -- src/app/paper/page.test.tsx
```

## UI accessibility

The `/paper` page follows WCAG 2.2 AA where its code can enforce it: labelled
controls, captions and `scope` on every table, focusable scroll regions, explicit
empty/missing/unreadable states, and a focus target after the inline close
confirmation. Contrast, focus and announcement findings are recorded in
`docs/ui-accessibility-audit-2026-11-24.md`; the reusable gate for new pages is
`docs/ui-accessibility-checklist.md`.
