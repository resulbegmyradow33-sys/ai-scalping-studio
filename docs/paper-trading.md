# Paper trading engine

Phase 12 slices one and two: a persistent, restart-safe paper-trading ledger, a
deterministic session runner, a read/write HTTP API over the ledger, and a
disabled MT5 demo adapter skeleton. There is still no scheduler and no frontend
page; a session row is a record, not a running process.

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

## HTTP API

`scalping.api.routes_paper` mounts under `/api/paper`. It reads and writes the
ledger record; it does not start a runner, so an opened session stays `running`
until something drives it and calls close.

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

### Reads during a write

Every GET opens a `mode=ro` connection (`PaperLedger.open_reader`), runs plain
queries in autocommit and closes. It runs no migration, takes no write
transaction and cannot create the file, so it keeps working while the session
runner holds `BEGIN IMMEDIATE`; WAL readers see the last committed snapshot. A
GET that lands while the first writer is still creating the schema sees an empty
page instead of a 500.

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
- The Paper Trading frontend page (next slice).
- Limit orders and partial fills. `orders.status` already has `pending`, and
  `resume` reports pending rows, but the runner only sends market orders.
- Exit fills in `fills`. The broker settles exits as `trades`; the exit price
  and reason live there.
- Probability vectors on signals. The column exists and is written when a
  caller supplies one; the current policy contract returns a discrete action.

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

```bash
cd backend
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/test_api_paper.py tests/test_paper_mt5.py -q
uv run pytest
```
