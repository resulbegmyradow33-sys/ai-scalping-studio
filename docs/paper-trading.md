# Paper trading engine

First Phase 12 slice: a persistent, restart-safe paper-trading ledger and a
deterministic session runner. This is a tested library, not a service. There is
no HTTP API, no MT5 adapter and no scheduler yet.

## No real-money execution

The project runs in **paper trading only** mode. The modules in
`backend/src/scalping/paper/` import no network client and hold no credentials.
`PaperLedger` can only write to a local SQLite file, and `PaperSession` can only
call `Simulator.step`. No code path in this slice reaches a broker, an exchange
or a terminal. A real or demo execution adapter would have to be added as a
separate, explicitly enabled module; it does not exist here.

## Scope

Implemented:

- `scalping.paper.PaperLedger` — SQLite persistence with WAL, foreign keys, a
  schema version and explicit `BEGIN IMMEDIATE` transactions.
- `scalping.paper.PaperSession` — drives the existing
  `scalping.env.simulator.Simulator` one bar at a time and records every step.
- Restart safety: reopening a ledger and resuming a session replays the
  recorded actions, verifies the account against the stored state and only then
  appends. Duplicate history is impossible by construction.

Deliberately reused instead of re-implemented:

- fill prices, spread, slippage, commission, SL/TP and P&L come from
  `scalping.env.broker.PaperBroker` through the simulator;
- risk gates and position sizing come from `scalping.env.risk.RiskEngine`;
- the observation window and bar timeline come from the simulator.

The roadmap names `broker/adapter.py`, `broker/paper.py` and
`broker/mt5_demo.py`. The repository already has a broker *engine*
(`env/broker.py`) and this slice adds a persistence layer, not a second fill
model, so it lives in `scalping/paper/`. The adapter interface and the disabled
MT5 skeleton are still open; see “Not built yet”.

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

## Not built yet

- Broker adapter interface (`broker/adapter.py` in the roadmap) and the
  disabled MT5 demo skeleton (`broker/mt5_demo.py`). No credentials, no socket.
- API endpoints for paper account, positions, orders and history.
- Session scheduler and a long-running process separate from training.
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

```bash
cd backend
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/test_paper_ledger.py -q
uv run pytest
```
