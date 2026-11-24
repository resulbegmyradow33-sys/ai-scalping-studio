# Risk management

The independent risk layer for Phase 11 slice 1: kill switches and limits that
sit outside every model. The engine is `scalping.env.risk.RiskEngine`; the
simulator owns the state it reads and applies it in
`scalping.env.simulator.Simulator`. There is no second risk implementation.

## Limits

All Phase 11 limits live on `RiskConfig` and are **disabled at `0`**. A default
`RiskConfig` therefore reproduces the pre-Phase-11 engine exactly, and existing
runs and tests are unchanged.

| Field | Default | Semantics | Reset rule |
|---|---|---|---|
| `weekly_loss_pct` | `0.0` | Blocks entries and force-closes when `(week_start_equity - equity) / week_start_equity >= pct` | New week anchor at the 17:00 UTC session roll |
| `max_drawdown_pct` | `0.0` | Blocks entries and force-closes when `(peak_equity - equity) / peak_equity >= pct` | Equity recovers above the threshold; `peak_equity` is the episode peak |
| `consecutive_loss_pause` | `0` | Blocks entries once `consecutive_losses >= N` (a loss is `Trade.pnl < 0`) | Win, timed pause expiry, or the next session roll when `pause_bars == 0` |
| `max_spread_points` | `0.0` | Blocks entries when `spread_points > cap` (strict); never force-closes | Next bar; evaluated per bar only |
| `min_equity_frac` | `0.0` | Emergency stop: blocks entries and force-closes when `equity < frac * ContractSpec.initial_balance` (strict) | Equity recovers above the floor |
| `pause_bars` | `0` | Latch duration for a tripped weekly/drawdown/emergency or consecutive-loss switch: the next `pause_bars` decision bars stay blocked | Entries resume at the bar whose timestamp equals `paused_until`; `0` leaves the switch condition-driven only |

Boundary rules are deliberate and tested at the exact limit: the weekly,
drawdown and consecutive checks trip **at** the threshold (`>=`), the emergency
and spread checks trip **strictly beyond** it (`<` / `>`). A missing anchor
(`week_start_equity`/`peak_equity` is `None` or non-positive) skips the switch
instead of inventing a value.

`spread_points` is `bar.spread / ContractSpec.point`; the simulator fills it
from the current bar, so the cap is expressed in contract points, not price
units. The default contract (`point = 0.001`) turns a 0.20 spread into 200
points.

When a weekly, drawdown or emergency switch trips with a position open, the
simulator force-closes it at the current bar's liquidating quote and records the
switch name as `Trade.reason` (`weekly_loss_limit`, `max_drawdown`,
`emergency_stop`). Consecutive-loss and spread switches block entries only;
HOLD and CLOSE always pass through.

## Anchors

Both anchors reuse the existing 17:00 UTC session approximation, which stands in
for the New York 17:00 roll and ignores US daylight saving:

- **Day:** a bar at or after 17:00 UTC anchors to its own date, earlier to the
  previous date. The simulator rolls `day_start_equity` when this anchor
  changes.
- **Week:** the `(iso_year, iso_week)` of the day anchor. The week rolls at the
  first session anchor whose ISO week differs, which with the fixed
  approximation is **Monday 17:00 UTC**. The simulator rolls
  `week_start_equity` there.

`peak_equity` is the maximum equity marked during the episode and resets only
on `reset()`. It is the same peak the reward's drawdown penalty uses.

## The model cannot bypass a limit

Every limit is frozen at `RiskEngine` construction. `size`, `allow_new` and
`allow_new_extended` accept prices and an `AccountState` snapshot; none accepts
a risk override, and no action space exposes a limit. A policy chooses BUY,
SELL, HOLD or CLOSE, and every BUY/SELL passes the same gate. `allow_new` keeps
its pre-Phase-11 signature and delegates to `allow_new_extended`, so old callers
and new callers cannot diverge. A test asserts both signatures have exactly
`(self, account, ts)`.

The simulator adds no privileged path: the only entry call is
`allow_new_extended`, and the kill-switch force-close runs after every mark,
outside the policy's action.

## Acceptance evidence

`backend/tests/test_risk_acceptance.py` is the Phase 11 non-bypass suite. It
runs the simulator and `DslEvaluator` on the *same* bars and settings and
compares the closed trades, so a second gate path cannot drift silently.

Proven:

- A generated DSL `StrategySpec` cannot carry or request a risk limit. It has no
  `risk` field, a payload with one is rejected, and `evaluate`/`run` take no
  settings or limit argument.
- DSL sizing nodes (`AtrMultiple`, `FixedFraction`) are identity only: an
  extreme `AtrMultiple(1000)` and `FixedFraction(1.0)` produce the same fills,
  lots and PnL as a spec with no sizing node. Every lot equals
  `RiskEngine.size(...)` for the balance at entry.
- Daily loss, weekly loss, max drawdown, consecutive-loss, emergency and spread
  switches each block the next entry in the raw simulator and in the evaluator;
  the manual loop and the evaluator produce identical trade rows. Weekly,
  drawdown and emergency also force-close the open position with the switch
  name recorded in `Trade.reason`.
- `action_masks()` agrees with `_entry_reason` and the step rejection for every
  gate, both per-gate and on a property-style sweep of a synthetic path.
- An always-enter spec cannot open a second position (`max_positions`) and its
  lots never exceed the notional cap.
- `RiskEngine` exposes no override parameter, `RiskConfig`/`Settings` are frozen
  and the engine binds its config behind read-only properties.
- No orchestrator job kind accepts a `risk` key: the payload is a permanent
  failure (one attempt, permanent event, no ledger or archive written).

Remaining, not proven by these tests:

- Partial fills and broker-side rejection/requote/margin paths.
- Execution delay and next-bar fills; the paper broker fills on the decision
  bar's quote with ATR slippage.
- Broker-enforced risk (a real server-side limit or margin call). This suite
  proves the local engine cannot be bypassed, not that an external broker would
  enforce the same limits.

## Not covered yet

This slice implements the equity and entry limits only. Still open for later
Phase 11 slices:

- **Partial fills:** the paper broker fills the full requested lot; no partial
  fill model exists.
- **Rejected orders:** orders are assumed accepted once sized. Broker-side
  rejection, margin rejection and requote paths are not modelled.
- **Execution delay:** entries and exits fill on the decision bar's quote with
  ATR slippage; there is no order latency or next-bar fill delay.
- **Slippage limit and data-quality block:** listed in the Phase 11 roadmap,
  not in this slice.
- **Slippage stress:** one-axis cost sensitivity and Monte Carlo trade
  resampling exist (see `docs/backtesting-methodology.md`); the risk engine has
  no per-run slippage multiplier of its own.
