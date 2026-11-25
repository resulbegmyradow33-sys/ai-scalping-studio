# Risk management

The independent risk layer for Phase 11 slice 1: kill switches and limits that
sit outside every model. The engine is `scalping.env.risk.RiskEngine`; the
simulator owns the state it reads and applies it in
`scalping.env.simulator.Simulator`. There is no second risk implementation.

## Limits

All Phase 11 limits live on `RiskConfig` and are **disabled at `0`** (the
boolean flag at `False`). A default `RiskConfig` therefore reproduces the
pre-Phase-11 engine exactly, and existing runs and tests are unchanged.

| Field | Default | Semantics | Reset rule |
|---|---|---|---|
| `weekly_loss_pct` | `0.0` | Blocks entries and force-closes when `(week_start_equity - equity) / week_start_equity >= pct` | New week anchor at the 17:00 UTC session roll |
| `max_drawdown_pct` | `0.0` | Blocks entries and force-closes when `(peak_equity - equity) / peak_equity >= pct` | Equity recovers above the threshold; `peak_equity` is the episode peak |
| `consecutive_loss_pause` | `0` | Blocks entries once `consecutive_losses >= N` (a loss is `Trade.pnl < 0`) | Win, timed pause expiry, or the next session roll when `pause_bars == 0` |
| `max_spread_points` | `0.0` | Blocks entries when `spread_points > cap` (strict); never force-closes | Next bar; evaluated per bar only |
| `max_slippage_points` | `0.0` | Blocks entries when the estimated entry slippage `slippage_points > cap` (strict); never force-closes | Next bar; evaluated per bar only |
| `data_quality_block` | `False` | Blocks entries when the decision bar's `data_quality_ok` is false (ATR missing/NaN/non-positive, spread missing/NaN/negative, or a stale bar gap); never force-closes | Next bar; evaluated per bar only |
| `min_equity_frac` | `0.0` | Emergency stop: blocks entries and force-closes when `equity < frac * ContractSpec.initial_balance` (strict) | Equity recovers above the floor |
| `pause_bars` | `0` | Latch duration for a tripped weekly/drawdown/emergency or consecutive-loss switch: the next `pause_bars` decision bars stay blocked | Entries resume at the bar whose timestamp equals `paused_until`; `0` leaves the switch condition-driven only |

Boundary rules are deliberate and tested at the exact limit: the weekly,
drawdown and consecutive checks trip **at** the threshold (`>=`), the emergency,
spread and slippage checks trip **strictly beyond** it (`<` / `>`). A missing
anchor (`week_start_equity`/`peak_equity` is `None` or non-positive) or a
missing measurement (no spread, no ATR so no slippage estimate) skips the
switch instead of inventing a value.

`spread_points` is `bar.spread / ContractSpec.point`; the simulator fills it
from the current bar, so the cap is expressed in contract points, not price
units. The default contract (`point = 0.001`) turns a 0.20 spread into 200
points.

`slippage_points` uses the same estimate the fill pays,
`ATR * CostConfig.slippage_atr_frac`, divided by the contract point. With the
defaults (`slippage_atr_frac = 0.05`, `point = 0.001`) an ATR of 0.5 price is
25 points, so `max_slippage_points = 20` refuses that entry and
`max_slippage_points = 25` allows it. The gate is recomputed from the current
decision bar, never cached across bars.

### Data-quality gate

`data_quality_block` is off by default because it changes what a run trades.
When on, an entry is refused with `data_quality_block` if any of these holds on
the decision bar:

- ATR is missing, `NaN`, infinite or `<= 0`. Without the flag the same bar is
  still refused, but with the older `no_atr` reason.
- Spread is missing, `NaN`, infinite or negative. Without the flag the
  simulator sanitises a bad spread to `0.0` and the entry can proceed.
- The gap from the previous bar exceeds the frame's stale tolerance:
  `max(10 minutes, median positive bar gap)`. Timestamps are schedule metadata,
  not market observations, and the cadence is fixed before the run; a missing
  H1 bar (120-minute gap against a 60-minute cadence) is stale, while normal H1
  spacing is not.

The simulator reports the flag (`Simulator.data_quality_ok`) and the slippage
estimate (`Simulator.slippage_points`) on every bar, even when the gates are
off, so a caller can inspect them. `RiskEngine` reads them only under the
configured limit/flag.

When a weekly, drawdown or emergency switch trips with a position open, the
simulator force-closes it at the current bar's liquidating quote and records the
switch name as `Trade.reason` (`weekly_loss_limit`, `max_drawdown`,
`emergency_stop`). Consecutive-loss, spread, slippage and data-quality gates
block entries only; HOLD and CLOSE always pass through. The gates are evaluated
in that order, so a tripped equity switch reports its own reason before a
per-bar spread, slippage or quality block.

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
a risk override, and no action space exposes a limit. The snapshot carries the
measured `spread_points`, `slippage_points` and `data_quality_ok` values, which
are data the engine judges, not switches it obeys: the limits themselves live
only in the frozen `RiskConfig`. A policy chooses BUY, SELL, HOLD or CLOSE, and
every BUY/SELL passes the same gate. `allow_new` keeps its pre-Phase-11
signature and delegates to `allow_new_extended`, so old callers and new callers
cannot diverge. A test asserts both signatures have exactly `(self, account,
ts)` and that `Simulator.__init__` has no slippage/data-quality argument.

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
- Daily loss, weekly loss, max drawdown, consecutive-loss, emergency, spread,
  slippage and data-quality gates each block the next entry in the raw
  simulator and in the evaluator; the manual loop and the evaluator produce
  identical trade rows. Weekly, drawdown and emergency also force-close the
  open position with the switch name recorded in `Trade.reason`.
- `action_masks()` agrees with `_entry_reason` and the step rejection for every
  gate, both per-gate and on a property-style sweep of a synthetic path.
- The slippage cap allows the exact cap and blocks one point above it; the
  data-quality gate blocks missing/NaN ATR, missing spread and a stale bar gap,
  while the default-off run on the same degraded bars is unchanged.
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

This slice implements the equity, execution-cost and data-quality entry limits
only. Still open for later Phase 11 slices:

- **Partial fills:** the paper broker fills the full requested lot; no partial
  fill model exists.
- **Rejected orders:** orders are assumed accepted once sized. Broker-side
  rejection, margin rejection and requote paths are not modelled.
- **Execution delay:** entries and exits fill on the decision bar's quote with
  ATR slippage; there is no order latency or next-bar fill delay. The slippage
  cap therefore gates the modelled estimate, not a realised fill.
- **Realised-slippage attribution:** no comparison of estimated against actual
  fill slippage exists in paper trading.
- **Whole-frame data audit:** the data-quality gate reads only the decision
  bar's ATR, spread and gap. It does not validate the slice (duplicates, OHLC
  consistency, outliers, provenance, synthetic flag); that stays with
  `scalping.data.validate`, `/api/data/quality` and the promotion gates.
- **Slippage stress:** one-axis cost sensitivity and Monte Carlo trade
  resampling exist (see `docs/backtesting-methodology.md`); the risk engine has
  no per-run slippage multiplier of its own.
