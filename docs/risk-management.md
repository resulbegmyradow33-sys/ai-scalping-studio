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
- **Slippage stress:** costs are configurable but there is no per-run slippage
  multiplier or Monte Carlo trade resampling yet.
