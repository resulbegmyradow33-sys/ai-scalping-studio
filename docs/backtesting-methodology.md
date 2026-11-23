# Backtesting methodology: Monte Carlo and cost sensitivity

Phase 11 slice 2. Two fragility checks that sit on top of the existing
simulator run, both pure library code with a CLI surface. Neither one trains a
model, and neither one is a promotion gate.

The base run is always the same: `scalping.backtest.engine.run_backtest` drives
the single simulator, which owns sizing, fills, stops and risk. The modules
here only read the closed `Trade` list or rerun the same policy with different
execution costs. There is no second accounting implementation.

## Monte Carlo trade-sequence analysis

`scalping.backtest.montecarlo` answers one question: given the trades we
observed, how much of the equity path was luck of ordering?

A backtest is a single permutation of a finite trade set. The simulator's
equity curve, drawdown and ruin path all come from that one ordering. The
module reorders or resamples the trades, compounds each path from a starting
balance and reports the distribution.

| Config field | Default | Meaning |
|---|---|---|
| `n_paths` | `1000` | paths to generate |
| `seed` | `0` | one RNG; same seed gives identical paths |
| `mode` | `shuffle` | `shuffle` or `bootstrap` |
| `block_size` | `5` | block length for a bootstrap; `1` is the IID bootstrap |
| `confidence_levels` | `(0.05, 0.5, 0.95)` | quantile keys `p05`/`p50`/`p95` |
| `starting_balance` | `10 000` | compounding base, normally `ContractSpec.initial_balance` |
| `ruin_fraction` | `0.0` | ruin floor as a fraction of the starting balance, in `[0, 1)` |

Each trade's realised P&L is treated as a return on the starting balance,
`r_i = pnl_i / starting_balance`, and a path compounds multiplicatively:
`balance_k = starting_balance * Π(1 + r_i)`. That keeps the observed dollar
scale while letting gains and losses compound; it does not re-simulate a
different lot size. A balance at or below zero stops the path, because an
account cannot trade through zero.

`max_drawdown` is the deepest drop from a running peak as a positive fraction
of the peak, measured from the starting balance through every post-trade
balance. It matches `backtest.metrics.compute`.

A path is **ruined** when any post-trade balance touches the floor
`ruin_fraction * starting_balance`. Ruin is a touch, not an ending state: a
path that recovers still counts, because the account would have breached the
floor on the way. `probability_of_ruin` is the ruined share of all paths.

The result carries the per-path outcomes (`index`, `final_pnl`,
`ending_balance`, `max_drawdown`, `ruined`), percentile tables for all three
series, the worst path (lowest ending balance, ties by path index) and the path
count. An empty trade list returns a documented zero-path result with zeroed
percentiles; a single trade returns `n_paths` identical paths and sets
`degenerate`. Neither case raises.

### Shuffle, IID bootstrap and block bootstrap

`shuffle` permutes the whole observed list. It preserves the exact multiset of
trades, so **terminal wealth is invariant**: multiplication is commutative and
the product of `(1 + r_i)` does not depend on order. This is arithmetic, not a
bug. Under `shuffle`, the terminal-P&L percentiles are flat and the information
is in `max_drawdown` and the ruin rate. Read the shuffle report as "how bad
could the path have been with these trades", never as a P&L confidence
interval.

`bootstrap` draws trades with replacement, so the sampled multiset changes and
terminal P&L varies too. Plain IID bootstrap has a specific blind spot: it
destroys **streak clustering**. If losses arrive in runs (a losing regime, a
widening spread, a news session), IID resampling scatters them and makes the
drawdown distribution look milder than the data supports.

`bootstrap` with `block_size > 1` uses a circular block bootstrap: it samples
contiguous runs of trades and concatenates them, so short-range dependence
survives inside a block. It still cannot preserve dependence longer than the
block, and the block size is a modelling choice, not a measured quantity. A
block size of one is exactly the IID bootstrap. With a block equal to the
sample length, every path is a circular rotation of the original.

`MonteCarloConfig` defaults to `shuffle`, the least-assumption reorder. The CLI
defaults to `bootstrap` with block size 5, because a flat terminal-P&L band is
easy to misread.

### Limits

- The trades are the ones a single historical run produced. Resampling cannot
  tell you how the strategy behaves on unseen data and cannot repair selection
  bias in which run was chosen.
- Each trade reuses its realised costs; the paths are not re-priced.
- No path-dependent sizing: a real account would resize after a drawdown, and
  that feedback is not modelled.
- A narrow distribution means the result is insensitive to ordering, not that
  the strategy is profitable.

## Parameter sensitivity

`scalping.backtest.sensitivity` answers a different question: how much of the
headline result depends on the cost assumptions we picked?

Each axis reruns the **same policy** on the **same features** through the same
`run_backtest`, changing exactly one execution cost:

| Axis | What moves | Default grid |
|---|---|---|
| `spread_multiplier` | `FeatureFrame.spread` and `CostConfig.spread_floor` | `0.5, 1, 2, 3` |
| `slippage_atr_frac` | absolute `CostConfig.slippage_atr_frac` | `0, ½×, 1×, 2×` the base value |
| `commission_per_lot` | absolute `CostConfig.commission_per_lot` | `0, ½×, 1×, 2×` the base value |

The spread axis has to scale the frame's `spread` array because that is what
fills read; `spread_floor` alone cannot move a fill at backtest time. Slippage
and commission are absolute values, so the report states the actual cost, not
just a multiplier.

Risk, contract, feature and label settings never move. `_settings_for` changes
one cost field and nothing else, and `_features_for` returns the same frame
object for the absolute axes. That rules out look-ahead by construction: no
axis changes what the policy sees or what it is asked to predict. It also means
the axes are one-at-a-time; the true cost surface can have interactions this
grid does not see.

`SensitivityRow` carries the parameter and value, its multiplier relative to
the base cost, trades, net profit, profit factor, expectancy in R, max drawdown,
total costs and the run id. Rows are sorted by value, so the caller's grid order
cannot change the report. The result also holds an unmodified `base` row.

`breakeven_multiplier` is the multiple of the base cost where net profit
crosses zero between two adjacent rows, linearly interpolated. It is `None`
when the sign never changes, when the axis is flat (for example, a hold policy
with no trades), or when the base cost is zero so no multiplier exists. It is
one crossing of one grid, not a precisely located economic threshold.

Note that changing costs can change the trade count on a real strategy: P&L
feeds the risk sizing, so a worse run can size differently and open a different
number of trades. The grid reports that honestly instead of holding the trade
set fixed.

## Commands

Both commands reuse the stored M1 frame, build features once and run the same
`hold`/`rule` baselines as `backtest`. `rule` is the default here because a hold
policy has no trades to analyse.

```bash
cd backend
uv run python -m scalping.cli backtest montecarlo \
  --symbol XAUUSD --policy rule --root ../data-ticks5-store \
  --n-paths 2000 --seed 7 --mode bootstrap --block-size 5 \
  --ruin-fraction 0.5 --confidence 0.05,0.5,0.95 \
  --out runs/mc.json

uv run python -m scalping.cli backtest sensitivity \
  --symbol XAUUSD --policy rule --root ../data-ticks5-store \
  --spread 0.5,1,2,3 --commission 0,3.5,7,14 \
  --out runs/sensitivity.json
```

`backtest montecarlo` prints one JSON line with the trade count, path count,
mode/seed/block size, starting balance, ruin threshold and probability, the
three percentile tables and the worst path. `--out` writes the full report,
including every path outcome. `backtest sensitivity` prints the base row and
every axis with its rows and breakeven; `--out` writes the same payload with an
added `out` path. Both exit 2 with one JSON error line on bad input.

The legacy flat `backtest --policy ... --out ...` form is unchanged. Its
`--out` is now validated in the handler rather than by `argparse`, because the
same parser hosts the two subcommands.

## What these checks prove, and do not

Together they test fragility, not edge. A run that survives a 3× spread and
whose drawdown distribution stays tolerable under a block bootstrap is more
robust than one that does not, but both statements are conditional on the same
historical trade list. Neither is out-of-sample evidence and neither overrides
the promotion gates. If the base strategy has no edge, a narrow Monte Carlo
distribution simply says it loses consistently.

Neither module reads the test split or retrains anything, so neither can
manufacture a result that the simulator did not already produce.
