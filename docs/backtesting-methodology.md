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

## Candidate evidence bundle (discovery)

`scalping.discovery.evidence` builds an immutable `CandidateEvidence` for one
DSL spec on one dataset build. It is the promotion-grade evidence input, not a
promotion: it records folds, stress and intervals so a separate policy can
judge them. A bundle on its own proves that the measurements were made
correctly, never that the candidate has an edge.

### What the bundle contains

| Section | Fields |
|---|---|
| identity | `spec_hash`, `spec_version`, `complexity`, `dataset_id`, `dataset_checksum`, `symbol`, `timeframe`, `code_version`, `settings_digest`, `seed`, `trial_count`, `search_budget` |
| policy | folds, purge/embargo request, `min_trades_per_fold`, cost multipliers, regime minimum, bootstrap samples and confidence, observation window |
| splits | half-open `DatasetBundle` row offsets for train, validation and test start/end, so a policy can tell a selection fold from the locked holdout |
| folds | half-open train/test row ranges and their first/last ISO UTC timestamps, effective purge/embargo, trades, net profit, profit factor, win rate, expectancy in R, max drawdown, per-regime rows, per-stress rows |
| aggregate | total trades, pooled expectancy and total R, pooled profit factor, worst fold drawdown, positive-fold fraction, 95% bootstrap intervals, pooled per-regime counts and regime coverage |
| locked OOS | optional `LockedOOSEvidence` (`checksum`, `evidence_checksum`, identity, policy id/version, test rows/timestamps, metrics, intervals, regimes, stress); it is part of the bundle checksum. `checksum` is canonical JSON sha256 without `checksum`/`created_at`; `evidence_checksum` is the pre-attach bundle checksum |
| stamps | `checksum` (sha256 of canonical JSON without `checksum`/`created_at`) and `created_at` |

Every number is finite or the build fails. A profit factor of `inf` (wins with
no losses) is rejected rather than clamped: the fold has no loss evidence. The
identity is part of the checksum, so a stale dataset, dataset checksum, spec,
code version or execution-settings digest is detectable before a policy reads a
metric. `settings_digest` hashes the contract, risk and cost blocks, because
settings are not in the dataset manifest and two bundles priced differently must
not share identity.

The schema version is stored and must equal the module's current version.

### Fold geometry

Rows are `DatasetBundle` positions. The discovery timeline starts at the first
train row and ends at the manifest's validation end; the final test rows are
never sliced or named to an evaluator. It is cut into `folds + 1` contiguous
blocks. Block 0 is initial history; block `k` is fold `k`'s test block.

```text
train: [0, test_start - gap)      expanding
gap:   purge + embargo bars       immediately before the test block
test:  [test_start, test_end)     no overlap with another fold
```

`gap = max(config.purge_bars, manifest.purge_bars) +
max(config.embargo_bars, manifest.embargo_bars)`. The manifest purge is already
at least the label horizon. The DSL policy has no fitted parameters, so the
training range is recorded for provenance rather than refitted; "expanding"
means every later fold's recorded history includes all earlier test blocks
except the trailing purge/embargo rows that precede the new test block.

Each test block is simulated by the same `DslEvaluator` the search uses: same
costs, risk engine, weekly session and bar order. The first `window - 1` bars of
a fold are warm-up inside the block, so a fold needs more than `window` rows.
Regime labels come from the causal rule engine on that fold slice and are
attached to trades at their entry timestamp; an unlabelled entry is `unknown`
and counts against coverage.

### Confidence intervals

Two 95% percentile intervals are reported from pooled per-trade `r_multiple`
values: one for mean R (expectancy) and one for total R. The bootstrap is iid
with replacement, `bootstrap_samples` draws and one fixed seed. It is
deterministic for identical inputs. Trade R values are already net of costs and
are not re-priced. Iid resampling destroys streak clustering, so an interval
can look tighter than a chronological sequence supports; a block bootstrap is a
later extension, not silently assumed.

### Fail-closed codes

`build_evidence` and `verify_evidence` raise `EvidenceError` with a stable
`code`; no partial bundle is returned.

| Code | Trigger |
|---|---|
| `insufficient_folds` | fewer than 3 folds, or too few rows to fill the blocks |
| `fold_geometry` | empty validation/test split, empty train/test range, overlapping test blocks, test rows inside the timeline |
| `fold_too_small` | a test block is no longer than the observation window |
| `fold_trades_below_minimum` | a fold closes fewer trades than the policy minimum |
| `missing_cost_stress` | no multiplier configured, or every configured multiplier is `<= 1.0` (a no-op stress) |
| `missing_regime_evidence` | the rule engine returned no labels |
| `regime_coverage_below_minimum` | too few trades have a regime at entry |
| `non_finite_evidence` | any stored metric, R multiple or CI bound is not finite |
| `holdout_dataset` | the manifest id names a test/holdout split |
| `symbol_mismatch` / `timeframe_mismatch` | config and manifest scope disagree |
| `complexity_mismatch` | the configured complexity differs from the spec |
| `invalid_trial_context` | `trial_count` exceeds `search_budget` |
| `frame_mismatch` | the supplied feature frame is not the bundle's source frame |
| `invalid_spec` | the DSL spec fails validation |
| `invalid_settings` | the supplied settings cannot be canonicalised |
| `stale_dataset` / `stale_dataset_checksum` / `stale_spec` / `stale_code_version` / `stale_settings` | verification identity differs |
| `schema_mismatch` | the bundle schema version is not the current one |
| `checksum_mismatch` / `missing_section` | a bundle was edited or is incomplete |

### What is still missing

- **The locked final OOS call.** This slice never reads the test split, so the
  final holdout is still untouched; evaluating it once is a later, explicit
  decision that must not be used for tuning. The production policy sets
  `require_locked_oos=True`, so every bundle without a valid `LockedOOSEvidence`
  field rejects with `missing_locked_oos`. That rejection is honest and expected,
  not a bug: the holdout slice attaches the section.
- **Multiple-testing correction.** Trial count and search budget are recorded,
  and `scalping.discovery.promotion.PROMOTION_V1` caps both and limits how many
  folds may overlap the validation/selection window, but no deflated metric or
  false-discovery control is applied here. A policy pass bounds the exposure; it
  does not undo it.
- **Parameter-neighbourhood and cross-symbol stability.** One spec on one
  dataset build is measured; a sensitivity sweep is not.
- **Benchmark and champion comparison on identical windows.** `hold`, `ret_5`,
  the legacy incumbent and the current champion are not run by this module. The
  promotion policy compares a candidate with a champion *bundle* only when both
  name the same dataset id/checksum, settings digest, code version, split
  boundaries and full fold geometry; it cannot fill in a missing champion
  measurement.
- **A block bootstrap.** Iid intervals are stated as a simplification.
- **Any paper or live claim.** The bundle contains historical simulation
  evidence only.

### Evidence vocabulary

A bundle is `implemented` when the code runs and its tests pass. It is not
`validated` until an independent chronological OOS protocol and a promotion
policy accept it, not `paper-traded` until the exact spec runs in the paper
engine, and never a profitability claim. Discovery produces candidates; a
bundle is not a promotion, and a validation winner after N trials is a
multiple-testing warning rather than evidence of a durable edge.

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
