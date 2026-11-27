# Backtesting methodology: Monte Carlo and cost sensitivity

Phase 11 slice 2. Two fragility checks that sit on top of the existing
simulator run, both pure library code with a CLI surface. Neither one trains a
model, and neither one is a promotion gate.

The base run is always the same: `scalping.backtest.engine.run_backtest` drives
the single simulator, which owns sizing, fills, stops and risk. The modules
here only read the closed `Trade` list or rerun the same policy with different
execution costs. There is no second accounting implementation.

## Panel: the Backtesting page (`/backtesting`)

The dashboard page is a thin client over two existing endpoints. It adds no
accounting of its own: every displayed number is the response's `metrics`.

| Purpose | Endpoint | What it returns |
|---|---|---|
| Run selector | `GET /api/runs` | registry rows; the page keeps only `status == completed` |
| Replay | `POST /api/backtest` | `{run_id, symbol}` → `{report_path, metrics}` |

The request model has exactly two fields. There is no cost-multiplier input on
this page because `BacktestRequest` does not accept one; cost stress and
walk-forward live on `POST /api/evaluate`, which the page does not call.

The response has no equity or trade array. `BacktestResponse` is `report_path`
plus `metrics`; the page renders an equity block only if a future response adds
a finite `equity` array, and otherwise prints an explicit "API does not return
an equity curve" state. It never rebuilds a curve from `net_profit` or
`max_drawdown`.

`metrics_payload` maps non-finite floats (a `profit_factor` of `inf`) to JSON
`null`; the metric cards show those as `—`. The route answers `404` for an
unknown run, `409` for a live or failed run (it cannot be replayed), and `422`
for a malformed symbol; each becomes a labelled error rather than a result.

Monte Carlo trade-sequence analysis remains CLI-only
(`scalping.cli backtest montecarlo`); no HTTP route exposes it, so the page
shows a "CLI only" note and no distribution chart. The same note names
`POST /api/evaluate` as the separate home of walk-forward and cost stress.

Accessibility: the selector is a labelled `<select>` with a hint, the
available-run table has a `<caption>` and `scope="col"`/`scope="row"` headers
inside a focusable scroll region, results are announced through one polite live
region, and the equity block is a focusable `role="region"` with a text summary
in its caption.

Tests cover the loaded state, a missing run list (`404`), a run-list API error,
a `404`/`409` backtest, an absent metrics object, the optional equity array,
symbol validation, and the CLI-only note. `frontend/e2e/pages.spec.ts` adds the
no-runs branch.

What this page proves: that a stored policy can be replayed through the
backtest endpoint and that the returned metrics are displayed unchanged. What
it does not prove: that the policy is profitable, out-of-sample-valid or
promotable. Those require the walk-forward, cost-stress and holdout evidence
described below.

## Execution delay and rejected orders

The simulator fills a decision at the decision bar by default. Two deterministic
execution assumptions can be enabled, both off by default:

- **Entry latency** (`execution_delay=1`): BUY/SELL fills at the **next bar's
  close**; gates are re-checked at the fill bar and the signal/arm is frozen at
  the decision bar. CLOSE is delayed too; SL/TP, timeout and kill exits stay immediate.
- **Rejected orders** (`RejectionConfig`):
  - `max_gap_atr` rejects a delayed entry when the fill close moved more than
    `N × decision-bar ATR` (`info["blocked"] == "rejected_price_gap"`);
  - `reject_zero_volume` rejects entries on zero-volume bars
    (`"rejected_no_liquidity"`).

Both are part of the run id when enabled. A rejection is **not** a partial fill:
the project has no L2/queue data, so partial fills remain blocked rather than
simulated from bar volume.

```bash
uv run python -m scalping.cli backtest --policy rule --execution-delay 1 \
  --max-gap-atr 0.5 --reject-zero-volume --root ../data --out ../runs/bt.json
```

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

### Determinism contract

The bundle checksum must be a function of the inputs, not of how many threads
the runtime happens to use. Two implementation choices enforce that:

- **Checksummed sums never rely on Polars' threaded `group_by().sum()`.** A
  parallel aggregation merges partial sums in an order that follows the thread
  pool and the column's chunk layout, so its last bits can move between two
  runs of the same input (issue #145). Per-regime rows are therefore reduced in
  Python over rows sorted by `(regime, entry_ts)`, using `math.fsum` for net
  profit, total R and expectancy. An exactly rounded sum is independent of
  addend order. The pooled profit factor uses the same reduction. The pooled
  total R and expectancy keep NumPy's single-threaded pairwise sum, which is
  deterministic for a fixed array; the bootstrap matches that statistic so a
  point estimate cannot fall outside its own interval for arithmetic reasons.
- **The locked-OOS section uses the same reducer.** `scalping.discovery.holdout`
  imports `_regime_rows` from the evidence module, so its sealed regime rows
  get the same order-independent arithmetic and the same identity behaviour.

Thread-count independence is tested by building the same evidence in fresh
interpreters with `POLARS_MAX_THREADS=1` and `=4` and requiring identical
checksums, regime rows and trade counts. Pinning the pool is not required.

What the contract does **not** promise:

- **Same environment only.** Changing the bars, dataset artifact, code-version
  digest, settings, seed or library version can change a checksum, and is meant
  to. A different NumPy/BLAS build, CPU architecture or Python patch level can
  still move a floating-point result; exact order-independent sums remove one
  class of drift, not every class. Cross-machine equality is not claimed.
- **Floating point is compared exactly.** The checksum is over canonical JSON,
  with no tolerance band. A last-bit difference is a mismatch, because a
  tolerant comparison would hide exactly the drift the contract exposes.
- **Determinism is not validation.** Identical arithmetic on the same history
  is not independent out-of-sample evidence, not a promotion and not a
  profitability claim.

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

## Locked final holdout (one-shot)

`scalping.discovery.holdout` turns the final test split into a spend-once
resource. The protocol is deliberately separate from search and from the
promotion policy: search never sees test rows, the policy decides whether a
candidate deserves a holdout, and this module records the one time that
decision is acted on.

### Ledger and identity

`HoldoutLedger(path)` is a small SQLite ledger (WAL, foreign keys, schema
version, `busy_timeout`, `synchronous = NORMAL`, matching the discovery
archive). Two tables matter:

| Table | Purpose |
|---|---|
| `holdouts` | one row per `(dataset_id, dataset_checksum)`, with `status` (`sealed`/`spent`), the sealed `spec_hash`, `symbol`, `timeframe`, `code_version`, `settings_digest`, the unlocking `policy_id`/`policy_version`/`evidence_checksum`, `spent_at`, and the immutable `result_json`/`result_checksum`. |
| `access_log` | append-only rows with timestamp, action (`seal`/`unlock`/`evaluate`), outcome (`granted`/`refused`), dataset identity, actor, reason and detail. SQLite triggers reject `UPDATE` and `DELETE`, so a refusal cannot be quietly erased. |

A `HoldoutSpec` carries the identity that is sealed. A new independent dataset
version/checksum is a new key and therefore a new seal. You cannot re-seal a
spent `(dataset_id, dataset_checksum)` for a different spec: the same data
build is the same holdout.

### Protocol

1. **Seal.** `ledger.seal(HoldoutSpec(...))` records the identity while the test
   rows remain untouched. Re-sealing the same identity is idempotent; a
   conflicting identity for the same dataset build is refused.
2. **Unlock.** `ledger.unlock(spec, decision=..., policy=...)` requires an
   accepted `PromotionDecision` from the supplied policy version. The decision's
   dataset id/checksum, spec hash, code version and settings digest must match
   the sealed identity. A rejected decision, a mismatched identity or a policy
   version mismatch leaves the row `sealed`. The transition `sealed -> spent`
   is one conditional `UPDATE`, so concurrent unlock attempts produce exactly
   one winner. A second unlock for the same `(dataset_id, dataset_checksum)`
   raises `HoldoutSpent`, even when it names a different spec.

   The decision must be **sealed** by `evaluate_promotion`; a hand-constructed
   or mutated decision is refused. `policy` is the **target production policy**
   the holdout is being spent for and must be in
   `scalping.discovery.promotion.PRODUCTION_POLICIES`. Because `PROMOTION_V1`
   requires the `locked_oos` section that does not exist before the spend, the
   eligibility decision comes from a named non-production policy with
   `require_locked_oos=False`. `unlock` records the target policy's id, version
   and digest in the holdout row/token/result, so the final promotion sees a
   matching section and re-evaluates the attached bundle with `PROMOTION_V1`
   (`require_locked_oos=True`). The eligibility decision is preserved in
   `decision_json`, so the audit distinguishes the two checks.
3. **Evaluate once.** `evaluate_holdout(features_or_bundle, spec, config,
   ledger=..., unlock_token=...)` validates the single-use token, claims the one
   evaluation slot, slices the manifest's final test rows, and runs the existing
   `DslEvaluator` with the configured costs and stress multipliers. The result
   records trades, net profit, profit factor, win rate, expectancy in R, total
   R, max drawdown, per-regime rows, per-stress rows and iid bootstrap intervals.
   It is written to `holdouts.result_json` in the same transaction that clears
   the token. Re-running the same token, or the same holdout with any other
   token, raises `holdout_result_exists`; the result is never recomputed.
4. **Verify.** `verify_holdout_result(result, evidence)` re-derives the result
   checksum and checks the dataset id/checksum, spec hash, code version,
   settings digest, symbol and timeframe against the evidence identity.
5. **Attach.** `attach_holdout(evidence, result)` returns a **new, type-exact**
   `CandidateEvidence` with a frozen `LockedOOSEvidence` section; the input
   bundle is not mutated. The section's `checksum` is the canonical
   `locked_oos_checksum(section)`, and its `evidence_checksum` is the
   pre-attachment evidence bundle's checksum. The test range must equal
   `splits.test_start_row/end_row`, start at or after
   `splits.validation_end_row`, and carry ordered timestamps after the last
   validation fold. The section is what `require_locked_oos=True` recognises.
   Attaching the same result twice is idempotent, and a result unlocked by a
   different evidence checksum cannot be attached.

Nothing reads the test rows before unlock. A refused unlock never receives a
feature frame; `evaluate_holdout` validates the token and claims the evaluation
before slicing. If the process dies after the claim but before the result is
stored, the holdout stays `spent` with no result. That is the one-shot rule:
losing an evaluation is safer than silently repeating it.

### Refusal codes

`HoldoutError` carries a stable `code`. `HoldoutSpent` is the specific
already-spent refusal.

| Code | Meaning |
|---|---|
| `holdout_not_sealed` | no sealed row exists for this dataset build |
| `holdout_conflict` | a different identity is already sealed for the same dataset build |
| `decision_rejected` | the promotion decision was not accepted |
| `decision_policy_mismatch` | the decision's policy id/version is not the supplied policy |
| `decision_identity_mismatch` | the decision's dataset/spec/code/settings identity differs from the seal |
| `holdout_identity_mismatch` | the requested spec/code/settings differs from the sealed row |
| `holdout_spent` | the dataset build already consumed its one evaluation |
| `holdout_invalid_token` | the token is unknown, belongs elsewhere or was already consumed |
| `holdout_evaluation_in_progress` | another caller claimed the evaluation slot |
| `holdout_result_exists` | a result is already stored and cannot be recomputed |
| `holdout_evaluation_failed` | the one simulation failed after the slot was claimed |
| `non_finite_holdout` | a stored metric, R multiple or CI bound is not finite |
| `stale_*` / `checksum_mismatch` | verification identity or result checksum differs |
| `evidence_checksum_mismatch` | the result was unlocked by a different evidence bundle |
| `locked_oos_conflict` / `locked_oos_mismatch` | the evidence already carries a different holdout section |

### What a new holdout requires

A genuinely new holdout needs a new independent dataset version with a new
`dataset_checksum` (new data or a separately frozen build), a new `HoldoutSpec`
seal, its own accepted promotion decision, and its own single evaluation. A new
candidate on the same dataset build is **not** a new holdout: the spent key
stays spent forever. This is the point of the ledger, not an inconvenience.

### Honest limits

One holdout result is still one sample. The protocol prevents reuse and records
who spent it; it does not make the sample large, add a multiple-testing
correction, or remove selection bias that happened during search. A single
holdout pass can be unlucky in either direction, one symbol/timeframe is not a
cross-market validation, and an iid bootstrap interval ignores serial
dependence. A passing holdout is not a profitability claim and not paper
trading. Reusing the same data under another name, or "just checking" a variation
after seeing the result, destroys the property this ledger exists to protect.

### What is still missing

- **Wire the holdout into the autonomous path.** The protocol, ledger and
  evidence attachment exist and are tested, but discovery/orchestration do not
  yet call them, and no real XAUUSD holdout has been spent. The production policy
  sets `require_locked_oos=True`, so every bundle without a valid frozen
  `LockedOOSEvidence` field rejects with `missing_locked_oos`; that rejection is
  honest and expected, not a bug. The holdout slice attaches the section, after
  which the policy can evaluate the complete evidence bundle.
- **Multiple-testing correction.** Trial count and search budget are recorded,
  and `scalping.discovery.promotion.PROMOTION_V1` caps both and limits how many
  folds may overlap the validation/selection window. The screening control in
  `scalping.discovery.selection` now applies a Bonferroni/Sidak-adjusted mean-R
  test and a deflated effect floor, but no evidence bundle or promotion policy
  consumes its report automatically. A policy pass bounds the exposure; it does
  not undo it.
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

## Writing the dataset artifact and evidence bundle (discovery)

`scalping.discovery.persist` is the production writer that closes issue #136.
Before it, `discovery reproduce` expected an artifact and a bundle that only a
test fixture could create. The writer uses the same builder, artifact module and
evidence module every other command uses; it introduces no second arithmetic
path.

### What it writes

```text
<store-root>/datasets/<dataset_checksum>/artifact.json   manifest (written last)
<store-root>/datasets/<dataset_checksum>/X.npy
<store-root>/datasets/<dataset_checksum>/y_direction.npy
<store-root>/datasets/<dataset_checksum>/y_return.npy
<store-root>/datasets/<dataset_checksum>/ts.npy
<store-root>/datasets/<dataset_checksum>/split.npy
<out-path>                                             evidence JSON (canonical)
<out-path without .json>.spec.json                     strategy sidecar
```

The dataset directory is keyed by `manifest.checksum` (the blake2b digest of
`X`, `y_direction`, `y_return`, `ts` and `split`). `save_dataset_artifact`
verifies that digest against the bundle arrays and the feature frame's
timestamps before writing, stages the artifact in a unique sibling directory,
re-reads it with digest verification and only then moves it onto its final name.
A failure leaves no temp directory and no partial dataset. Re-saving the same
bundle is idempotent. A directory that already holds a **different** dataset
checksum, a manifest that lies about its arrays, or a corrupt artifact is a
`PersistError` (`artifact_conflict`, `dataset_checksum_mismatch`,
`artifact_corrupt`), never an overwrite.

`save_evidence_bundle` re-derives the bundle checksum, requires the spec to hash
to the recorded `spec_hash`, `complexity` and `spec_version`, writes the sidecar
first and the bundle second, each atomically. An existing file at the bundle
path is replaced because `--out` is a report location, not an identity. The
sidecar uses exactly the path `discovery reproduce` looks up, so a replay needs
no `--spec`.

`persist_evidence_run` is the convenience the CLI calls: read the stored bars,
build the causal feature frame and a chronological dataset with the standard
purge/embargo builder, persist the artifact, build the evidence and persist the
bundle plus sidecar. It returns the artifact directory, bundle path, sidecar
path and the evidence identity for the JSON line.

### Producer command

```bash
cd backend
uv run python -m scalping.cli discovery evidence \
  --symbol XAUUSD --timeframe M1 \
  --spec runs/discovery/candidate.spec.json \
  --out runs/discovery/candidate.evidence.json \
  --root ../data-ticks5-store \
  --folds 3 --regime-coverage-min 1.0 \
  --stress 1.5,2.0 --bootstrap-samples 1000 --seed 0 \
  --search-budget 100 --trials 4
```

`--spec` accepts a spec JSON path or inline JSON and is required: the evidence
bundle stores a hash, and a hash cannot be inverted. `--root` is the same bar
store and artifact root every other command uses; `--out` is the bundle path and
its sidecar is written beside it. `--purge`/`--embargo` control the dataset
gap, `--fold-purge`/`--fold-embargo` the fold gap. `--stress ''` is refused by
the evidence builder because promotion evidence needs at least one multiplier
above 1.0. The command prints one compact JSON line:

```json
{"persisted": true, "dataset_id": "XAUUSD:M1:h1:...", "dataset_checksum": "...",
 "artifact": "...", "bundle": "...", "spec": "...", "evidence_checksum": "...",
 "folds": 3, "validation_end_row": 1308, "test_start_row": 1309}
```

### Read it back, then reproduce

`load_dataset_artifact` re-verifies the per-file digests with
`load_dataset_arrays` **and** recomputes the dataset checksum from the loaded
arrays, so a manifest that claims a checksum its arrays do not produce is
caught. `load_evidence_bundle` parses with the strict field loader from
`scalping.discovery.reproduce` and re-derives the bundle checksum. The replay
contract and its refusal codes are described in
[Reproducing a candidate experiment](#reproducing-a-candidate-experiment-discovery)
below; the producer is not a second reproduce implementation and does not
evaluate the test split.

The exact replay of a producer-written bundle is also exercised end to end by
`tests/test_discovery_persist.py`: save, load, `reproduce_bundle`, equal
checksums.

## Multiple-testing and data-snooping control (discovery)

`scalping.discovery.selection` answers one narrow question: after a search
scored `m` candidates and kept the best, is the winner's mean R still above a
threshold that accounts for the search? It is a screening control. It never
promotes, never selects and never writes the archive.

### Inputs

| Config field | Default | Meaning |
|---|---|---|
| `trial_count` | `1` | candidates actually scored before this winner, including rejected ones |
| `search_budget` | `100` | the run's declared cap; a trial count above it is an inconsistent record |
| `family_alpha` | `0.05` | family-wise error rate, not a per-test rate; must be in `(0, 0.5)` |
| `min_effect` | `0.0` | economically meaningful mean R before deflation; a floor, never a target |
| `complexity` | `0` | the candidate's structural cost; only a logarithm enters the penalty |
| `folds` | `3` | contiguous chunks used for the minimum-sample check, not an OOS split |
| `seed` | `0` | makes the naive percentile bootstrap reproducible |

`selection_report(trades, config)` reads each closed trade's `r_multiple`
(already net of costs) in chronological order. It refuses, with a stable
`SelectionError` code:

- `trial_count < 1`, a non-positive or above-`1 000 000` `search_budget`, or
  `trial_count > search_budget`;
- a non-finite `r_multiple`, `family_alpha`, `min_effect` or computed statistic;
- fewer than `max(folds * 5, 10)` trades;
- a zero-variance sample, where the standard error is zero and a t-test would
  be undefined.

The fold count is a diagnostic, not walk-forward evidence: the ordered trades
are cut into contiguous chunks and every chunk must hold at least five trades.
Balanced chunking makes that automatic once the total floor is met. The
implementation also bounds `folds` to `1..100`, `complexity` to `1 000 000` and
`seed` to `2**63 - 1`, so a nonsense integer cannot make the arithmetic or the
RNG call undefined.

### Family-wise significance

The test is one-sided over the mean R multiple:

```text
t      = mean_r / (s / sqrt(n))          with n - 1 degrees of freedom
p      = P(T > t)                        one-sided, unadjusted
alpha_B = family_alpha / m               Bonferroni per-test rate
alpha_S = 1 - (1 - family_alpha) ** (1/m)   Sidak per-test rate
alpha*  = min(alpha_B, alpha_S)           the value that gates
t*      = t quantile at 1 - alpha*
threshold_r = t* * s / sqrt(n)
```

Bonferroni is always at most Sidak for `m > 1`, and Bonferroni controls the
family-wise error rate without assuming independent trials. Sidak is reported
for reference only. The report also exposes the equivalent
`bonferroni_p_value = min(1, m * p)`.

With one trial and `family_alpha = 0.05`, this reduces to the ordinary
one-sided 95% t-test. With `m` trials it is not the ordinary test repeated: the
critical value moves with `m`, and `threshold_r` grows.

### Deflated effect floor

Significance alone is not economic substance, and the winner of many trials is
biased upward even when the null is true. The required mean R is raised by a
penalty scaled by the standard error:

```text
search_term     = sqrt(2 * ln(max(m, 1)))
complexity_term = 0.25 * ln(1 + complexity)
penalty_r       = min(1.0, 0.5 * se * (search_term + complexity_term))
deflated_r      = max(min_effect, threshold_r + penalty_r)
```

`sqrt(2 ln m)` is the leading order of the expected maximum of `m` independent
standard normals, the intuition shared with a deflated Sharpe ratio. The
complexity term charges for a larger hypothesis space. The `0.5` factor keeps
the penalty from double-counting the critical value. The `1.0 R` cap prevents
an extreme trial count from producing an unbounded requirement. These
coefficients are stated heuristics, not calibrated constants; no decimal of the
deflated threshold should be read as precision.

The report passes only when both conditions hold:

```text
mean_r >= deflated_r
bonferroni_p_value <= family_alpha
```

The second condition is arithmetically implied by the first up to floating-point
rounding; it is checked so the threshold and p-value cannot disagree silently.

### Naive interval

`naive_ci` is a seeded percentile bootstrap of the mean, 1 000 resamples,
two-sided at `1 - family_alpha`. It is called naive because it ignores the
number of trials and the selection of this candidate. It is iid by
construction: it destroys streak clustering and can look tighter than a
chronological trade sequence supports. A block bootstrap is not implemented.

### Promotion advice

`promotion_advice(report)` returns a recommendation and reasons, never a
promotion:

| Recommendation | When |
|---|---|
| `accept` | the report passed the adjusted threshold and deflated floor |
| `reject` | the report failed **and** the naive interval's upper bound is below `min_effect`, so the required effect is ruled out |
| `insufficient_evidence` | the report failed but the interval still leaves the required effect possible, or the supplied sample is below a stricter `min_trades` floor |

Statistical non-significance is reported as insufficient evidence, not as
proof of no edge. `accept` means only "this screening control did not stop the
candidate". The named, versioned promotion policy still judges identity, fold
geometry, locked out-of-sample evidence, cost stress, regime coverage and the
champion comparison. The advice object has no policy authority and cannot be
passed to the archive or the promotion path.

### Limits

- **The trial count must be truthful.** Unrecorded trials, a changed search
  space, discarded candidates or adaptive stopping all invalidate the
  correction. The module cannot detect them.
- **No look-elsewhere guarantee.** The correction covers the finite number of
  trials the caller declares. It says nothing about an unbounded or uncountable
  search space, and it is not a p-hacking panacea: a lucky result can still
  pass, and a manipulated search can still produce a report that looks honest.
- **iid and normal-ish assumptions.** A t-test and an iid bootstrap assume
  trade R multiples are sufficiently independent and similarly distributed.
  Serial dependence, regime shifts, cost changes and path-dependent sizing can
  make both the p-value and the interval optimistic.
- **The deflated penalty is a heuristic.** It is bounded, monotonic in trials
  and complexity, and deliberately coarse. It is not a posterior probability,
  a false-discovery rate or a calibrated expected maximum.
- **No parameter-neighbourhood, cross-symbol or out-of-sample evidence.** One
  candidate's trade list is screened. Parameter stability and the locked
  holdout are separate gates.
- **Not a promotion.** A passing report is a candidate for independent
  evidence, never a champion and never a profitability claim.

```python
from scalping.discovery.selection import (
    SelectionConfig,
    promotion_advice,
    selection_report,
)

report = selection_report(
    trades,
    SelectionConfig(
        trial_count=42,
        search_budget=100,
        complexity=12,
        min_effect=0.05,
        folds=3,
        seed=7,
    ),
)
advice = promotion_advice(report)
print(report.to_dict())
print(advice.recommendation, advice.reasons)
```

## Reproducing a candidate experiment (discovery)

`scalping.discovery.reproduce` answers master prompt §36.3: given the identity a
bundle recorded, can the experiment be re-run today and produce the same
evidence? It is a replay, not a re-search: the spec, seed, folds and costs are
fixed by the stored bundle, and nothing is promoted.

### The reproducibility contract

A replay is possible only when all of these still exist and still agree:

| Input | Where it comes from | Checked against |
|---|---|---|
| evidence bundle | one JSON file written from `CandidateEvidence.to_dict()` | its own `checksum`, `schema_version` and required sections |
| strategy spec | `spec=` / `--spec`, or the sidecar `<bundle stem>.spec.json` | `spec_hash`, `complexity`, `spec_version` |
| dataset bars | `<store-root>/<symbol>/<timeframe>/<yyyy>/<mm>.parquet` | rebuilt dataset checksum |
| dataset artifact | `<store-root>/datasets/<dataset_checksum>/artifact.json` | artifact file digests, `dataset_id`, `dataset_checksum`, symbol, timeframe |
| settings | `Settings()` or `--settings` YAML | `settings_digest` |
| code version | `scalping.discovery.run.code_version()` | stored `code_version` |

The bundle cannot carry the strategy itself: a hash cannot be inverted. The
sidecar is a location convention only—its content is validated against
`spec_hash`, so it is not a trusted shortcut.

The dataset artifact is written by
`scalping.datasets.artifact.save_dataset_arrays` through the production writer
`scalping.discovery.persist.save_dataset_artifact` (see the producer section
above). Its manifest supplies the split edges, feature groups and horizon that
the evidence payload does not store, and its files are digest-checked on load. The arrays are then rebuilt
from the bars and must reproduce the recorded checksum; the artifact is not a
source of arrays that could bypass a changed bar.

### What a replay does

```text
read bundle -> verify checksum/schema -> load spec -> check spec hash
  -> check settings digest -> check code version -> load artifact manifest
  -> rebuild dataset from bars -> rebuild evidence with stored policy
  -> compare checksum -> equal: CandidateEvidence
                        different: ReproduceMismatch(sections, differences)
```

`ReproduceMismatch` carries the rebuilt bundle, the expected and reproduced
checksums and the first differing leaf paths (for example
`aggregate.expectancy_r`). The original file is never rewritten.

The dataset checksum covers the full array, including the test rows, because
that is the stored identity. Reproduction rebuilds those rows to prove the
checksum, but no fold, metric or gate reads them: evidence stops at the
validation end and every fold's `test_end_row` is checked to stay inside the
discovery timeline. There is no test/holdout evaluation and no promotion.

### Refusals and exit codes

Identity differences raise `ReproduceError` with a stable `code`; none of them
falls back to a rerun under different inputs.

| Code | Meaning | CLI exit |
|---|---|---|
| `bundle_missing`, `bundle_invalid` | no readable evidence file | 2 |
| `spec_missing`, `spec_invalid` | the strategy was not supplied or does not parse | 2 |
| `dataset_missing` | no artifact for the stored checksum | 1 |
| `dataset_corrupt`, `dataset_manifest_invalid`, `dataset_checksum_invalid` | artifact files or manifest do not verify, or the stored checksum is not a hex digest | 2 |
| `bars_missing`, `bars_invalid` | the bar store cannot rebuild the features | 2 |
| `settings_invalid` | the supplied settings cannot be canonicalised | 2 |
| `spec_mismatch`, `spec_complexity_mismatch`, `spec_version_mismatch` | a different strategy | 1 |
| `settings_mismatch` | a different execution contract, risk or cost block | 1 |
| `code_version_mismatch` | a different arithmetic tree | 1 |
| `dataset_mismatch`, `dataset_checksum_mismatch`, `dataset_columns_mismatch` | the stored data or feature registry changed | 1 |

Exit code 0 means the checksum was reproduced. Exit code 1 means the stored
identity does not match this environment (or the rebuilt evidence differs).
Exit code 2 means the input cannot be read or reproduced at all. Every result
prints one JSON line:

```json
{"reproduced": true, "checksum": "...", "expected": "...", "mismatches": []}
```

A mismatch names the differing sections, for example
`{"reproduced": false, "checksum": "...", "expected": "...", "mismatches": ["aggregate"]}`.
`--out` writes the full report (including the rebuilt bundle) to a separate
path.

### Honest limits

- **Same environment only.** Determinism assumes the same bars, artifact, code
  version digest, settings, seed and library set. NumPy/BLAS builds, CPU
  architecture or a different Python patch level can move a floating-point
  result and change a checksum. That is reported as a mismatch, not hidden.
  Polars thread count is not one of these inputs: checksummed regime sums are
  order-independent by construction, so one thread and many threads agree (see
  "Determinism contract").
- **Deterministic seeds, not statistical replication.** The bootstrap and the
  simulator use fixed seeds, so the same inputs give the same numbers. That
  says the arithmetic replays; it does not say the result would repeat on
  another sample.
- **Floating point is compared exactly.** The checksum is over canonical JSON,
  so a value that differs in the last bit is a difference. There is no
  tolerance band: a tolerant comparison would hide the drift the contract is
  meant to expose.
- **No network, no look-ahead.** Nothing in the module opens a socket, retrains
  anything or evaluates the holdout.
- **Reproduction is not validation.** A replayed bundle is the same historical
  simulation evidence. It is still not independent OOS evidence, not a
  promotion and not a profitability claim.

```bash
cd backend
uv run python -m scalping.cli discovery reproduce \
  --bundle runs/discovery/candidate.evidence.json \
  --store-root ../data-ticks5-store \
  --settings configs/settings.yaml \
  --out runs/discovery/reproduce.report.json
```

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

## Adversarial coverage (evidence, promotion, holdout, reproduce)

`backend/tests/test_evidence_adversarial.py` attacks the gates with forged input
instead of replaying a happy path. It covers, one refusal per test:

- a `CandidateEvidence` subclass whose `to_dict` injects a `locked_oos` section,
  and a plain mapping that resembles a bundle: both are `invalid_evidence`, and
  `evidence_checksum`/`verify_evidence` raise `TypeError` on the subclass;
- a locked section with a forged checksum (`locked_oos_checksum`), an
  `evidence_checksum` bound to another bundle (`locked_oos_evidence_checksum`),
  wrong `test_rows` or unordered timestamps (`locked_oos_geometry`), a
  zero-trade section (`insufficient_trades`; the constructor refuses before
  that), catastrophic metrics, and a CI lower bound of exactly `0.0`
  (`min_ci_lower_r`, while `1e-9` passes);
- the public `promote_with_evidence` entry point with an unregistered test
  policy and with an edited copy of `PROMOTION_V1` (`non_production_policy`),
  the read-only `PRODUCTION_POLICIES` mapping (`TypeError` on mutation), and the
  reserved `manual-override` id;
- direct `DiscoveryArchive._promote_evidence` calls with no capability
  (`TypeError`), a forged one (`unauthorized`), a valid capability over a
  tampered payload (checksum mismatch), and a valid capability over evidence the
  policy rejects (`promotion_rejected`): a capability authorises the caller, it
  never authorises the numbers;
- a second promotion without a stored champion bundle
  (`champion_evidence_missing`), re-promoting the champion itself
  (`champion_same_spec`), and champion bundles that differ in dataset, dataset
  checksum, settings, code version or fold geometry;
- exploratory and manual champions refused by `ChampionProvider`
  (`NoEvidenceChampion`), and the audited `replaces_non_evidence` transition
  that records the legacy champion before the first governed promotion;
- reproduction of a tampered bundle and tampered locked section
  (`checksum_mismatch`), and a locked bundle that reproduces the historical base
  checksum while dropping `locked_oos`.

The tests were mutation-checked: disabling the locked-OOS checksum re-derivation,
the exact-type gate, the archive capability verification, or the champion
dataset-checksum comparison each makes the matching test fail. The mutations were
reverted and are not part of the tree. These tests prove the gates refuse hostile
input; they do not add an edge, a holdout spend, or a profitability claim.
