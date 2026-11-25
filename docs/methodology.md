# Methodology: from market data to a governed paper champion

**Snapshot:** 2026-11-25, branch `feat/methoddoc`, based on `main` at `626409f`.
This document describes the method the repository actually implements at that
commit. It consolidates the rules already stated in the master prompt, the
evidence documents and the code; it does not replace any of them.

**What this document is for.** A reviewer should be able to read this end to end
and answer four questions without opening ten other files:

1. What evidence is required before a claim may be made?
2. What happens at each pipeline stage, what does it guarantee, and how does it
   fail closed?
3. Which statistical rules are enforced, and which are only documented?
4. What does the project explicitly refuse to claim?

**Precedence.** `docs/requirements/master-prompt.md` §§28–38 and `AGENTS.md` are
authoritative. Where this document and a dated status page disagree, this
document follows the code at `626409f`; the disagreement is reported in
[Document drift](#document-drift) rather than smoothed over.

**No formulas are duplicated here.** The one-line summaries below are orientation.
The full definitions stay in
[`backtesting-methodology.md`](backtesting-methodology.md),
[`model-validation.md`](model-validation.md) and
[`local-learning-engine.md`](local-learning-engine.md).

---

## 1. Evidence ladder

The project uses six words. They are not synonyms, and a higher rung is never
inferred from a lower one. `AGENTS.md` lists five (implemented, trained,
validated, paper-traded, profitable); the acceptance harness adds a distinct
**tested** level, and this document keeps that distinction because "the modules
connect on synthetic data" is weaker than market evidence. The table also names
the artifact a reviewer can actually inspect.

| Level | Means exactly | Required artifact | State at `626409f` |
|---|---|---|---|
| **implemented** | Code exists on the reviewed branch and its unit/integration tests pass. | Commit SHA, the module paths, and the exact `pytest` command and passing output. A module is not an experiment. | Many modules; targeted tests pass (see [Verification](#verification)). |
| **tested** | The real modules connect end to end in the fixed order, on synthetic data, with each stage leaving an artifact. | `runs/acceptance/<acceptance-id>/report.json` from `scalping.cli acceptance run`, with stage statuses and a `claims` block. The harness writes `implemented: true`, `tested: true`, `validated: false`, `paper_traded: false`, `profitable: false`. | Harness exists and its tests pass. It proves plumbing, not an edge. |
| **trained** | A model was actually fitted and a reproducible run artifact exists. For a DSL candidate, "trained" does not apply; the equivalent record is an evaluated trial in the discovery archive. | Classical ML: `runs/ml/ml-<model>-<dataset-suffix>/model.joblib` plus `metrics.json`, with `dataset_id`, split counts, feature columns, seed, model spec and validation metrics. RL tooling has its own run directory through `scalping.rl.registry`, but it is an experimental baseline, not the autonomous engine. | Training code exists; no trained model artifact is committed. No trained claim is made. |
| **validated** | Independent chronological out-of-sample evidence and realistic cost-stress evidence exist. | A `CandidateEvidence` JSON (schema version 3) whose `locked_oos` section is attached from a **spent** holdout, plus the `HoldoutLedger` row with its `result_json` and a re-derived bundle checksum. A sealed, accepted `PromotionDecision` is required before promotion, but policy acceptance is not a separate rung. A passing replay is required before the result is relied on; it is an acceptance check, not part of the definition. | The evidence and holdout machinery is implemented and tested. No real XAUUSD holdout has been spent and no committed bundle carries a real `locked_oos` section. The committed smoke is explicitly **not** validation. |
| **paper-traded** | The exact validated strategy version ran in the paper engine with immutable attribution. | The orchestrator's paper SQLite ledger (`paper.sqlite`, under the archive root) with `sessions.model_version` and every `signals.model_version` equal to the canonical spec hash, the full DSL payload in `sessions.config_json.extra.dsl`, closed `trades` rows, and recorded `source`/`synthetic` provenance. The `paper.advance` job result adds `session_id`, `candidate_id`, `promotion_id` and `spec_hash`. Matured `OutcomeBatch` fingerprints make the record reproducible. | Runner, continuous mode and governed `paper.advance` exist. No evidence-mode champion exists, so no real paper-traded claim is made. |
| **profitable** | Cautious forward evidence from matured, attributed, non-synthetic paper outcomes over a sample declared adequate **before** the result, with cost and regime review. Never a guarantee. | Matured `OutcomeBatch` records plus `outcome_fingerprint`, the forward report, and a written review that states the sample, interval, costs, regimes and decision. No code gate currently converts these into the word "profitable". | None. The only documented forward test (24 trades) has a 95% interval crossing zero and is reported as inconclusive in [`forward-test.md`](forward-test.md). |

### 1.1 The artifact rule

A prose sentence is not evidence. Each rung is tied to a file or database a
reviewer can re-read, and every identity-bearing artifact is checksum-addressed:

- dataset: `manifest.checksum`, plus `artifact.json` and the `.npy` arrays;
- evidence: canonical JSON sha256, plus a spec sidecar;
- promotion: the evidence checksum and the named policy id/version/digest stored
  in the discovery archive;
- holdout: the sealed identity, the append-only access log and the immutable
  `result_checksum`;
- paper: the canonical spec hash and the session ledger;
- matured outcomes: the `OutcomeBatch` fingerprint.

Run directories (`runs/`, `data-*`, model weights) are gitignored. A committed
summary under `docs/evidence/` is a report about an artifact, not the artifact
itself.

### 1.2 What the ladder forbids

- A validation-slice winner is a **candidate**, never a champion.
- Passing CI, a merged PR or the synthetic acceptance harness is not `tested` in
  the market sense and never `validated`.
- A trained model is not validated; a validated candidate is not paper-traded;
  paper trading is not profitability.
- Promotion is not a rung. It is a gate that may turn validated evidence into a
  governed champion. Passing it is policy compliance, not proof of future
  returns.
- NO TRADE and "no champion change" are valid terminal outcomes at every rung.

---

## 2. The pipeline

```
market data
  -> causal features
  -> chronological dataset
  -> bounded discovery search        (validation slice only)
  -> candidate evidence bundle       (train + validation folds, cost stress)
  -> promotion policy                (fail-closed, named, versioned)
  -> sealed locked holdout           (one spend per dataset build)
  -> paper trading                   (evidence-mode champion only)
  -> matured outcomes                (closed trades, fingerprint)
  -> next research cycle             (proposal, not automatic promotion)
```

| Stage | Reads | Writes | May promote? |
|---|---|---|---|
| Data | provider bars | Parquet store, provenance sidecar, validation/quality report | No |
| Features | stored bars, settings/costs | `FeatureFrame`, feature cache entry | No |
| Dataset | feature frame | dataset manifest, arrays, checksum | No |
| Discovery | validation slice | candidate rows, trials, lineage, decision | No; decisions are `no_trade`, `no_change`, `rejected` |
| Evidence | spec + dataset of train/validation folds | `CandidateEvidence` JSON + sidecar | No |
| Promotion | evidence, named policy, current identity | sealed decision; archive evidence row and champion row | Yes, only inside the archive API |
| Holdout | sealed identity, accepted eligibility decision | one immutable result, access log | No; it supplies the missing OOS section |
| Paper | evidence-mode champion, stored bars | paper ledger rows | No |
| Outcomes | paper ledger | `OutcomeBatch` + fingerprint | No; proposal only |
| Next cycle | new data, matured outcomes, approved proposals | queue jobs and audit events | No automatic promotion |

### 2.1 Data ingestion

**Inputs.** Provider bars (CSV, JSON, Dukascopy, synthetic, tick aggregate) and
the optional provenance sidecar `_meta/<symbol>/<timeframe>.json`.

**Guarantees.** The bar schema and UTC timestamps are enforced. Validation
reports and never repairs. The provenance sidecar records source, synthetic
flag, row count, time bounds, checksum and creation time. A changed frame gets a
new checksum; the old sidecar is replaced only by an explicit write. A synthetic
run is labelled synthetic and cannot become real market evidence by default.

**Provenance rule.** `source`/`synthetic` travel from the sidecar into the
dataset manifest, the evidence identity and checksum, the locked holdout seal
and the promotion decision. A field that is missing is `None` (unknown); it is
never defaulted to `False`. Promotion is fail-closed: a bundle with
`synthetic is None` is rejected with `provenance_unknown`, `synthetic is True`
with `synthetic_data`, and only an explicit `False` reaches a production policy.
A non-production demo policy may opt into synthetic data with
`allow_synthetic=True`; the production policy cannot be built with it. A
reproduced bundle must match the stored manifest provenance or is refused with
`provenance_mismatch`.

**Failure modes.** Structural errors are reported as validator issues:
missing schema columns, non-UTC or non-datetime timestamps, unsorted timestamps,
duplicates, nulls, non-finite prices, OHLC inconsistency, crossed quotes,
negative spread or volume. Time gaps and robust-z outliers are warnings. An
unregistered symbol or timeframe is a request error. A missing sidecar is
unknown provenance, not an assumed real source.

**Fail-closed codes.** Validator issue codes (for example
`duplicate_timestamps`, `ohlc_inconsistent`, `crossed_quotes`,
`negative_spread`, `negative_volume`, `price_outliers`, `volume_outliers`,
`time_gaps`), quality status `fail` with `status_rule`, plus `store_empty` and
`provenance_missing`. The data-file trigger refuses an unusable layout with
`unknown_layout`, `invalid_symbol` or `unknown_timeframe`.

**Known gap.** `dataset.build` reads the store and builds; it does **not** call
the quality report first. The validator is an advisory gate a caller must
consult. `GET /api/data/quality` is read-only and never blocks a build.

See [`data-pipeline.md`](data-pipeline.md).

### 2.2 Features

**Inputs.** Stored bars, `Settings`, and the feature registry groups.

**Guarantees.** Features are causal and built from a 59-column registry in seven
groups (price, momentum, volatility, volume, structure, session,
multi-timeframe). A selected column that the source cannot support is refused
before a dataset is built. The disk cache is keyed by the data fingerprint, the
column registry, the cache version and the full cost block; a cache hit must be
byte-identical to a fresh build. A truncated input frame cannot change the
features of the rows it still contains.

**Failure modes.** A missing capability (`bar_volume`, `real_quotes`,
`modeled_spread`, `tick_flow`, `m1_source`) raises `FeatureUnavailableError` and
names every missing input. A wrong matrix shape, mismatched aligned arrays or a
non-finite value after warm-up is a `ValueError`. A corrupt cache entry is
quarantined and rebuilt; a cache write failure does not fail the run. A feature
formula change that does not change the column list is invisible unless
`CACHE_VERSION` is bumped — a review obligation, not a runtime check.

**Fail-closed codes.** No stable string codes; the exception class and message
are the contract. Do not silently substitute a feature for a missing capability.

See [`feature-cache.md`](feature-cache.md) and [`data-pipeline.md`](data-pipeline.md).

### 2.3 Dataset

**Inputs.** A causal `FeatureFrame`, symbol/timeframe, label horizon, split
fractions, purge and embargo bars, optional feature groups and capability
declarations.

**Guarantees.** Splits are chronological, never random. The gap at each boundary
is `max(horizon, purge_bars) + embargo_bars`, so purge is at least the label
horizon. Rows without a complete future horizon are dropped from the tail.
Purged rows stay visible with `split == -1` so a report can show their cost. The
dataset identity is `symbol:timeframe:h<horizon>:<blake2b-prefix>`, and the
manifest records columns, splits, purge/embargo, checksum and creation time. The
array artifact is written atomically, verified on load, and re-checksummed from
the loaded arrays. The test split is retained for the holdout; no downstream
search may read it.

**Failure modes.** Bad fractions, too few usable rows, empty validation/test
ranges, or purge/embargo that consume the test are `ValueError`s. A manifest
that lies about its arrays, a directory that already holds a different dataset
checksum, or a corrupt artifact is a typed persistence refusal, never an
overwrite.

**Fail-closed codes.** `PersistError` with `artifact_conflict`,
`dataset_checksum_mismatch` or `artifact_corrupt`; `ArtifactError` for an
untrusted manifest. The builder's geometry errors have no stable string code.

### 2.4 Discovery search

**Inputs.** A **validation-only** frame, a seed, an attempt budget, a population
and generation policy, the exploratory gate, cost-stress multipliers and the
baseline list.

**Guarantees.** Search spends an exact attempt budget: duplicates, invalid ASTs
and over-complex candidates each spend an attempt, and `trials + skipped`
equals the cap when the run stops on budget. Candidates are deduplicated by
canonical hash; every evaluated trial is persisted with dataset id, seed and
metrics; lineage is recorded. Search is deterministic for the same inputs and
seed. Candidates and baselines run on the same frame, settings, observation
window, weekly session gate and costs. The test split is never copied, scored
or named to an evaluator. The evaluator refuses a dataset id whose tokens name
`test`, `tests`, `holdout`, `final` or `locked`.

**Failure modes.** An empty store or empty validation split is an error, not an
empty search. A default `GateCriteria` is empty and accepts every candidate;
that is exploratory mode, not promotion. A flat candidate can win a
net-profit comparison with zero trades and is recorded as `no_trade`. Search
stop reasons are explicit: `budget`, `patience`, `exhausted`, `stopped`.

**Fail-closed codes.** Decision vocabulary `no_trade`, `no_change`, `rejected`;
`promoted` is never written by discovery. Evaluator and gate refusals include
`missing_cost_stress`, `missing_metric`, `missing_calibration`, `missing_regime`
and the dataset-id guard's `ValueError`. `DiscoveryRunError` is a message
contract, not a stable code.

**Important.** Search fitness is not promotion evidence. A validation winner
after N trials is a multiple-testing warning.

See [`autonomous-discovery.md`](autonomous-discovery.md).

### 2.5 Candidate evidence bundle

**Inputs.** One DSL spec, the dataset bundle (train + validation timeline), fold
count, purge/embargo, cost-stress multipliers above 1.0, regime-coverage floor,
bootstrap samples and confidence, observation window, trial count and search
budget, code version and settings digest.

**Guarantees.** At least three folds. Folds are contiguous, non-overlapping test
blocks that end exactly at the validation boundary. Training is expanding from
the first train row; the gap before each test block is
`max(config.purge, manifest.purge) + max(config.embargo, manifest.embargo)`.
The final test rows are never sliced or named to an evaluator. Every numeric
field must be finite; an infinite profit factor is rejected, not clamped. The
bundle carries its spec hash, dataset id/checksum, symbol, timeframe, code
version, settings digest, seed, trial context, fold metrics, stress rows,
regime rows and bootstrap intervals. Its checksum is canonical JSON sha256
excluding `checksum` and `created_at`; the schema version must equal the current
one (3).

**Failure modes.** A flat candidate, too few folds, overlapping test blocks, a
test block no longer than the window, too few trades per fold, a no-op cost
stress, missing regime labels, non-finite evidence, or a mismatched
dataset/spec/settings identity all refuse the whole bundle. No partial bundle is
returned.

**Fail-closed codes.** `EvidenceError.code`, including
`insufficient_folds`, `fold_geometry`, `fold_too_small`,
`fold_trades_below_minimum`, `missing_cost_stress`,
`missing_regime_evidence`, `regime_coverage_below_minimum`,
`non_finite_evidence`, `holdout_dataset`, `symbol_mismatch`,
`timeframe_mismatch`, `complexity_mismatch`, `invalid_trial_context`,
`frame_mismatch`, `invalid_spec`, `invalid_settings`, the `stale_*` family,
`schema_mismatch`, `checksum_mismatch` and `missing_section`.

The trigger table is in
[`backtesting-methodology.md`](backtesting-methodology.md#fail-closed-codes).
A bundle proves the measurements were made correctly; it never proves an edge.

### 2.6 Promotion policy

**Inputs.** A `CandidateEvidence` bundle, a named/versioned `PromotionPolicy`,
the **current** identity (`ExpectedIdentity`: code version, dataset checksum,
settings digest), and, when a champion exists, the champion's evidence bundle
and spec hash.

**Guarantees.** The policy is fail-closed and cannot be default-constructed into
a no-op. It has no force flag. `policy_id` must be non-empty and cannot be the
reserved `manual-override`. `version >= 1`. Thresholds must be real:
`min_folds >= 3`, `min_profit_factor > 1`, drawdown/positive-fold/regime
fractions in `(0, 1]`, at least one stress multiplier above 1.0, positive
complexity/trial caps. `PROMOTION_V1` cannot disable `require_locked_oos` and
must keep a selection-fold share of at least 0.05. A decision is accepted only
when the rejection list is empty, the bundle checksum re-derives, every required
section exists, and the current identity matches. Decisions are sealed; a
hand-constructed or mutated accepted decision is refused. The public path
accepts only a policy digest registered in `PRODUCTION_POLICIES`. Candidate and
champion must share dataset id/checksum, settings digest, code version, scope,
split boundaries, complete fold geometry, stress multipliers, bootstrap settings
and observation window; a different window is a rejection, not a normalisation
problem.

**Failure modes.** Missing, stale, edited or mismatched evidence rejects.
A production promotion with no locked-OOS section rejects with
`missing_locked_oos` by design. A non-production/test policy is refused with
`non_production_policy`. A champion that is exploratory or manual has no
evidence bridge; replacing it is a separate audited transition
(`replaces_non_evidence`), not a comparison. A candidate cannot tie the
champion; improvement must be at least `min_improvement_r`.

**Fail-closed codes.** Grouped:
`missing_evidence`, `invalid_evidence`, `missing_section`,
`checksum_mismatch`, `schema_mismatch`, the `stale_*` family;
`insufficient_folds`, `fold_geometry`, `inconsistent_evidence`,
`selection_fold_share`; `insufficient_trades`, `min_expectancy_r`,
`min_profit_factor`, `max_drawdown`, `min_positive_fold_fraction`,
`min_regime_coverage`, `min_ci_lower_r`, `min_bootstrap_samples`,
`min_bootstrap_confidence`; `max_complexity`, `trial_exposure`,
`search_exposure`; `missing_cost_stress`, `cost_stress`,
`stress_degradation`; `missing_locked_oos`, `locked_oos_checksum`,
`locked_oos_evidence_checksum`, `locked_oos_identity`,
`locked_oos_geometry`; `champion_evidence_missing`,
`champion_identity_missing`, `champion_improvement`,
`incomparable_evidence`, `champion_invalid_evidence`,
`champion_checksum_mismatch`, `champion_spec_mismatch`,
`champion_same_spec`, `champion_dataset_mismatch`,
`champion_dataset_checksum_mismatch`, `champion_settings_mismatch`,
`champion_code_version_mismatch`, `champion_scope_mismatch`,
`champion_splits_mismatch`, `champion_folds_mismatch`,
`champion_cost_stress_mismatch`, `champion_bootstrap_mismatch`,
`champion_window_mismatch`; `non_production_policy`.

**Acceptance means policy compliance.** It is not proof of an edge, not
out-of-sample validation of future returns and not a profitability claim. The
full policy table and rejection rationale are in
[`local-learning-engine.md`](local-learning-engine.md#promotion-policy-p2).

### 2.7 Locked holdout

**Inputs.** A `HoldoutSpec` identity (`dataset_id`, `dataset_checksum`, spec
hash, symbol, timeframe, code version, settings digest), an **accepted**
eligibility decision from a named non-production policy with
`require_locked_oos=False`, the target production policy, and later one feature
frame.

**Guarantees.** One holdout row per `(dataset_id, dataset_checksum)`. Sealing is
idempotent for the same identity and refused for a conflicting one. Unlock
requires an accepted, sealed decision whose identity matches the seal and which
names the target production policy. The `sealed -> spent` transition is one
conditional `UPDATE`, so concurrent unlocks have exactly one winner. A second
unlock raises, even for a different spec. The evaluation token is single-use;
the result is written in the same transaction that clears the token. Re-running
a token or re-evaluating a spent holdout raises; a result is never recomputed.
The access log is append-only with SQLite triggers rejecting `UPDATE` and
`DELETE`. Test rows are not read before unlock. If the process dies after
claiming the slot but before storing the result, the holdout stays spent with no
result: losing an evaluation is safer than silently repeating it. `attach_holdout`
returns a new bundle with a frozen `LockedOOSEvidence` section; the section
checksum and the pre-attach evidence checksum bind it to that bundle. The
production policy re-evaluates the attached bundle with `PROMOTION_V1`.

**Failure modes.** A new candidate on the same dataset build is **not** a new
holdout. Renaming or reusing the data build does not reset the key. A refused
unlock never receives a feature frame. A hand-constructed or mutated decision is
refused.

**Fail-closed codes.** `holdout_not_sealed`, `holdout_conflict`,
`decision_rejected`, `decision_policy_mismatch`,
`decision_identity_mismatch`, `holdout_identity_mismatch`, `holdout_spent`,
`holdout_invalid_token`, `holdout_evaluation_in_progress`,
`holdout_result_exists`, `holdout_evaluation_failed`, `non_finite_holdout`,
the `stale_*`/`checksum_mismatch` family, `evidence_checksum_mismatch`,
`locked_oos_conflict` and `locked_oos_mismatch`.

**At this commit.** The protocol and ledger are implemented and tested. No real
holdout has been spent. Production promotion therefore rejects every bundle
without a real locked-OOS section; that is the intended fail-closed state.

See [`backtesting-methodology.md`](backtesting-methodology.md#locked-final-holdout-one-shot).

### 2.8 Paper trading

**Inputs.** An **evidence-mode** champion read by `ChampionProvider`, stored
bars, settings and the frozen `RiskEngine`, and a scope (`<symbol>:<timeframe>`).

**Guarantees.** The champion must have been promoted by a policy digest in
`PRODUCTION_POLICIES`, with a matching stored policy id/version; exploratory or
manual champions are refused. The stored canonical payload is revalidated as a
DSL `StrategySpec` and its recomputed hash must equal the archived candidate id.
A missing, invalid or non-evidence champion is an auditable NO TRADE with a
reason, never a legacy fallback. The runner stores the canonical spec hash on
`model_version` for every signal and the full payload in the session config, so
the strategy can be reconstructed from the ledger. Data provenance is written
once at session creation; unknown is `None`, never an invented `false`; a later
known disagreement is a refusal. Bars must be ordered and gap-free: a gap is a
refusal, not a repaired bar. Resume replays recorded actions through a fresh
simulator and checks balance and position before appending. The spec never sees
`settings.risk`; sizing, stops and kill switches belong to the simulator's
`RiskEngine`. No network client, credentials, broker route or real-money path
exists.

**Failure modes.** `NoChampionError` → `no_champion`; `NoEvidenceChampion` →
`no_evidence_champion`; `InvalidChampionSpec` → `invalid_champion`. A known
provenance change is `SessionMismatchError`; a missing bar in the pending range
is `BarGapError` (retryable `bar_gap`); an out-of-order bar is `BarOrderError`
(permanent). Resume divergence is `ResumeError`; an ambiguous or unreadable
outcome is `OutcomeError`. The `paper.advance` handler records NO TRADE as a
success result, not a job failure.

**Fail-closed codes.** Exception names and the handler's `reason_class` values
above, plus ledger `LedgerError`/`DuplicateError`. Synthetic data may exercise
the engine but is refused by the outcome trigger unless explicitly opted in.

**At this commit.** `paper.advance` resolves the champion and runs the
continuous runner. In the committed state no governed champion exists, so the
observable production behaviour is NO TRADE. The promotion id is recorded in the
job result/audit, not inside the ledger; the ledger joins to it by spec hash and
session id.

See [`paper-trading.md`](paper-trading.md).

### 2.9 Matured outcomes

**Inputs.** The paper ledger's sessions and closed trades.

**Guarantees.** `matured_outcomes` returns **closed trades only**; open positions
are counted separately and never become rows. Each row carries entry/exit
timestamps, side, prices, lot, P&L, R multiple, exit reason, canonical strategy
hash, dataset id when known, and the session's `source`/`synthetic` provenance.
`outcome_fingerprint` is sha256 over a canonical payload that excludes
`trade_id` and `open_trades`, sorts rows by their own fields and includes
provenance, so a re-created ledger hashes identically while a provenance change
changes the hash. The trigger dedupes on that fingerprint. A `synthetic: true`
batch is refused by default (`allow_synthetic=False`); opting in is explicit and
auditable. The handler is proposal-only: it returns `queue_discovery` or
`no_action`, has a read-only job-store facade and cannot enqueue, train or
promote.

**Failure modes.** A batch with no closed trade, no strategy hash, no dataset id,
an unreadable session config, a ledger that is missing/unreadable, or a
synthetic batch is refused with a recorded reason. A scan bound can defer a
session (`truncated`), never silently drop it. A cursor crash after enqueue
re-emits the same trigger; the job dedupe key makes it a duplicate, not a second
job.

**Fail-closed codes.** Trigger refusal reasons `empty_batch`,
`synthetic_outcome`, `no_strategy_hash`, `no_dataset_id`, `outcome_error`,
`no_ledger`, `ledger_unavailable`, `truncated`; the handler returns
`no_action` with `refused: "synthetic_outcome"` for a synthetic batch. No
promotion or training code is reachable from this path.

### 2.10 Next cycle

**Inputs.** A new/changed data file, a matured outcome batch, or an approved
research proposal.

**Guarantees.** Triggers are deduplicated by content hash or outcome
fingerprint. A new data file enqueues exactly one `dataset.build`; it does not
enqueue discovery, because a filename cannot supply a gate, budget and seed. An
approved proposal can be wired to emit one `discovery.run` through
`ProposalQueueTriggerSource`, but only a caller that installs the source gets
that behaviour, and approval is never automatic. A matured outcome only
proposes; it cannot execute. All of these paths are caller-driven scans; there
is no market clock or filesystem watcher inside the trigger source.

**Failure modes.** A wrong data layout or scope is refused (`unknown_layout`,
`invalid_symbol`, `unknown_timeframe`). A proposal whose status is not approved,
whose scope is unknown, whose data requirements are empty, or whose dataset hint
looks like a holdout is refused (`not_approved`, `missing_scope`,
`unknown_scope`, `missing_data`, `holdout_data`, `invalid_proposal`,
`truncated`). A queued job is not an execution; a rejected proposal does not
retract an already queued job.

**Fail-closed codes.** The trigger/handler reason lists above. The loop is
deliberately not closed: no code auto-promotes and no code auto-trains from
paper outcomes.

See [`orchestrator.md`](orchestrator.md#matured-paper-outcomes-p5-slice-2).

---

## 3. Statistical rules

### 3.1 Chronology, purge and embargo

No random financial split is permitted. Splits are chronological
`[train][gap][validation][gap][test]`; `split == -1` marks purged rows. At the
dataset boundary the gap is `max(horizon, purge_bars) + embargo_bars`; at an
evidence fold the gap is `max(config.purge, manifest.purge) +
max(config.embargo, manifest.embargo)`. Purge exists because two labels whose
horizons overlap share future bars even when their rows do not. Embargo adds
separation beyond that overlap. The test split is retained for the one-shot
holdout, not for model selection.

Full geometry: [`model-validation.md`](model-validation.md#1-chronology-not-random-splits)
and [`backtesting-methodology.md`](backtesting-methodology.md#fold-geometry).

### 3.2 No tuning on the test split

The search evaluator is bound to one validation dataset id and refuses an id
whose tokens name the test/holdout split. Discovery has no store handle and no
path to the test rows. The evidence builder cuts folds across the train +
validation timeline and stops at the validation boundary. In the DSL discovery
path, the only code that **evaluates** test rows is the holdout evaluator, after
an unlock; reproduction rebuilds the full array only to verify the stored
checksum and computes no metric from the test rows. Classical ML has its own
explicit one-shot test read (`ml train --evaluate-test`, `ml evaluate --split
test`); that is a separate model path whose result must still be reported as a
single test read, never used for tuning. Hyperparameters are selected on
train/validation. Reading the test split "just to check" a variation destroys
the property the ledger exists to protect.

### 3.3 The holdout is one-shot

A holdout is keyed by the dataset build `(dataset_id, dataset_checksum)`. A
second candidate on the same build cannot get a second holdout, and a spent key
stays spent forever. A genuinely new holdout requires a new independent dataset
version/checksum, a new seal, its own accepted eligibility decision and its own
single evaluation. The ledger records who spent it and why; it does not make the
sample larger, remove selection bias, or correct multiple testing.

### 3.4 Multiple-testing screening

The screening control answers one narrow question: after `m` scored candidates,
is the winner's mean R above a threshold that accounts for the search? It
computes one-sided Bonferroni and Sidak per-test rates
(`alpha* = min(family_alpha/m, 1 - (1-family_alpha)**(1/m))`) and a deflated
effect floor that raises the required mean R by a bounded penalty for trial
count and complexity. It reports `accept`, `reject` or `insufficient_evidence`
and a naive seeded percentile interval; it never promotes, selects or writes the
archive. A passing report means only "this screening control did not stop the
candidate".

The correction is only as honest as the trial count. Unrecorded trials, a
changed search space, discarded candidates or adaptive stopping invalidate it,
and the module cannot detect them. It gives no look-elsewhere guarantee for an
unbounded search space. `PROMOTION_V1` separately caps `trial_count` and
`search_budget` at 500; that bounds exposure but does not undo it. No evidence
bundle or promotion policy automatically consumes the selection report.

Full thresholds and limits: [`backtesting-methodology.md`](backtesting-methodology.md#multiple-testing-and-data-snooping-control-discovery).

### 3.5 Bootstrap limits

The evidence bundle and the selection control use an **iid** percentile
bootstrap of per-trade R multiples. It is deterministic for a fixed seed, but it
destroys streak clustering: a losing regime, widening spread or news session is
scattered, so the interval can look tighter than a chronological sequence
supports. A block bootstrap is not implemented in either place. `PROMOTION_V1`
requires at least 500 draws and a confidence at least 0.95, and the lower bound
must be strictly above 0R; a weak interval rejects.

Monte Carlo trade-sequence analysis does offer a circular block bootstrap, but it
resamples one historical trade list and reuses realised costs. It is a fragility
check, not out-of-sample evidence and not a promotion gate.

### 3.6 Minimum evidence is a policy choice, not a natural law

`PROMOTION_V1` asks for 3 folds, 60 pooled trades, +0.05R expectancy, profit
factor 1.20, 25% worst-fold drawdown, 2/3 positive folds, 85% regime coverage,
1.5x and 2.0x cost stress, a complexity cap of 40, at most 500 trials, and
at most 40% of folds overlapping the validation/selection window. These numbers
encode how much evidence this project demands; they are not calibrated
statistics. A pass is policy compliance.

### 3.7 Reporting rules

Every trading result reports sample size, confidence interval, trade count,
drawdown, costs and regime breakdown. Accuracy or win rate alone is never a
trading result. Cost stress is part of the headline, not an appendix. A single
metric comparison is allowed only in exploratory search, never in promotion.
A failed candidate stays in the experiment log; deletion is not evidence.

---

## 4. Legacy baselines

The legacy components are comparison assets, not the discovery engine. Their
classification is in `AGENTS.md` and the import/consumer audit is in
[`legacy-inventory.md`](legacy-inventory.md).

| Component | Role | Allowed use |
|---|---|---|
| `flow_momentum`, `FlowMomentumRule`, `PolicyCard`, `strategy_card.json`, nightly calibration | Legacy incumbent and historical benchmark; still the panel/API runtime | Reproduce and compare. Never seed or constrain DSL search. Its self-promotion path is not evidence for the autonomous engine. |
| `strategy/signals.py` five fixed families + `strategy/bandit.py` LinUCB | Legacy finite-menu selector | Optional benchmark/ablation. Do not expand the arm list as a substitute for discovery. |
| PPO / behaviour cloning / `nightly_rl.py` | Experimental learning baseline | Compare fairly; cannot promote itself without the same evidence gates. |
| `hold`, `ret_5`, incumbent flow rule | Rule-free baseline suite | Identical-window comparison only. |

### 4.1 How a comparison avoids contaminating search

1. **Import boundary.** The legacy families are not imported by `discovery/`,
   `strategy/dsl/`, `orchestrator/` or `paper/`. The DSL search space is the
   bounded primitive grammar, independent of any named strategy. Baselines are
   not seeds, mutation templates or mandatory inputs.
2. **Same execution surface.** `hold` and `ret_5` are scored on the exact
   validation slice, with the same `Settings`, observation window, weekly
   session gate and costs as the candidates. A win caused by a cheaper baseline
   is not a win. The best baseline's selection metric is the comparison
   reference; a candidate must strictly exceed it before discovery hands it on
   as `rejected` (a candidate record, not a promotion).
3. **Separate accounting.** Baseline results are rows in the run report, not
   candidate trials. They do not enter the trial count used for multiple-testing
   correction. The search's trial count is the candidate denominator.
4. **Same-window champion comparison.** A promotion challenger is compared with
   the champion's stored evidence only when both name the same dataset
   id/checksum, settings digest, code version, scope, split boundaries, fold
   geometry, stress multipliers, bootstrap settings and observation window. A
   different window is a rejection. A missing champion measurement is not filled
   in or normalised.
5. **No fallback.** The autonomous path never falls back to a legacy handwritten
   strategy. A missing or invalid DSL champion is an auditable NO TRADE. The
   legacy panel path is separate and may still replay or self-promote its card;
   that path is not a governed champion and must not be cited as one.

### 4.2 What is not yet compared

There is no automated side-by-side of the legacy flow card, the five legacy
families, LinUCB or PPO against a DSL candidate on the **same locked holdout**.
The holdout module evaluates the candidate spec; it does not run baselines. A
reviewer who needs that comparison must run it as a separate, identical-window
experiment and record it separately. Without that, a claim that the DSL engine
"beats the legacy system" is unproven.

The legacy panel can auto-promote through the nightly script and an HTTP
endpoint, gated only by holdout trade count, positive expectancy and beating the
incumbent. That is a legacy runtime decision with no DSL evidence and no
`PROMOTION_V1`; it is documented in
[`legacy-inventory.md`](legacy-inventory.md#42-auto-promotion-yes-twice-with-no-human-gate)
and is outside the autonomous promotion path.

---

## 5. Reproduction contract

`scalping.cli discovery reproduce` answers one question: given the identity a
bundle recorded, can the experiment be re-run today and produce the same
evidence? It is a replay, not a re-search. The spec, seed, folds and costs are
fixed by the stored bundle.

### 5.1 What must still agree

| Input | Source | Checked against |
|---|---|---|
| Evidence bundle | canonical JSON from `CandidateEvidence.to_dict()` | its own checksum, schema version and required sections |
| Strategy spec | `--spec` or the sidecar `<bundle stem>.spec.json` | `spec_hash`, `complexity`, `spec_version` |
| Dataset bars | `<store-root>/<symbol>/<timeframe>/<yyyy>/<mm>.parquet` | rebuilt dataset checksum |
| Dataset artifact | `<store-root>/datasets/<dataset_checksum>/artifact.json` | file digests, `dataset_id`, checksum, symbol, timeframe |
| Settings | `Settings()` or `--settings` YAML | `settings_digest` |
| Code version | `scalping.discovery.run.code_version()` | stored `code_version` |

The producer side is `discovery evidence`, which writes the dataset artifact,
the evidence JSON and the spec sidecar. The sidecar is a location convention,
not a trusted shortcut: its content is validated against `spec_hash`.

### 5.2 What a replay does

```
read bundle -> verify checksum/schema -> load spec -> check spec hash
  -> check settings digest -> check code version -> load artifact manifest
  -> rebuild dataset from bars -> rebuild evidence with stored policy
  -> compare checksum -> equal: CandidateEvidence
                        different: ReproduceMismatch(sections, differences)
```

The original file is never rewritten. Exit code 0 means the checksum was
reproduced; 1 means the stored identity does not match this environment (or the
rebuilt evidence differs); 2 means the input cannot be read or reproduced.
Refusal codes include `bundle_missing`, `bundle_invalid`, `spec_missing`,
`spec_invalid`, `dataset_missing`, `dataset_corrupt`,
`dataset_manifest_invalid`, `dataset_checksum_invalid`, `bars_missing`,
`bars_invalid`, `settings_invalid`, `spec_mismatch`,
`spec_complexity_mismatch`, `spec_version_mismatch`, `settings_mismatch`,
`code_version_mismatch`, `dataset_mismatch`, `dataset_checksum_mismatch` and
`dataset_columns_mismatch`.

### 5.3 Environment limits

- **Same environment only.** Determinism assumes the same bars, artifact, code
  version, settings, seed and library set. NumPy/BLAS builds, CPU threading or a
  different Python patch level can move a floating-point result and change a
  checksum. That is reported as a mismatch, not hidden.
- **Pin Polars threads.** The committed real-store smoke found that Polars'
  default group-by summation order made two reproduce runs differ in the last
  bits of regime sums (~1e-16 relative), producing different checksums. With
  `POLARS_MAX_THREADS=1` the bundle and reproduce agreed exactly. The smoke
  environment was Python 3.12.13, NumPy 2.5.5 and Polars 2.1.0.
- **Floating point is compared exactly.** The checksum is over canonical JSON;
  there is no tolerance band. A tolerant comparison would hide the drift the
  contract exists to expose.
- **Deterministic seeds are not statistical replication.** The arithmetic
  replays; it does not say the result would repeat on another sample.
- **No network, no retraining, no holdout.** Reproduction evaluates no test
  rows, spends no holdout and promotes nothing.
- **Hardware is a budget, not a proof.** The target is a Mac mini M2 with 8 GB.
  One heavy job at a time. The `M2_8GB_PROFILE` guard refuses at 6000 MB system
  RSS, 4500 MB process RSS, load 1m 6.0 or 2048 MB free disk; the daemon uses it
  by default, while a bare `Orchestrator` still defaults to `AllowAllGuard`.
  Measured peaks (for example ~884 MiB for a 200k-bar M1 pipeline suite) are
  local observations, not an acceptance result or an SLA.

Full contract and command examples:
[`backtesting-methodology.md`](backtesting-methodology.md#reproducing-a-candidate-experiment-discovery).
Environment evidence: [`research-smoke-2026-11-24.md`](research-smoke-2026-11-24.md),
[`m2-measurements-2026-11-24.md`](m2-measurements-2026-11-24.md) and
[`mac-mini-optimization.md`](mac-mini-optimization.md).

---

## 6. Accepting a candidate

A reviewer accepts a candidate only when every item below is answered with an
artifact, not a narrative. "No" or "unknown" is a rejection, not a maybe.

### 6.1 Before the run

- [ ] One falsifiable hypothesis, written before the result.
- [ ] Dataset id and checksum; symbol, timeframe, row count, UTC coverage.
- [ ] Provenance sidecar reviewed; `synthetic` known, not guessed.
- [ ] Split geometry stated: train/validation/test rows, purge and embargo,
      untouched test window.
- [ ] Baselines named: `hold`, `ret_5`, the relevant legacy baseline, the
      current champion.
- [ ] Search budget, population/seed, complexity cap and selection metric fixed.
- [ ] Cost model and stress multipliers fixed (at least one above 1.0).
- [ ] Regime coverage floor and bootstrap settings fixed.
- [ ] Stopping rule and NO-TRADE rule stated.
- [ ] Holdout plan stated: which dataset build, who may unlock, what decision
      authorises it.
- [ ] Rollback target named.

### 6.2 During the run

- [ ] Data quality report is `ok`, or every warning is explained and accepted;
      no silent repair or backfill.
- [ ] Feature availability matches the declared capabilities; no `flow_*`
      inferred from bars.
- [ ] Dataset is chronological; purge is at least the horizon; test rows were
      not scored.
- [ ] Every attempt is recorded, including duplicates, invalid ASTs and
      gate-rejected candidates; `trials + skipped` is honest.
- [ ] Candidate and baselines used the same frame, window, session and costs.
- [ ] No candidate was selected on the test split.

### 6.3 Evidence and promotion

- [ ] Bundle schema is current; checksum re-derives; every number is finite.
- [ ] At least 3 folds; folds are contiguous, non-overlapping and end at the
      validation boundary.
- [ ] Per-fold cost stress above 1.0 exists and remains profitable with PF > 1;
      degradation is within the policy cap.
- [ ] Regime coverage meets the floor; unlabelled entries are counted honestly.
- [ ] Bootstrap interval meets the minimum sample/confidence and its lower bound
      clears the policy floor.
- [ ] Trial count and search budget are truthful and within the cap.
- [ ] Complexity is within the cap; no post-hoc simplification of the reported
      candidate.
- [ ] The promotion policy is named, versioned, registered and non-empty;
      `PROMOTION_V1` was not weakened.
- [ ] The bundle's `source`/`synthetic` are explicit; unknown is
      `provenance_unknown` and synthetic is `synthetic_data` on the production
      policy.
- [ ] Candidate and champion were compared on identical identity, geometry and
      provenance; a different window or tape was rejected, not normalised.
- [ ] A rejection lists stable codes and the observed/required values.

### 6.4 Holdout and paper

- [ ] The holdout was sealed before the candidate was chosen, including its
      `source`/`synthetic`; the spend cannot switch tapes.
- [ ] The eligibility decision was accepted and bound to the seal.
- [ ] The unlock token was single-use; the result is stored and never recomputed.
- [ ] The attached `locked_oos` section matches the dataset's test rows and the
      final validation boundary.
- [ ] The production policy re-evaluated the attached bundle and accepted it.
- [ ] Any second evaluation attempt was refused (`holdout_spent`,
      `holdout_invalid_token` or `holdout_result_exists` as applicable).
- [ ] The paper champion is evidence-mode and promoted by a registered policy
      digest; exploratory/manual champions are refused.
- [ ] The ledger records the canonical spec hash on the session and every
      signal, plus dataset id and known `source`/`synthetic`.
- [ ] Provenance is `synthetic: false` and a real `source`; unknown is not
      accepted as real.
- [ ] Risk limits were frozen and enforced; no spec, action or job payload
      raised them.
- [ ] Matured outcomes contain only closed trades; the fingerprint is stable and
      matches the batch.
- [ ] Any forward claim states the sample, interval, costs and regimes, and does
      not call a 24-trade result proof.

### 6.5 Reproduction and reporting

- [ ] `discovery reproduce` exits 0 in the pinned environment, or the mismatch
      is explained and the claim is withdrawn.
- [ ] Code version, settings digest, dataset checksum and seed are recorded.
- [ ] The report says implemented / tested / trained / validated / paper-traded /
      profitable separately, with the exact artifact for each.
- [ ] Failed candidates and rejection reasons remain in the archive.
- [ ] Rollback restores a candidate that was champion of that scope before; it
      does not promote a never-champion.
- [ ] No data, secrets, model weights or run stores were committed.

If any box is unchecked, the candidate may remain a research record. It may not
be described as validated, paper-traded or profitable.

---

## 7. Claims the project does not make

These are explicit non-claims, not modesty. Each one is a statement a reviewer
should reject if it appears in a PR, a dashboard or a release note.

- **Not an autonomous profitable trading AI.** The repository contains a
  strategy-independent discovery foundation and a governed evidence path, not a
  proven money-making system.
- **Not a closed accepted loop.** No committed artifact shows a real holdout
  spend, a runtime promotion, a governed champion in paper, or a matured paper
  outcome that triggered the next research cycle. The paper-outcome path is
  proposal-only.
- **No real out-of-sample validation of any DSL discovery candidate at this
  commit.** Evidence folds sit inside train + validation. The final test split is
  locked and unspent. A bundle without `locked_oos` is not validation.
- **No profitability claim.** The documented 24-trade forward test has a 95%
  interval crossing zero; it is positive evidence, not proof. A passing
  promotion policy is policy compliance, not a return forecast.
- **No real-money or broker execution.** Real-money routing is out of scope and
  disabled. Paper trading is the only execution target. An MT5 demo adapter is a
  disabled skeleton, not a live path.
- **No universal edge.** One run is one symbol, one timeframe, one dataset
  build. Cross-symbol, cross-timeframe, cross-regime and parameter-neighbourhood
  stability are not established by a single bundle.
- **No claim that the legacy panel is the autonomous engine.** The legacy card
  remains the panel runtime and can self-promote through its own path. It is a
  baseline/historical benchmark, not evidence for the DSL champion.
- **No claim that a document, PDF or LLM "understands" the market.** Ingestion is
  deterministic parsing and heuristic extraction. Documents are untrusted
  source material; they never become orders or trusted rules.
- **No order-book, tick-by-tick or Level-2 model claim.** Tick data is
  aggregated to bars; no Level-2 feed exists.
- **No statistical guarantee from multiple-testing screening.** The correction
  covers the declared finite trials, not an unbounded search space. A lucky
  result can still pass; the module cannot detect hidden trials.
- **No serial-dependence correction in the evidence interval.** The iid
  bootstrap can be optimistic. A block bootstrap exists only in the separate
  Monte Carlo fragility tool, not in promotion evidence.
- **No M2 acceptance.** Local wall/RSS numbers are measurements on one noisy
  machine, not an SLA or a passed budget. The bare orchestrator does not enforce
  the guard unless configured.
- **No claim that green CI, a merged PR or the synthetic acceptance harness is
  market evidence.** Those prove code and plumbing.
- **No claim that a failed candidate is disproven forever.** A rejection is a
  decision on the evidence available; it is not proof that no variant could
  ever work.
- **No claim that a promotion cannot lose.** The champion can lose; NO TRADE and
  no champion change are valid outcomes.

---

## Document drift

Two dated status documents predate the current code and are stale on these
points. They are left as written; this document records the conflict instead of
copying it.

| Document | Stale claim | Code at `626409f` |
|---|---|---|
| `STATUS.md` (snapshot `693f8a3`) | Says #136 is open, that `reproduce` has a consumer but no producer, and that `paper.advance` does not require promotion-policy provenance. | `discovery evidence` writes the dataset artifact and evidence bundle (#159); `paper.advance` resolves an evidence-mode champion through `ChampionProvider` (#156) and writes provenance-aware sessions (#160). |
| `docs/task-board.md`, second-refresh block | Says no merged command or handler calls the promotion policy, and that `paper.advance` uses the slice runner without promotion provenance. | The promotion **policy** still has no CLI/orchestrator call site, so "no runtime promotion" remains true. `paper.advance` does enforce the governed champion. The evidence producer now exists. |

`docs/autonomous-system-audit-2026-11-24.md` is a historical snapshot by design;
its technical limitations remain relevant unless the current-state addendum or
the code says otherwise.

---

## Verification

What this document checked, and what it did not.

**Checked against code and docs at `626409f`:**

- evidence ladder vocabulary and artifacts: `AGENTS.md`, `STATUS.md`,
  `docs/acceptance-harness.md`, `backend/src/scalping/acceptance/run.py`;
- pipeline identities and refusals: `discovery/{evidence,promotion,holdout,persist,reproduce,run,search,evaluate}.py`,
  `paper/{continuous,autonomous,ledger}.py`, `orchestrator/{handlers,paper_triggers}.py`,
  `datasets/builder.py`, `features/registry.py`, `data/{validate,quality,provenance}.py`;
- statistical rules: `docs/backtesting-methodology.md`,
  `docs/model-validation.md`, `discovery/selection.py`;
- legacy classification and import boundary: `docs/legacy-inventory.md`;
- reproduction and environment limits: `docs/evidence/discovery-smoke-2026-11-24.json`,
  `docs/m2-measurements-2026-11-24.md`, `docs/mac-mini-optimization.md`.

**Targeted tests run on 2026-11-25** (worktree `backend`, `PYTHONPATH=src`,
interpreter from the parent worktree `.venv`, Python 3.12.13, NumPy 2.5.5,
Polars 2.1.0):

```text
tests/test_discovery_persist.py tests/test_locked_holdout.py
tests/test_discovery_selection.py                         67 passed
tests/test_promotion_policy.py tests/test_discovery_reproduce.py
tests/test_acceptance.py tests/test_paper_continuous.py   190 passed
```

This is **not** a full-suite run, an M2 benchmark, a market-data run, a holdout
spend or a promotion. No test count is offered as a quality or profitability
metric.

**Not verified here, and therefore not claimed:**

- No committed artifact shows a real holdout spend. A local gitignored
  `HoldoutLedger` could exist outside the repository; it is not evidence in the
  repository.
- No committed artifact shows a real governed champion, a real paper session
  chosen by promotion, or matured real-market outcomes.
- The forward-test statistics in `docs/forward-test.md` were read, not re-run.
- The M2 wall/RSS numbers were read from the dated evidence file, not re-measured
  on this worktree.
- No real market store was opened; no candidate was searched, trained, validated,
  paper-traded or promoted while writing this document.
- The exact RL run-directory layout was not exercised; RL is an experimental
  baseline and no trained RL artifact is claimed.
