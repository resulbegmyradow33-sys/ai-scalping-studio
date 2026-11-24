# Local learning engine

Phase 10 turns ad-hoc training runs into a registry-backed loop: a candidate is
trained, measured, gated, and only then promoted; the previous champion stays
recoverable and the samples are compared for drift. This page covers slice 2 —
the candidate → champion gate (`scalping.registry.gate`), the discovery
promotion policy (`scalping.discovery.promotion`) and drift diagnostics
(`scalping.registry.drift`). The SQLite registry itself is described by its
module docstring and `docs/data-pipeline.md`; the scheduler is a later slice.

## Candidate → champion gate

`GateCriteria` is a frozen dataclass of thresholds. **Every threshold is
disabled at `0`**, and an empty cost-stress tuple or missing evidence disables
that check too; this follows the `RiskConfig` convention, so a default
`GateCriteria()` accepts every candidate and changes no existing run. Passing
values is the caller's choice, made explicit at the call site.

| Criterion | Default | Enabled behaviour | What rejects |
|---|---|---|---|
| `min_trades` | `0` (off) | pass when `trades >= value` | fewer closed trades than required |
| `min_expectancy_r` | `0.0` (off) | pass when `expectancy_r >= value` | negative or too small expectancy |
| `min_profit_factor` | `0.0` (off) | pass when `profit_factor >= value` | profit factor below the floor |
| `max_drawdown` | `0.0` (off) | pass when `max_drawdown <= value`; fraction in `[0, 1]` | deeper peak-to-equity drop than allowed |
| `max_ece` | `0.0` (off) | pass when `ece <= value` | calibration worse than allowed |
| `require_positive_cost_stress` | `()` (off) | each multiplier must exist in the stress evidence and have `net_profit > 0` | a missing stress run, or one that loses money |
| `min_regime_coverage` | `0.0` (off) | pass when labelled trades / headline trades `>= value` | too many trades whose entry regime is unknown |

Boundary rules are deliberate and tested at the exact value: the three `min_*`
criteria and `min_regime_coverage` use `>=`, `max_drawdown` and `max_ece` use
`<=`, and cost stress is strictly positive — a break-even stressed run is a
rejection, because paying more costs without losing money is not evidence that
the edge survives them.

A recommended starting policy is stricter than the defaults:

```python
GateCriteria(
    min_trades=100,
    min_expectancy_r=0.05,
    min_profit_factor=1.3,
    max_drawdown=0.25,
    max_ece=0.10,
    require_positive_cost_stress=(1.5, 2.0),
    min_regime_coverage=0.85,
)
```

The values are policy, not statistics. They encode how much evidence this
project demands before a model may trade; they cannot prove the model is good.
A candidate that clears them has passed checks a human chose, nothing more.

### Missing evidence rejects

An enabled criterion whose input is absent produces a rejection, never a silent
pass. The codes are:

| Code | Meaning |
|---|---|
| `missing_metric` | a metric the criterion needs (`trades`, `expectancy_r`, `profit_factor`, `max_drawdown`) is absent or not a number |
| `missing_calibration` | `max_ece` is enabled but no ECE was supplied |
| `missing_cost_stress` | a required multiplier has no `net_profit` entry |
| `missing_regime` | `min_regime_coverage` is enabled but the regime table is absent or empty, or coverage cannot be measured |
| `min_trades` / `min_expectancy_r` / `min_profit_factor` / `max_drawdown` / `max_ece` / `cost_stress` / `min_regime_coverage` | the evidence is present and fails the policy |

`GateDecision.rejections` holds one `"code: message"` string per failed check;
`GateDecision.codes` returns the codes alone. `to_dict()` is JSON-safe and keeps
the criteria, the measured values and the summary, so a log or dashboard can
explain a rejection without re-running the evaluation.

### Promotion and forced overrides

`promote_candidate(registry, model_id, decision, scope=..., reason=...)` is the
only gate-aware promotion path; `ModelRegistry.promote` keeps its signature and
stays the primitive.

- An accepted decision promotes and writes the gate summary into the promotion
  reason (`"<human reason>; gate accepted: trades=…, expectancy_r=…"`).
- A rejected decision raises `PromotionRefused` (a `RegistryError` subclass)
  and leaves the champion untouched.
- `force=True` is the escape hatch. It requires a non-empty human `reason`, and
  the stored reason begins with `forced: <reason>; gate rejected (…)`, so the
  audit trail records that the policy was overridden instead of hiding it.
- The promotion chain still records `previous_model_id`, and the registry's
  `rollback` restores an earlier champion with its own reason.

Comparison against the current champion's *measured* metrics is not part of
this gate: the caller decides which candidates reach it, and the gate only judges
the evidence it is handed. Strategy promotion — where the comparison *is* part of
the policy — lives in the discovery archive and is described in the next
section.

## Strategy champion (discovery archive)

A model registry champion and a strategy champion are different objects. Models
are promoted through `ModelRegistry` + `GateCriteria`; discovered strategies are
promoted through `scalping.discovery.archive.DiscoveryArchive`, which already
holds the candidate's canonical spec, its trials and its lineage. `run_discovery`
records `rejected`/`no_change`/`no_trade` and never promotes; promotion is an
explicit governance call.

| Call | Purpose |
|---|---|
| `promote_exploratory(candidate_id, *, scope, reason, force=False, metric="expectancy_r")` | research/ablation only: one-metric comparison, never the paper champion |
| `promote_with_evidence(archive, candidate_id, evidence, policy, *, scope, reason, expected_identity)` | production path: evaluates the policy internally and stores the evidence bundle |
| `promote_manual_override(archive, candidate_id, override, *, scope, reason)` | stores a human override as `mode = "manual"`; never paper-eligible |
| `champion(scope)` | current candidate, canonical spec, newest accepted metrics, promotion id/time/reason |
| `champion_history(scope)` | every promotion and rollback of the scope, oldest first |
| `compare_to_champion(candidate_id, scope, metric)` | candidate value, champion value, delta, `better` |
| `rollback(scope, *, to_candidate_id, reason)` | restore a previous champion, recording `rolled_back` |

A **scope** is conventionally `<symbol>:<timeframe>`
(`scalping.discovery.champion.scope_for`), and one scope has exactly one champion
(`PRIMARY KEY (scope)`). A trial belongs to a scope when its `dataset_id`
contains every scope token. Each promotion and rollback runs in
`BEGIN IMMEDIATE` and writes the pointer, the history row and the decision in one
transaction, so the live champion and the audit trail cannot disagree.

Schema v3 adds `mode` (`evidence`, `exploratory`, `manual`) and an
`evidence_checksum` pointer to both `champion_promotions` and `champions`, plus an
immutable `evidence` table keyed by the bundle checksum. Schema v4 adds
`policy_id`, `policy_version` and `policy_digest` to the champion rows and
`dataset_checksum`/`settings_digest` to `trials`; legacy rows keep NULL and are
never treated as production evidence. A rollback restores the mode, evidence
pointer and policy identity of the candidate's last promotion. The paper provider
accepts only `mode = "evidence"` **whose policy digest is in the production
allow-list**; a research, manual or unregistered-policy champion is visible in the
audit trail but cannot trade.

`promote_exploratory` refuses, writing nothing: an unknown candidate, a candidate
with no trial on the scope's dataset, a candidate with no **accepted** trial
there, and — unless `force` — a candidate without a measured metric or without a
strict improvement over the champion's newest accepted value. The evidence gate
is not forceable: `force` overrides the comparison only, requires a reason, and
prefixes the stored reason with `forced: `. Rollback only restores a candidate
that was champion of that scope before; a candidate that never won is promoted
instead.

There is no `promote` method. The metric-only path was renamed to
`promote_exploratory` on purpose: a name that looks like the production call must
not hide a one-metric comparison. Production promotion is
`promote_with_evidence`, described below.

## Promotion policy (P2)

`scalping.discovery.promotion` separates **exploratory search** from **production
promotion**. Search ranks candidates on validation fitness. Promotion reads a
:class:`scalping.discovery.evidence.CandidateEvidence` bundle — folds, cost
stress, regimes, bootstrap intervals, trial exposure — and returns a
`PromotionDecision`.

`PromotionPolicy` is a frozen dataclass with no usable default. The constructor
refuses an empty or degenerate policy: a non-empty `policy_id` that is not the
reserved `manual-override`, `version >= 1`, at least three folds, positive sample
and complexity/trial caps, a profit factor above 1, fractions in `(0, 1]`, at
least one cost-stress multiplier above 1, a bootstrap confidence of at least 0.95
from at least 500 draws, a stress-degradation cap of at least 1, and a
selection-fold share in `[0, 1]`. `policy.digest` hashes every field, so the audit
row names the exact policy object. `PROMOTION_V1` is the documented starting
policy:

| Field | `PROMOTION_V1` | Why |
|---|---|---|
| `policy_id` / `version` | `promotion-v1` / `1` | every decision is attributable to a named, versioned policy |
| `min_folds` | `3` | fewer folds say nothing about stability |
| `min_trades` | `60` | about 20 pooled trades per fold |
| `min_expectancy_r` | `0.05` | a margin over a cost-paying break-even, not a rounding error |
| `min_profit_factor` | `1.20` | losses may not nearly match wins |
| `max_drawdown` | `0.25` | worst fold's peak-to-equity drop, a policy cap, not a survival claim |
| `min_positive_fold_fraction` | `2/3` | a majority of chronological windows must pay |
| `min_regime_coverage` | `0.85` | almost every trade must be attributable to a regime |
| `min_ci_lower_r` | `0.0`, strict | the 95% bootstrap lower bound must **exceed** the floor |
| `require_cost_stress` | `1.5, 2.0` | each multiplier must exist in every fold's stress rows |
| `max_complexity` | `40` | structural node cap |
| `max_trials` | `500` | caps the candidate's trial count and the search budget |
| `min_improvement_r` | `0.05` | a challenger must improve on the champion by at least this much |
| `min_bootstrap_confidence` | `0.95` | the aggregate interval must be at least 95% |
| `min_bootstrap_samples` | `500` | fewer draws are not a usable interval |
| `max_stress_degradation` | `2.0` | a stressed fold keeps at least half the base expectancy, stays profitable and keeps PF > 1 |
| `require_locked_oos` | **`True`** | **nothing promotes until the holdout slice attaches a valid `LockedOOSEvidence` section** |
| `max_selection_fold_share` | `0.4` | at most 40% of folds may overlap the validation/selection window; `0` disables the check |

`evaluate_promotion(evidence, policy, *, champion_evidence=None,
champion_spec_hash=None, expected_identity=None)` runs `verify_evidence` against
the **expected** code version, dataset checksum and settings digest, then the
structural and threshold checks, and returns a sealed `PromotionDecision`:
`accepted`, a tuple of `PromotionRejection(code, message, observed, required)`,
the policy id/version/digest, the re-derived evidence checksum, the dataset
checksum, the settings digest and the champion checksum. A decision is accepted
only when the rejection list is empty. Any missing, stale, edited or mismatched
input is a rejection; there is no `force` parameter. A decision cannot be
hand-built or `dataclasses.replace`d into an acceptance: it carries a
module-private seal over its content.

A **selection fold** is a fold whose test block intersects
`[splits.validation_start_row, splits.validation_end_row)`. Those folds sit in the
slice the search selected on, so they are not independent walk-forward evidence.
`max_selection_fold_share=0.4` requires a clear majority of folds to lie outside
the selection window: one of three passes, two of four does not. A bundle whose
share is higher is rejected rather than silently reweighted.

Fold geometry is also checked directly: every fold's test block must stay inside
`[splits.train_start_row, splits.validation_end_row]`, the folds must be
contiguous, and the last fold must end exactly at `validation_end_row`. The
initial-history block before the first fold is allowed; a gap between folds is
not. A fold that reaches into the test split is a `fold_geometry` rejection, not
a wider window.

### Rejection codes

| Code | Meaning |
|---|---|
| `missing_evidence` / `invalid_evidence` | no bundle, or an object of the wrong type |
| `checksum_mismatch` | the stored checksum does not re-derive from the payload |
| `schema_mismatch` | the bundle layout version is not the one this policy reads |
| `missing_section` | empty identity field (including `dataset_checksum`/`settings_digest`), absent split boundaries, aggregate, evidence policy or bootstrap section |
| `inconsistent_evidence` | fold numbering/gaps, aggregate trade or positive-fold counts disagree with the folds |
| `insufficient_folds` / `insufficient_trades` | below the policy minimum |
| `min_expectancy_r` / `min_profit_factor` / `max_drawdown` | metric outside the policy |
| `min_positive_fold_fraction` / `min_regime_coverage` | too few paying folds, or too many unattributed trades |
| `min_ci_lower_r` | bootstrap lower bound does not exceed the policy floor |
| `missing_cost_stress` | a required multiplier is absent from a fold's stress rows, or the bundle's policy records no multiplier above 1.0 |
| `max_complexity` | structural complexity above the cap |
| `trial_exposure` / `search_exposure` | candidate trials or the recorded search budget exceed `max_trials` |
| `selection_fold_share` | too many folds overlap the validation/selection window |
| `fold_geometry` | a fold leaves the train/validation range, reaches into the test split, or leaves a gap |
| `min_bootstrap_samples` / `min_bootstrap_confidence` | the aggregate interval is weaker than the policy |
| `cost_stress` | a required stressed fold is not profitable or has PF <= 1 |
| `stress_degradation` | a stressed fold loses more expectancy than `max_stress_degradation` allows |
| `missing_locked_oos` | the bundle has no valid `locked_oos` dataclass section, or its identity/policy/test range does not match; **the normal state until the holdout slice lands** |
| `non_production_policy` | the policy digest is not in `PRODUCTION_POLICIES` (constructor/refusal text, not a `PromotionRejection`) |
| `stale_trial_evidence` | the accepted trial lacks the same `dataset_checksum`/`settings_digest`, or predates schema v4 |
| `champion_evidence_missing` / `champion_identity_missing` | a champion was named without its bundle or without a spec hash |
| `incomparable_evidence` | candidate or champion lacks `dataset_checksum`, `settings_digest`, `code_version`, splits, folds or fold identity |
| `champion_invalid_evidence` | the stored champion bundle fails `verify_evidence` (edited, wrong schema, no aggregate) |
| `champion_checksum_mismatch` | the champion bundle was edited |
| `champion_spec_mismatch` / `champion_same_spec` | the champion bundle is for another spec, or the candidate *is* the champion |
| `champion_dataset_mismatch` / `champion_dataset_checksum_mismatch` | different dataset id or build |
| `champion_settings_mismatch` | different execution settings (the digest differs) |
| `champion_code_version_mismatch` / `champion_scope_mismatch` | different code version, symbol or timeframe |
| `champion_splits_mismatch` | candidate and champion were cut from different dataset split boundaries |
| `champion_folds_mismatch` | fold rows, ISO timestamps, purge/embargo or fold count differ |
| `champion_cost_stress_mismatch` | the two bundles were stressed with different multiplier sets |
| `champion_bootstrap_mismatch` | different bootstrap sample count or confidence |
| `champion_window_mismatch` | different observation window |
| `champion_improvement` | expectancy improvement is below `min_improvement_r`, or cannot be measured |

The champion comparison is deliberately strict. Comparing a 3-fold bundle with a
5-fold champion, the same folds built from another dataset checksum, or a run
priced with another settings digest is a rejection rather than a normalisation
problem. The `>= min_improvement_r` boundary carries a 1e-9R tolerance only
because floats cannot represent `0.25 - 0.20` exactly; a tie still fails.

### Production promotion path

`promote_with_evidence(archive, candidate_id, evidence, policy, *, scope, reason,
expected_identity)` is the only production call. It evaluates the policy
**inside**; there is no accepted-decision argument to forge:

- it reads the current champion from the archive. An **evidence** champion uses
  its stored evidence as the comparison baseline; an evidence champion whose
  stored evidence is missing is rejected with `champion_evidence_missing`. An
  exploratory or manual champion is replaced as an explicit, audited first
  evidence promotion: `detail_json.comparison = "replaces_non_evidence"` and
  `replaces_non_evidence` keeps the old candidate id, mode, promotion id and
  reason. A true first promotion records `comparison = "first"`.
- it requires `evidence.spec_hash == candidate_id`, so evidence cannot be
  detached from the spec it measured.
- it refuses unless the internal decision is accepted, then calls the private
  `archive._promote_evidence` with a process-private HMAC capability over the
  candidate/evidence/policy identity. There is no public
  `archive.promote_evidence`; a direct call without a valid capability is
  rejected.
- the archive re-builds the payload with `evidence_from_dict`, re-derives its
  checksum and runs `verify_evidence`; an arbitrary blob or mismatched digest is
  `ArchiveError`. It then requires an **accepted trial on the evidence dataset**
  whose `dataset_checksum` and `settings_digest` match (`no_accepted_trial` /
  `stale_trial_evidence`), stores the immutable bundle in the `evidence` table
  keyed by its checksum, and writes the champion pointer, history row and
  `promoted` decision in one transaction.
- the history row stores **no metric** (`metric`/`value` are NULL) and both the
  row and the decision detail record `mode = "evidence"`, the policy
  id/version/digest, the evidence checksum, the champion checksum, the candidate
  spec hash, the dataset id/checksum, the settings digest and the code version.

An evidence promotion is therefore distinguishable from an exploratory or manual
one by inspection: `mode = "evidence"`, `metric IS NULL` and
`detail_json.mode == "evidence"` mark the production path.

### Manual override is not evidence

`manual_override(evidence, policy, *, reason, owner)` records a human decision in
a separate `ManualOverride` object. It requires a non-empty reason and owner,
re-evaluates the policy, copies the original `PromotionRejection` tuple verbatim,
fixes `policy_id` to `"manual-override"` and keeps the original policy
id/version/digest and dataset id beside it. It has no `accepted` field,
`is_policy_evidence` is always `False`, and `promote_with_evidence` raises
`TypeError` if handed one. An override on evidence the policy already accepts is
refused: there is nothing to override.

`promote_manual_override(archive, candidate_id, override, *, scope, reason)` is
the only path that stores one. It writes `mode = "manual"`, keeps the original
rejections in the detail and still requires a known candidate with a trial on the
dataset. The provider refuses a manual champion for paper.

The rule is simple: an override can be stored for audit, but it can never be
mistaken for a passed policy, and the next promotion decision must not cite it as
validation.

### Locked-OOS contract (for the holdout slice)

`CandidateEvidence.locked_oos` is a real frozen `LockedOOSEvidence` field, not a
dictionary read from `to_dict`. It must carry: `checksum`, `evidence_checksum`,
`dataset_id`, `dataset_checksum`, `symbol`, `timeframe`, `spec_hash`,
`spec_version`, `complexity`, `code_version`, `settings_digest`, `policy_id`,
`policy_version`, `test_start_row`, `test_end_row`, `test_rows`,
`test_start_ts`, `test_end_ts`, `trades`, `net_profit`, `profit_factor`,
`win_rate`, `expectancy_r`, `total_r`, `max_drawdown`,
`expectancy_r_ci95_low/high`, `total_r_ci95_low/high`, `bootstrap_samples`,
`bootstrap_confidence`, `bootstrap_method`, `regime_coverage`, plus optional
`regimes`, `stress`, `created_at` and `schema_version`. The section enters the
bundle checksum. `_locked_oos` requires its identity, policy id/version, symbol,
timeframe, spec version and test-row range to match the bundle and the applied
policy; a subclass overriding `to_dict` cannot inject one because the policy
requires `type(evidence) is CandidateEvidence`.

### What acceptance means

Acceptance is **policy compliance**, not proof of an edge. The bundle contains
historical simulation evidence; a policy pass says the measured sample, stability,
cost stress, interval, complexity and search exposure cleared the checks this
project chose. None of that predicts future profitability. The identity includes
the dataset checksum and the execution-settings digest, so a bundle built with
different costs cannot be promoted as if it were the same experiment, and the
selection-fold cap limits — but does not remove — the fact that the search saw the
validation slice.

The locked final OOS section is now mandatory for the production policy. Until
the holdout slice attaches it, every `PROMOTION_V1` decision rejects with
`missing_locked_oos`, so **no evidence promotion can happen**. That is the
intended fail-closed state; do not relax it to make a test pass.

Two structural limits are worth stating. The bundle records its fold/CI
`EvidencePolicy` but not the promotion policy id/version, so the decision,
`policy_digest` and archive row are what attribute a promotion to
`promotion-v1`; the checksum binds the numbers, not the policy. And the audit
trail is local SQLite, not a signed ledger: it proves what this process wrote, not
that a different process could not have written something else.


`scalping.discovery.champion.ChampionProvider` reads the champion and returns a
`StrategySpec` that has passed `validate()` and whose recomputed hash still
matches the archived id. It refuses anything except `mode = "evidence"`:
exploratory and manual champions raise `NoEvidenceChampion`. `NoChampionError`
and `InvalidChampionSpec` are raised instead of falling back to a hand-written
strategy. The module imports the archive and the DSL only; `scalping.paper` is
never imported from `scalping.discovery`.

### Honest limits

- The exploratory `promote_exploratory` numbers come from validation trials. No
  walk-forward, untouched test-slice or cost-stress rule is part of it; if the
  search touched the validation split, the promotion inherits that data-snooping
  risk. That is why it is not the production path.
- The evidence path checks a walk-forward bundle and requires a locked-OOS
  section, but one holdout evaluation is one sample. Acceptance cannot substitute
  for forward paper evidence.
- One metric, higher is better, in the exploratory path. A lower-is-better metric
  such as drawdown must not be passed to it.
- Promotion is policy, not proof: passing the policy says the candidate cleared
  the checks this project chose on historical folds, nothing about the future.
- Drift and decay monitoring is not automatic. Nothing re-tests the champion on a
  schedule or rolls back on its own; `compare_to_champion` and `rollback` are
  tools a human or a later orchestrator slice has to call.
- The audit trail survives a restart and a crash (WAL, one transaction), but an
  OS-level power loss can still drop the newest committed transaction.

## Drift diagnostics

Drift answers "did the new sample move?", not "is the model broken?". All
functions are pure over 1-D arrays; non-finite values are dropped before any
statistic, and `drift_from_datasets` counts the dropped rows per feature so a
mostly-missing column cannot look stable.

| Metric | What it compares | Default | Notes |
|---|---|---|---|
| `population_stability_index(reference, current, bins=10)` | feature or score histogram | bins = 10 reference quantiles | directional (the reference defines the bins); empty-bin share floored at `1e-6`; identical samples return exactly `0.0`; a constant reference is compared as "equal" vs "anything else" |
| `ks_statistic(reference, current)` | empirical CDFs | — | returns `(D, p)`; stdlib `math` only, no scipy dependency; `p` is the asymptotic tail probability with effective sample size, not an exact permutation test |
| `drift_from_datasets(reference_bundle, current_bundle, ...)` | every feature column of a `FeatureFrame` or a mapping of arrays | PSI threshold `0.2` | `DriftReport` holds per-feature PSI/KS, `drifted`, notes and a human summary |
| `prediction_drift(reference_probabilities, current_probabilities, ...)` | per-row maximum class probability, plus mean ECE when outcomes are supplied | PSI threshold `0.2` | confidence PSI/KS; `drifted` is set by PSI alone, ECE is reported as `ece_delta` for context |

`drifted` uses a strict `psi > threshold`; a PSI exactly at `0.2` is not drift.
The conventional reading is PSI below `0.1` = no meaningful shift, `0.1–0.2` =
worth watching, above `0.2` = a significant shift — a rule of thumb, not a
significance test.

Mean ECE averages the per-class ECE over the classes that actually occur in the
outcome vector. A class the model never sees is excluded rather than scored,
and missing outcomes leave the ECE fields `None` with a note instead of inventing
a value. A feature whose PSI cannot be computed appears with `psi=None` and a
note, never as "stable".

Drift detection is descriptive. PSI has no null distribution, KS's p-value is
asymptotic and uncorrected for multiple features, and a market that simply got
quieter can move every number without the model being wrong. These metrics are
a reason to look at fresh evidence; they do not promote, reject or retrain
anything by themselves.

## Not in this slice

- Scheduler and job cancellation, one-heavy-job enforcement, RAM budget.
- Champion-vs-candidate metric comparison inside the *model* gate; the strategy
  counterpart is the discovery promotion policy (previous section).
- A locked final OOS section produced inside this module. The holdout slice
  attaches it; until then `require_locked_oos=True` rejects every promotion.
- Drift-triggered retraining: the report exists, the automation that reads it
  does not.
- Regime-level promotion rules; `min_regime_coverage` only checks that labels
  exist, not that performance is stable inside each regime.
- An orchestrator that calls `promote_with_evidence` on its own. Promotion is
  still an explicit governance call; nothing in this slice schedules it.
