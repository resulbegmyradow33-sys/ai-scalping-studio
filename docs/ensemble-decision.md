# Ensemble and decision engine

Phase 9. The goal is to combine models **only where out-of-sample evidence
supports it**, and to make the resulting decision explainable. Slice 1 builds the
combination rules, the out-of-fold plumbing and the decision engine. Slice 2 adds
regime-dependent weights, also from out-of-fold rows only. Neither slice trains on
the real store.

The roadmap file names are `ensemble/stacking.py`, `confidence/calibration.py`
and `decision/engine.py`. The code keeps the combination rules in
`scalping.ensemble.core`, regime selection in `scalping.ensemble.regime` and the
decision rules in `scalping.ensemble.decision`; the `confidence/` package
(temperature scaling, drift) is a later slice. The public names match the
roadmap's intent.

## Out-of-fold predictions

`fit_oof(members, bundle, cv_config)` asks
`scalping.ml.cv.fold_geometry` for the purged expanding-window boundaries. The
purge/embargo arithmetic is not copied: `cross_validate` and `fit_oof` resolve
the same `CVPlan` from the same bundle. The fold gap is

```
max(cv_config.purge_bars, bundle.manifest.purge_bars, bundle.manifest.horizon)
+ cv_config.embargo_bars
```

so a wider boundary purge requested at dataset build time is never narrowed
inside the ensemble. Each member is built fresh per fold from a factory; passing
a fitted estimator would leak.

The first fold block is only ever a training window, so it has no out-of-fold
prediction. The scored blocks tile the rest of the training split. That is the
honest set: a row is either scored by a fit that never saw it, or it is not
scored at all.

Two constraints are deliberate:

- every fold's training window must contain the full class set, otherwise the
  member cannot produce a real probability for every class;
- a member's `classes_` must match the shared class order; columns are
  reordered, never guessed.

Validation/test rows are never selected. A NaN, negative, non-finite or
all-zero probability row rejects the member instead of being repaired.

## Combination rules

| Rule | Class | Output |
|---|---|---|
| Soft voting | `SoftVotingEnsemble` | unweighted mean of member probabilities |
| Weighted voting | `WeightedVotingEnsemble` | weighted mean; weights normalised to sum to 1 |
| Hard voting | `HardVotingEnsemble` | one-hot majority over member argmax classes |
| Stacking | `StackedEnsemble` | multinomial logistic regression on OOF probabilities |

`EnsembleConfig` carries the member names, mode, weights, `min_members`, a
probability floor, the renormalisation flag and the meta-learner seed. It
rejects an empty or one-member ensemble, duplicate names, weights that do not
name exactly the members, negative weights, a non-positive weight sum, and
fewer than `min_members` positive weights. A zero weight mutes a member; at
least two members must stay active.

`probability_floor` clamps every member probability before averaging so one
confident member cannot zero a class. It requires `renormalise=True` because
the clamp breaks the row sum.

Hard voting does not break ties. `vote()` returns the tie mask and
`predict_proba()` emits a uniform row, which the decision engine rejects. A
uniform row is the honest representation of "no majority", not a probability a
member produced.

The meta-learner concatenates OOF probabilities in `config.members` order,
encodes the target in the OOF class order and maps the fitted columns back
through `meta_.classes_`. Class weights are deliberately not balanced: the
meta-learner should learn the calibration of the OOF rows, not re-balance them.
`StackedEnsemble.fit` needs every class in the OOF target.

## Disagreement

`disagreement(probabilities)` is the mean pairwise **total variation** distance
between member distributions:

```
TV(p, q) = 0.5 * sum_i |p_i - q_i|
score    = mean over pairs (p, q) of TV(p, q)
```

It is 0 for identical members, 1 when two members put their mass on disjoint
classes (opposite corners of the simplex), and bounded in `[0, 1]` by
construction. Total variation is the share of probability mass that would have
to move to turn one distribution into the other, which is easier to explain in a
log than an entropy proxy.

Mean pairwise TV is used instead of `1 - max_vote_share` because the latter
rewards a larger class set and ignores the size of the losing coalitions. TV
also does not care which class happens to be the argmax, so a confident
minority cannot hide behind a quiet majority.

## Regime-dependent selection

Slice 2 lives in `scalping.ensemble.regime`. It keeps the evidence base of slice
1: every weight is derived from an `OOFResult`, so the rows that chose the
weights were scored by fits that never saw them. Validation and test rows are
never read.

### Labels at decision time

`align_regime_labels(bundle, classify(bars))` joins `ts` + `regime` onto the
bundle rows with a **backward as-of** rule: a row takes the newest label whose
timestamp is at or before it. A label that starts after the row can never be
attached, so the join itself adds no future information. The rule engine is
causal per row (every window runs backwards from the row), so computing labels
on the full frame and reading only the rows that reach the ensemble is safe. A
row before the first label becomes `unknown`.

### Weights

For a regime bucket with at least `min_rows_per_regime` OOF rows (default 30):

```
L_m = mean OOF log loss of member m on the bucket's rows
w_m = (1 / L_m) / sum_j (1 / L_j)
```

Inverse loss is a deterministic heuristic, not a fitted optimiser: there is no
hyperparameter search, row order does not matter, and every weight stays
positive, so a regime changes how much each member counts but never drops one.
The global weights use every OOF row and are the fallback for:

- a bucket with fewer than `min_rows_per_regime` rows (`source="global"` in the
  table and the report);
- a regime label at decision time that never appeared in the OOF rows
  (`rows=0`, global weights). Refusing to combine on an unseen label would turn
  a coverage gap into a silent NO TRADE.

`RegimeWeightedEnsemble.predict_proba(probabilities, regimes)` averages each row
with its own regime's weights. The member set is fixed, so selection is a
weighting rule, not a different model per regime.

### Report

`regime_selection_report(oof, regimes, table)` returns:

- per-regime rows: bucket size, share of OOF rows, weight source, the weights and
  the losses they came from, each member's metrics on the bucket, the
  regime-weighted ensemble and the global-weight ensemble on the same bucket;
- coverage and fallback usage in rows and shares;
- an overall comparison of the regime ensemble, the global ensemble and the best
  single member on the same OOF rows, sorted by accuracy then name.

The payload always carries `descriptive_only: True` and no promotion flag.

### Limits

- Regime labels are heuristic descriptions of the past. Different thresholds move
  rows between buckets, and a bucket is not a forecast.
- Buckets are small. The `min_rows_per_regime` guard avoids fitting weights on a
  handful of rows, but it also means coverage is partial; the report counts the
  fallback instead of hiding it.
- The weights and the scores come from the **same** OOF rows. The comparison is
  in-sample for the weights and therefore optimistic; it is not an independent
  validation and cannot be read as one.
- Inverse loss is not optimal. It is explainable, deterministic and cheap, which
  is why it is used before any tuned meta-model is considered.
- No profitability claim. A better prediction row still has to pass the
  simulator, the cost stress and a regime-coverage gate.

## Out-of-distribution rows

`scalping.ensemble.OODDetector` fits a diagonal-shrunk Gaussian on the
**training features** and scores each decision row with its Mahalanobis
distance. Rows above the training quantile (default `0.99`) are flagged
`unknown`.

```python
from scalping.ensemble import OODConfig, OODDetector

detector = OODDetector(OODConfig(quantile=0.99)).fit(train_features)
decision = engine.decide(
    probabilities, disagreement=0.10, unknown=bool(detector.is_ood(row))
)
```

Rules:

- `unknown=True` blocks a **new** entry with reason `out_of_distribution`;
- with an open position the normal hold/exit rules still decide — an OOD score
  is not a risk model and cannot force a close;
- the detector needs the same feature order and width as training; a mismatch
  is an error, not a silent zero;
- `to_dict()` records quantile, threshold, feature count and training rows for
  a report. The detector itself is not persisted; that is Phase 10.

The score is descriptive. A low score does not mean the row is tradeable, and a
high score does not predict a loss.

## Decision rules

`DecisionEngine.decide()` returns a `Decision` with `action`, `probabilities`,
`classes`, `disagreement`, `regime`, `reason`, `model_version` and `ts`. The
engine is a pure function of probabilities, disagreement, position and
thresholds. `disagreement` is a required argument; a silent zero default would
let a caller bypass the cap and write a false score into the log. `ts` and
`model_version` are log metadata, not rule inputs.

The engine cannot bypass risk or the simulator. `BUY`/`SELL` mean "the signal is
strong enough to ask for an entry", not "an order was placed". Sizing, stops,
cost vetoes and the paper ledger live in later layers.

| Condition (flat book) | Action | Reason |
|---|---|---|
| disagreement > cap | `NO_TRADE` | `disagreement_above_cap` |
| top probability < entry threshold | `NO_TRADE` | `below_entry_threshold` |
| top class is the flat class `0` | `NO_TRADE` | `flat_class` |
| top minus second < min edge | `NO_TRADE` | `edge_below_min_edge` |
| top class `+1` and all gates pass | `BUY` | `entry_long` |
| top class `-1` and all gates pass | `SELL` | `entry_short` |

With an open position the engine only says whether to keep it:

| Condition | Action | Reason |
|---|---|---|
| held class is argmax and probability >= exit threshold | `HOLD` | `hold_long` / `hold_short` |
| otherwise | `NO_TRADE` | `exit_long` / `exit_short` |

The disagreement cap gates **new** risk. With an open position the score is
logged but does not force an exit: an ensemble disagreement is not a risk
model. Flipping a position is left to the risk layer. `HOLD` is never emitted
for a flat book.

Every decision carries a non-empty reason, not just `NO_TRADE`. `Decision`
rejects an empty reason, an unknown action, mismatched probabilities and
classes, negative/non-finite values and a row sum that is not 1.
`to_dict()` keys probabilities by class label and round-trips through JSON.

`compare_to_best_member(y_true, member_probabilities, ensemble_probabilities)`
scores each member and the combined row on the same rows with
`scalping.nn.evaluate.classification_metrics`, and sorts by the same key as
`compare_predictors` (accuracy descending, name ascending). The combined row is
named `ensemble`, and a member cannot reuse that name. If the ensemble does not
win, the simpler member stays; that is the Phase 9 exit criterion.

## What is not built

- **Regime-conditional promotion.** Regime weights are computed and reported, but
  no gate promotes them. The report is descriptive and shares rows with the
  weights; an independent evaluation is a later step.
- **Regime switching between member sets.** The member set is fixed for every
  regime. Only the weights change.
- **Temperature scaling.** No post-hoc calibration of member or ensemble
  probabilities. ECE/MCE are reported, not optimised.
- **Drift monitoring.** Feature/prediction drift over time is not tracked;
  OOD scoring is a per-row distance against the training distribution.
- **Persistence and API.** Decisions are not written to a store and not served.
  The decision log schema exists as `Decision.to_dict()`; storage is Phase 10.
- **Cost-aware veto inside the engine.** Costs and the simulator gate live
  outside this module.
- **Member inference fits.** `fit_oof` produces OOF evidence; fitting members on
  the full training split for live inference is the caller's step.

## Honest note

An ensemble is not a trading edge. It can be better calibrated, less noisy or
better than its best member on prediction metrics, and still lose money after
spread, slippage and the cost stress. The only claim these slices support is a
prediction comparison on fixed rows. Regime weights add a second descriptive
comparison on the same rows, not an independent one. Promotion requires the
simulator, a cost stress and regime coverage; until those pass, the simpler model
stays.
