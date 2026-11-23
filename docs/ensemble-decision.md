# Ensemble and decision engine

Phase 9, slice 1. The goal is to combine models **only where out-of-sample
evidence supports it**, and to make the resulting decision explainable. This
slice builds the combination rules, the out-of-fold plumbing and the decision
engine. It does not select members by regime yet, and it does not train on the
real store.

The roadmap file names are `ensemble/stacking.py`, `confidence/calibration.py`
and `decision/engine.py`. This slice keeps the core in
`scalping.ensemble.core` and the decision rules in `scalping.ensemble.decision`;
the `confidence/` package (temperature scaling, drift) is a later slice. The
public names match the roadmap's intent.

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

- **Regime-dependent selection.** Regime tables exist in
  `scalping.regime.strategy`, but members are not chosen or weighted by regime
  yet. A regime-conditional ensemble needs its own leakage-safe OOF split.
- **Temperature scaling.** No post-hoc calibration of member or ensemble
  probabilities. ECE/MCE are reported, not optimised.
- **Drift and OOD scoring.** No feature/prediction drift monitor, no
  out-of-distribution score, no confidence bands beyond disagreement.
- **Persistence and API.** Decisions are not written to a store and not served.
  The decision log schema exists as `Decision.to_dict()`; storage is Phase 10.
- **Cost-aware veto inside the engine.** Costs and the simulator gate live
  outside this module.
- **Member inference fits.** `fit_oof` produces OOF evidence; fitting members on
  the full training split for live inference is the caller's step.

## Honest note

An ensemble is not a trading edge. It can be better calibrated, less noisy or
better than its best member on prediction metrics, and still lose money after
spread, slippage and the cost stress. The only claim this slice supports is a
prediction comparison on fixed rows. Promotion requires the simulator, a cost
stress and regime coverage; until those pass, the simpler model stays.
