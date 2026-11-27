# Classical machine learning

First Phase 5 slice. The goal is an honest baseline, not a leaderboard: a model
must beat a majority/prior baseline on validation before it is worth a
simulator run, and it is never promoted from this page alone.

## Implemented models

| Name | Task | Estimator | Notes |
|---|---|---|---|
| `logistic` | direction | `LogisticRegression` | balanced class weights, multinomial |
| `random_forest` | direction | `RandomForestClassifier` | 300 trees, depth 10, leaf 20 |
| `extra_trees` | direction | `ExtraTreesClassifier` | 300 trees, depth 12, leaf 20 |
| `hist_gradient` | direction | `HistGradientBoostingClassifier` | 300 iterations, depth 6 |
| `sgd` | direction | `SGDClassifier` | log loss, balanced weights, no shuffle, `partial_fit` |
| `lightgbm` | direction | `LGBMClassifier` | 500 trees, 31 leaves, balanced weights |
| `xgboost` | direction | `XGBClassifier` | 500 trees, depth 6, histogram, label-encoded wrapper |
| `isolation_forest` | anomaly | `IsolationForest` | unsupervised outlier score, **never** a classifier |

Dependencies: scikit-learn, LightGBM and XGBoost. On macOS both boosting
libraries need the OpenMP runtime (`brew install libomp`); the imports are lazy,
so the rest of the CLI and simulator keep working without it. CI's Linux wheels
ship OpenMP.

An SVM is still **not implemented**; the roadmap keeps it in Phase 5 rather than
pretending the list is complete. SGD and isolation forest are implemented below,
with their own semantics.

## Task types: direction versus anomaly

`scalping.ml.models` exports two disjoint registries and one task map:

- `MODEL_NAMES` — the direction classifiers above. These are the names
  `POST /api/ml/train` and `ml train` accept, they expose `predict_proba` over
  the dataset's class labels, and they are the only models eligible to become a
  promotion candidate.
- `ANOMALY_MODEL_NAMES` — `("isolation_forest",)`. It has no classes, no
  direction probability and no promotion path.
- `ALL_MODEL_NAMES` = both registries; `MODEL_TASKS` maps every name to
  `"direction"` or `"anomaly"`.
- `model_task(name)`, `is_promotion_candidate(name)` and
  `build_anomaly_model()` make the distinction explicit in code.

`build_model("isolation_forest")` raises with a pointer to the anomaly module,
and `train_classifier(model="isolation_forest")` raises before fitting. That is
deliberate: an anomaly score is a statement about how unusual a row is, not a
probability that the next bar rises. Forcing it into `predict_proba` would
manufacture a direction signal the model never learned. `is_promotion_candidate`
returns `False` for every anomaly name, and every anomaly report carries
`promotion_eligible: false`.


## Data contract

`train_classifier` calls the same `datasets.builder.build_dataset` as every
other experiment:

- chronological train/validation/test;
- purge at least the label horizon, plus configurable embargo;
- `split == -1` rows are visible for audit and never fitted;
- the dataset id, feature columns and spec are stored with the model;
- the test split is untouched unless `evaluate_test=True` is explicit.

The first task is binary direction over `horizon` bars: `{-1, +1}`. Exact-zero
returns are dropped by default because on H1 they are about one row in four
thousand; keeping that third class inflates accuracy while the model never
predicts it. The number of dropped rows is recorded as `flat_dropped`.
`--include-flat` restores the old behaviour for comparison.

Labels use future closes only as targets; inputs come from the causal feature
matrix.

## Metrics

Validation reports:

- accuracy and balanced accuracy;
- multiclass log loss and Brier score;
- reliability bins, expected calibration error (ECE) and a guarded maximum
  calibration error (MCE ignores bins with fewer than 20 rows);
- per-class precision/recall/F1/support;
- confusion matrix;
- rows in each split and purged rows;
- baseline: majority-class accuracy and train-prior log loss.

Accuracy alone is not accepted. A model that beats the baseline on log loss but
loses money after costs is still rejected.

## Temperature scaling (optional)

`ml/calibration.py` can fit one multiclass temperature `T > 0` so that
`p_T = softmax(log(p) / T)`. For `p = softmax(z)` this is exactly
`softmax(z / T)`: `T > 1` softens an overconfident model, `T < 1` sharpens an
underconfident one, and `T = 1` is the identity. The fit minimises multiclass
log loss with a deterministic grid over `log T` plus a golden-section
refinement; it uses no scipy and no random seed. The search range is
`T ∈ [0.05, 20]`, and a flat objective keeps `T = 1` instead of drifting to a
grid edge.

```bash
uv run python -m scalping.cli ml train ... --calibrate temperature
```

The fit runs on **validation rows only**. The test split is never used to fit
and is not re-scored with the temperature, so `test_metrics` stays the raw
model output. The report gains a `calibration` object:

```json
{
  "method": "temperature",
  "temperature": 1.42,
  "fitted_on": "validation",
  "rows": 4427,
  "improved": true,
  "before": {"log_loss": 0.694, "brier": 0.25, "ece": 0.043, "mce": 0.08},
  "after":  {"log_loss": 0.691, "brier": 0.249, "ece": 0.021, "mce": 0.05}
}
```

`before` and `after` are measured on the same validation rows that were used
to fit `T`, so `after` is optimistic: it is an in-sample view of the
calibration, not an out-of-sample validation of it. The honest check is to fit
on one window and report on a later untouched window, which this slice does not
do yet.

The temperature optimises log loss, so `after.log_loss <= before.log_loss` by
construction (the search includes `T = 1`). ECE, MCE and Brier usually improve
but are not guaranteed to: a single number cannot fix a model whose ranking is
wrong, and temperature scaling preserves the predicted class and the ranking of
class probabilities. It cannot add information.

Promotion evidence still reads the recorded probabilities and metrics. The
headline `validation` and `test` blocks keep their raw numbers, and the
`calibration` object is a separate diagnostic; a fitted temperature never
substitutes for walk-forward, cost-stress or OOS evidence.

## Precision-target rejection gate (optional)

Calibration fixes the probability scale; it does not decide when a prediction
is too weak to trade. `ml/rejection.py` selects a top-probability threshold on
validation rows: the lowest threshold whose accepted rows reach
`target_precision`, optionally with a `min_coverage` floor.

```python
from scalping.ml.rejection import select_precision_threshold

choice = select_precision_threshold(
    validation_probabilities, validation_labels,
    target_precision=0.60, min_coverage=0.02,
)
```

The sweep accepts whole tie groups, so rows sharing the boundary probability
are never split. If no threshold satisfies both floors, the result has
`satisfied=False` and `threshold=None`; the caller must keep rejecting rather
than silently trade a gate that failed its own validation. `rejection_summary`
reports precision and coverage for a fixed threshold, and `RejectionGate`
bundles the threshold with its validation profile.

This is a selection rule, not evidence: a threshold fitted on validation still
needs an untouched window, cost stress and a live/paper check.

`ml train --reject-precision 0.60 [--reject-min-coverage 0.02]` runs the gate
selection after fitting and records a `rejection` object in `metrics.json`:

```json
{
  "threshold": 0.58,
  "precision": 0.63,
  "coverage": 0.21,
  "accepted": 930,
  "total": 4427,
  "target_precision": 0.6,
  "min_coverage": 0.02,
  "satisfied": true
}
```

The threshold is selected on the raw validation probabilities, because the
saved model does not apply temperature scaling; `satisfied=false` means the
requested gate could not be met and must not be used silently. Training does not
apply the gate to its test metrics or to the saved model's `predict_proba`.

`ml evaluate --use-rejection-gate` reuses that saved threshold instead of
`--threshold`. It refuses a missing/unsatisfied gate and a fitted threshold
below 0.5 (the policy's directional floor); the evaluation result's `threshold`
field then shows the value actually used.

## Commands

```bash
cd backend
uv run python -m scalping.cli ml train \
  --root ../data-ticks5-store \
  --symbol XAUUSD --timeframe H1 \
  --model sgd --horizon 5 \
  --groups price,momentum,volatility \
  --purge 5 --embargo 5 \
  --incremental-chunks 8 \
  --out runs/ml
```

Optional final test read:

```bash
uv run python -m scalping.cli ml train ... --evaluate-test
```

Output is one JSON line with the dataset id, split counts, validation metrics,
baseline, optional test metrics and artifact paths.

## Incremental SGD

Two ways to fit `sgd` in chunks, both deterministic:

- `ml train --incremental-chunks N` with `N > 1`;
- the library helper `scalping.ml.incremental.incremental_fit(estimator, X, y,
  chunks=N, classes=...)`.

Chunking semantics are fixed and documented:

- chunks are **contiguous** slices in the caller's row order (chronological for
  a dataset bundle), never shuffled inside the helper; the first blocks absorb
  the division remainder, so `(rows, N)` fully determines the split;
- the class set is computed once from the full `y` and passed to every
  `partial_fit`, so a chunk that happens to contain one class cannot shrink the
  output columns;
- `chunks=1` is a normal batch `fit`; `chunks>1` requires `partial_fit`, and a
  model without it (any tree ensemble here) raises instead of silently falling
  back;
- `SGDClassifier.partial_fit` rejects `class_weight="balanced"`, so the helper
  computes balanced weights once from the full training labels, clears the
  parameter and passes them as `sample_weight` on each chunk. The weighting
  therefore does not depend on chunk boundaries.

Honest limitation: N chunk passes are **not** the same optimisation as one batch
`fit` with `max_iter=N`. A batch epoch visits every row in order; chunking visits
chunk 1, 2, … and only returns to chunk 1 on the next pass, so the two take a
different path through the same rows while the learning-rate schedule advances by
chunk size. What is guaranteed is reproducibility: the same seed, rows and chunk
count give bit-identical coefficients, and on the synthetic learnable test frame
validation accuracy matches the batch fit and log loss stays within 0.05 absolute.
On real data measure the gap with `ml cv` or a paired run before choosing; do not
assume it away.

`sgd` also fits the raw feature matrix: there is no scaler in this slice. On the
real H1 features its probabilities saturate near 0/1 and its calibration is
poor, so read it as a fast linear baseline rather than a calibrated probability
source. A standardiser would have to be fitted on training rows only and is left
for a later slice.

## Anomaly detection (not a classifier)

`scalping.ml.anomaly` fits an isolation forest on the **training** rows only and
scores later rows with `score = -model.score_samples(X)`, so **higher means more
anomalous**. There is no `predict_proba` and no class. The operational threshold
is the training-score quantile:

```
threshold = quantile(train_scores, spec.quantile)   # default 0.99
flagged   = score > threshold
```

Fit and scoring are separated: validation/test rows are never used to fit the
detector or to choose the threshold. `AnomalyScore` reports, per split, the row
count, threshold, anomaly rate, and the score mean/std/min/max (all finite by
construction). `AnomalyReport.to_dict()` adds the dataset id, columns, spec,
split counts and `promotion_eligible: false`.

The optional **separation** number exists only when labels are supplied. By
default the label is "large move": `|future return| >=` the same quantile of the
**training** absolute returns (`--magnitude-quantile`, default 0.9), computed on
train rows only. With both classes present the report includes a ROC AUC of the
anomaly score against that label; a single-class label vector reports
`auc: null` with a reason instead of a made-up number. Pass
`--no-magnitude-labels` (or `magnitude_quantile=None`) to omit it entirely.

```bash
cd backend
uv run python -m scalping.cli ml anomaly \
  --root ../data-ticks5-store \
  --symbol XAUUSD --timeframe H1 \
  --groups price,momentum,volatility \
  --quantile 0.99 --splits validation \
  --out runs/anomaly-report.json
```

`--splits` defaults to `validation`; `test` is available but, like every other
one-shot OOS read, is not spent by default.

Limitations, stated plainly:

- a high anomaly score is not a BUY/SELL signal, not a probability and not an
  edge; it describes unusualness relative to the training window;
- the isolation forest is fitted on feature rows, so regime or volatility shift
  changes what counts as normal; a score is only meaningful against the same
  training window;
- an AUC against large moves is a diagnostic, not a strategy backtest; it does
  not pay costs and is not walk-forward evidence;
- an anomaly run is **never** a promotion candidate (`is_promotion_candidate`
  and `promotion_eligible` both say so). Use it to generate a falsifiable
  hypothesis and test that hypothesis on a new window.

## Feature importances

`scalping.ml.models.feature_importances(estimator)` is the documented accessor.
It returns a 1-D, finite, non-negative copy aligned with the fitted feature
columns for `random_forest`, `extra_trees`, `lightgbm` and `xgboost`, and `None`
for models that do not expose `feature_importances_` (`logistic`, `sgd`,
`hist_gradient`) or for an unfitted estimator. The XGBoost wrapper delegates to
its fitted booster. No `|coef_|` substitute is invented for the linear models:
coefficient magnitudes are not comparable to split/impurity counts, and the
accessor would be lying about what it measures.

Importances describe where the fitted model spent its splits. They are not
causal effects, not out-of-sample evidence and not a reason to drop features
without testing the resulting model.


## Purged cross-validation

`ml cv` answers "is this model/spec stable across time?" without spending the
validation or test rows. It builds the same dataset as `ml train`, then keeps
only `split == 0` rows and cuts them into `n_splits + 1` contiguous blocks:

- fold `k` tests block `k + 1`;
- fold `k` fits on every earlier block (expanding window);
- the last `max(purge_bars, spec.purge_bars, horizon)` training rows are
  dropped, because their labels overlap the test block's future bars;
- `embargo_bars` extra rows are dropped immediately before the test block;
- a fold is rejected if the training window falls below `min_train_rows` or if
  either side is empty.

The gap is therefore `max(config.purge_bars, spec.purge_bars, horizon) +
embargo_bars` bars, the same purge rule the dataset builder uses at its own
boundaries plus any wider purge the caller set for `ml train`. Folds are
chronological, contiguous and non-overlapping. At least two folds are required;
one fold has no variance to report. Validation and test rows are never selected:
they stay untouched for the one-shot final decision, and CV cannot inflate a
score by reusing them.

```bash
cd backend
uv run python -m scalping.cli ml cv \
  --root ../data-ticks5-store \
  --symbol XAUUSD --timeframe H1 \
  --model logistic --horizon 5 \
  --groups price,momentum,volatility \
  --folds 5 --purge 5 --embargo 5 \
  --min-train-rows 50
```

`--embargo` defaults to 5, matching `ml train`; `--include-flat` and
`--min-train-rows` mirror the training knobs. Output is one JSON line: dataset
id, `spec`, feature `columns`, split counts, every fold's boundaries and metrics
(accuracy, balanced accuracy, log loss, Brier, ECE/MCE), and the mean/std of
each metric across folds. `split_counts` uses the same kept-row semantics as
`ml train`: train/validation/test exclude flat rows and `flat_dropped` is
reported separately. `--out` optionally writes the full report; the `out` path
itself only appears on stdout, so the file is a reproducible record. No model
artifact is produced because CV selects, it does not promote.

Honest limitations: there is no hyperparameter search yet, so CV compares a
fixed spec rather than tuning it. Folds share their early history, so they are
not independent samples; a small `min_train_rows` or few folds gives noisy
mean/std. The module does not run the simulator and does not read the test
split, so a good CV score is evidence about stability, not a trading result.

## Feature-group ablation

`ml ablation` answers one fixed question: for **this** dataset, model and spec,
how much does the validation score change when a feature group is removed or
kept alone? The 59 columns are partitioned into seven registered groups
(`price`, `momentum`, `volatility`, `volume`, `structure`, `session`,
`multi_timeframe`), and the protocol is:

- `all` — every selected group (the reference);
- `without:<group>` — leave one group out;
- `only:<group>` — that group alone.

That is `1 + 2 * 7 = 15` fits for the full registry; a smaller `--groups`
selection costs `1 + 2 * n`. The dataset is built once, so every run shares the
same rows, chronological split, purge/embargo, seed, hyperparameters and
flat-row policy. There is no hyperparameter search, so the comparison isolates
the columns rather than the tuning. `--groups` accepts the same comma-separated
names as `ml train`.

Each run reports accuracy, balanced accuracy, log loss, Brier score and ECE, plus
`delta_vs_all` for every metric. The sign is `run - all`: a negative accuracy
delta means the run scored worse than the full matrix. The `all` run's delta is
`null`. The report also carries the shared dataset id, spec, split counts, the
majority/prior baseline and a `protocol` block that states the scored split and
that the test split was not scored.

```bash
cd backend
uv run python -m scalping.cli ml ablation \
  --root ../data-ticks5-store \
  --symbol XAUUSD --timeframe H1 \
  --model logistic --horizon 5 \
  --groups price,momentum,volatility \
  --purge 5 --embargo 5 \
  --out runs/ablation/price-momentum-volatility.json
```

Output is one JSON line; `--out` writes the same full report without the `out`
path, so the file is a reproducible record. Identical inputs produce an
identical payload: the result carries no timestamp. The run is refused with a
clear error for an empty selection, an unknown group, a single-group selection
(leave-one-out would score an empty matrix) or a group that expands to no
columns.

Limits, stated plainly:

- **Dependence, not causality.** Removing a group removes its columns, not the
  information they encode. Correlated groups compensate for each other, so a
  near-zero delta does not mean the group is useless, and a large delta does not
  prove the group is the cause of the score.
- **Not a trading edge.** A delta is measured on validation classification
  metrics, before simulator costs, slippage and trade selection. A group that
  improves log loss can still lose money; `ml evaluate` is the next step.
- **One spec, one split.** The protocol does not tune, does not bootstrap and
  reports no significance test. A validation split is a few hundred rows: small
  deltas are noise, and re-running with another window is the honest check.
- **Test rows stay locked.** `evaluate_test` is ignored here on purpose; an
  ablation must not spend the one-shot split.

## Trading evaluation

A classifier is not a strategy until it pays the simulator's costs. The
evaluation policy is deliberately small:

- flat: BUY when `p_up >= threshold`, SELL when `p_down >= threshold`;
- long: CLOSE when `p_down >= exit_threshold`;
- short: CLOSE when `p_up >= exit_threshold`;
- the simulator still owns sizing, SL/TP, time stop and every risk block.

```bash
uv run python -m scalping.cli ml evaluate   --run runs/ml/ml-logistic-316312dff683   --root ../data-ticks5-store   --split validation --stress 2,3
```

This runs on one chronological split (validation by default), reports the same
net/PF/expectancy/drawdown/cost fields as every other strategy and a hold
baseline, and repeats the run at the requested cost multipliers. The test split
is available with `--split test`, but should be read once per candidate.

Limitations of this first slice: `--calibrate temperature` reports a fitted
temperature but it is not a promotion gate, and the threshold policy is not
trained end-to-end as a trading objective.

## Regime breakdown

The trading evaluation can also answer *where* the trades happened. It joins
the same closed trades the headline metrics were computed from to the regime
label that was current at each trade's **entry** bar, then reports one row per
regime:

| Field | Meaning |
|---|---|
| `regime` | the label at entry (rule engine names, or labels from a parquet) |
| `trades` | closed trades that entered in this regime |
| `win_rate` | share with `r_multiple > 0` |
| `expectancy_r` | mean R multiple |
| `total_r` | sum of R multiples |
| `pnl` | net PnL after costs |
| `profit_factor` | gross win / gross loss; `null` when a bucket has wins but no losses |

`unmatched_trades` counts entries that precede the first label (or carry a
missing/`NaN` label). Those trades are not dropped and not put in a fake
bucket: `sum(rows.trades) + unmatched_trades == headline trades` always holds,
and the code raises if it does not.

```bash
uv run python -m scalping.cli ml evaluate \
  --run runs/ml/ml-logistic-316312dff683 \
  --root ../data-ticks5-store \
  --regime                      # rule labels, computed causally
  # or: --regime-labels labels.parquet   # ts + regime columns
```

JSON output gains a `"regime"` object with `"rows"` and `"unmatched_trades"`;
the existing keys are unchanged, and without either flag `"regime"` is `{}`.

The join is a backward as-of on the entry timestamp: a label that starts after
the entry can never be attached, and an entry exactly on a label timestamp uses
that label. Rule labels are computed from the same sliced bars the policy
trades, using the existing causal rule engine; no label uses data after its
bar. The trade-off is that the rule windows warm up inside the split, so the
first bars default to `sideways`/`normal` rather than borrowing history from
before the split.

Limitation, stated plainly: these buckets are small (a validation split may
produce single-digit trades per regime), they describe one already-evaluated
run, and they are not a promotion gate. A good bucket can be luck; a bad one
can be one trade. Cost-stress runs stay headline-only so the table never mixes
two cost assumptions. Use the breakdown to generate a hypothesis to test on a
new window, not to select the run that looks best.

## HTTP API

The same training and evaluation is available over HTTP so a panel can run a
real experiment and read its metrics without a shell. The endpoints live under
`/api/ml` and are thin wrappers around `scalping.api.ml_runner.MLRunner`; no
request field names a path.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/ml/models` | Direction model names `POST /train` accepts, from `scalping.ml.models.MODEL_NAMES`; anomaly detection is CLI/library only |
| `POST` | `/api/ml/train` | Validate, prepare and start a run; `202 {run_id}` |
| `GET` | `/api/ml/runs` | `{run_id, status, created_at, model, symbol, timeframe, dataset_id}` rows, newest first |
| `GET` | `/api/ml/runs/{run_id}` | Status, config, validation/test prediction metrics, calibration, split counts, columns, spec, dataset id |
| `POST` | `/api/ml/runs/{run_id}/control` | `{"action": "cancel"}` for a live run; `200` or `409` |
| `POST` | `/api/ml/evaluate` | Trade a finished run on one split, optional cost multipliers and regime table |

```bash
curl -s -X POST localhost:8000/api/ml/train -H 'content-type: application/json' -d '{
  "symbol": "XAUUSD", "timeframe": "H1", "model": "logistic",
  "horizon": 5, "groups": ["price", "momentum", "volatility"],
  "purge_bars": 5, "embargo_bars": 5,
  "split": "validation", "stress": [2, 3], "regime": true
}'
```

`/train` performs validation, the store read, the feature build and the dataset
geometry check **before** the `202`, so a bad field is `422`, an unusable store
is `409`, and only a reserved run reaches `runs/ml/<run_id>/`. The response is
an opaque alphanumeric token; it is never a path fragment and the routes reject
anything that is not `^[A-Za-z0-9]{1,64}$`.

`/evaluate` reads the symbol, timeframe and spec from the run's own artefact and
refuses to trade when the stored data no longer reproduces the run's
`dataset_id`. It returns the headline metrics, the hold baseline, one entry per
stress multiplier (`x2`, `x3`, …) and, when `regime` is true, the descriptive
`{rows, unmatched_trades}` table. There is no path field, so an evaluation
cannot be pointed at another instrument or directory.

### Run lifecycle and the one-job rule

```
preparing (request thread) -> running -> completed
                                      -> failed
                                      -> cancelling -> cancelled
```

One heavy ML job runs per process. `MLJobSlot` is shared by `/train` and
`/evaluate`; a second start while a fit or a stress pass is alive is `409
another ML job is running`. The slot is released in the worker's `finally`, so
success, failure and cancellation all free it. A second `409` is not a queue:
nothing is retried in the background.

`cancel` sets a cooperative flag and moves the visible status to `cancelling`.
The estimator fit and each simulator pass are single blocking calls that cannot
be interrupted mid-call; the worker observes the flag before the fit, immediately
after it, and between evaluation stages, and only then writes the terminal
`cancelled`. A cancel that arrives during a long fit therefore leaves the run
`cancelling` until the fit returns. That is the honest state, not a stuck job.

Limits: the API refuses a stored slice above 400 000 bars or a selection above
128 feature columns instead of materialising an unbounded float32 matrix. There
is no dataset cache; a run holds one feature frame while it works, but the
**feature matrix** is reused from `<runs_root>/.feature-cache` when the bars,
costs and registry match. See [feature-cache.md](feature-cache.md).

### What the API does not decide

The `validation`/`test` numbers are prediction metrics (accuracy, balanced
accuracy, log loss, Brier, ECE/MCE against the train prior). The `evaluate`
numbers are trading metrics after the simulator's costs. Neither is a promotion
decision: the API reports what happened on the chosen split and never writes a
strategy card, changes the registry champion or claims an edge. A good Brier
score with negative expectancy is still a rejection, and a positive regime
bucket can be one trade.

### Panel: `/labs/ml`

The web panel at `/labs/ml` is a client of these endpoints only. It reads the
symbol, timeframe and model lists from `/api/symbols`, `/api/timeframes` and
`/api/ml/models`; it never hard-codes a model name. A start posts `/train`, then
polls `/runs/{id}` every 2 s while the status is non-terminal (`running`, then
`cancelling`; `preparing` is polled defensively) and stops on the terminal
status; the timer is cleared on unmount. A `409` is shown
as the one-job busy state, not retried. Cancel posts `/control` and the button
shows `cancelling` until the worker writes the terminal state.

Results show validation/test prediction metrics (with an explicit "not a
trading result" note), split counts, dataset id, feature-column count and the
spec summary. The evaluate panel sends the split, cost multipliers and regime
flag to `/evaluate` and renders the headline metrics against the hold baseline,
one row per stress multiplier and the per-regime table — or an explicit empty
state. A missing test split, an empty run list and an unknown run (`404`) each
get their own message; the panel never fills a gap with a plausible number.

## Artifacts

The CLI writes:

```
runs/ml/ml-<model>-<dataset-suffix>/
├── model.joblib   # estimator + classes + selected columns
└── metrics.json   # dataset id, spec, metrics, baseline, split counts, calibration
```

An API run owns a directory named by its opaque token and additionally keeps
the status and report sidecars:

```
runs/ml/<run_id>/
├── meta.json      # status timeline, config, dataset id, error
├── report.json    # validation/test metrics, calibration, evaluation
├── model.joblib
└── metrics.json
```

Both sidecars are written atomically; `meta.json` is written last on success so
a reader that sees `completed` also sees the report.

`load_model(path)` returns `(model, columns)`. A future SQLite registry
(Phase 10) will index these directories, add champion/rollback and drift.

## First real run (not promoted)

XAUUSD H1, horizon 5, all 59 features, 20,654 train / 4,427 validation /
4,408 test rows, purge 5 + embargo 5, flat rows dropped:

| Model | Val accuracy | Val balanced | Val log loss | Test accuracy | Test log loss |
|---|---|---|---|---|---|
| Majority/prior baseline | 54.3% | — | **0.6919** | — | — |
| Logistic | 51.6% | 51.8% | 0.6921 | 49.3% | 0.6961 |
| Random forest | 50.5% | 50.8% | 0.6937 | 50.6% | 0.6985 |
| HistGradientBoosting | 51.5%* | 51.5%* | 0.6972* | — | — |
| LightGBM | 51.9% | 52.0% | 0.7073 | 51.0% | 0.7421 |
| XGBoost | 51.8% | 51.9% | 0.7154 | 50.7% | 0.7523 |

Logistic validation ECE is **0.043** (bins concentrated around 0.47–0.53); the
guarded MCE is small once one-row bins are excluded. The model is reasonably
calibrated but still uninformative: calibration does not create edge.

\* hist-graded numbers are from the earlier 3-class run; the binary rerun is
pending. **No model, including LightGBM and XGBoost, beats the prior baseline;
none is promoted.** This is a useful
negative result: the 59 causal features do not predict 5-bar direction on H1
with these baselines. The next attempt must add evidence (flow/macro/regime
interactions, different horizon, or a different task) rather than blindly adding
capacity.

### Trading evaluation of the same logistic model

Validation split, threshold 0.55, simulator costs included:

| Cost | Trades | Net | PF | Expectancy | Max DD | Costs |
|---|---|---|---|---|---|---|
| ×1 | 407 | **−$2,002** | 0.83 | −0.131 R | 22.3% | $1,503 |
| ×2 | 226 | −$2,002 | 0.71 | −0.231 R | 20.9% | $1,731 |
| ×3 | 185 | −$2,006 | 0.65 | −0.274 R | 20.5% | $2,114 |

Holding was flat (net $0). The classifier is worse than doing nothing after
costs; the trading evaluation exists to catch exactly this, which accuracy and
log loss alone did not.

## What this proves

- The pipeline can train a genuine classical classifier on leakage-safe,
  time-ordered data.
- Validation metrics are reproducible and compared with a simple baseline.
- Artifacts are versioned by dataset id and spec.
- On the first real H1 run the baselines lose to the prior; the system reports
  that instead of hiding it.

## What it does not prove

- No out-of-sample trading result yet: the simulator/cost-stress evaluation is
  the next Phase 5 task.
- No hyperparameter search (Phase 16 of the brief): purged CV compares a fixed
  spec, it does not tune it, and it never reads the test split.
- No promotion. A model in `runs/ml/` is an experiment, not the live strategy.
- Incremental SGD is reproducible but not proven equivalent to the batch fit on
  real data; the gap must be measured per dataset.
- Anomaly scores and their AUC against large moves are diagnostics. No anomaly
  run has been turned into a validated trading policy, and none may be promoted.
- No out-of-sample calibration check: the temperature is fitted and reported
  on the same validation rows. Calibration is not an edge and cannot repair a
  model that has none.
