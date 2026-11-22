# Classical machine learning

First Phase 5 slice. The goal is an honest baseline, not a leaderboard: a model
must beat a majority/prior baseline on validation before it is worth a
simulator run, and it is never promoted from this page alone.

## Implemented models

| Name | Estimator | Notes |
|---|---|---|
| `logistic` | `LogisticRegression` | balanced class weights, multinomial |
| `random_forest` | `RandomForestClassifier` | 300 trees, depth 10, leaf 20 |
| `extra_trees` | `ExtraTreesClassifier` | 300 trees, depth 12, leaf 20 |
| `hist_gradient` | `HistGradientBoostingClassifier` | 300 iterations, depth 6 |

`scikit-learn` is the only new dependency. LightGBM, XGBoost, SGDClassifier,
isolation forest and SVM are **not implemented yet**; the roadmap keeps them in
Phase 5 rather than pretending the list is complete.

## Data contract

`train_classifier` calls the same `datasets.builder.build_dataset` as every
other experiment:

- chronological train/validation/test;
- purge at least the label horizon, plus configurable embargo;
- `split == -1` rows are visible for audit and never fitted;
- the dataset id, feature columns and spec are stored with the model;
- the test split is untouched unless `evaluate_test=True` is explicit.

The first task is 3-class direction over `horizon` bars: `{-1, 0, +1}`. Labels
use future closes only as targets; inputs come from the causal feature matrix.

## Metrics

Validation reports:

- accuracy and balanced accuracy;
- multiclass log loss and Brier score;
- per-class precision/recall/F1/support;
- confusion matrix;
- rows in each split and purged rows;
- baseline: majority-class accuracy and train-prior log loss.

Accuracy alone is not accepted. A model that beats the baseline on log loss but
loses money after costs is still rejected.

## Commands

```bash
cd backend
uv run python -m scalping.cli ml train \
  --root ../data-ticks5-store \
  --symbol XAUUSD --timeframe H1 \
  --model logistic --horizon 5 \
  --groups price,momentum,volatility \
  --purge 5 --embargo 5 \
  --out runs/ml
```

Optional final test read:

```bash
uv run python -m scalping.cli ml train ... --evaluate-test
```

Output is one JSON line with the dataset id, split counts, validation metrics,
baseline, optional test metrics and artifact paths.

## Artifacts

```
runs/ml/ml-<model>-<dataset-suffix>/
├── model.joblib   # estimator + classes + selected columns
└── metrics.json   # dataset id, spec, metrics, baseline, split counts
```

`load_model(path)` returns `(model, columns)`. A future SQLite registry
(Phase 10) will index these directories, add champion/rollback and drift.

## What this proves

- The pipeline can train a genuine classical classifier on leakage-safe,
  time-ordered data.
- Validation metrics are reproducible and compared with a simple baseline.
- Artifacts are versioned by dataset id and spec.

## What it does not prove

- No out-of-sample trading result yet: the simulator/cost-stress evaluation is
  the next Phase 5 task.
- No hyperparameter search (Phase 16 of the brief), no purged cross-validation,
  no reliability curve, no LightGBM/XGBoost.
- No promotion. A model in `runs/ml/` is an experiment, not the live strategy.
