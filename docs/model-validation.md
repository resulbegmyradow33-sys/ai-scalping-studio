# Model validation methodology

This document defines how the project decides whether a model is any good. It
applies to classical ML, neural networks and RL alike. The default is
skepticism: a positive backtest is not proof of future profit.

## 1. Chronology, not random splits

Financial rows are dependent. A random train/test split leaks near-duplicate
bars and future information. Every dataset uses chronological cuts:

```
[train][purge/embargo][validation][purge/embargo][test]
```

`scalping/datasets/builder.py` builds those cuts from a causal
`FeatureFrame`:

- `horizon`: how many future bars the label uses;
- `purge_bars`: at least the horizon, because two labels whose windows overlap
  share future bars;
- `embargo_bars`: extra separation after the purge;
- `train_fraction`, `validation_fraction`: time-ordered fractions;
- `split == -1` marks purged rows so reports can show the cost.

Labels use future closes **only as targets**. Inputs come from
`build_features`, which is causal and tested for it.

## 2. Labels

`build_dataset` produces:

- `y_direction`: sign of the future return, `{-1, 0, +1}`;
- `y_return`: simple future return over the horizon;
- `ts`: decision timestamps;
- `split`: train/validation/test/purged.

Trade-related labels must be priced with the same spread, commission and
slippage assumptions as the simulator. Classification accuracy alone is not a
trading metric.

## 3. Baselines

No model is evaluated in isolation. The minimum comparison set:

1. majority class / zero return;
2. `flow_momentum` rule card;
3. buy-and-hold or hold-flat where relevant;
4. random actions with the same trade frequency;
5. the previous champion model.

A neural model that beats only the majority class has not earned a seat.

## 4. Metrics

Prediction quality:

- log loss, Brier score, calibration curve;
- precision/recall by class, especially the entry class;
- MAE/RMSE for return regression;
- confidence coverage: accuracy in the top-confidence bucket.

Trading quality, always through the paper simulator:

- net P&L, profit factor, expectancy in R, win rate;
- maximum drawdown, Sharpe/Sortino where the sample supports it;
- trade count and average holding time;
- total costs and cost share;
- performance by regime and by timeframe.

Report sample size and confidence intervals. A 24-trade forward test is
positive evidence, not proof.

## 5. Cost stress

Every candidate is re-run with:

- spread ×2 and ×3;
- commission ×2;
- slippage ×2;
- a wider-spread frame (adjust ask/bid, not just the floor).

A strategy that only survives the cheapest assumptions is not deployable.

## 6. Walk-forward and purged CV

- `backtest/evaluate.py` already runs chronological walk-forward folds and
  reports train/test expectancy trends and an overfit warning.
- Phase 5 adds purged time-series CV with embargo for classical ML.
- Phase 11 adds Monte Carlo trade-sequence analysis, parameter sensitivity and
  regime-conditional reports.

The final test split is touched **once** per candidate. Hyperparameters are
selected on train/validation, never on test.

## 7. Calibration and uncertainty

Raw softmax is not confidence. Before a probability is used to size or veto a
trade it must be calibrated (reliability curve, Brier, log loss, temperature
scaling) and compared with ensemble disagreement. Low confidence maps to
`NO TRADE`, which is a valid decision.

## 8. Promotion gate

A candidate replaces the champion only when all hold:

1. implementable and reproducible (data checksum + config + seed);
2. trained on a documented dataset version;
3. positive out-of-sample expectancy after real costs;
4. survives cost stress;
5. beats the current champion on the same windows;
6. has enough trades and regime coverage;
7. passes leakage and reproducibility tests;
8. can be rolled back.

A failed candidate is kept in the experiment log; deletion is not a substitute
for evidence.

## 9. Leakage checks

Existing patterns to reuse:

- a truncated frame must produce bit-identical features for the rows it still
  contains;
- no `shift(-n)` in feature code;
- purge/embargo boundaries must be respected by tests;
- labels for row `i` may use `close[i+h]`, features may not;
- scalers/normalisers are fitted on train only;
- stacking uses out-of-fold predictions.

## 10. Current state

Implemented:

- causal features (59 columns), feature groups, chronological dataset builder
  with purge/embargo and manifest checksums;
- walk-forward evaluation, cost stress, forward test on unseen 2026 data;
- rule card promotion gate with a holdout.

Not implemented yet:

- classical ML cross-validation and calibration;
- neural-network training/evaluation;
- ensemble stacking;
- SQLite experiment registry and rollback;
- Monte Carlo and parameter-sensitivity reports.

## 11. Commands

```bash
cd backend
uv run python -m scalping.cli dataset build \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --groups price,momentum,volatility --horizon 5 \
  --purge 5 --embargo 5 \
  --out ../runs/dataset_h1_h5.json
```

The manifest records dataset id, columns, split ranges, purge/embargo,
checksum and creation time. Models must record the dataset id they trained on.
