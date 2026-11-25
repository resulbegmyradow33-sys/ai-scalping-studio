# Model comparison — XAUUSD H1 — 2026-11-25

**Status: descriptive comparison, negative result. NOT validated, NOT profitable, NOT promoted.**
**No champion change, no discovery change, no test tuning.**

Question: on one fixed chronological XAUUSD H1 split, do logistic regression,
LightGBM, or a small MLP beat the `ret_5` rule and the train-prior baseline on
prediction metrics, and do any of them survive simulator costs?

The machine-readable record is
[`docs/evidence/model-comparison-2026-11-25.json`](evidence/model-comparison-2026-11-25.json).

## Answer

**No.** No arm beat the train prior on accuracy or log loss, the `ret_5` rule
did not beat it either, and every arm lost money at base cost and at 2× and 3×
cost stress. The best-calibrated model still lost. This is a negative result,
not a ranking to promote from.

## Inputs

| Item | Value |
|---|---|
| Branch / base | `feat/modelcompare` / `06c853c` |
| Source store | `.worktrees/v1/data-ticks5-store` (read-only; no cache or dataset written) |
| Store checksum / rows | `0930bedd3c64554621c599ad6ee20214` / 29,586 |
| Span | 2021-05-24 00:00 → 2026-05-22 20:00 UTC |
| Synthetic | no (`source: ticks:data-ticks5`) |
| Dataset id / checksum | `XAUUSD:H1:h5:316312dff683` / `316312dff68323f14136c9a86db76767` |
| Features | 59 columns, 29,521 dataset rows |
| Train / validation / test | 20,654 / 4,427 / 4,408 (flat labels dropped: 12; purged: 20) |
| Code version / settings digest | `05c600129db13136` / `2226685b488a17e799a58f8d7e1da7db74be94352663575c0447516c0951c5ed` |
| Runtime | Python 3.12.13, scikit-learn 1.9.1, LightGBM 4.7.0, libomp present, `OMP_NUM_THREADS=1` |

The dataset id is the same `316312dff683` recorded in
[`classical-ml.md`](classical-ml.md); the logistic and LightGBM numbers below
reproduce that page exactly, which confirms the same rows, features and split.

## Fixed protocol

Everything was fixed before the test split was read. There is no hyperparameter
search, no threshold search and no re-run after seeing test.

| Knob | Value |
|---|---|
| Task | direction over 5 bars, labels `{-1, +1}` |
| Feature groups | all seven (59 columns) |
| Train / validation / test | 70% / 15% / 15%, chronological |
| Purge / embargo | 5 / 5 bars |
| Flat rows | dropped (exact-zero direction labels) |
| Seed | 0 |
| Entry / exit threshold | 0.55 / 0.55 |
| Simulator window | 60 bars |
| Cost stress | base, 2×, 3× (spread, slippage, commission) |
| Initial balance / risk | $10,000; 0.5% per trade; 3% daily loss gate |
| Episode termination | equity ≤ 80% of initial, i.e. −$2,000 |
| Test split | read once for all four pre-registered arms in the same run |
| Tuning | none |
| Promotion | none; this file is not promotion evidence |

Arms: `logistic` (balanced multinomial, repo default), `lightgbm` (repo default
500 trees), `mlp` (one 32-unit hidden layer, `StandardScaler` fitted on train
only, 60 epochs), and `ret_5` (sign of the 5-bar return, long-only, the
discovery baseline).

## Prediction metrics

Prior = always predict the train majority class; log loss uses the train class
priors (`−1: 0.49177`, `+1: 0.50823`). The rule has no probabilities, so log
loss, Brier and ECE are undefined for it and are left blank rather than filled
with a fake hard-label score.

### Validation (4,427 rows)

| Arm | Accuracy | Balanced | Log loss | Brier | ECE | MCE |
|---|---|---|---|---|---|---|
| **Prior baseline** | **0.5428** | — | **0.691873** | — | — | — |
| logistic | 0.5162 | 0.5179 | 0.692120 | 0.4990 | 0.0427 | 0.0582 |
| lightgbm | 0.5193 | 0.5204 | 0.707306 | 0.5124 | 0.0684 | 0.2605 |
| mlp | 0.4990 | 0.5012 | 0.726882 | 0.5298 | 0.1038 | 0.3982 |
| ret_5 rule | 0.5055 | 0.5018 | — | — | — | — |

### Test (4,408 rows, one read, no tuning)

| Arm | Accuracy | Balanced | Log loss | Brier | ECE | MCE |
|---|---|---|---|---|---|---|
| **Prior baseline** | **0.5461** | — | **0.691766** | — | — | — |
| logistic | 0.4927 | 0.4880 | 0.696115 | 0.5029 | 0.0412 | 0.0839 |
| lightgbm | 0.5100 | 0.5118 | 0.742050 | 0.5386 | 0.1077 | 0.6437 |
| mlp | 0.5023 | 0.5108 | 0.793372 | 0.5627 | 0.1477 | 0.4851 |
| ret_5 rule | 0.4982 | 0.4939 | — | — | — | — |

Deltas against the prior:

- logistic log loss: **+0.000247** validation, **+0.004349** test (worse);
  accuracy **−2.67 pp** and **−5.33 pp**;
- lightgbm: **+0.015433** / **+0.050284** log loss, accuracy **−2.35 pp** /
  **−3.61 pp**;
- mlp: **+0.035008** / **+0.101606** log loss, accuracy **−4.38 pp** /
  **−4.38 pp**;
- ret_5: accuracy **−3.73 pp** / **−4.79 pp**.

The validation ranks do not survive on test: logistic falls from 51.6% to
49.3%, LightGBM from 51.9% to 51.0%, and `ret_5` from 50.6% to 49.8%. None of
these differences is a stable signal; all are below the prior on both splits.

## Trading metrics

The classifier policy enters when `p_up ≥ 0.55` (or `p_down ≥ 0.55`), closes on
the opposite probability at 0.55, and the simulator owns sizing, stops, targets,
time stop, daily-loss gate and the equity floor. `ret_5` is long-only; it is
pushed through the same `evaluate_model` path with a degenerate probability
mapping, so its trading result uses the same engine and cost code.

A net near **−$2,000** means the episode hit the 80%-of-initial termination
floor. Those nets are truncated at the floor, not the policy's unbounded path
loss; trade count, profit factor and expectancy are the comparable fields.

### Validation

| Arm | Cost | Trades | Net | PF | Expectancy | Max DD | Costs |
|---|---|---|---|---|---|---|---|
| hold | ×1 | 0 | $0.00 | 0.000 | 0.0000 R | 0.00% | $0.00 |
| logistic | ×1 | 407 | −$2,002.26 | 0.8272 | −0.1313 R | 22.29% | $1,503.20 |
| logistic | ×2 | 226 | −$2,002.47 | 0.7067 | −0.2310 R | 20.93% | $1,731.20 |
| logistic | ×3 | 185 | −$2,005.83 | 0.6472 | −0.2745 R | 20.49% | $2,113.69 |
| lightgbm | ×1 | 1,075 | −$2,018.69 | 0.9070 | −0.0430 R | 21.97% | $4,591.79 |
| lightgbm | ×2 | 194 | −$2,012.32 | 0.5884 | −0.2378 R | 20.98% | $1,949.19 |
| lightgbm | ×3 | 149 | −$2,026.75 | 0.4812 | −0.3106 R | 21.10% | $2,131.84 |
| mlp | ×1 | 321 | −$2,034.84 | 0.7067 | −0.1472 R | 22.23% | $1,677.57 |
| mlp | ×2 | 165 | −$2,011.42 | 0.4934 | −0.2892 R | 20.46% | $1,594.50 |
| mlp | ×3 | 142 | −$2,005.14 | 0.4445 | −0.3384 R | 20.10% | $2,005.93 |
| ret_5 | ×1 | 245 | −$2,028.19 | 0.6609 | −0.1954 R | 20.47% | $1,203.71 |
| ret_5 | ×2 | 162 | −$2,038.83 | 0.5297 | −0.3082 R | 20.55% | $1,487.77 |
| ret_5 | ×3 | 137 | −$2,018.36 | 0.4720 | −0.3552 R | 20.32% | $1,785.28 |

### Test

| Arm | Cost | Trades | Net | PF | Expectancy | Max DD | Costs |
|---|---|---|---|---|---|---|---|
| hold | ×1 | 0 | $0.00 | 0.000 | 0.0000 R | 0.00% | $0.00 |
| logistic | ×1 | 296 | −$1,751.65 | 0.7718 | −0.1574 R | 18.14% | $808.19 |
| logistic | ×2 | 260 | −$2,015.19 | 0.7025 | −0.2189 R | 20.17% | $1,409.97 |
| logistic | ×3 | 139 | −$2,038.17 | 0.5168 | −0.3836 R | 20.38% | $1,329.82 |
| lightgbm | ×1 | 886 | −$2,013.76 | 0.8787 | −0.0607 R | 21.00% | $2,672.85 |
| lightgbm | ×2 | 194 | −$2,005.53 | 0.5663 | −0.2530 R | 20.75% | $1,476.12 |
| lightgbm | ×3 | 131 | −$2,017.12 | 0.4124 | −0.3888 R | 20.71% | $1,612.84 |
| mlp | ×1 | 561 | −$2,012.96 | 0.8158 | −0.0935 R | 23.12% | $1,994.99 |
| mlp | ×2 | 251 | −$2,003.89 | 0.6425 | −0.2022 R | 20.58% | $1,977.30 |
| mlp | ×3 | 205 | −$2,008.72 | 0.5647 | −0.2382 R | 20.58% | $2,492.06 |
| ret_5 | ×1 | 1,150 | −$651.14 | 0.9708 | −0.0139 R | 18.63% | $3,991.78 |
| ret_5 | ×2 | 556 | −$2,008.12 | 0.8263 | −0.0861 R | 20.24% | $3,903.71 |
| ret_5 | ×3 | 208 | −$2,018.56 | 0.6075 | −0.2372 R | 20.30% | $2,431.27 |

## Findings

1. **No prediction edge.** Every arm is below the train prior on accuracy and
   log loss, on both splits. The closest is logistic, still worse by
   +0.000247 nats on validation and +0.004349 on test.

2. **More capacity did not add signal; it hurt calibration.** Validation ECE
   rises from 4.27% (logistic) to 6.84% (LightGBM) to 10.38% (MLP); test ECE
   rises from 4.12% to 10.77% to 14.77%. LightGBM's test MCE is 0.644, so at
   least one large bin is badly overconfident.

3. **The rule is not a benchmark winner either.** `ret_5` scores 50.55% and
   49.82% accuracy, below the 54.28% / 54.61% prior, and loses money at every
   cost level.

4. **Accuracy and calibration are not edge.** Logistic has the best calibration
   and the smallest log-loss gap to the prior, and still loses −$2,002 on
   validation and −$1,752 on test at base cost (PF 0.83 / 0.77, negative
   expectancy, 22.3% / 18.1% drawdown). The best prediction row is a losing
   strategy.

5. **The least-bad trading result does not pass.** `ret_5` on test loses
   −$651.14 at base cost with PF 0.971 and −0.0139 R expectancy: still negative,
   and at 2× costs it loses −$2,008.12. No arm is positive at base or under
   stress on either split.

6. **The equity floor truncates losses.** Most losing runs terminate at
   −$2,000, which is why their nets cluster there. The differences that remain
   are trade count, PF and expectancy, not the capped net. `ret_5` on test is
   the one base run that did not reach the floor and still lost.

7. **Validation and test disagree.** The logistic validation accuracy (51.6%)
   is 2.3 pp above its test accuracy (49.3%); LightGBM is 0.9 pp above. A
   single-split rank is noise, and the test read did not change any decision.

**Negative results, recorded explicitly:** logistic, LightGBM, MLP and `ret_5`
are all rejected as promotion candidates. None beat the prior baseline or the
rule on prediction, and none survives costs. No candidate advances, no champion
changes, no discovery parameter changes.

## What this shows and does not show

**Shows:** the comparison ran on the real read-only store; the libomp/LightGBM
arm worked; the logistic and LightGBM prediction numbers reproduce
`classical-ml.md` exactly; every pre-registered arm fails both the prediction
baseline and the cost gate; the harness reports that instead of hiding it.

**Does not show:** that the features contain no signal at any horizon, that a
tuned model cannot do better, or that the market is inefficient. It tests one
horizon (5 bars), one feature set (59 columns), one split, one seed and fixed
hyperparameters. It is not walk-forward evidence, not a locked-holdout spend and
not a trading result with an independent sample.

## Limits

- One store, one symbol, one timeframe, one chronological split, one seed.
- No hyperparameter search, no walk-forward, no locked holdout, no bootstrap.
- The test split was read once for four arms in the same run; it was not used
  to select or tune anything, but four test reads is more than one candidate's
  one-shot read. Treat the test columns as descriptive.
- The MLP is a fixed 60-epoch budget and did not converge (scikit-learn
  `ConvergenceWarning`); it is a bounded baseline, not a tuned network.
- The rule is label-only; log loss, Brier and ECE do not exist for it.
- Thresholds 0.55/0.55 and the cost multipliers are fixed, not optimised.
- Losing runs are truncated at −$2,000 by the simulator's equity termination;
  net comparisons near that floor are not full path losses.
- Multiple comparisons were made without a significance correction; the
  consistent direction (all arms below prior, all arms negative) is the useful
  part, not any single delta.
- No regime breakdown was run.

## Reproduction

```bash
cd backend
OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run python scripts/model_comparison.py
```

The script rebuilds features without a cache, rebuilds the same dataset, fits
the fixed arms, scores validation and test once, runs the simulator at base/2×/3×
costs, and rewrites the evidence JSON. Runtime on this machine was 7.6 s with a
peak RSS of 433 MB.
