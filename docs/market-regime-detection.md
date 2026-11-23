# Market regime detection

The regime engine describes **what the recent path looked like**, not what the
next bar will do. It is an independent classifier: it does not read the trading
feature matrix, the strategy card or the RL agent, so a regime report cannot
inherit their assumptions. A regime label is never a signal by itself.

## 1. Contract

`scalping.regime.classify(frame, config=None)` takes canonical M1 (or derived)
bars and returns one row per bar, sorted by `ts`, aligned to the input:

| Column | Values | Meaning |
|---|---|---|
| `trend` | `strong_up`, `weak_up`, `sideways`, `weak_down`, `strong_down` | direction plus efficiency of the last `trend_window` closes |
| `trend_strength` | `strong`, `weak`, `none` | coarse version of the same decision |
| `efficiency_ratio` | `[0, 1]` | net move / sum of absolute bar moves |
| `volatility` | `low`, `normal`, `high` | ATR(14) against its own trailing median (480 bars) |
| `vol_ratio` | `>= 0` | the ATR ratio behind `volatility` |
| `expansion` | `contracting`, `stable`, `expanding` | ATR now versus ATR ten bars ago |
| `expansion_ratio` | `> 0` | the ratio behind `expansion` |
| `breakout` | `none`, `up`, `down` | first close outside the prior 20-bar range |
| `unusual` | `true`/`false` | latest log return is a `>= 4σ` outlier of its trailing 60-bar distribution |
| `return_z` | float | the z-score behind `unusual` |
| `regime` | `unusual`, `breakout_up`, `breakout_down`, `up_*`, `down_*`, `sideways_*` | one summary label for screens and tables |

`regime` precedence: `unusual` first, then a fresh breakout, otherwise
`<direction>_<volatility>` (`up_high`, `sideways_low`, ...). A trend that keeps
making new highs is therefore `up_*`; only the bar that leaves a range is
`breakout_up`.

## 2. Causality

Every axis uses only bars at or before the row's timestamp:

- trend: `close` compared with `close.shift(trend_window)` and a rolling sum of
  absolute diffs;
- volatility: ATR(14) and its trailing median, both backward looking;
- expansion: ATR against `ATR.shift(expansion_window)`;
- breakout: rolling high/low of the **previous** `breakout_window` bars, with
  the previous close required to be still inside that range;
- unusual: rolling mean/std of log returns over the trailing window.

The test suite mutates every bar after a cut and requires the classified prefix
to stay identical (`assert_frame_equal`), which is stricter than "the numbers
look plausible".

## 3. Defaults and scaling

`RegimeConfig` defaults are research choices for M1 bars:

| Parameter | Default | Note |
|---|---|---|
| `trend_window` | 60 | one hour on M1 |
| `trend_threshold` | 0.30 | below this the path is sideways |
| `strong_trend_threshold` | 0.50 | half the path is net movement |
| `atr_window` | 14 | bars |
| `vol_window` | 480 | eight hours on M1 |
| `vol_low_ratio` | 0.75 | ATR at or below 75% of its own median |
| `vol_high_ratio` | 1.50 | ATR at or above 150% of its own median |
| `expansion_window` | 10 | bars |
| `expansion_ratio` | 1.30 | expanding ATR |
| `contraction_ratio` | 0.77 | contracting ATR |
| `breakout_window` | 20 | range before the event |
| `breakout_atr_buffer` | 0.0 | optional extra ATR beyond the range |
| `return_window` | 60 | z-score window |
| `unusual_z` | 4.0 | outlier threshold |

Windows are counted in **bars**. On M15 or H1 the same defaults describe a much
longer horizon; pass a config with scaled windows when comparing timeframes.
Thresholds are not fitted to the forward store and must not be tuned on it.

## 4. Tests

`backend/tests/test_regime_rules.py`:

- a clean exponential trend labels `up`, a quiet sine range labels `sideways`;
- a sustained volatility jump labels `high` (the 480-bar baseline is long
  enough that the label does not normalise itself immediately);
- a one-bar range exit is a breakout event and does not repeat on every bar of
  the following drift;
- future bars cannot change earlier labels (truncated and mutated fixtures);
- classification is deterministic and independent of input row order;
- empty and short frames keep the column contract;
- impossible thresholds are rejected at construction.

## 5. Statistical models

`scalping.regime.fit_predict(frame, model, config)` trains on the first
`train_fraction` of the rows (default 70%) and labels every row:

| Model | Implementation | Output states |
|---|---|---|
| `kmeans` | scikit-learn `KMeans`, `n_init=10` | `state_0..state_{k-1}` |
| `gmm` | scikit-learn `GaussianMixture`, full covariance | `state_0..state_{k-1}` |
| `hmm` | hmmlearn `GaussianHMM`, diagonal covariance, Viterbi | `state_0..state_{k-1}` |
| `supervised` | `HistGradientBoostingClassifier` fitted to the rule `regime` label | the rule labels seen in training |

Contract:

- feature matrix from `regime/features.py` (eight causal columns: 1/5/15-bar
  returns, realised volatility, ATR/price, range/ATR, range efficiency, volume
  z-score);
- `StandardScaler` is fitted on the training slice only;
- unsupervised ids are canonicalised by the mean training `ret_1` of each
  cluster, so `state_0` is the lowest-return state regardless of initialisation;
- the feature frame is validated for nulls/NaN before it reaches a model;
- an unknown model name, a frame below `min_rows` (120) or a training slice
  smaller than `3 * n_states` is rejected.

The supervised baseline is a distillation check: it measures how much of the
rule structure the eight features can reproduce. It does not replace the rule
and its accuracy is not a trading result.

Dependencies: `scikit-learn` (BSD-3) and `hmmlearn` (BSD-3). Both install as
ARM64 wheels, add no service or credential, and the fixtures run in well under
a second; a 5-year M1 frame is a few hundred thousand rows and stays far below
the 1 GB Phase 4 budget.

## 6. Evaluation

`regime/evaluate.py`:

- `transition_matrix(labels)` — row-normalised state-to-state probabilities with
  every observed pair present (including zeros);
- `dwell_stats(labels)` — runs, bars, share, mean and max run length per state.

A label with a low mean run length flickers and should not drive a slow
strategy, even if its average forward return looks good. Transition and dwell
tables are the sanity check before any strategy-by-regime table is read.

## 7. Strategy × regime table

`regime/strategy.py` joins closed trades to the regime that was current at their
**entry bar** and summarises each strategy per regime:

```python
from scalping.regime import strategy_regime_table, trades_with_regime

joined = trades_with_regime(result.trades, regimes)      # per trade
table = strategy_regime_table(result.trades, regimes)    # per strategy × regime
```

- The join is a backward as-of on `entry_ts`: a label that starts after the
  entry can never be attached, and an entry exactly on a label timestamp uses
  that label.
- A trade that entered before the first label is reported as `unknown`, never
  dropped, so missing coverage is visible.
- Columns: `strategy`, `regime`, `trades`, `wins`, `win_rate`, `expectancy_r`,
  `total_r`, `total_pnl`, `profit_factor`.
- `profit_factor` is `null` when a bucket has wins but no losses; a finite
  value would invent a denominator.
- The input is the simulator's closed trades, so spread, commission and
  slippage are already in `pnl`/`r_multiple`. The table does not re-price.

How to read it: compare one strategy across regimes (which weather suits it),
not two strategies in different regimes with different sample sizes. A cell with
a handful of trades is noise; report `trades` next to every average. The table
is descriptive evidence, not a promotion gate.

## 8. API

`GET /api/regime?symbol=XAUUSD&timeframe=M1&limit=500`

- `rows` — every stored bar that was classified;
- `current` — the latest bar's axes (`trend`, `trend_strength`, `volatility`,
  `expansion`, `breakout`, `unusual`, `regime`);
- `counts` — bars and share per summary regime, largest first; the bars sum to
  `rows`;
- `series` — the last `limit` bars (`1..5000`, default 500) with `ts`, `regime`,
  `trend` and `volatility` for the timeline.

Only the timeline is capped: counts always cover the whole stored slice. An
empty store is a normal `200` with `rows: 0`, `current: null` and empty lists,
matching `/api/data/status`; it is not an error.

Rule-based labels are cheap enough to compute per request. Statistical models
are research tools and are not fitted inside a request; a fitted model belongs
in the run registry (Phase 10) before it can serve labels.

The panel page `/intelligence` reads this endpoint: current axes, the
full-slice distribution and a timeline of the last bars. It renders the API
values verbatim and states that a regime is not a forecast.

## 9. Limits

- Regime labels are descriptive, not predictive. Do not report "the regime
  predicts the next move"; the correct claim is "strategy X performed like this
  inside regime Y on this dataset".
- The volatility axis is relative to the symbol's own recent history, so "high"
  does not compare gold with an index.
- Unsupervised states have no inherent meaning beyond their ordering by mean
  return; a narrative name ("accumulation", "distribution") must be backed by
  the per-state statistics, not assigned by hand.
- The API endpoint and Market Intelligence page are the next slice of Phase 4;
  they reuse this row-aligned contract, the transition/dwell tables and the
  strategy-by-regime report.
