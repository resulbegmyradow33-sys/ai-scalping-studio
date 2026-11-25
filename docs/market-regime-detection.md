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


### One-run regime report

`regime_metrics(trades, labels)` is the single-run counterpart of the strategy
table: it groups one backtest's closed trades by the regime at entry and reports
`trades`, `wins`, `win_rate`, `expectancy_r`, `total_r`, `total_pnl`,
`profit_factor` and `avg_hold_minutes`. `profit_factor` is `null` for a bucket
with wins but no losses; trades before the first label stay in `unknown`.

Use it to answer "how did *this* run behave in each regime", then compare the
strategy table across runs. Both are descriptive and trade-count-aware; neither
promotes a strategy.

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

## 9. Strategy Intelligence page

`/strategies` answers a different question from `/intelligence`: *what is the
system running, and how is this slice distributed across regimes?* It reads
three existing read-only endpoints and adds no derived numbers.

| Section | Source | What it shows |
|---|---|---|
| Regime distribution | `GET /api/regime` | `counts` as a captioned table (`regime`, `bars`, `share`); every value is the response field, the percent is `share * 100` |
| Panel card | `GET /api/strategy/card` | `active: true` → `legacy` badge for `flow_momentum`; `active: false` → `fallback` badge for the legacy M1 momentum |
| DSL champions | `GET /api/autonomy/champions` | `readable: false` → "unreadable", never "no champions"; readable and empty → "no champion", which means NO TRADE |
| Strategy × regime | not exposed | an explicit gap notice with the reason |

The strategy × regime table is **not on the wire**. `strategy_regime_table`
in `regime/strategy.py` needs the closed `Trade` rows of a simulator pass plus
the regime labels; trade rows exist only inside an evaluation, not as a stored
read-only resource. The page therefore does not join or aggregate anything in
the browser. For a finished ML run, `POST /api/ml/evaluate` returns a
per-regime breakdown under `regime.rows` (`trades`, `win_rate`, `expectancy_r`,
`total_r`, `pnl`, `profit_factor`) plus `unmatched_trades`; that is one run's
evaluation, not a card × regime table.

Honesty rules applied on the page:

- an unreadable champion archive is not rendered as an empty list;
- `mode` on a champion row is displayed verbatim and the page notes that the
  backend defaults it to `dsl` when the archived payload names no mode;
- an empty slice (`rows: 0`) shows the import hint and keeps the card
  classification, which does not depend on bars;
- a failed request renders the error detail and never falls through to the
  empty state;
- the page has one `role="status"` live region for the settled regime summary;
  loading is not announced.

Adding a read-only `strategy × regime` endpoint later requires a stored trade
source (a paper session or a completed evaluation artefact) and the same
backward as-of join that `trades_with_regime` already implements. Until then
the page states the absence.

## 10. Limits

- Regime labels are descriptive, not predictive. Do not report "the regime
  predicts the next move"; the correct claim is "strategy X performed like this
  inside regime Y on this dataset".
- The volatility axis is relative to the symbol's own recent history, so "high"
  does not compare gold with an index.
- Unsupervised states have no inherent meaning beyond their ordering by mean
  return; a narrative name ("accumulation", "distribution") must be backed by
  the per-state statistics, not assigned by hand.
- `/api/regime`, `/intelligence` and `/strategies` reuse the row-aligned
  contract; the transition/dwell tables and the strategy-by-regime report stay
  research-side until an endpoint exposes a stored trade source.
