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

## 5. Limits

- Rule labels are descriptive, not predictive. Do not report "the regime
  predicts the next move"; the correct claim is "strategy X performed like this
  inside regime Y on this dataset".
- The volatility axis is relative to the symbol's own recent history, so "high"
  does not compare gold with an index.
- Statistical models (K-Means, GMM, HMM and a supervised baseline) and the
  strategy-by-regime report are the next slices of Phase 4; they will reuse the
  same row-aligned contract and the same leakage test.
