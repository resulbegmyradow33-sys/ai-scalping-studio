# Neural-network research

Phase 6 adds sequence models (MLP, RNN/LSTM/GRU, CNN/TCN) on top of the
leakage-safe dataset builder. The goal is a **reusable trainer and an honest
evaluator**, not a claim that a neural network can predict gold. Every result
still has to pass the simulator and a cost stress before anyone calls it an
edge.

This document grows with the phase; the dataset contract below is the first
slice.

## 1. Why windows

A GRU/TCN does not see one feature row. It sees the last `window` rows as a
sequence, so it can learn path shapes (a slow drift, a volatility burst, a
range compression) that a single-row classifier cannot express.

The label stays the one Phase 3 built: `y_direction` at row `i` is the sign of
the return over the next `horizon` bars. The window **ends** at `i`; the target
is never shifted again inside this module.

## 2. Dataset contract

`scalping.nn.build_windowed_dataset(bundle, config)` takes a
`DatasetBundle` from `scalping.datasets.builder` and returns:

| Field | Shape | Meaning |
|---|---|---|
| `X_train` | `(n, window, features)` float32 | training windows |
| `y_train` | `(n,)` int64 | direction labels at the window end |
| `ts_train` | `(n,)` datetime | end timestamps (join key back to bars) |
| `X_validation`, `...` | same | validation split |
| `X_test`, `...` | same | test split, untouched by training |
| `mean`, `std` | `(features,)` float32 | normaliser fitted on training rows only |
| `dataset_id` | string | bundle id + window + clip |

Rules:

1. **Train-only normalisation.** Mean/std are computed from rows whose
   `split == train`, then applied to every split and clipped to `±clip`
   (default `5`). A wild test period cannot move the scale.
2. **No cross-split context.** A window is kept only if all `window` rows
   belong to the same split. The first `window - 1` rows of validation and test
   are therefore not predicted; that is deliberate and visible in `n`.
3. **Purged rows are never targets.** The builder's `split == -1` rows are
   excluded, and because a window must be entirely inside a split, a purged
   bar cannot sneak in as context either.
4. **Constant features** get `std = 1` (not a division by zero) and normalise
   to zero.
5. **Memory guard.** The tensor is materialised in RAM. `WindowConfig.max_bytes`
   (default 2 GiB) refuses an oversized build with a message pointing at a
   longer timeframe or the lazy loader planned for Phase 14.

Example sizes (`window=60`, 59 features, float32):

| Source | Rows | Approximate tensor |
|---|---|---|
| H1, 1 year | 8,760 | ~120 MB |
| M15, 1 year | 35,040 | ~0.5 GB |
| M1, 1 month | ~43,200 | ~0.6 GB |
| M1, 1 year | ~525,600 | ~7.4 GB → rejected |

## 3. Usage

```python
from scalping.datasets.builder import build_dataset
from scalping.features.build import build_features
from scalping.nn import WindowConfig, build_windowed_dataset

features = build_features(frame)                 # causal 59-column matrix
bundle = build_dataset(features, horizon=1, purge_bars=1)
data = build_windowed_dataset(bundle, WindowConfig(window=60))

data.counts()          # {"train": ..., "validation": ..., "test": ...}
data.X_train.shape     # (n_train, 60, 59)
```

## 4. Tests

`backend/tests/test_nn_data.py`:

- window `k` equals the normalised `X[end-window+1:end+1]` for its timestamp;
- the normaliser equals the training rows' statistics, and mutating test rows
  leaves `mean`, `std` and `X_train` byte-identical;
- every input row of every window belongs to the target's split (no purged
  context);
- constant features do not divide by zero;
- deterministic builds, memory guard, config validation, non-finite rejection.

## 5. Limits and next slices

- Windows are materialised; M1 history needs H1/M15, a smaller window or a lazy
  loader (Phase 14). Do not silently reduce `window` to make RAM fit — record
  the choice.
- The dataset alone says nothing about profitability. Model scores are
  prediction metrics; the simulator + cost stress decides.
- Next slices: model zoo (MLP/RNN/LSTM/GRU/CNN/TCN), trainer (MPS with CPU
  fallback, early stopping, checkpoints, cancellation, one-job policy) and the
  shared evaluator against Phase 5 and the rule.
