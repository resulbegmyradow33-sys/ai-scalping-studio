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

## 5. Model zoo

`scalping.nn.build_model(name, n_features=..., window=..., seed=...)` returns a
`torch.nn.Module` that maps `(batch, window, features)` to logits
`(batch, classes)`:

| Name | Family | Notes |
|---|---|---|
| `mlp` | flattened MLP | one hidden layer |
| `deep_mlp` | flattened MLP | four hidden layers + dropout |
| `residual_mlp` | residual MLP | two pre-norm residual blocks |
| `rnn` | vanilla RNN | last hidden state |
| `lstm` | LSTM | last hidden state |
| `gru` | GRU | last hidden state |
| `cnn` | 1D CNN | one conv layer, same length |
| `dilated_cnn` | dilated 1D CNN | dilations 1, 2, 4 |
| `cnn_lstm` | CNN + LSTM | conv front end, LSTM over steps |
| `tcn` | causal TCN | left-padded residual blocks, configurable kernel/levels |

Shared choices:

- `hidden` (default 64), `dropout` (default 0.1), `n_classes` (default 3) and
  `seed`; `build_model` seeds `torch` before construction so the same seed
  gives the same initial weights.
- The output is always **logits**; the trainer applies cross-entropy.
- The TCN's receptive field is
  `1 + 2 * (kernel - 1) * sum(dilations)`. With the default kernel 3 and levels
  1, 2, 4 it is 29 bars. Windows shorter than that are allowed; the early steps
  simply see padded zeros.
- Only the TCN promises **causality inside the window**: `forward_sequence`
  computes per-step states with left-only padding. The test feeds two inputs
  that differ from step 10 and requires steps 0–9 to stay identical. The other
  models may look both ways inside the window; the window itself ends at the
  prediction bar, so that is not look-ahead.

Determinism, shapes, the input contract, the receptive-field formula and the
TCN prefix test live in `backend/tests/test_nn_models.py`.

## 6. Trainer

`scalping.nn.train_classifier(data, config, checkpoint=..., control=..., resume_from=...)`
fits one model and writes a resumable checkpoint.

| Setting | Default | Notes |
|---|---|---|
| `model` | `gru` | any name from the model zoo |
| `epochs` | 20 | hard cap |
| `batch_size` | 64 | seeded permutation each epoch |
| `learning_rate` | 1e-3 | AdamW |
| `weight_decay` | 1e-4 | |
| `grad_clip` | 1.0 | `clip_grad_norm_` before every step |
| `patience` | 5 | early stop when validation loss stalls |
| `min_delta` | 1e-4 | minimum improvement counted |
| `hidden` / `dropout` | 64 / 0.1 | model width and regularisation |
| `device` | `auto` | MPS when available, else CPU |
| `checkpoint_every` | 1 | epochs between atomic saves |
| `max_minutes` | unset | optional wall-clock budget |

Resource and safety rules:

- **one job per process** — a non-blocking `JobSlot` rejects a second training
  call with `TrainError`, so two runs cannot fight over RAM or the MPS device;
- **explicit MPS** — `device="auto"` prefers MPS and falls back to CPU;
  `device="mps"` on a machine without it is an error, never a silent CPU run;
- **cancellation** — call `JobControl.cancel()`; the flag is checked every batch
  and the last complete checkpoint remains loadable. A cancelled run reports
  `stopped="cancelled"`;
- **atomic checkpoints** — `torch.save` goes to a temp file and is moved onto
  the target, so an interrupt cannot leave a truncated `.pt`;
- **no invented classes** — validation labels absent from training are an
  error instead of an always-wrong class the model can never predict.

Checkpoint payload: model name, config, classes, `n_features`, `window`,
`dataset_id`, history, best epoch/loss/accuracy, `model_state` (last epoch),
`best_state` (best validation epoch) and the optimizer state. `resume_from`
requires the same model name, feature count and window, then continues from the
next epoch.

The best weights are stored, but evaluation is the next slice: a trainer
result is a validation-loss curve, not a trading result.

## 7. Limits and next slices

- Windows are materialised; M1 history needs H1/M15, a smaller window or a lazy
  loader (Phase 14). Do not silently reduce `window` to make RAM fit — record
  the choice.
- The dataset alone says nothing about profitability. Model scores are
  prediction metrics; the simulator + cost stress decides.
- Next slice: the shared evaluator against Phase 5 and the rule (simulator +
  cost stress), plus the CLI/API wiring for training runs.
