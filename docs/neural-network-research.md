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

## 7. Evaluator

`scalping.nn.evaluate_checkpoint(checkpoint, data, split="test")` rebuilds the
checkpoint's model (best validation weights when available) and scores one
split:

| Metric | Meaning |
|---|---|
| `accuracy` | share of correct labels |
| `balanced_accuracy` | mean per-class recall; a majority-class model cannot hide here |
| `log_loss` | mean `-log p(true)` with probabilities clipped at `1e-12` |
| `brier` | multiclass Brier score, `mean(sum((p - onehot)^2))` |
| `per_class` | precision, recall, F1 and support for every class |
| `confusion` | counts in the checkpoint's class order |

`compare_predictors(y_true, {"gru": (...), "rule": (...)}, classes=...)` scores
any mix of label-only and probability predictors on the same rows and sorts by
accuracy. The result always includes `rows` and `trading_claim: false`.

The rule comparison is the repo's `ret_5` momentum baseline
(`momentum_rule_labels`), mapped to `-1/0/+1` and scored on the same window-end
timestamps as the network. It is **not** the validated `flow_momentum` card;
that card needs tick flow and is judged by the strategy layer.

Classical ML models (Phase 5) plug into the same two functions: pass their
label array, and their probability matrix when available. Nothing in the
evaluator imports `scalping.ml`, so the two phases stay independent.

Boundary: these are prediction metrics. A high accuracy is not a trading edge;
the simulator with spread, commission and slippage plus a cost stress is still
required before any profitability word is used.

## 8. CLI

```bash
cd backend
uv run python -m scalping.cli nn train \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --model gru --window 60 --epochs 20 \
  --checkpoint ../runs/nn/gru_h1.pt --out ../runs/nn/gru_h1.json

uv run python -m scalping.cli nn evaluate \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --checkpoint ../runs/nn/gru_h1.pt
```

- `train` builds features and the chronological dataset from the store, fits
  the chosen model, writes the checkpoint and a report, and prints one JSON
  line (checkpoint, dataset id, best epoch, validation accuracy, stop reason,
  device, window counts).
- `evaluate` reloads the checkpoint, rebuilds the dataset and scores one split
  (`--split`, default `test`). It refuses a checkpoint whose `dataset_id` does
  not match the rebuilt data, so a wrong root/timeframe/window cannot be scored
  silently. Use the same `--horizon`, `--purge`, `--clip`,
  `--train-fraction` and `--validation-fraction` as training when they differed
  from the defaults.
- `--device auto` prefers MPS, `cpu` forces the CPU and `mps` fails if the
  machine has no MPS.
- Checkpoints and reports live under `runs/`, which is gitignored.

The subprocess test `backend/tests/test_cli_nn.py` runs `synth → derive H1 →
nn train → nn evaluate` with a tiny model, so the whole path is exercised
without market data or committed weights. A real TCN/GRU H1 and M15 run still
needs the local stores.

## 9. Benchmarking

```bash
cd backend
uv run python -m scripts.benchmark_nn \
  --models mlp,gru,tcn,transformer,patchtst,itransformer \
  --n-features 59 --window 60 --batch 8 --repeats 5 --hidden 64 \
  --out ../runs/nn/benchmark.json
```

`scalping.nn.benchmark_models` builds each model at the same input shape and
measures:

| Field | Meaning |
|---|---|
| `parameters` / `parameter_mb` | exact parameter count and its float32 weight size |
| `latency_ms_median` | median forward latency over `repeats` runs after `warmup` |
| `throughput_windows_per_s` | `batch / median latency` |
| `rss_high_water_mb` | process high-water RSS (`ru_maxrss`, normalised) |
| `device` | `cpu` or `mps` |

Caveats are part of the result: RSS is process-wide and monotonic, so it is a
smoke signal rather than isolated per-model memory — run one model per process
for that. Latency is forward-only on a fixed random tensor; training time,
optimizer memory, gradient checkpointing and data loading are out of scope.
`device="mps"` synchronises around each measurement; CPU runs pin one torch
thread for comparability.

The benchmark does not rank models by accuracy. Prediction metrics come from
`nn evaluate` (shared evaluator), and trading claims still require the
simulator plus cost stress.

## 10. Limits and next slices

- Windows are materialised; M1 history needs H1/M15, a smaller window or a lazy
  loader (Phase 14). Do not silently reduce `window` to make RAM fit — record
  the choice.
- The dataset alone says nothing about profitability. Model scores are
  prediction metrics; the simulator + cost stress decides.
- Next: the simulator + cost-stress gate for any trading claim, and the
  Phase 5 comparison once `scalping.ml` lands.
