# Transformer models

Phase 7 adds compact Transformer encoders to the sequence-model framework.
They use the same `WindowedDataset`, trainer, checkpoints and evaluator as the
GRU/TCN zoo, so a Transformer result is directly comparable to the rule and to
the other models on the same rows.

The target hardware is a Mac mini M2 with 8 GB. The config guard below is part
of the design, not an afterthought: a tiny model that finishes a run is worth
more than a large one that swaps.

## 1. Variants

| Model name | Attention | Pooling | Use |
|---|---|---|---|
| `transformer` | bidirectional over the window | mean over steps | offline classification; the window ends at the prediction bar |
| `causal_transformer` | upper-triangular mask | last token | streaming-safe representation; step `t` never reads `t + 1` |
| `patchtst` | bidirectional over patches | mean over patches | PatchTST-style patching; attention sees a few tokens instead of every bar |
| `itransformer` | bidirectional over variates | mean over variates | iTransformer-style; one token per feature, modelling cross-feature structure |

Both prepend a learned positional embedding, use pre-norm encoder layers with
GELU feed-forward blocks, and output logits. `forward_sequence` exposes the
per-step states so causality can be tested exactly like the TCN.

## 2. Config guard

`TransformerConfig`:

| Setting | Default | Bound |
|---|---|---|
| `n_heads` | 4 | `>= 1`, must divide `hidden` |
| `n_layers` | 2 | `1..8` |
| `ff_mult` | 4 | `>= 1` |
| `max_sequence` | 512 | `8..4096` |

`window > max_sequence`, a head count that does not divide `hidden`, or a layer
count outside the range is a construction error. The default model at 59
features, 60 bars and `hidden=32` stays under two million parameters
(`test_default_model_stays_small`).

### PatchTST-style patching

`patchtst` splits the window into patches of `patch_len` bars with an optional
`stride` (default: non-overlapping). Each patch is flattened to
`patch_len * features`, projected to `hidden`, and becomes one encoder token:

```
n_patches = (window - patch_len) // stride + 1
```

A trailing partial patch is **dropped**, not zero-padded: `window=25` with
`patch_len=8` gives 3 patches and bar 24 is unused, visible in `n_patches`.
`max_sequence` caps the number of patches (the attention length), not the raw
window. `window < patch_len`, a zero patch/stride or a head count that does not
divide `hidden` raise at construction.

```bash
uv run python -m scalping.cli nn train ... --model patchtst --window 60 --hidden 32
```

With `window=60`, the default 8-bar patches give 7 tokens, which keeps the
encoder tiny on the M2.

### iTransformer-style variate attention

`itransformer` transposes the input so every **feature** becomes one token: the
token embedding is a linear projection of that feature's whole window. Attention
then mixes features rather than bars. The model is deliberately **not** causal:
the window already ends at the prediction bar, and the point of this variant is
cross-feature structure.

`max_sequence` caps the number of variate tokens (`n_features`), and the window
is still fixed and checked on every call.

```bash
uv run python -m scalping.cli nn train ... --model itransformer --window 60 --hidden 32
```

## 3. Causality

`tests/test_transformers.py` mutates the input from step 10:

- `causal_transformer` steps 0–9 stay identical (masked attention);
- `transformer` steps change even before step 10 (bidirectional by design).

The bidirectional variant is not a leak: the window itself ends at the
prediction bar, so all inputs are in the past relative to the label. The causal
variant exists for streaming or prefix-style use.

## 4. Usage

```bash
cd backend
uv run python -m scalping.cli nn train \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --model transformer --window 60 --hidden 32 --epochs 20 \
  --checkpoint ../runs/nn/transformer_h1.pt --out ../runs/nn/transformer_h1.json

uv run python -m scalping.cli nn evaluate \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --checkpoint ../runs/nn/transformer_h1.pt
```

`hidden` must be divisible by 4 with the default heads; use `hidden=32` or
`hidden=64` (not 30).

## 5. Status and limits

Implemented: encoder, causal temporal encoder, config guard, causality tests,
parameter-count smoke test, trainer/CLI/evaluator integration.

Open in Phase 7: an optional TFT and a measured benchmark table against
GRU/TCN (latency and peak memory, not just parameters). No profitability claim is made from
prediction accuracy; the simulator and cost stress remain the gate.
