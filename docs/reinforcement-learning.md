# Reinforcement learning

The RL track asks a narrow question: can an agent learn the scalping decision
from the same simulator the rules are judged on? The honest answer so far is
**stable but not better than the validated rule**. This document records what
exists, how it is trained and evaluated, and what is still open. It is not a
claim of profitability.

## 1. Environment

`TradingEnv` wraps the deterministic accounting loop (`env/simulator.py`):

- **Observation** — the last `window` (60) feature rows, the current position
  state, and an optional 3-column flow context (signed imbalance, its excess
  over the gate threshold, 5-bar change).
- **Actions** — `HOLD`, `BUY`, `SELL`, `CLOSE` (0–3).
- **Episode** — starts at `valid_from + window - 1`, terminates if equity falls
  to 80% of the initial balance, truncates at `max_steps` or data end, and
  never ends with an open position (data-end/max-step quote close).
- **Costs** — spread, commission and ATR-scaled slippage are the simulator's,
  not the agent's to choose.

`action_masks()` reports which actions the loop would accept: position state,
session hours, daily/weekly loss, drawdown and kill switches, consecutive-loss
pause, spread limit, flow/signal gate, missing ATR and lot-size. Evaluation can
enforce the mask (`run_backtest(mask_actions=True)`, `backtest --mask-actions`);
training still uses plain PPO, so a masked action is proposed and rejected
rather than never sampled (see open items).

## 2. Reward

Per step, with `initial` the account's starting balance:

```text
reward = (equity - equity_before) / initial * 100
         - 50 * max(0, new_drawdown_depth - previous_depth)
         - 0.05 * opened
         + 0.02 * risk_ok
         clipped to [-5, +5]
```

The drawdown term charges **new depth only**, not every bar at depth, so
"never trade" is not the optimal policy. `opened` is 1 only on the entry step;
`risk_ok` is a one-time credit on the same gate-approved step. A reward ablation
report across these terms is still open.

## 3. Training

`rl/train.py` uses stable-baselines3 PPO:

| Setting | Default |
|---|---|
| `total_steps` | 200,000 |
| `n_envs` | 2 |
| `lr` | 3e-4 |
| `n_steps` | 1,024 |
| `batch_size` | 256 |
| `gamma` | 0.99 |
| `ent_coef` | 0.01 |
| `checkpoint_every` | 10,000 steps |
| `random_start` | true |
| `entry_penalty` | 0.05 |
| `gate_signals` | true |
| `window` | 60 |
| `timeframe` | M1 |

- Python, NumPy and torch seeds are set once; the run records library versions,
  data hash and resolved config.
- `VecNormalize` normalises observations and rewards; the normaliser is saved
  with the model.
- Threads are pinned for reproducible CPU training (`rl/runtime.py`).
- Learning happens in chunks; after each chunk the model, normaliser, metrics
  and manifest are written atomically under `runs/<run_id>/`, so a crash leaves
  the previous checkpoint readable.
- Pause/resume/cancel are cooperative callbacks; a paused run keeps its
  checkpoint, a cancelled run stops at the next safe boundary.
- One heavy job per process is enforced (the PPO gate); nightly jobs use the
  same registry.

## 4. Imitation warm start

`docs/rl-imitation.md` documents the experiment: collect `(observation,
action)` pairs by replaying the validated `flow_momentum` rule, fit a supervised
policy (class-weighted cross-entropy, validation split, early stopping), then
fine-tune with PPO. Two teacher definitions were tried — the rule's requested
action and the action the environment actually executed. The executed-action
teacher removes the "policy learns rejected entries" problem.

## 5. Baselines and evaluation

No RL result is reported alone. The comparison set:

1. **hold** — the simulator with no trades;
2. **random** — same action distribution, same episode;
3. **rule** — the validated `flow_momentum` card, replayed by the same
   simulator;
4. the previous checkpoint / champion.

Evaluation is chronological and out-of-sample; cost stress and an untouched
forward store remain the gate. Prediction metrics from the neural network
framework are not a trading result.

Reported results (H1, real spread, sequential simulator; see
`docs/rl-imitation.md` for the table):

| Experiment | 2024+ | 2025+ |
|---|---|---|
| rule (live card) | +0.068 R, PF 1.11, 266 trades | +0.103 R, PF 1.12, 123 trades |
| PPO from scratch | +0.099 R, PF 1.25 | −0.049 R, PF 0.69 |
| BC + PPO (stateless teacher) | +0.012 R, PF 1.07, 452 trades | +0.020 R, PF 1.05, 242 trades |
| BC only (stateless) | −0.117 R, PF 0.74, 41 trades | — |
| BC + PPO (executed teacher) | −0.018 R, PF 0.96, 403 trades | — |

Interpretation: imitation makes PPO stable, but it does not beat the rule. The
sample is small and split-dependent; no promotion is justified.

## 6. Registry and artefacts

`rl/registry.py` writes one directory per run:

```text
runs/<run_id>/
  manifest.json      # listed by the API
  config.json        # resolved config + library versions + data hash
  model.zip          # SB3 policy
  vecnormalize.pkl   # observation/reward normaliser
  metrics.json       # latest snapshot
  metrics.jsonl      # append-only history
  bandit.npz         # optional strategy bandit
```

Writes are atomic (temp file + `os.replace`). A model is never edited in place.

## 7. Open items (Phase 8)

- Mask-aware training (`sb3-contrib` MaskablePPO); evaluation masking exists.
- `REDUCE` position action and partial exits.
- SAC / TD3 / DQN / A2C comparisons on the same split and costs.
- Seed ensembles / policy averaging.
- Execution-delay assumption in the environment.
- Reward ablation report; measured benchmark for PPO.
- Promotions still require simulator + cost stress + untouched forward data.

## 8. Reproducing a run

```bash
cd backend
uv run python -m scripts.nightly_rl --help
# or the API: POST /api/train, then GET /api/runs/<run_id>
```

Artifacts are local and gitignored; no model weights, `runs/` content or data
are committed.
