# Architecture audit — 2026-11-22

Audit of `main` (then `feat/v1`) at commit `e3c8095`, before the audit documents were added.
Method: source-tree inventory, targeted reading of every module named below,
`rg` for classes and capabilities, dependency files, test collection, and the
recorded test/evaluation runs. This is an engineering audit, not a line-by-line
security review.

The requirements baseline is `docs/requirements/master-prompt.md`.

## Executive summary

The repository is a **working, tested single-symbol research and paper-replay
system** for XAUUSD, with an unusually honest evaluation chain:

- causal feature engineering, realistic paper broker, walk-forward evaluation,
  cost stress, and a promoted rule card;
- a validated `flow_momentum` rule with a positive forward test;
- a local PPO/behavior-cloning research path that is stable but weaker than the
  rule.

It is **not yet** the multi-model, multi-symbol, self-managing platform in the
master prompt. The largest gaps are:

1. no classical ML or deep-learning model zoo (only PPO's small MLP);
2. no regime, ensemble, confidence or Transformer engines;
3. no SQLite experiment/model registry, drift detection or rollback;
4. no multi-symbol contract registry (XAUUSD-centric);
5. only four frontend pages (no ML/NN/RL/regime/paper/monitoring labs);
6. no persistent paper-trading ledger or broker adapter;
7. this audit adds 2 of the 15 required documents; 13 are still missing.

Nothing was deleted or rewritten during this audit.

## Current architecture

```
data providers (synthetic / CSV / Dukascopy)
        │
        ▼
ParquetStore  ──►  resample (M5/M15/H1/D1)
        │
        ▼
feature builder (59 causal columns, M5/M15 context)
        │
        ├──► labels (5 rule families, counterfactual R)
        │
        ▼
TradingEnv / Simulator ── PaperBroker + RiskEngine
        │                         │
        │                         └── spread, commission, slippage, SL/TP,
        │                             daily loss, notional cap, time stop
        ├──► rule policy (flow_momentum card)
        └──► PPO (SB3, MLP 64×64, optional BC warm start)
                    │
                    ▼
        backtest / walk-forward / cost stress
                    │
                    ▼
        FastAPI + WebSocket ──► Next.js panel (4 pages)
```

## Backend module inventory

| Area | Files | What actually exists | Status |
|---|---|---|---|
| Data schema/store | `data/base.py`, `data/store.py`, `data/resample.py` | typed BAR_SCHEMA, atomic Parquet partitions, UTC, coverage, resampling | DONE |
| Providers | `data/synthetic.py`, `data/csv_provider.py`, `data/dukascopy.py` | deterministic synthetic with regimes, broker CSV, Dukascopy ticks with bid/ask volume | DONE (XAU/XAG for Dukascopy) |
| Feature engineering | `features/build.py`, `context.py`, `indicators.py`, `liquidity.py`, `structure.py`, `prepare.py` | 59 causal features: returns, candle anatomy, RSI/MACD/EMA, ATR/realised vol/regime, volume, spread, structure/BOS/CHoCH, FVG/sweep, order-block distance, session, M5/M15 context | DONE for XAUUSD M1 source |
| Labels | `strategy/labeling.py`, `strategy/signals.py` | 5 rule families (momentum, reversal, sweep, fvg, trend), counterfactual labels priced by the same broker, causal bandit context | DONE |
| Strategy selection | `strategy/bandit.py` | LinUCB, bias-augmented, offline fit, frozen for evaluation | DONE (one selector, not an ensemble) |
| Live rule | `strategy/flow.py`, `rule.py`, `calibrate.py`, `card.py` | signed-flow gate, momentum rule, grid calibration on train, holdout gate, versioned card | DONE |
| Simulator | `env/simulator.py`, `env/broker.py`, `env/risk.py`, `env/trading_env.py` | Gymnasium env, paper broker, ATR SL/TP, time stop, spread/commission/slippage, daily loss, notional cap, position limit, equity stop | DONE |
| Backtest | `backtest/engine.py`, `metrics.py`, `walkforward.py`, `evaluate.py` | sequential replay, 9+ metrics, walk-forward folds, overfit warning, cost stress | PARTIAL vs master prompt (no purged CV, Monte Carlo, parameter sensitivity, partial fills, latency) |
| RL | `rl/train.py`, `policy.py`, `registry.py`, `callbacks.py`, `runtime.py` | PPO (MlpPolicy 64×64), VecNormalize, random starts, checkpoints, resume/pause/stop, BC warm start, run registry | PARTIAL (PPO only; no SAC/TD3/DQN/A2C; CPU only) |
| API | `api/app.py`, `routes.py`, `runner.py`, `train_runner.py`, `schemas.py` | health, data status/download, train, runs, run detail/control, backtest, evaluate, sim start/control, strategy card/calibrate, WS stream | DONE for current scope |
| CLI | `cli.py` | data synth/csv, backtest, serve | DONE |
| Research scripts | `scripts/*.py` | MT CSV ingest, tick aggregation, flow aggregation, Dukascopy fetch, flow training, nightly calibration, nightly RL lab, forward ingest/test/update | DONE as scripts, not as scheduler modules |

### Data assets on this machine (not tracked)

| Store | Coverage | Rows |
|---|---|---|
| `data-ticks5-store` | 2021-05-24 → 2026-05-22, XAUUSD | 1,773,331 M1 |
| `data-flow-store` | same period, Tickmill tick imbalance | 1,773,331 M1 |
| `data-real-store` | 2009-03 → 2026-01, broker M1 (no volume) | 5,888,800 M1 |
| `data-rich-store` | 2025-06 → 2025-11, Tickmill ticks | 176,713 M1 |
| `data-forward-store` | 2026-05-28 → 2026-11-15, Dukascopy ticks | 167,666 M1 |
| `data-forward-flow` | same | 167,666 M1 |
| `data-macro` | FRED DXY/rates, Forex Factory calendar | n/a |

## Models and learning — what exists

| Component | Reality | Evidence |
|---|---|---|
| PPO policy | `MlpPolicy`, `net_arch=[64,64]`, observation 3,563 dims with selector (60×59 + 5 position + 3 flow + 15 strategy) | `rl/train.py` |
| Behavior cloning | supervised warm start from the rule, inverse-frequency weights, validation early stopping | `rl/train.py::_behaviour_clone` |
| Bandit | LinUCB, 5 arms, 15-dim context, frozen after offline fit | `strategy/bandit.py` |
| Live card | flow_momentum, H1, 0.85 quantile / 480 bars, 3/6 ATR, 48-bar hold | `backend/config/strategy_card.json` |
| Deep learning zoo | **none** — no CNN, RNN, LSTM, GRU, TCN, Transformer, AE, VAE | no modules/deps |
| Classical ML | **none** — no sklearn/XGBoost/LightGBM/Optuna | `pyproject.toml` |
| Regime engine | only a 3-state volatility feature and synthetic regime generator | `features/build.py`, `data/synthetic.py` |
| Ensemble/confidence | only LinUCB arm selection; no probability calibration, stacking, disagreement, NO TRADE engine | `strategy/bandit.py` |
| Model registry | JSON manifests + run directories under `runs/`; no SQLite, champion table, rollback | `rl/registry.py` |
| Continuous learning | two ad-hoc loops: nightly rule calibration, data-gated RL lab; no scheduler module, drift detection or promotion DB | `scripts/nightly_calibrate.py`, `nightly_rl.py` |

## API and frontend inventory

API routes: `/api/health`, `/api/data/status`, `/api/data/download`,
`/api/train`, `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/control`,
`/api/backtest`, `/api/evaluate`, `/api/sim/start`, `/api/sim/control`,
`/api/strategy/card`, `/api/strategy/calibrate`, `/ws/sim`.

Frontend pages: `/` (simulator + live card + forward result), `/training`,
`/performance`, `/data`. Components cover charts, metrics, positions, runs,
strategy table and the strategy card.

Missing UI from the brief: market intelligence, classical ML lab, neural
network lab, RL lab, strategy intelligence by regime, autonomous learning,
paper-trading ledger, system monitoring.

## Test and quality evidence

Last executed runs:

- backend: `uv run pytest` → **539 passed** (116.7 s);
- frontend: `npm test` → **184 passed**;
- `ruff check src tests scripts` → clean;
- `mypy src` → clean (49 modules);
- `npm run build` → passed (earlier in the same branch).

Coverage is strong for the implemented scope (data, features, env, broker,
risk, labels, bandit, RL smoke, API, strategy card). There are no tests for
classical ML, deep-learning architectures, regime detection, ensembles,
confidence, SQLite or paper-trading persistence because those modules do not
exist.

## Requirements gap matrix

Legend: DONE / PARTIAL / MISSING / EXPERIMENTAL.

| # | Requirement | Status | Evidence / gap |
|---|---|---|---|
| 1 | Zero-cost, local, no LLM | DONE | no API clients; PyTorch/SB3 local |
| 2 | Mac mini M2, 8 GB, MPS/CPU | PARTIAL | small MLP, CPU pinning, checkpoints; no RAM manager, MPS unused |
| 3 | 10 instruments, symbol specs | MISSING | Dukascopy only XAU/XAG; one gold `ContractSpec` |
| 4 | Tick–D1 timeframes | PARTIAL | M1/M5/M15/H1/D1; no M30/H4; ticks aggregated, not stored |
| 5 | CSV/Parquet/JSON import | PARTIAL | CSV/synthetic/Dukascopy; Parquet store; no JSON importer |
| 6 | Data validation engine | PARTIAL | schema/UTC/coverage/atomic; no duplicate/outlier/gap report/provenance engine |
| 7 | Feature framework | PARTIAL | 59 causal features; no configurable selection; flow is a side table |
| 8 | Genuine bid/ask/order-flow features | PARTIAL | signed flow built from tick volumes; no book/L2, no DeepLOB |
| 9 | Multi-timeframe analysis | PARTIAL | M5/M15 context; no independent encoders/fusion; no M30/H4 |
| 10 | Classical ML zoo | MISSING | dependencies absent |
| 11 | Deep-learning zoo | MISSING | only PPO MLP; no CNN/RNN/LSTM/GRU/TCN/AE |
| 12 | Transformer research | MISSING | no module/dependency |
| 13 | RL algorithms | PARTIAL | PPO + BC; no SAC/TD3/DQN/A2C; actions no REDUCE |
| 14 | Market regime engine | PARTIAL | vol_regime feature; no KMeans/GMM/HMM/supervised engine |
| 15 | Strategy research/selection | PARTIAL | 5 families + flow rule + LinUCB; no regime-performance matrix or NO TRADE engine |
| 16 | Ensemble/decision intelligence | PARTIAL | LinUCB only; no stacking/voting/regime-dependent selection |
| 17 | Confidence/uncertainty | MISSING | no calibration/Brier/disagreement/OOD |
| 18 | Continuous learning | PARTIAL | nightly calibration + RL lab; no SQLite, drift, rollback, incremental learning |
| 19 | Experiment/HPO | MISSING | no Optuna/sweeps |
| 20 | Independent risk engine | PARTIAL | risk %, daily loss, notional, positions, SL/TP, timeout; no weekly loss, DD/consecutive-loss kill switch, max spread, data-quality block |
| 21 | Realistic backtesting | PARTIAL | walk-forward, cost stress; no purged CV/embargo, Monte Carlo, sensitivity, latency, partial fills |
| 22 | Paper trading engine | PARTIAL | simulator/broker; no persistent ledger, broker adapter, MT5 demo adapter |
| 23 | Model registry/knowledge DB | MISSING | JSON/Parquet only; no SQLite, champion/rollback |
| 24 | Dashboard sections | PARTIAL | 4 pages; missing 6 laboratory/monitoring sections |
| 25 | Scheduler | PARTIAL | two background loops/scripts; no configurable scheduler |
| 26 | Tests | DONE for current scope | 539 backend, 184 frontend; no ML/DL/regime tests because absent |
| 27 | Required docs | PARTIAL | 2 of 15 complete (this audit + roadmap); 13 missing |
| 28 | Acceptance criteria | MISSING | most ML/DL/ensemble/paper criteria unmet |

## Architectural risks and bottlenecks

1. **Single-symbol coupling.** `ContractSpec` is gold-shaped; features assume
   M1; the runner/card default to XAUUSD. Multi-symbol needs a registry first,
   before any model work.
2. **Feature scale vs data.** 59 features on ~15k H1 bars (or 1.7M M1) is easy
   to overfit. Any new model must reuse the purged/chronological split and cost
   stress, or it will produce another unstable result.
3. **Observation size.** PPO flattens 60×59 into 3,548 inputs. A temporal
   encoder is the natural next architecture, but data, not capacity, is the
   binding constraint.
4. **Forward sample size.** The clean forward test has 24 trades. Bigger models
   cannot be validated on that sample yet.
5. **No experiment database.** `runs/` directories are readable but not
   queryable; champion/rollback and drift need SQLite.
6. **One training job at a time.** 8 GB RAM means the scheduler must enforce a
   single heavy job and a RAM budget; none exists yet.
7. **Data provenance is thin.** The `provider` column exists, but synthetic vs
   real, source file, import time and checksum are not recorded per dataset.

## What this audit does not claim

- No live broker connection or real-money execution.
- No multi-symbol support.
- No classical ML, deep-learning zoo or Transformer implementation.
- No regime/ensemble/confidence engine.
- No SQLite model registry or automatic champion promotion for ML models.
- The `flow_momentum` forward result is positive but small-sample; it is not a
  guarantee of future profitability.
- PPO/BC is experimental and weaker than the rule; it is not the live strategy.

## Post-audit progress (2026-11-22, Phase 2 first slice)

- Added `data/registry.py` with ten symbol specs and M30/H4 timeframe rules.
- Added `data/validate.py` (bars/ticks; errors vs gap warnings, no repair).
- Added `data/provenance.py` and seeded sidecars for the existing XAUUSD stores.
- Added JSON import and `data derive` CLI commands.
- Added `/api/symbols`, `/api/data/validate`, and provenance on `/api/data/status`.
- Added tests for registry, validation, provenance, JSON import, API and CLI.

Still missing from Phase 2: real data providers for the eight non-metal symbols,
statistical outlier/cross-source checks, and a stored tick table. These are
tracked in the roadmap.

Phase 3 first slice (same date): `features/registry.py` (seven groups covering
59 columns) and `datasets/builder.py` (chronological split, purge/embargo,
checksum, manifest) plus `cli dataset build` and leakage/split tests.

## Next step

See `docs/implementation-roadmap.md` for the dependency-ordered plan and
`STATUS.md` for the one-page collaboration matrix. Phase 1 ends with this audit
and the roadmap; no large-scale code modification starts before approval.
