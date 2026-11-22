# Master task board

One-by-one breakdown of `docs/requirements/master-prompt.md`. A box is checked
only when the task is **implemented and tested**. Partial work stays unchecked
with a `PARTIAL` note. GitHub phase issues mirror these lists.

Legend: `[x]` done · `[ ]` todo · `PARTIAL` some code exists but the task is not
met end to end · `BLOCKED` missing data/dependency/decision.

## Phase 1 — Audit and documentation

- [x] Read the backend, frontend, API, strategies, PPO, imitation, features,
      data, backtest, broker, config, training, dashboard, tests, deps, docs.
- [x] `docs/architecture-audit.md` with evidence and gaps.
- [x] `docs/implementation-roadmap.md` with order, files, RAM and tests.
- [x] `STATUS.md` one-page matrix.
- [x] `docs/requirements/master-prompt.md` canonical brief.
- [x] Team workflow, CI, PR/issue templates, CODEOWNERS, security policy.

## Phase 2 — Data foundation

- [x] Ten-symbol registry with contract/point/lot/spread/session/precision.
- [x] Contract mapping for the ten symbols.
- [x] M1, M5, M15, M30, H1, H4, D1 timeframe ladder.
- [x] CSV and JSON import; Parquet store.
- [x] Bar validation: duplicates, UTC/order, OHLC, crossed quotes, negative
      spread/volume, gaps.
- [x] Tick validation: sorted quotes, crossed book, non-finite/negative values.
- [x] Provenance sidecar with source, synthetic flag, coverage and checksum.
- [x] API `/api/symbols`, `/api/data/validate`, provenance in data status.
- [x] `data derive` CLI for higher timeframes.
- [ ] PARTIAL: separate raw/cleaned/feature/training/validation/test layers.
      The store + `_meta` + dataset manifests exist; a raw/clean split does not.
- [ ] Real providers for EURUSD, GBPUSD, USDJPY, AUDUSD, NAS100, US30, USOIL,
      BTCUSD. **BLOCKED on free sources/licences** (issue #15).
- [ ] Statistical outlier detection and cross-source reconciliation.
- [ ] Tick table layer for querying ticks without pre-aggregation.
- [ ] Data-quality report in the dashboard.
- [ ] Per-symbol sessions enforced in the simulator.
- [ ] Dataset import is chunked automatically for multi-GB files.

## Phase 3 — Features and datasets

- [x] Feature registry: seven groups covering all 59 causal columns.
- [x] Canonical-order selection and `select_matrix`.
- [x] Chronological train/validation/test split.
- [x] Purge ≥ label horizon and configurable embargo.
- [x] `split == -1` audit rows, checksum and manifest.
- [x] `dataset build` CLI.
- [x] Leakage/label-alignment/CLI tests.
- [ ] Feature availability rules (e.g. signed flow only with tick volumes).
- [ ] H4/M30 multi-timeframe alignment tests.
- [ ] Dataset id wired into every trainer/run config.
- [ ] Persisted array format for large datasets (memmap/parquet).
- [ ] Feature ablation runner.
- [ ] Consecutive-candle and log-return features explicitly named.
- [ ] Cumulative volume delta and order-flow features from real ticks.

## Phase 4 — Market regime engine

- [x] Rule-based trend/volatility/expansion/breakout/unusual regimes.
- [x] K-Means regime model.
- [x] Gaussian mixture regime model.
- [x] Hidden Markov regime model.
- [x] Random forest / gradient boosting regime baseline.
- [ ] Autoencoder/representation-based regime research.
- [x] Strategy × regime performance table.
- [ ] API endpoint and Market Intelligence UI.
- [x] `docs/market-regime-detection.md`.
- [x] Tests: determinism, no leakage, transition sanity.

## Phase 5 — Classical ML

- [x] Add scikit-learn (LightGBM/XGBoost still open).
- [x] Logistic regression, random forest, extra trees, HistGradientBoosting.
- [x] LightGBM and XGBoost (lazy imports; macOS needs `libomp`).
- [ ] SGDClassifier incremental path.
- [ ] Isolation forest anomaly detector.
- [ ] SVM only if timing is practical.
- [x] Direction task (3-class, configurable horizon).
- [x] Train/validation split, purge/embargo and dataset id.
- [x] Brier, log loss, balanced accuracy, per-class metrics, confusion, baseline.
- [x] Flat-zero rows dropped by default with an audited count; `--include-flat` opt-in.
- [ ] Return/volatility/regime/trade-outcome tasks.
- [ ] Purged time-series CV with embargo.
- [x] Reliability curve, ECE and guarded MCE.
- [ ] Calibration thresholds / probability rejection gate.
- [ ] Feature importance and ablation.
- [x] Simulator/cost-stress trading evaluation for a saved classifier.
- [ ] Regime breakdown of the trading evaluation.
- [ ] Model registry metadata in SQLite (Phase 10).
- [x] `docs/classical-ml.md`.

## Phase 6 — Neural network framework

- [x] Windowed tensor dataset with train-only normalisation.
- [ ] PyTorch trainer: MPS when reliable, CPU fallback.
- [ ] Gradient clipping, early stopping, checkpoints, resume, cancellation.
- [ ] RAM guard and one-job policy.
- [x] MLP, deep MLP, residual MLP.
- [x] Vanilla RNN, LSTM, GRU.
- [x] 1D CNN, dilated CNN, CNN-LSTM.
- [x] Causal TCN with configurable receptive field.
- [ ] Shared evaluator against classical ML and the rule.
- [ ] Checkpoint round-trip and causality tests.
- [ ] `docs/neural-network-research.md`.

## Phase 7 — Transformers

- [ ] Transformer encoder.
- [ ] Causal temporal Transformer.
- [ ] PatchTST-style patching.
- [ ] iTransformer-style variate attention.
- [ ] TFT only if smaller models justify it.
- [ ] Config guard (d_model/layers/heads/sequence/batch).
- [ ] Memory smoke test and benchmark.
- [ ] Same evaluator/registry.
- [ ] `docs/transformer-models.md`.

## Phase 8 — Reinforcement learning expansion

- [x] PPO with Gymnasium environment.
- [x] Behavior-cloning warm start.
- [x] VecNormalize, random starts, checkpoints, pause/resume.
- [x] Rule, hold and random baselines.
- [ ] Action masking for flow/risk block.
- [ ] REDUCE position action.
- [ ] SAC evaluation.
- [ ] TD3 evaluation.
- [ ] DQN evaluation.
- [ ] A2C baseline.
- [ ] Seed ensembles / policy averaging.
- [ ] Execution-delay assumption in the environment.
- [ ] Reward ablation report.
- [ ] `docs/reinforcement-learning.md`.

## Phase 9 — Ensemble and confidence

- [ ] Out-of-fold stacking.
- [ ] Weighted averaging and voting.
- [ ] Regime-dependent model selection.
- [ ] Probability calibration and temperature scaling.
- [ ] Brier/log loss/calibration curves.
- [ ] Ensemble disagreement score.
- [ ] Out-of-distribution/unknown regime detection.
- [ ] Decision engine: BUY/SELL/HOLD/NO TRADE.
- [ ] Decision log with models, probabilities, regime, reason.
- [ ] Compare ensemble vs best member; keep the simpler winner.
- [ ] `docs/ensemble-decision.md`.

## Phase 10 — Autonomous learning and registry

- [ ] SQLite schema: datasets, experiments, models, promotions, backtests,
      paper runs, drift, jobs.
- [ ] Dataset versioning and checksums.
- [ ] Model versioning, champion pointer and rollback.
- [ ] Candidate → champion gate (trades, OOS, cost stress, regime coverage).
- [ ] Drift detection: features, predictions, calibration.
- [ ] Configurable scheduler with pause/resume/cancel.
- [ ] One heavy job/RAM budget enforcement.
- [ ] Failed training cannot stop paper replay.
- [ ] Migrate nightly rule/RL jobs to the scheduler.
- [ ] `docs/local-learning-engine.md`.
- [ ] Tests: promote, reject, rollback, drift, cancel.

## Phase 11 — Risk and backtesting

- [ ] Max weekly loss.
- [ ] Max drawdown kill switch.
- [ ] Consecutive-loss pause.
- [ ] Max spread and slippage limits.
- [ ] Data-quality trade block.
- [ ] Emergency kill switch.
- [ ] Risk limits cannot be bypassed by a model.
- [ ] Purged CV with embargo integrated in evaluation.
- [ ] Monte Carlo trade-sequence analysis.
- [ ] Parameter sensitivity.
- [ ] Ablation runner.
- [ ] Regime-specific reports.
- [ ] Execution delay and rejected-order assumptions.
- [ ] Partial fills only where data supports them.
- [ ] `docs/risk-management.md`, `docs/backtesting-methodology.md`.
- [ ] Tests: kill switch, embargo, Monte Carlo determinism.

## Phase 12 — Paper trading

- [ ] Broker adapter interface.
- [ ] Persistent paper ledger (SQLite).
- [ ] Track signals, orders, fills, spread, slippage, commission, P&L,
      model/strategy versions and rejection reasons.
- [ ] Restart-safe paper session runner.
- [ ] MT5 demo adapter skeleton, disabled, no credentials.
- [ ] API: account, positions, orders, history.
- [ ] `docs/paper-trading.md`.
- [ ] Tests: ledger invariants, restart recovery, no real-money path.

## Phase 13 — Dashboard

- [ ] Market Intelligence page.
- [ ] Classical ML Lab.
- [ ] Neural Network Lab.
- [ ] RL Lab.
- [ ] Strategy Intelligence by regime.
- [ ] Autonomous Learning (candidate/champion/rollback/drift).
- [ ] Backtesting (equity, DD, Monte Carlo, walk-forward).
- [ ] Paper Trading.
- [ ] System Monitoring (RAM/CPU/storage/jobs/data quality).
- [ ] Every number comes from an API/artifact; no fake metrics.
- [ ] Frontend tests and build.

## Phase 14 — Mac mini optimisation

- [ ] RAM budget manager and job caps.
- [ ] MPS benchmark and fallback policy.
- [ ] Torch thread pinning and memory profiling.
- [ ] Chunked dataset loading.
- [ ] Benchmark table: LightGBM, GRU, TCN, small Transformer, PPO.
- [ ] Cancellation/resume under memory pressure.
- [ ] `docs/mac-mini-optimization.md`.

## Phase 15 — Acceptance

- [ ] All 15 required documents complete and accurate.
- [ ] Full backend/frontend tests, lint, types, build, e2e.
- [ ] End-to-end: import → validate → dataset → train → evaluate → promote/
      reject → paper → monitor → rollback.
- [ ] Acceptance checklist met or each gap documented.
- [ ] Final report separates implemented / trained / validated / profitable.

## Cross-cutting requirements

- [x] Local, zero-cost, no external/local LLM, no paid API.
- [x] CPU fallback; MPS optional.
- [x] Realistic paper broker with spread, commission, slippage, SL/TP.
- [x] Walk-forward, cost stress and untouched forward test.
- [x] Independent risk engine with daily loss/notional/position limits.
- [ ] Independent risk engine with weekly/DD/consecutive-loss/spread blocks.
- [x] Public repository, CI, branch protection, secret scanning, CodeQL.
- [ ] License choice (owner decision; currently none).
- [ ] Real-money execution: **not a goal**; must stay disabled.

## Unavailable by design

- [ ] DeepLOB / L2 order-book models — **BLOCKED**: no genuine L2 data.
- [ ] Order-book imbalance, absorption, price impact — **BLOCKED**: same.
- [ ] Local/external LLM — **not used** by policy.
- [ ] Cloud-hosted neural networks — **not used**.
- [ ] Paid market data — **not used**.
