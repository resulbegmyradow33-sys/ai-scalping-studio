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
- [x] Statistical outlier detection and cross-source reconciliation.
- [ ] Tick table layer for querying ticks without pre-aggregation.
- [ ] Data-quality report in the dashboard.
- [x] Per-symbol sessions enforced in the simulator.
- [x] Chunked CSV parse; frame still materialised (a 4 GB file still needs ~6 GB).

## Phase 3 — Features and datasets

- [x] Feature registry: seven groups covering all 59 causal columns.
- [x] Canonical-order selection and `select_matrix`.
- [x] Chronological train/validation/test split.
- [x] Purge ≥ label horizon and configurable embargo.
- [x] `split == -1` audit rows, checksum and manifest.
- [x] `dataset build` CLI.
- [x] Leakage/label-alignment/CLI tests.
- [x] Feature availability rules (e.g. signed flow only with tick volumes).
- [ ] H4/M30 multi-timeframe alignment tests.
- [ ] Dataset id wired into every trainer/run config.
- [x] Persisted array format for large datasets (memmap/parquet).
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
- [x] API endpoint and Market Intelligence UI.
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
- [x] Purged time-series CV with embargo.
- [x] Reliability curve, ECE and guarded MCE.
- [ ] Calibration thresholds / probability rejection gate.
- [ ] Feature importance and ablation.
- [x] Simulator/cost-stress trading evaluation for a saved classifier.
- [x] Regime breakdown of the trading evaluation (as-of join, partition identity tested).
- [x] Model registry metadata in SQLite (Phase 10).
- [x] `docs/classical-ml.md`.

## Phase 6 — Neural network framework

- [x] Windowed tensor dataset with train-only normalisation.
- [x] PyTorch trainer: MPS when reliable, CPU fallback.
- [x] Gradient clipping, early stopping, checkpoints, resume, cancellation.
- [x] RAM guard and one-job policy.
- [x] MLP, deep MLP, residual MLP.
- [x] Vanilla RNN, LSTM, GRU.
- [x] 1D CNN, dilated CNN, CNN-LSTM.
- [x] Causal TCN with configurable receptive field.
- [ ] PARTIAL: shared evaluator + rule baseline done; classical ML comparison and simulator/cost gate still open.
- [x] Checkpoint round-trip and causality tests.
- [x] `docs/neural-network-research.md`.

## Phase 7 — Transformers

- [x] Transformer encoder.
- [x] Causal temporal Transformer.
- [x] PatchTST-style patching.
- [x] iTransformer-style variate attention.
- [ ] TFT only if smaller models justify it.
- [x] Config guard (d_model/layers/heads/sequence/batch).
- [ ] PARTIAL: parameter-count smoke test done; measured benchmark pending.
- [ ] Same evaluator/registry.
- [x] `docs/transformer-models.md`.

## Phase 8 — Reinforcement learning expansion

- [x] PPO with Gymnasium environment.
- [x] Behavior-cloning warm start.
- [x] VecNormalize, random starts, checkpoints, pause/resume.
- [x] Rule, hold and random baselines.
- [ ] PARTIAL: flow-gate mask + `masked_action` landed (#76); risk/account gates are
      not in the mask and nothing wires the mask into training or eval yet.
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

- [x] Out-of-fold stacking (reuses the purged CV fold geometry).
- [x] Weighted averaging and voting (soft, weighted, hard with tie → NO_TRADE).
- [ ] Regime-dependent model selection.
- [ ] PARTIAL: Brier/log loss/ECE/MCE var; temperature scaling yok.
- [x] Brier/log loss/calibration curves.
- [x] Ensemble disagreement score (mean pairwise total variation).
- [ ] Out-of-distribution/unknown regime detection.
- [x] Decision engine: BUY/SELL/HOLD/NO TRADE (ordered gates, reason required).
- [ ] PARTIAL: `Decision` kaydı model/probability/disagreement/regime/reason taşıyor; kalıcı log yok.
- [ ] PARTIAL: karşılaştırma var; 'daha basit olanı tut' otomatik değil.
- [ ] `docs/ensemble-decision.md`.

## Phase 10 — Autonomous learning and registry

- [ ] PARTIAL: datasets/models/promotions/jobs var; experiments, backtests,
      paper runs ve drift tabloları yok.
- [x] Dataset versioning and checksums (dataset_id + digest).
- [x] Model versioning, champion pointer and rollback (scope'lu).
- [ ] Candidate → champion gate (trades, OOS, cost stress, regime coverage).
- [ ] Drift detection: features, predictions, calibration.
- [ ] Configurable scheduler with pause/resume/cancel.
- [ ] One heavy job/RAM budget enforcement.
- [ ] Failed training cannot stop paper replay.
- [ ] Migrate nightly rule/RL jobs to the scheduler.
- [ ] `docs/local-learning-engine.md`.
- [ ] Tests: promote, reject, rollback, drift, cancel.

## Phase 11 — Risk and backtesting

- [x] Max weekly loss (Monday 17:00 UTC anchor; default off).
- [x] Max drawdown kill switch (peak-to-equity; force-closes; default off).
- [x] Consecutive-loss pause (blocks new entries, never CLOSE; default off).
- [ ] PARTIAL: max spread points landed; no slippage cap and no rejected-order model.
- [ ] Data-quality trade block.
- [x] Emergency kill switch (equity below a fraction of initial balance; default off).
- [x] Risk limits cannot be bypassed by a model (frozen at construction, no override).
- [ ] Purged CV with embargo integrated in evaluation.
- [x] Monte Carlo trade-sequence analysis (shuffle + block bootstrap, ruin, DD).
- [x] Parameter sensitivity (one-axis spread/slippage/commission grid).
- [ ] Ablation runner.
- [ ] Regime-specific reports.
- [ ] Execution delay and rejected-order assumptions.
- [ ] Partial fills only where data supports them.
- [x] `docs/risk-management.md`, `docs/backtesting-methodology.md`.
- [x] Tests: kill switch, embargo, Monte Carlo determinism.

## Phase 12 — Paper trading

- [ ] Broker adapter interface.
- [x] Persistent paper ledger (SQLite; sessions, signals, orders, fills, trades, equity).
- [x] Track signals, orders, fills, spread, slippage, commission, P&L,
      model/strategy versions and rejection reasons.
- [x] Restart-safe paper session runner (replay + state verification, no inferred state).
- [x] MT5 demo adapter skeleton, disabled, no credentials (every op raises).
- [x] API: account, positions, orders, history (`/api/paper`).
- [x] `docs/paper-trading.md`.
- [x] Tests: ledger invariants, restart recovery, no real-money path.

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


---

## Phase 16 — Autonomous strategy discovery (CORE OBJECTIVE)

- [ ] Confirm existing strategies are optional baselines only, not required
      owner-supplied rules or a fixed search menu.
- [ ] Define a safe, versioned strategy DSL/AST; generated strategies cannot
      execute arbitrary Python or shell code.
- [ ] Build candidate generation from reusable causal primitives, conditions,
      lookback windows, entry/exit/holding logic and NO TRADE.
- [ ] Implement bounded evolutionary/genetic or grammar-based search; add
      Bayesian/random search only where justified.
- [ ] Add candidate mutation, crossover, deduplication, complexity limits,
      reproducible seeds, experiment budgets and cancellation.
- [ ] Connect candidate evaluation to realistic simulator costs, risk controls,
      walk-forward validation and untouched OOS data.
- [ ] Track trial count and control data-snooping/multiple-testing risk.
- [ ] Evaluate stability across symbols, timeframes, regimes and cost stress.
- [ ] Store full candidate lineage, dataset/code/config hashes and rejection
      reasons in the registry.
- [ ] Compare candidates with simple baselines and champion; allow no candidate
      to win and allow NO TRADE.
- [ ] Tests: no look-ahead, deterministic replay, DSL safety, overfit rejection,
      candidate lineage, no-trade selection and cost sensitivity.

## Phase 17 — Knowledge ingestion and hypothesis research

- [ ] Local import for PDF, TXT, Markdown, HTML and DOCX where supported;
      CSV/JSON/Parquet remain market-data paths.
- [ ] Preserve originals, checksums, source metadata, parser versions and
      page/section references.
- [ ] Extract text/tables and report OCR/scanned-document limitations.
- [ ] Build a local searchable document index with source citations.
- [ ] Classify claims, concepts, examples, opinions and empirical observations.
- [ ] Convert document ideas into structured, falsifiable research hypotheses.
- [ ] Link each hypothesis to source passages and experiment outcomes.
- [ ] Route hypotheses to strategy discovery; never treat prose as proven alpha.
- [ ] Ingest completed backtests, paper logs and broker exports as feedback.
- [ ] Tests: parser errors, provenance, source traceability, duplicate imports,
      hypothesis schema and no direct document-to-order path.
- [ ] Explicit capability note: no mandatory LLM; document extraction is not
      equivalent to training a language model or human-level book comprehension.

## Phase 18 — Autonomous Learning Orchestrator and closed-loop operation

- [ ] Detect new datasets/documents and completed paper sessions.
- [ ] Route imports to validation, indexing, dataset and hypothesis pipelines.
- [ ] Plan and queue eligible research, training, evaluation and drift jobs.
- [ ] Persist jobs; support pause/resume/cancel/retry/crash recovery.
- [ ] Enforce M2 RAM/CPU/disk budgets and one heavy job at a time.
- [ ] Keep paper execution isolated from training failures.
- [ ] Connect strategy discovery, ML/DL/RL, model registry, champion gate and
      paper-trading feedback into one auditable lifecycle.
- [ ] Log why every automated job was started, skipped, promoted or rejected.
- [ ] Provide owner controls: pause, safe mode, disable research and rollback.
- [ ] Tests: new-data trigger, job recovery, resource refusal, paper isolation,
      promotion/rejection/rollback and full closed-loop integration.

## Phase 19 — Autonomous paper-trading acceptance

- [ ] Run the system without owner-authored entry/exit rules.
- [ ] Demonstrate autonomous BUY/SELL/HOLD/CLOSE/REDUCE/NO TRADE decisions.
- [ ] Complete historical, walk-forward, untouched OOS and cost-stress gates.
- [ ] Complete a sustained paper-trading trial with immutable trade attribution.
- [ ] Demonstrate learning from matured outcomes without data leakage.
- [ ] Demonstrate champion protection and automatic rollback.
- [ ] Keep real-money execution disabled; optional MT5 demo remains explicitly
      configured and disabled by default.
- [ ] Final report distinguishes implemented, tested, trained, validated,
      paper-traded and profitable; no profitability guarantee.

## New project-wide acceptance rule

- [ ] The system can import new data and research material, independently
      generate strategy candidates, test them, reject overfit candidates,
      choose NO TRADE, paper trade without a human-selected strategy, analyse
      outcomes, and run another validated learning cycle.
- [ ] No named strategy, indicator recipe or trading methodology supplied by
      the owner is required for the core system to function.
- [ ] All autonomy remains bounded by immutable risk controls and resource
      limits.
