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
- [x] H4/M30 multi-timeframe alignment tests (#133): a higher-timeframe value
      is visible only after its bucket closes; chained resampling is pinned.
      M30/H4 are data-layer timeframes, not feature inputs yet.
- [ ] Dataset id wired into every trainer/run config.
- [x] Persisted array format for large datasets (memmap/parquet).
- [ ] Feature ablation runner.
- [ ] PARTIAL: `log_ret_3` is a documented read-only alias for the log-return
      column and the naming is pinned by tests (#133); consecutive-candle
      features do not exist.
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
- [x] Shared evaluator against classical ML and the rule.
- [ ] PARTIAL: simulator/cost-stress gate for network predictions (the
      evaluator is prediction-only; `ml evaluate` has the trading path).
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
- [ ] PARTIAL: mask now covers flow/signal, session, daily/weekly loss, drawdown,
      kill switches, consecutive losses, spread, ATR and lot-size gates, and
      `run_backtest(mask_actions=True)` / `backtest --mask-actions` enforce it in
      evaluation. Training still uses plain PPO (no MaskablePPO dependency).
- [ ] REDUCE position action.
- [ ] SAC evaluation.
- [ ] TD3 evaluation.
- [ ] DQN evaluation.
- [ ] A2C baseline.
- [ ] Seed ensembles / policy averaging.
- [ ] Execution-delay assumption in the environment.
- [ ] Reward ablation report.
- [x] `docs/reinforcement-learning.md`.

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

- [ ] PARTIAL: SQLite schema — datasets/models/promotions/jobs var; experiments,
      backtests ve paper runs tabloları yok.
- [x] Dataset versioning and checksums (dataset_id + digest).
- [x] Model versioning, champion pointer and rollback (scope'lu).
- [ ] PARTIAL: aday → şampiyon kapısı fail-closed promotion policy (#117) ve
      one-shot locked holdout (#121) ile implemented+tested; ancak hiçbir komut
      discovery çalışmasından evidence bundle yazmıyor (#136 açık), dolayısıyla
      kapı uçtan uca çalıştırılamıyor ve runtime'da çağıran yok.
- [x] Drift detection: features (PSI/KS), predictions and calibration (ECE).
- [ ] PARTIAL: konfigüre edilebilir scheduler — kalıcı kuyruk, pause/resume/cancel
      ve daemon döngüsü var (#118); tetikleyiciler yalnızca dosya ve matured
      paper-outcome (#122), veri/doküman/araştırma tetikleyicileri yok.
- [ ] PARTIAL: bir ağır iş/RAM bütçesi — NN/ML job slot ve RAM guard var; PPO
      için global slot yok.
- [ ] Failed training cannot stop paper replay.
- [ ] Migrate nightly rule/RL jobs to the scheduler.
- [x] `docs/local-learning-engine.md`.
- [x] Tests: promote, reject, rollback, drift, cancel.

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
- [x] Classical ML Lab (`/labs/ml`, real API only).
- [ ] Neural Network Lab.
- [ ] RL Lab.
- [ ] Strategy Intelligence by regime.
- [ ] Autonomous Learning (candidate/champion/rollback/drift).
- [ ] Backtesting (equity, DD, Monte Carlo, walk-forward).
- [x] Paper Trading page over the real `/api/paper` ledger (#128); read-only
      except closing a running session, every number from a response.
- [ ] System Monitoring (RAM/CPU/storage/jobs/data quality).
- [ ] PARTIAL: read-only autonomy status API exists (#134: `/api/autonomy`
      status/jobs/events/champions, all GET, missing store = `configured:false`);
      no autonomy dashboard page consumes it yet.
- [ ] PARTIAL: ML Lab ve Paper Trading sayfaları gerçek API kullanıyor; diğer
      lab ve autonomy sayfaları henüz yok. Test sayısı burada yazılmaz; CI
      sonucu için STATUS.md'ye bakın.

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
- [x] Independent risk engine with weekly/DD/consecutive-loss/spread blocks;
      #135 proves a spec, a raw action and a job payload all hit the same frozen
      `RiskEngine`. Slippage cap and rejected-order model stay open in Phase 11.
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

- [x] Confirm existing strategies are optional baselines only, not required
      owner-supplied rules or a fixed search menu (spec §28).
- [x] Define a safe, versioned strategy DSL/AST; generated strategies cannot
      execute arbitrary Python or shell code.
- [x] Build candidate generation from reusable causal primitives, conditions,
      lookback windows, entry/exit/holding logic and NO TRADE.
- [x] Bounded evolutionary search is connected to the DSL evaluator and
      end-to-end discovery command (#104); it remains validation-slice only.
- [x] Add candidate mutation, crossover, deduplication, complexity limits,
      reproducible seeds, experiment budgets and cancellation.
- [ ] PARTIAL: simulator fills/costs and risk controls are connected through
      DslEvaluator (#102/#104). A chronological fold evidence bundle (#114), a
      one-shot locked holdout (#121) and a bundle reproducer (#130) exist, but
      `discovery run` neither builds an evidence bundle nor spends a holdout;
      the producer is issue #136.
- [ ] PARTIAL: trial counts and caps are recorded per search; a data-snooping
      correction/penalty policy is still missing (only attempt caps apply).
- [ ] Evaluate stability across symbols, timeframes, regimes and cost stress.
- [x] Store full candidate lineage, dataset/code/config hashes and rejection
      reasons in the registry.
- [x] Compare candidates with hold and ret_5 on the same validation slice;
      allow no_trade / no_change outcomes.
- [x] Replay a stored evidence bundle from its identity and refuse settings,
      code-version, dataset or spec drift (#130).
- [ ] PARTIAL: a fail-closed `PROMOTION_V1` policy compares candidate and
      incumbent on identical evidence (#117), but no runtime path calls it and
      no command writes the bundle it reads (#136).
- [ ] PARTIAL: no-look-ahead, determinizm, lineage ve DSL safety testleri var;
      overfit reddi, no-trade seçimi ve cost sensitivity evaluator diliminde.

## Phase 17 — Knowledge ingestion and hypothesis research

- [ ] PARTIAL: TXT/MD/HTML/DOCX var; PDF kasten desteklenmiyor (opsiyonel parser).
- [x] Preserve originals, checksums, source metadata, parser versions and
      page/section references.
- [ ] PARTIAL: metin ve DOCX paragrafları var; tablo yapısı PR #119'da açık
      (henüz merge değil), OCR yok.
- [x] Build a local searchable document index with source citations (FTS5/LIKE).
- [x] Classify claims, concepts, examples, opinions and empirical observations
      (deterministic heuristic triage, #107; not semantic reading).
- [x] Deterministic hypothesis drafts from sentence patterns with source refs
      (#106); this is not semantic book understanding and not tested alpha.
- [x] Link each hypothesis to source passages and experiment outcomes.
- [x] Route source-linked hypotheses to audited experiment proposals (#123,
      provenance hardened in #132). Prose cannot create a strategy or an order.
- [ ] PARTIAL: proposals are stored with an approval/audit trail; nothing
      converts an approved proposal into a discovery run automatically.
- [ ] Ingest completed backtests, paper logs and broker exports as feedback.
- [x] Tests: parser errors, provenance, source traceability, duplicate imports,
      hypothesis schema and no direct document-to-order path.
- [x] Explicit capability note: no mandatory LLM; document extraction is not
      equivalent to training a language model or human-level book comprehension.

## Phase 18 — Autonomous Learning Orchestrator and closed-loop operation

- [ ] PARTIAL: dosya tetikleyicisi (size+mtime+sha256, kalıcı cursor) ve matured
      paper-outcome trigger'ı (#122, yalnızca proposal üretir) var; veri ve
      doküman yönlendirmesi yok.
- [ ] PARTIAL: ingestion/index pipeline'ları hazır ve dosya tetikleyicisi
      `knowledge.ingest` işini kuyruğa alabiliyor; dataset build ve discovery
      otomatik tetiklenmiyor.
- [x] Production handlers registered: knowledge ingest, dataset build,
      discovery, paper advance and matured-outcome analysis (#115).
- [x] Persist jobs; support pause/resume/cancel/retry/crash recovery.
- [ ] PARTIAL: measured guard (RSS/load/disk, fail-closed) merged in #112; the
      daemon (#118) defaults to the `m2_8gb` profile, but the plain `Orchestrator`
      still defaults to `AllowAllGuard`, so callers can bypass measurement.
- [ ] Keep paper execution isolated from training failures.
- [ ] PARTIAL: discovery, evidence bundle (#114), fail-closed promotion policy
      (#117), one-shot holdout (#121), champion governance, paper runner and
      matured outcomes exist as separate pieces; one auditable lifecycle and
      evidence-only champion enforcement are still missing. Nothing persists a
      bundle from a run yet (#136).
- [x] Log why every automated job was started, skipped, promoted or rejected
      (append-only event tablosu).
- [ ] PARTIAL: pause/safe mode/disable research var; champion rollback discovery
      arşivinde var (#109), orchestrator seviyesinde rollback kontrolü yok.
- [ ] PARTIAL: trigger, recovery, resource refusal ve kontrol testleri var;
      paper isolation ve uçtan uca döngü yok.

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


## Current autonomous priority update — 2026-11-24 (second refresh)

Verified main snapshot: `693f8a37ed5f5fdfbabb7bbbed0bd468468c9875`.
This section supersedes stale status statements above where they conflict with
the current merged mainline. Detailed evidence and limitations are in
`docs/current-state-addendum-2026-11-24.md`.

1. **Discovery is a validation-slice experiment, not promotion-grade selection.**
   #104 is merged; candidates are searched/evaluated with simulator costs and
   compared with hold/ret_5. `discovery run` builds no evidence bundle and never
   reads the test split.
2. **Promotion policy is fail-closed and tested, but unreachable at runtime.**
   #117 `PROMOTION_V1` refuses an empty policy, a force flag, and any bundle
   without a locked OOS section; `require_locked_oos` cannot be disabled on a
   production policy. No merged command or handler calls it.
3. **Locked holdout exists; no real one has been spent.** #121 seals a holdout
   by `(dataset_id, dataset_checksum)`, grants one single-use token off an
   accepted decision, and writes an immutable result. Spending it requires the
   missing evidence producer (#136).
4. **Reproduction has a consumer but no producer.** #130 replays a stored bundle
   and returns a typed mismatch on any drift; nothing writes the bundle or the
   dataset artifact it needs (#136 open).
5. **Knowledge extraction is heuristic; routing is audited.** #106/#107 merged,
   #123 routes hypotheses to experiment proposals (#132 hardens provenance).
   Table structure is open PR #119. Proposals do not become discovery runs.
6. **Orchestration is assembled, not one pipeline.** Queue/events/controls, five
   handlers (#115), measured guard (#112), daemon with `m2_8gb` default (#118),
   paper-outcome trigger (#122) and integration tests (#129) exist; the bare
   `Orchestrator` still defaults to `AllowAllGuard`.
7. **Paper has a page and a ledger, not a governed loop.** #103/#116/#128 give a
   DSL runner, append-only continuation, matured outcomes and a `/paper` page;
   `paper.advance` still uses the slice runner and enforces no promotion
   provenance.
8. **Risk is non-bypassable in tests.** #135 proves the DSL spec, a raw action
   and a job payload all reach the same frozen `RiskEngine`. PPO training is
   still plain PPO (#108 masks evaluation only).
9. **Main is not green.** `tests/test_api_autonomy.py` calls a removed
   `DiscoveryArchive.promote`; 4 tests fail on `6113197`/current tree. The newest
   green main CI run is `45422130310` at `2bbb312`.

### Next work, strict order

- [ ] Persist the candidate evidence bundle and dataset artifact from a discovery
      run and invoke `evaluate_promotion`/`promote_with_evidence` (issue #136).
- [ ] Fix `test_api_autonomy.py` so main is green again.
- [ ] Enforce promotion provenance in `paper.advance`: only a sealed accepted
      decision may select the champion; otherwise auditable NO TRADE.
- [ ] Record every candidate attempt and add multiple-testing/data-snooping
      controls, complexity penalties and uncertainty reporting inside the bundle.
- [ ] Convert an approved experiment proposal into a discovery run without
      letting prose become a strategy or an order.
- [ ] Prove the complete loop on real XAUUSD data within measured M2 8 GB limits.

Do not mark an item complete because a module, issue, PR or unit test exists.
Require code integration, regression tests and reproducible acceptance evidence.
