# Master task board

One-by-one breakdown of `docs/requirements/master-prompt.md`. A box is checked
only when the task is **implemented and tested**. Partial work stays unchecked
with a `PARTIAL` note. GitHub phase issues mirror these lists.

Legend: `[x]` done · `[ ]` todo · `PARTIAL` some code exists but the task is not
met end to end · `BLOCKED` missing data/dependency/decision.

## Snapshot — 2026-11-25, main `54eef5e`

Verified `main`: `54eef5e7a25854c78f2bef2d26e4c42d50f3ab5e` (#220). CI run
`45627617157` is green at this SHA: backend `2,435 passed, 8 skipped`, lint
clean, mypy `153 source files`; frontend `287 passed` in 34 files and build
compiled; CodeQL `45627617898` success. Test counts are not repeated per phase
below — use that run or a newer CI run, never an estimate.

The autonomous track added more than 20 new modules since the 2026-11-24
snapshot (`693f8a3`): 14 new backend source modules, 24+ new test modules and 10
frontend source files. A module plus a passing unit test is `implemented`, not
`validated` and not `paper-traded` (`AGENTS.md`, "Evidence vocabulary").

Present in code at this commit: the evidence-bundle producer (#159), fail-closed
`PROMOTION_V1` (#117), one-shot holdout (#121), bundle reproduction
(#130/#169), tradable-best candidate gate (#174), provenance-aware paper
champion with the continuous runner (#156/#164), paper-completion trigger
(#189), audited local controls (#175), fail-closed legacy UI switch (#177),
slippage cap and data-quality entry block (#186), M30/H4 context (#187),
feature ablation (#188), OOF regime weights (#190), optional PDF extraction
(#215), provenance fail-closed through dataset → evidence → promotion (#216),
read-only discovery archive API (#217), evidence-champion paper scheduler
(#218), M30/H4 resample boundary tests (#219) and the runtime promotion command
with one-shot owner approval (#220).

### Current blockers

Canonical list and per-area evidence: `STATUS.md`.

1. **No validated candidate.** The locked holdout is unspent, no bundle carries
   a real `locked_oos` section, and `PROMOTION_V1` rejects every bundle. The
   merged real-data H1 attempt (#191) ended `no_trade`; NO TRADE is a valid
   result, not validation.
2. **The holdout spend is unexercised.** `discovery promote` (#220) is the
   runtime call site and is tested, including the second-spend refusal, but no
   owner-approved spend has run against real data. This is the only PARTIAL
   left in the promotion path; it needs owner approval, not code.
3. **No governed paper champion and no paper run.** `--schedule-paper` (#218)
   enqueues `paper.advance` for an evidence champion on a new decidable bar, but
   no evidence champion exists, so every scope is refused and no session runs.
4. **No model edge.** Classical models are at or below the prior baseline; PPO
   is experimental and mostly degenerate; no committed neural checkpoint.
5. **No target-hardware acceptance.** No measured full loop on the M2 8 GB
   machine; the bench paper stage failed with `BarGapError`; walk-forward is
   unmeasured.
6. **Documentation drift.** Snapshots below lag the code in places; this
   snapshot and the corrected boxes supersede them.

Closed by the #215–#220 merges: PDF is now an optional extra with page
references, provenance is fail-closed, the archive has a read-only API, the
paper scheduler exists, M30/H4 resample boundaries are pinned by tests, and the
holdout/promotion command exists. What is not closed is the evidence itself:
holdout spend, champion, paper run.

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
- [x] Tick table layer for querying ticks without pre-aggregation (`TickStore`).
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
      Resample bucket alignment, left-closed boundaries and stored-derivation
      parity are covered in `test_store.py` (#219).
- [x] Opt-in M30/H4 higher-timeframe context features with closed-bucket
      causality (#187).
- [ ] Dataset id wired into every trainer/run config.
- [x] Persisted array format for large datasets (memmap/parquet).
- [x] Feature ablation runner (#188): reproducible feature-group ablation with
      a recorded run manifest.
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

- [x] Add scikit-learn.
- [x] Logistic regression, random forest, extra trees, HistGradientBoosting.
- [x] LightGBM and XGBoost (lazy imports; macOS needs `libomp`).
- [x] SGDClassifier incremental path (#184).
- [x] Isolation forest anomaly detector (#184).
- [ ] SVM only if timing is practical.
- [x] Direction task (3-class, configurable horizon).
- [x] Train/validation split, purge/embargo and dataset id.
- [x] Brier, log loss, balanced accuracy, per-class metrics, confusion, baseline.
- [x] Flat-zero rows dropped by default with an audited count; `--include-flat` opt-in.
- [ ] Return/volatility/regime/trade-outcome tasks.
- [x] Purged time-series CV with embargo.
- [x] Reliability curve, ECE and guarded MCE.
- [ ] Calibration thresholds / probability rejection gate.
- [x] Feature importance and ablation (#184 importances, #188 ablation runner).
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
- [x] Config guard (hidden/heads/layers/sequence).
- [x] Memory smoke test and measured benchmark (parameters, latency, throughput, RSS).
- [x] Same evaluator.
- [ ] PARTIAL: shared model registry is Phase 10 work.
- [x] `docs/transformer-models.md`.

## Phase 8 — Reinforcement learning expansion

- [x] PPO with Gymnasium environment.
- [x] Behavior-cloning warm start.
- [x] VecNormalize, random starts, checkpoints, pause/resume.
- [x] Hold baseline in the RL run (`baseline_hold_net_profit`).
- [ ] PARTIAL: rule baseline exists in backtest/evaluator; random baseline and
      both in the RL run manifest are open.
- [ ] PARTIAL: flow mask (#76) is merged; full risk/session/spread/ATR/lot mask
      and evaluation wiring is #181 (no #181 commit is on `54eef5e`). Training
      still uses plain PPO (no MaskablePPO dependency).
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
- [x] Regime-dependent model selection: OOF-only regime-dependent member
      weights (#190).
- [x] Brier/log loss/ECE/MCE metrics and optional temperature scaling (#176).
- [x] Brier/log loss/calibration curves.
- [x] Ensemble disagreement score (mean pairwise total variation).
- [ ] Out-of-distribution/unknown regime detection.
- [x] Decision engine: BUY/SELL/HOLD/NO TRADE (ordered gates, reason required).
- [ ] PARTIAL: `Decision` kaydı model/probability/disagreement/regime/reason taşıyor; kalıcı log yok.
- [ ] PARTIAL: karşılaştırma var; 'daha basit olanı tut' otomatik değil.
- [x] `docs/ensemble-decision.md`.

## Phase 10 — Autonomous learning and registry

- [ ] PARTIAL: SQLite schema — datasets/models/promotions/jobs var; experiments,
      backtests ve paper runs tabloları yok.
- [x] Dataset versioning and checksums (dataset_id + digest).
- [x] Model versioning, champion pointer and rollback (scope'lu).
- [x] Aday → şampiyon kapısı: fail-closed promotion policy (#117), one-shot
      locked holdout (#121), evidence bundle + dataset artifact (#159),
      provenance fail-closed (#216) ve runtime `discovery promote` (#220)
      merged. Gerçek holdout harcaması yapılmadı; o ayrı ve owner onaylı bir act.
- [x] Runtime holdout + promotion command: bundle identity check, pre-holdout
      policy, one holdout spend, `PROMOTION_V1` re-evaluation,
      `promote_with_evidence`, archive/champion row and a second-spend refusal
      (#220; `tests/test_discovery_promote.py`). Code merged; **no real spend**
      — the one-shot token is owner-supplied and has not been used.
- [ ] PARTIAL: gerçek locked holdout harcaması ve champion kaydı — kod hazır,
      owner onayı yok. `discovery promote --spend-holdout` yalnızca owner token
      ile çalışır; gerçek veride denenmedi.
- [x] Drift detection: features (PSI/KS), predictions and calibration (ECE).
- [x] Konfigüre edilebilir scheduler: kalıcı kuyruk, pause/resume/cancel ve
      daemon döngüsü (#118); tetikleyiciler veri dosyası (#158), matured
      paper-outcome (#122/#189, proposal only) ve evidence champion için
      `paper.advance` (#218, `--schedule-paper`). Scheduler saat/feed okumaz;
      karar barı store'da kapandıktan sonra tetikler.
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
- [x] Max spread points, slippage cap and data-quality entry block (#186).
- [x] Data-quality trade block (#186).
- [ ] Rejected-order model.
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

- [x] Market Intelligence page (`/intelligence`, `/api/regime` only).
- [x] Classical ML Lab (`/labs/ml`, real API only).
- [ ] Neural Network Lab.
- [ ] RL Lab.
- [x] Strategy Intelligence by regime (#166). The strategy × regime table is
      not exposed by any endpoint yet; the page says so instead of deriving it.
- [x] Autonomous Learning page (`/autonomy`): read-only candidates, champions,
      jobs, events and drift from `/api/autonomy/*`; no control mutations.
- [ ] Backtesting (equity, DD, Monte Carlo, walk-forward).
- [x] Paper Trading page over the real `/api/paper` ledger (#128); read-only
      except closing a running session, every number from a response.
- [ ] System Monitoring (RAM/CPU/storage/jobs/data quality).
- [x] Data quality panel on the Data page (#165).
- [ ] PARTIAL: Neural Network Lab, RL Lab, Backtesting and System Monitoring
      pages do not exist. Test counts are not written here; use the CI run in
      STATUS.md.

## Phase 14 — Mac mini optimisation

- [ ] PARTIAL: measured orchestrator guard (RSS/load/disk, #112) exists; a
      general RAM budget manager for every job kind and PPO does not.
- [ ] MPS benchmark and fallback policy.
- [ ] Torch thread pinning and memory profiling.
- [ ] Chunked dataset loading.
- [ ] Benchmark table: LightGBM, GRU, TCN, small Transformer, PPO.
- [ ] Cancellation/resume under memory pressure.
- [x] `docs/mac-mini-optimization.md` exists; its measurements are partial
      (`docs/evidence/bench-2026-11-24.json`, `m2_acceptance=false`).

## Phase 15 — Acceptance

- [ ] All 15 required documents complete and accurate.
- [ ] Full backend/frontend tests, lint, types, build, e2e.
- [ ] End-to-end: import → validate → dataset → train → evaluate → promote/
      reject → paper → monitor → rollback. All runtime call sites now exist
      (#216 provenance, #218 scheduler, #220 promote), but the chain has not
      been validated: the merged #191 attempt ran the chain on real XAUUSD H1 to
      the promotion gate, recorded NO TRADE and stopped. No holdout spend, no
      promotion, no paper session.
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
      `RiskEngine`. Slippage cap and data-quality block landed in #186; the
      rejected-order model stays open in Phase 11.
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
      DslEvaluator (#102/#104). #159 closes #136: discovery can now persist an
      atomic dataset artifact, evidence bundle and spec sidecar, verify them by
      read-back and reproduce the checksum. The CLI producer exists, but
      discovery does not automatically spend the locked holdout or promote.
- [x] Add multiple-testing selection controls (#149): adjusted screening,
      deflated effect floor with trial/complexity penalty and seeded bootstrap
      CI. The advice is an additional screen, not a promotion decision; IID
      assumptions and limitations remain documented.
- [ ] Evaluate stability across symbols, timeframes, regimes and cost stress.
- [x] Store full candidate lineage, dataset/code/config hashes and rejection
      reasons in the registry.
- [x] Compare candidates with hold and ret_5 on the same validation slice;
      allow no_trade / no_change outcomes.
- [x] Replay a stored evidence bundle from its identity and refuse settings,
      code-version, dataset or spec drift (#130).
- [x] Fail-closed `PROMOTION_V1` policy compares candidate and incumbent on
      identical evidence (#117); #159 supplies the evidence bundle input and
      #220 connects the runtime call site. Promotion is an explicit
      owner-approved act (`discovery promote`), not an automatic scheduler job.
- [ ] PARTIAL: no-look-ahead, determinism, lineage and DSL safety tests exist;
      #169 makes evidence checksums thread-independent and #216 pins provenance.
      The real XAUUSD H1 smoke (#143) correctly chose NO TRADE; it did not
      validate a strategy. #174 (merged) requires a tradable best candidate;
      #191 records a second real-data attempt that stopped at the promotion
      gate.

## Phase 17 — Knowledge ingestion and hypothesis research

- [x] TXT/MD/HTML/DOCX var; PDF optional `pypdf` extra ile (`uv sync --extra
      pdf`), sayfa referanslı (#215). OCR yok.
- [x] Preserve originals, checksums, source metadata, parser versions and
      page/section references.
- [x] Metin, DOCX, tablo yapısı (#119) ve opsiyonel PDF sayfa çıkarımı (#215)
      var; OCR yok ve core kurulum `pypdf` içe aktarmaz.
- [x] Build a local searchable document index with source citations (FTS5/LIKE).
- [x] Classify claims, concepts, examples, opinions and empirical observations
      (deterministic heuristic triage, #107; not semantic reading).
- [x] Deterministic hypothesis drafts from sentence patterns with source refs
      (#106); this is not semantic book understanding and not tested alpha.
- [x] Link each hypothesis to source passages and experiment outcomes.
- [x] Route source-linked hypotheses to audited experiment proposals (#123,
      provenance hardened in #132). Prose cannot create a strategy or an order.
- [ ] PARTIAL: #153 adds an audited bridge: only approved, scope-checked,
      holdout-safe and deduplicated proposals may enqueue `discovery.run`.
      It does not approve proposals, create strategies from prose or place
      orders. Completed backtests/paper logs are not yet a full feedback corpus.
- [ ] Ingest completed backtests, paper logs and broker exports as feedback.
- [x] Tests: parser errors, provenance, source traceability, duplicate imports,
      hypothesis schema and no direct document-to-order path.
- [x] Explicit capability note: no mandatory LLM; document extraction is not
      equivalent to training a language model or human-level book comprehension.

## Phase 18 — Autonomous Learning Orchestrator and closed-loop operation

- [ ] PARTIAL: file triggers (size+mtime+sha256, durable cursor) and the
      matured paper-outcome trigger (#122, proposal only) exist. #153 adds the
      approved-proposal → `discovery.run` queue bridge; this is not automatic
      hypothesis approval or promotion.
- [ ] PARTIAL: ingestion/index and dataset/discovery handlers are registered
      (#115); data triggers can enqueue dataset builds, and approved proposals
      can enqueue discovery. Automatic data-quality approval and a full
      unattended research lifecycle remain open.
- [x] Production handlers registered: knowledge ingest, dataset build,
      discovery, paper advance and matured-outcome analysis (#115).
- [x] Persist jobs; support pause/resume/cancel/retry/crash recovery.
- [ ] PARTIAL: measured guard (RSS/load/disk, fail-closed) merged in #112; the
      daemon (#118) defaults to the `m2_8gb` profile, but the plain `Orchestrator`
      still defaults to `AllowAllGuard`, so callers can bypass measurement.
- [ ] Keep paper execution isolated from training failures.
- [ ] PARTIAL: #159 persists evidence bundles and dataset artifacts; #117
      supplies the fail-closed promotion policy; #121 protects one-shot holdout;
      #156 routes `paper.advance` through the continuous runner and
      evidence-governed champion; #216 adds provenance; #218 adds the scheduler;
      #220 adds the owner-approved promotion command. #151 proves only synthetic
      wiring. A complete real-market OOS → reviewed promotion → sustained paper
      → matured-feedback cycle is still unproven.
- [x] Log why every automated job was started, skipped, promoted or rejected
      (append-only event tablosu).
- [x] Persisted pause/safe-mode/disable-research controls exist in the
      orchestrator core. #175 adds an audited local HTTP control surface and
      #177 adds a fail-closed legacy UI fallback switch; both are merged.
- [ ] PARTIAL: trigger, recovery, resource refusal ve kontrol testleri var;
      `paper.advance` scheduler'ı (#218) var, gerçek paper session yok. Paper
      isolation ve uçtan uca döngü kanıtlanmadı.

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
      `docs/final-report-2026-11-25.md` is merged and carries a late addendum
      ("runtime chain complete, not validated"); the merged #191 attempt record
      is a stage-by-stage real-data attempt, not final acceptance.

## New project-wide acceptance rule

- [ ] The system can import new data and research material, independently
      generate strategy candidates, test them, reject overfit candidates,
      choose NO TRADE, paper trade without a human-selected strategy, analyse
      outcomes, and run another validated learning cycle.
- [ ] No named strategy, indicator recipe or trading methodology supplied by
      the owner is required for the core system to function.
- [ ] All autonomy remains bounded by immutable risk controls and resource
      limits.


## Historical priority update — 2026-11-24 (superseded by the snapshot above)

Verified main at that time: `693f8a37ed5f5fdfbabb7bbbed0bd468468c9875`. Kept for
traceability only. Where it conflicts with the 2026-11-25 snapshot, the snapshot
wins.

1. **Discovery is a validation-slice experiment, not promotion-grade selection.**
   Still true: `discovery run` builds no evidence bundle and never reads the
   test split.
2. **Promotion policy is fail-closed and tested, but unreachable at runtime.**
   Still true, and it is blocker 2 in the current list.
3. **Locked holdout exists; no real one has been spent.** Still true. The
   "missing evidence producer" clause is stale: #159 merged the producer.
4. **Reproduction has a consumer but no producer.** Stale. #159 added the
   producer; #130/#169 replay and compare it.
5. **Knowledge extraction is heuristic; routing is audited.** Table structure
   is merged (#119); PDF is still unsupported.
6. **Orchestration is assembled, not one pipeline.** Still true: five handlers,
   no evidence/promotion/holdout job, no `paper.advance` trigger.
7. **Paper has a page and a ledger, not a governed loop.** Partly stale:
   `paper.advance` now uses the continuous runner and refuses non-evidence
   champions (#156/#164). What is missing is a champion and a scheduler.
8. **Risk is non-bypassable in tests.** Still true; #186 added the slippage cap
   and data-quality block.
9. **Main is not green.** Stale. `test_api_autonomy.py` passes at `50fbca6`
   (CI run `45550807353`, backend `2,320 passed, 3 skipped`).

### Next work, strict order (updated 2026-11-25, late)

- [x] Make provenance fail closed through dataset → evidence → promotion:
      `source`/`synthetic` carried end to end; `PROMOTION_V1` refuses unknown
      and synthetic data (#216).
- [x] Add the runtime holdout + promotion command (CLI and/or handler) that
      spends one holdout, re-evaluates `PROMOTION_V1` and calls
      `promote_with_evidence`; refuse a second spend (#220). Code complete; the
      first real spend still needs owner approval.
- [ ] Record a real candidate experiment and a deliberately overfit negative
      control; publish the artifacts whether the decision is NO TRADE or a
      rejection. **Still open.**
- [x] Add a `paper.advance` trigger/scheduler so the daemon, not a human,
      creates advance jobs on new bars (#218, `--schedule-paper`). The scheduler
      reads stored bars, never a live feed; a gap still reaches the strict
      runner unchanged.
- [x] Add the optional pinned offline PDF parser (#215): `pypdf` is an optional
      extra and each page is a section with a page reference. Extraction is
      merged.
- [ ] PARTIAL: a recorded PDF → page/section → hypothesis run is not published.
      The parser exists; the end-to-end document-to-hypothesis artifact is open.
- [ ] Prove the complete loop on real XAUUSD data within measured M2 8 GB
      limits, and publish one report that does not promote lower evidence rungs.
      **Still open, and it now requires an owner-approved holdout spend.**

Do not mark an item complete because a module, issue, PR or unit test exists.
Require code integration, regression tests and reproducible acceptance evidence.
