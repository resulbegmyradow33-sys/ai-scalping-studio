# Implementation roadmap

Companion to `docs/architecture-audit.md` and
`docs/requirements/master-prompt.md`. This roadmap is dependency-ordered and
resource-bounded. No phase starts until its predecessors' exit criteria are
met.

## Current execution order — 2026-11-24

The autonomous core is now the critical path. This section takes precedence
when older phase prose suggests adding more model families before the learning
loop is connected.

1. **Evidence before promotion.** Extend discovery from one validation slice to
   chronological walk-forward folds with purge/embargo, an inaccessible locked
   OOS set, realistic costs/risk and stability reporting. Track every attempted
   candidate and address multiple-testing/data-snooping bias. The fold bundle
   (#114), fail-closed policy (#117), one-shot holdout (#121) and reproducer
   (#130) exist; what is missing is a producer that persists the bundle from a
   run (issue #136).
2. **Fail-closed champion governance.** `PROMOTION_V1` (#117) requires
   mandatory evidence, compares the candidate and incumbent on identical
   windows, keeps immutable lineage and rejects missing evidence. It has no
   runtime call site yet; validation-only #109 remains a foundation, not a
   production gate.
3. **Wire autonomy.** Register real handlers in the persistent orchestrator;
   add a scheduler/daemon, new-data/document/paper-completion triggers,
   eligibility/deduplication policies, measured RAM/CPU/disk limits and
   isolation between paper replay and training failures. Handlers (#115),
   guard (#112), daemon (#118) and paper-outcome trigger (#122, proposal only)
   are merged; routing remains manual.
4. **Connect research to discovery.** Preserve document provenance, parse
   supported sources honestly, route source-linked hypotheses to quantitative
   experiments, and never let prose create orders or bypass validation.
5. **Close the paper loop.** Load only a validated DSL champion, run it over
   appended market data, persist immutable decisions/fills/outcomes, wait for
   matured outcomes, analyse them, and enqueue the next eligible experiment.
   Missing/invalid champion means NO TRADE, not legacy fallback.
6. **Accept on target hardware.** Run reproducible end-to-end XAUUSD acceptance
   on the M2 8 GB machine under measured memory/disk budgets. Keep real-money
   execution disabled.
7. **Only then expand models/UI.** Merge the optional benchmark work when
   rebased and green, but defer TFT/SAC/TD3/DQN/Mamba/GNN or dashboard expansion
   unless an experiment justifies it.

### Mainline alignment

- As of `693f8a37ed5f5fdfbabb7bbbed0bd468468c9875`, #105–#118, #120–#124 and
  #128–#135 are merged.
- #107 is deterministic heuristic triage, not semantic understanding.
- #109 is champion history/provider/rollback infrastructure, not a
  promotion-grade OOS/walk-forward gate.
- #117 is a fail-closed promotion policy; #121 is a one-shot locked holdout;
  #130 is a bundle reproducer. None of them has a runtime producer (#136 open).
- #119 (knowledge table structure) is an open PR, not merged.
- `backend/tests/test_api_autonomy.py` fails on current main (removed
  `DiscoveryArchive.promote`); see `STATUS.md` for the CI run ids.
- #105 is optional benchmark infrastructure; do not let model-zoo expansion
  displace autonomous evidence and handler integration.

Working rules for every phase:

1. State the intended change and the files affected.
2. Check compatibility with the existing simulator, card, API and tests.
3. Estimate RAM/CPU and set a single-job resource budget.
4. Implement incrementally.
5. Run backend and frontend tests and lint/type checks.
6. Fix regressions.
7. Update the required documentation.
8. Report what was actually run, what passed, and what remains unproven.

No phase may train more than one heavy model at a time on the 8 GB machine.

## Phase 1 — Audit and documentation (DONE)

Deliverables:

- `docs/requirements/master-prompt.md`
- `docs/architecture-audit.md`
- `docs/implementation-roadmap.md`
- `STATUS.md`

Exit criteria: every requirement has a status; no production code changed;
audit evidence points to real files and tests. **Met.**

## Phase 2 — Data foundation and multi-symbol registry (IN PROGRESS)

Goal: make the data layer capable of the ten required instruments and the full
timeframe ladder, with validation and provenance, before any model is added.

Deliverables:

- `backend/src/scalping/data/registry.py`: `SymbolSpec` (tick size, contract
  size, min/max lot, lot step, commission, sessions, precision, provider
  symbols) and a registry for XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY, AUDUSD,
  NAS100, US30, USOIL, BTCUSD.
- Refactor `ContractSpec` reads to use the registry; keep the gold defaults for
  backward compatibility.
- `backend/src/scalping/data/validate.py`: duplicate timestamps, missing
  minutes, stale gaps, outliers, OHLC consistency (`low <= open/close <= high`),
  non-negative spread/volume, timezone checks; returns a structured report.
- `backend/src/scalping/data/provenance.py`: dataset id, source, checksum, row
  count, first/last bar, synthetic flag, import timestamp; persisted as JSON
  next to the store and exposed by the API.
- JSON bar importer; M30/H4 in `resample` and the CLI.
- `data/status` response extended with validation/provenance.

Affected files: `data/*`, `types.py`, `config.py`, `api/routes.py`,
`api/schemas.py`, `cli.py`, `tests/test_store.py`, new tests.

Dependencies: none beyond current stack. RAM: < 500 MB. Tests: unit for
registry, validation, provenance, JSON import, M30/H4 resampling.

Progress (2026-11-22):

- `data/registry.py`: ten specs, `contract_for`, `TIMEFRAMES` including M30/H4.
- `data/validate.py`: bars/ticks structural report; no repair.
- `data/provenance.py`: checksum + JSON sidecar; existing XAUUSD stores seeded.
- `data/json_provider.py` and `cli data json` / `cli data derive`.
- API: `/api/symbols`, `/api/data/validate`, status with provenance.
- Tests: registry, validation, provenance, JSON provider, API, CLI derive.

Remaining for exit: a real provider path for the eight non-metal symbols,
statistical outlier/cross-source reconciliation, and a tick-table layer.
Exit criteria: all ten specs load; the validator reports real gaps on the
existing stores; the API exposes provenance; existing tests stay green.

## Phase 3 — Feature registry and dataset builder (IN PROGRESS)

Goal: separate feature definitions from the 59-column monolith, support
configurable subsets, and build leakage-safe train/validation/test datasets.

Deliverables:

- `features/registry.py`: named feature groups, required columns, availability
  rules (e.g. signed flow only if tick volumes exist).
- Configurable feature sets (`core`, `price_action`, `volatility`, `flow`,
  `structure`, `multi_timeframe`); no silent fabrication.
- `datasets/builder.py`: chronological split, purge/embargo, label horizon
  handling, dataset manifest with checksums and versions.
- MTF alignment for M30/H4; tests that a truncated frame is bit-identical to the
  full frame (existing leakage test pattern).
- Feature importance/ablation helpers (no model training yet).

Affected files: `features/*`, new `datasets/*`, `features/build.py`,
`strategy/labeling.py`, tests.

Dependencies: Phase 2. RAM: < 1 GB. Tests: leakage, alignment, subset
stability, manifest reproducibility.

Progress (2026-11-22):

- `features/registry.py`: seven groups covering all 59 columns, canonical order,
  `select_matrix`.
- `datasets/builder.py`: chronological train/validation/test, purge >= horizon,
  embargo, `split == -1` audit rows, checksum, manifest.
- `cli dataset build`: outputs a manifest and split counts.
- Tests: group coverage, order, split gaps, label alignment, tail drop,
  deterministic manifest, CLI.

Progress (2026-11-24): #133 pins M30/H4 alignment semantics (a higher-timeframe
value reaches a decision bar only after its bucket closes) and the chained
resampling path. M30/H4 context is now an opt-in feature group (`htf`:
`m30_trend`, `m30_ret`, `h4_trend`, `h4_ret`, `h4_atr_frac`) appended after the
frozen 59 columns; the default matrix still consumes M5/M15 only (`htf_context`
is timeframe-agnostic). `log_ret_3` is a documented read-only alias for the
log-return column.

Remaining: availability rules (flow only with tick volumes), wiring the manifest
id into the trainer/registry, and a persisted array format for large datasets.

Exit criteria: a dataset can be rebuilt from a manifest and matches checksums;
no feature set uses future information.

## Phase 4 — Market regime engine

Goal: an independent regime classifier and a strategy-by-regime report.

Deliverables:

- `regime/features.py`, `regime/rules.py`, `regime/models.py` (K-Means, GMM,
  HMM, RandomForest/HistGradientBoosting; optional small MLP later).
- Regime labels: strong/weak up/down, sideways, high/low vol,
  expansion/contraction, breakout, unusual.
- `regime/evaluate.py`: stability, transition matrix, regime-conditional
  strategy performance, leakage-safe fitting.
- API: `GET /api/regime?...`; frontend Market Intelligence page later.

Dependencies: Phase 3 datasets. RAM: < 1 GB. Tests: deterministic labels on
synthetic regimes, no future leakage, transition sanity.

Exit criteria: regimes are reproducible; every strategy has a
regime-performance table; no claim that a regime predicts price by itself.

## Phase 5 — Classical ML engine

Goal: the first genuine non-RL learning models, with baselines and calibrated
comparison.

Deliverables:

- `ml/models.py`: logistic regression, random forest, extra trees,
  HistGradientBoosting, LightGBM, XGBoost, SGDClassifier, isolation forest;
  SVM only if timing is acceptable.
- Tasks: direction (1/3/5/10/15 bars), return regression, volatility, regime,
  trade outcome, anomaly.
- `ml/train.py`: purged time-series CV, calibration, Brier/log loss,
  feature importance, seeds, class weights.
- `ml/registry.py`: model artifacts + metadata (can start as JSON; SQLite in
  Phase 10).
- Baselines: majority class, momentum rule, buy-and-hold.
- Trading evaluation through the existing simulator, not accuracy alone.

Dependencies: Phase 3. New deps: scikit-learn, lightgbm, xgboost (verify ARM64,
license, memory). RAM: < 1.5 GB. Tests: leakage, reproducibility, serialization,
baseline comparison, calibration.

Exit criteria: at least one classical model beats the majority baseline on
out-of-sample log loss; none is promoted to live trading without the Phase 10
gate.

## Phase 6 — Neural network research framework

Goal: a reusable PyTorch trainer and the lightweight sequence models.

Deliverables:

- `nn/data.py`: windowed tensors from datasets, normalization fitted on train
  only, no shuffling across time for validation.
- `nn/trainer.py`: device selection (MPS if available and reliable, CPU
  fallback), AMP only if stable, gradient clipping, early stopping, checkpoints,
  resume, seed control, RAM guard, cancellation.
- Architectures: MLP/deep/residual MLP; GRU, LSTM; 1D CNN, dilated CNN,
  CNN-LSTM; causal TCN.
- `nn/evaluate.py`: shared metrics, calibration, directional accuracy,
  trading simulation, comparison with Phase 5 baselines.
- `nn/registry.py`: architecture config, weights path, data hash, metrics.

Dependencies: Phase 3; PyTorch already installed. RAM: 1–2 GB per run. Tests:
forward shapes, causality of TCN, checkpoint round-trip, deterministic eval,
overfit a tiny batch.

Exit criteria: at least TCN and GRU train and evaluate end to end on H1 and M15;
no model is described as profitable without a simulator run and cost stress.

## Phase 7 — Transformer research (conditional)

Goal: small, memory-bounded temporal Transformers, only after Phase 6 works.

Deliverables:

- `nn/transformers.py`: encoder, causal temporal encoder, PatchTST-style
  patching, iTransformer-style variate attention; optional TFT only if the
  smaller models justify it.
- Config guard: `d_model <= 64`, `layers <= 2`, `heads <= 4`, sequence <= 128,
  batch <= 128; reject larger configurations on 8 GB.
- Same trainer/evaluator/registry as Phase 6.

Dependencies: Phase 6. RAM: 1.5–3 GB; run alone. Tests: causal mask, shape,
memory smoke, checkpoint.

Exit criteria: a small Transformer completes training and evaluation; results
are reported even if it loses to TCN/GRU. No claim that Transformers are
superior.

## Phase 8 — Reinforcement learning expansion

Goal: audit PPO and test a small, justified set of alternatives.

Deliverables:

- Action masking (flow gate / risk blocks) and optional REDUCE action.
- Seed ensembles and policy averaging.
- SAC/TD3 only if the action space and compute justify it; DQN for discrete
  comparison; A2C as a cheap baseline. Do not run all at once.
- Reward/ablation report: entry penalty, drawdown penalty, cost sensitivity.
- Comparison against rule, classical ML, neural baseline, random, hold.

Dependencies: Phase 6 (shared evaluation). RAM: 1–2 GB. Tests: env contract,
masking, ensemble determinism, smoke train per algorithm.

Exit criteria: every RL result is reported against the rule on at least two
chronological splits and the forward window; RL remains experimental unless it
wins all gates.

## Phase 9 — Ensemble, decision and confidence engine

Goal: combine models only where out-of-sample evidence supports it.

Deliverables:

- `ensemble/stacking.py`: out-of-fold predictions, logistic meta-learner,
  weighted average, voting, regime-dependent selection.
- `confidence/calibration.py`: reliability curves, Brier, log loss, temperature
  scaling, ensemble disagreement, OOD score.
- `decision/engine.py`: BUY/SELL/HOLD/NO TRADE, thresholds, cost-aware veto,
  decision log with model/strategy versions, probabilities, regime, reason.
- API + persistence of decisions.

Dependencies: Phases 4–6. RAM: < 1 GB. Tests: no leakage in OOF, calibration
monotonicity, NO TRADE on low confidence, decision log schema.

Exit criteria: the ensemble is compared with its best member; if it does not
win, the simpler model stays. NO TRADE is exercised by tests.

## Phase 10 — Autonomous learning, registry and scheduler

Goal: turn ad-hoc scripts into a persistent, recoverable learning system.

Deliverables:

- SQLite schema: datasets, dataset_versions, experiments, models,
  model_versions, promotions, backtests, paper_runs, drift_reports, jobs.
- Dataset/model versioning, checksums, champion pointer, rollback.
- Candidate→champion gate: minimum trades, positive OOS expectancy, cost
  stress, regime coverage, beats champion; otherwise rejected.
- Drift detection: feature distribution, prediction distribution, calibration.
- `scheduler/jobs.py`: configurable jobs, pause/resume/cancel, one heavy job,
  RAM budget, retry policy; failed training cannot stop paper replay.
- Migrate the nightly calibration and RL lab onto the scheduler.

Dependencies: Phases 5–9. RAM: < 500 MB (scheduler/SQLite). Tests: promotion,
rejection, rollback, drift trigger, job cancellation, DB migration.

Progress (2026-11-24): the gate is no longer hypothetical. #117 adds a named,
versioned, fail-closed `PROMOTION_V1` policy (no empty policy, no force flag,
locked OOS required); #121 adds the one-shot holdout ledger that can attach the
locked section; #130 replays a stored bundle. The remaining gap is the producer:
no command persists an evidence bundle or dataset artifact from a discovery run
(#136), and `paper.advance` does not consume a promotion decision.

Exit criteria: a candidate can be trained, rejected, promoted and rolled back
with a queryable audit trail; the previous champion is always recoverable.

## Phase 11 — Risk and backtesting hardening

Goal: make the independent risk layer and the evaluation honest enough for
paper trading.

Deliverables:

- Risk: weekly loss cap, max drawdown kill switch, consecutive-loss pause, max
  spread, slippage limit, data-quality block, emergency stop; all outside the
  model.
- Backtest: purged CV with embargo, Monte Carlo trade sequences, parameter
  sensitivity, ablation runner, regime-specific reports, execution-delay and
  rejected-order assumptions; partial fills only where data supports them.
- Report costs and sample size with every metric.

Dependencies: Phase 3; can run in parallel with Phase 5. RAM: < 1 GB. Tests:
kill-switch triggers, embargo, Monte Carlo determinism, sensitivity bounds.

Progress (2026-11-24): #135 adds the Phase 11 non-bypass acceptance suite: a DSL
spec, a raw simulator action and an orchestrator job payload all reach the same
frozen `RiskEngine`, and divergence fails the test. Slippage cap, rejected-order
model and data-quality trade block remain open.

Exit criteria: risk blocks cannot be bypassed by a policy; every reported
strategy has walk-forward, stress and sample-size context.

## Phase 12 — Paper trading engine

Goal: a persistent paper ledger and a broker adapter interface.

Deliverables:

- `broker/adapter.py` (interface), `broker/paper.py` (persistent SQLite ledger),
  `broker/mt5_demo.py` skeleton (disabled, no credentials).
- Track signals, orders, positions, fills, spread, slippage, commission, P&L,
  model/strategy versions, rejection reasons.
- Session runner separate from the training process; restart-safe.
- API: paper account, positions, orders, history.

Dependencies: Phases 10–11. RAM: < 500 MB. Tests: ledger invariants, restart
recovery, no real-money path enabled, rejection audit.

Exit criteria: a paper session survives restart and reconciles every trade;
real execution remains impossible without an explicit future adapter.

## Phase 13 — Dashboard and monitoring

Goal: expose the new engines without fake metrics.

Deliverables:

- Pages: Market Intelligence, Classical ML Lab, Neural Network Lab, RL Lab,
  Strategy Intelligence, Autonomous Learning, Backtesting, Paper Trading,
  System Monitoring.
- API endpoints for each; empty/unknown states instead of invented values.
- Resource monitor: RAM/CPU/disk, job status, data quality, drift.
- Frontend tests for each page; no hardcoded performance numbers.

Dependencies: Phases 4–12. RAM: < 500 MB (UI). Tests: component tests, e2e
smoke, type check, build.

Progress (2026-11-24): the Paper Trading page (#128) reads `/api/paper` only and
renders an explicit dash/unknown rather than a fabricated zero. The read-only
autonomy status API (#134) exists, but no dashboard page consumes
`/api/autonomy` yet.

Exit criteria: every displayed number comes from an API or a documented local
artifact.

## Phase 14 — Mac mini optimisation and hardening

Goal: keep the whole system usable on 8 GB.

Deliverables:

- RAM budget manager and per-job caps; refuse jobs that exceed the budget.
- MPS benchmark and fallback policy; torch thread pinning; memory profiling.
- Chunked dataset loading; no full-store materialization in model code.
- Benchmark report: training time/RAM for LightGBM, GRU, TCN, small
  Transformer, PPO.
- Cancellation/resume tests under memory pressure.

Dependencies: Phases 6–13. RAM: measurement, caps < 6 GB. Tests: budget refusal,
resume after kill, MPS-unavailable fallback.

Exit criteria: a documented benchmark table and a scheduler that cannot start a
second heavy job.

## Phase 15 — Full validation and documentation

Goal: prove the acceptance criteria with executed tests, not claims.

Deliverables:

- All required docs listed in the master prompt.
- Full backend/frontend test run, lint, types, build, e2e.
- End-to-end scenario: import new data → validate → build dataset → train
  classical + neural candidates → evaluate → reject/promote → paper trade →
  monitor → rollback.
- A final status report that separates implemented / trained / validated /
  profitable.

Exit criteria: the acceptance checklist in the master prompt is met or each
unmet item has a documented reason.

## Priority queue

**P0 — unblock the platform**

1. Phase 2 data registry/validation/provenance.
2. Phase 3 dataset builder/leakage.
3. Phase 10 SQLite registry + promotion/rollback (can start after Phase 5).

**P1 — learning evidence**

4. Phase 4 regime engine.
5. Phase 5 classical ML baselines.
6. Phase 6 GRU/TCN framework and evaluation.

**P2 — advanced research**

7. Phase 7 Transformers.
8. Phase 8 RL expansion.
9. Phase 9 ensemble/confidence.

**P3 — product hardening**

10. Phase 11 risk/backtest.
11. Phase 12 paper ledger.
12. Phase 13 dashboard.
13. Phase 14 optimisation.
14. Phase 15 acceptance.

## Resource plan (8 GB M2)

| Work | Expected RAM | Expected time | Concurrency |
|---|---|---|---|
| Data validation / features | < 1 GB | seconds–minutes | with UI |
| LightGBM/XGBoost | < 1.5 GB | minutes | alone |
| GRU/TCN (H1/M15) | 1–2 GB | 5–30 min | alone |
| Small Transformer | 1.5–3 GB | 15–60 min | alone |
| PPO 300k steps | 1–2 GB | ~20–40 min | alone |
| Scheduler/SQLite/API | < 0.5 GB | continuous | always |

The scheduler must enforce: one heavy training job, a configurable RAM cap,
checkpoint before cancellation, and no training in the paper-trading process.

## What not to do

- Do not add all architectures before the data registry and dataset builder.
- Do not train on the forward store; it is the clean test.
- Do not select hyperparameters on the final test split.
- Do not promote a model on accuracy alone; require simulator P&L, cost stress
  and regime coverage.
- Do not claim a model is profitable because it trained.
- Do not run parallel heavy trainings on 8 GB.
- Do not enable real-money execution.

## Immediate next step after approval

Phase 2, in this order:

1. `data/registry.py` + tests for ten symbols.
2. `data/validate.py` + `data/provenance.py` + tests.
3. JSON importer and M30/H4 resampling.
4. API status extension and documentation.

No model work starts until Phase 2 and Phase 3 exit criteria are met.


---

# Autonomous self-learning implementation track (required)

This track extends the existing 15 engineering phases. It does not mark any
unfinished task as complete. The task board is the implementation checklist;
all items remain open until code, tests and reproducible evidence exist.

## Phase 16 — Autonomous strategy discovery

Build the safe strategy DSL/AST and bounded candidate search from causal market
primitives. The owner does not supply entry/exit rules. Start with one
resource-efficient search method (for example, grammar/evolutionary search)
and compare it with simple baselines. Add mutation/crossover or additional
optimisers only when measurable. Every candidate must be reproducible and
evaluated with realistic costs, walk-forward splits, untouched OOS data,
parameter sensitivity, complexity penalties and multiple-testing awareness.
No arbitrary generated code execution. The engine must be able to reject all
candidates and choose NO TRADE.

Exit criteria: an end-to-end run creates strategy candidates without
owner-authored rules, reproduces them from saved lineage, rejects an
intentionally overfit candidate, and records a champion/no-change decision.

Progress (2026-11-24): #104 runs search end to end on a validation slice and
archives a decision. #114 builds a chronological fold bundle, #121 enforces a
single holdout spend, #130 replays a bundle from identity. `discovery run`
still writes a decision report, not an evidence bundle; the producer is #136.
Reproduce commands are in `STATUS.md`.

## Phase 17 — Knowledge ingestion and hypothesis research

Add local document ingestion for supported PDF/TXT/Markdown/HTML/DOCX files,
with original-file preservation, checksums, parser metadata, page/section
references, searchable extracted text and honest OCR limitations. Convert
claims and ideas into traceable, falsifiable hypotheses; route those hypotheses
to quantitative strategy experiments. Document text must never directly
become a trusted strategy or order. This is structured knowledge extraction,
not a claim that the trading model has been trained on a book or that it has
human-level comprehension.

Exit criteria: import a sample document, trace a generated hypothesis back to
source passages, run its test, and record the outcome.

## Phase 18 — Autonomous Learning Orchestrator

Connect new-data detection, document ingestion, hypothesis generation,
dataset preparation, candidate search, ML/DL/RL jobs, evaluation, champion
governance, drift monitoring and paper feedback into one persistent workflow.
Enforce job dependencies, pause/resume/cancel, crash recovery, one heavy job,
RAM/disk limits and paper/training isolation. Log why each task was started,
skipped, promoted or rejected.

Exit criteria: a new dataset/document or matured paper outcome automatically
queues the appropriate eligible workflow; interrupted jobs recover safely;
the full chain is queryable and auditable.

Progress (2026-11-24): the daemon (#118) runs the persistent queue with the
`m2_8gb` guard profile by default; the file trigger and the matured
paper-outcome trigger (#122, proposal only) exist; five handlers (#115) and
end-to-end integration tests (#129) are merged. The bare `Orchestrator` still
defaults to `AllowAllGuard`, knowledge proposals do not auto-queue discovery, and
there is no orchestrator-level rollback control.

## Phase 19 — Autonomous paper-trading acceptance

Run the system without owner-authored entry/exit rules. Require historical,
walk-forward, untouched OOS and cost-stress gates before paper trials. Attribute
every paper trade to immutable data/model/strategy versions and learn only
after outcomes mature. Demonstrate champion protection, rejection, rollback,
NO TRADE and risk-control invariants. Real-money execution remains disabled.

Exit criteria: reproducible end-to-end demonstration of
import → validate → research → discover → backtest → independently validate →
paper trade → analyse outcomes → train candidate → champion decision →
monitor/rollback. This demonstrates system autonomy, not guaranteed
profitability.

## Revised priority guidance

Do not spend scarce M2 resources implementing every model architecture before
the autonomous loop works. Prioritise data integrity and leakage-safe datasets,
then registry/champion governance, a minimal but genuine strategy discovery
engine, knowledge ingestion, orchestration, robust evaluation and autonomous
paper trading. Expand model families only when experiments justify them.
