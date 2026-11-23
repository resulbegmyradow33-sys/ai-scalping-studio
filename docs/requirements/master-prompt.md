# Master requirements — AI Scalping Studio

Status: canonical project brief. This file restates the master prompt that
defines the target system. It is the shared contract between collaborators; the
audit and roadmap measure the repository against it.

- Repository: `https://github.com/resulbegmyradow33-sys/ai-scalping-studio`
- Target hardware: Mac mini M2, Apple Silicon, 8 GB unified memory, 256 GB SSD.
- Budget: **$0** for AI APIs, subscriptions, cloud inference, paid data.
- Default mode: **paper trading only**. Real-money execution is out of scope.
- No external or local LLM is required for inference or training.

## 1. Hard constraints

1. No OpenAI / Claude / Gemini / DeepSeek API.
2. No mandatory external or local LLM.
3. No cloud-hosted neural network.
4. No paid market-data provider.
5. No subscription-dependent service.
6. After dependencies and data are installed, training and inference must work
   offline.
7. Models train and run locally with ordinary Python ML/DL libraries.
8. Respect 8 GB RAM: small models, one training job at a time, CPU fallback,
   optional MPS, checkpoints, early stopping, chunked data loading.
9. The application must remain functional without MPS.

## 2. Required architecture

Modular components with documented interfaces:

1. Market Data Engine.
2. Data Validation Engine.
3. Feature Engineering Engine.
4. Multi-Timeframe Analysis Engine.
5. Market Regime Detection Engine.
6. Classical ML Engine.
7. Deep Learning Engine.
8. Neural Network Research Engine.
9. Reinforcement Learning Engine.
10. Strategy Research Engine.
11. Strategy Selection Engine.
12. Ensemble Decision Engine.
13. Prediction Confidence Engine.
14. Independent Risk Management Engine.
15. Backtesting Engine.
16. Paper Trading Engine.
17. Model Registry.
18. Continuous Learning Engine.
19. Model Evaluation Engine.
20. Performance Monitoring Engine.
21. Dashboard and Reporting Engine.

## 3. Market data

Instruments: XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY, AUDUSD, NAS100, US30,
USOIL, BTCUSD. Symbol-specific metadata: tick size, contract size, min/max lot,
lot step, spread, commission, sessions, price precision.

Timeframes: tick, M1, M5, M15, M30, H1, H4, D1. Import: CSV, Parquet, JSON
where appropriate.

Separate raw, cleaned, feature, training, validation, test and paper-trading
layers. Parquet partitioning. Avoid loading full history into memory.

Validation: missing data, duplicate ticks, timestamps, timezone, outliers,
gaps, price consistency, volume consistency, provenance. Never fabricate prices
to fill gaps. Distinguish real from synthetic data.

## 4. Features

Price action, returns, candle anatomy, consecutive candles. Momentum (RSI,
MACD, ROC, slopes, EMA relationships). Volatility (ATR, realised, Bollinger,
expansion/contraction). Volume/flow: tick volume, bid/ask volume, delta,
cumulative delta, imbalance — only when the source actually has it. Market
structure: swings, HH/HL/LH/LL, BOS, CHoCH, breakouts, S/R, sweep
approximations. Order flow: bid/ask imbalance, aggressor sides, spread changes,
book imbalance, absorption, impact, intensity — only with genuine tick/book
data. DeepLOB only with L2 data.

Configurable feature selection to prevent overfitting. Do not present
subjective ICT/SMC interpretations as objective facts.

## 5. Multi-timeframe intelligence

H4/H1 direction and structure; M15 setup and regime; M5 confirmation; M1
timing; tick execution context. Compare single-timeframe, multi-timeframe
fusion and independent encoders. Align strictly on information available at the
decision timestamp.

## 6. Classical ML

Logistic regression, random forest, extra trees, gradient boosting, XGBoost,
LightGBM, HistGradientBoosting, SVM where practical, isolation forest,
incremental SGD. Tasks: direction, return, volatility, regime, anomaly, trade
outcome, strategy performance. Baselines first. Complex models are not assumed
superior.

## 7. Deep learning

PyTorch, MPS when reliable, CPU fallback. Modular and independently evaluable:

- Feedforward: MLP, deep MLP, residual MLP.
- Recurrent: RNN, LSTM, GRU; bidirectional only offline.
- Convolutional: 1D CNN, dilated CNN, CNN-LSTM, residual CNN.
- TCN: causal, dilated, residual, configurable receptive field.
- Transformers: encoder, temporal, causal, lightweight, PatchTST,
  iTransformer, TFT where practical. Small configurations only.
- Order-book nets: DeepLOB and CNN/LSTM book encoders only with genuine L2.
- Unsupervised: AE, VAE, denoising AE, contrastive, clustering, self-supervised.
- Optional research: state-space, Mamba-style, GNN cross-asset — only with
  data, dependencies, hardware and evidence.

## 8. Learning tasks and labels

Direction (1/3/5/10/15 bars), future return, volatility, regime, breakout,
reversal, trade outcome, anomaly, uncertainty. Labels use future data only for
targets; inputs never do. Costs included in trade labels. Avoid misleading
accuracy on imbalanced classes.

## 9. Reinforcement learning

Audit and improve PPO. Evaluate SAC, TD3, DQN (discrete), A2C where appropriate
— not all at once. Gymnasium environments with BUY/SELL/HOLD/CLOSE/REDUCE.
Costs, spread, commission, slippage, execution delay, SL/TP, position limits,
max holding time, realised/unrealised P&L, drawdown. Rewards discourage
overtrading and excessive risk. Compare with buy-and-hold, random, rule,
classical ML and neural baselines. RL remains experimental until validated.

## 10. Regime detection

Strong/weak up/down, sideways, high/low volatility, expansion/contraction,
breakout, unusual. Evaluate rules, K-Means, GMM, HMM, RF, GB, neural
classifiers, autoencoder representations. Learn strategy performance by regime.

## 11. Strategy research

Momentum, trend, mean reversion, breakout, volatility expansion, price action,
flow momentum, order flow, liquidity, measurable ICT/SMC rules. Measure win
rate, PF, expectancy, average R, DD, Sharpe, Sortino, holding time, frequency,
costs, slippage sensitivity, performance by timeframe and regime. The engine
must be able to select **NO TRADE**.

## 12. Ensemble and decision intelligence

Combine classical ML, LSTM, GRU, TCN, CNN, Transformer, regime classifier,
order-flow model, RL policies and rules. Evaluate weighted ensembles, voting,
stacking, probability averaging and regime-dependent selection. Stacking uses
leakage-safe out-of-fold predictions. Every decision logs timestamp, symbol,
timeframe, model/strategy versions, probabilities, expected return, confidence,
regime, action and reason. Valid decisions: BUY, SELL, HOLD, NO TRADE.

## 13. Uncertainty and confidence

Calibration, Brier score, log loss, calibration curves, thresholds, ensemble
disagreement, predictive uncertainty, OOD detection. Distinguish high,
moderate, low and unknown confidence. Raw softmax is not confidence. Reject
unreliable signals.

## 14. Autonomous continuous learning

On new data: detect, validate, deduplicate, check timestamps, find gaps, update
metadata, build features, build labels when future bars exist, update datasets,
train candidates (classical and neural), evaluate, compare with champion,
compare by regime, save experiments, promote only validated candidates, keep
the previous stable model, monitor the new one.

Scheduled retraining, incremental learning where appropriate, controlled full
retraining, online learning where appropriate, checkpoints, resumption,
dataset/model versioning, experiment tracking, drift detection, rollback. New
data is not assumed to improve predictions. Never overwrite the champion with
an untested candidate.

## 15. Experiment and hyperparameter management

Select candidates, define configurations, run controlled experiments, optimize
hyperparameters (Optuna or lightweight equivalent), compare architectures,
record results, stop failures, checkpoint, resume. Search learning rate, batch
size, sequence length, hidden size, layers, dropout, weight decay, optimizer,
feature subsets, architecture. Strict resource limits. Never select on the
final test set.

## 16. Independent risk management

Max risk per trade, daily/weekly loss, max drawdown, consecutive losses, open
positions, max spread, slippage limits, sizing limits, SL validation, kill
switch. Deterministic and outside the model. Models cannot bypass or modify
limits. Block new trades on unsafe data or conditions. Paper only by default.

## 17. Realistic backtesting

Tick simulation where data exists; M1–H4; multiple symbols; variable spread,
commission, slippage, execution delay, partial fills where supported, rejected
orders, SL/TP, position management. Walk-forward, chronological validation,
purged CV/embargo where needed, untouched OOS, Monte Carlo trade sequences,
parameter sensitivity, ablation, stress, regime tests.

Prevent look-ahead, survivorship bias, leakage, overlapping-label leakage,
unrealistic fills, excessive optimisation. A positive backtest is not proof of
future profit.

## 18. Paper trading and broker architecture

Paper trading only. Broker adapter interface. Future MT5 demo adapter, not
enabled. Track signals, orders, positions, entries/exits, spread, slippage,
commission, size, SL/TP, realised/unrealised P&L, model and strategy versions.
Distinguish simulated from real execution.

## 19. Knowledge and model registry

SQLite for metadata and structured records; Parquet for large data; compact
checkpoints. Store dataset metadata/versions, feature definitions, architectures,
parameters, weights, training/validation history, backtests, paper results,
strategy/regime performance, drift, failed/successful experiments, promotion
history. Recover the previous stable state. Do not call ordinary storage
"neural memory".

## 20. Mac mini M2 optimisation

Configurable RAM budget, batch size, small configs, CPU fallback, optional MPS,
limited sequence lengths, Polars, streaming/chunked loading, checkpoints, early
stopping, gradient clipping, memory monitoring, cancellation/resumption, one
training job at a time, no mandatory Docker/LLM/oversized Transformer.

Priority: LightGBM/XGBoost → GRU → TCN → small LSTM → 1D CNN → CNN-LSTM →
lightweight Transformer → advanced only if justified.

## 21. Dashboard

Preserve and improve the frontend. Sections: market intelligence; neural
network lab; classical ML lab; RL lab; strategy intelligence; autonomous
learning; backtesting; paper trading; system monitoring. No fake metrics.

## 22. FOSS policy

Prefer Python, NumPy, Polars, Pandas, SciPy, scikit-learn, PyTorch, XGBoost,
LightGBM, Gymnasium, Stable-Baselines3, Optuna, SQLite, Parquet, FastAPI,
existing Next.js/charting. Before adding a dependency: license, ARM64, Python
compatibility, memory, maintenance, necessity. No popularity-driven additions.

## 23. Scheduler

Configurable local jobs for detection, validation, features, dataset prep,
training, NN experiments, evaluation, drift, comparison, paper analysis. Do not
train continuously. Pause/resume/cancel. A failed training job must not stop
paper trading.

## 24. Testing and quality

Tests for ingestion, validation, features, MTF alignment, labels, leakage,
training, serialization, loading, prediction reproducibility, online learning,
promotion, rollback, risk, backtesting, paper execution, API, frontend, memory.
Run unit/integration/regression/backend/frontend/type/lint/build. Never claim
tests passed without executing them.

## 25. Documentation

Required: `architecture-audit.md`, `implementation-roadmap.md`,
`system-architecture.md`, `local-learning-engine.md`,
`neural-network-research.md`, `transformer-models.md`,
`reinforcement-learning.md`, `market-regime-detection.md`,
`strategy-intelligence.md`, `model-validation.md`, `data-pipeline.md`,
`risk-management.md`, `backtesting-methodology.md`,
`mac-mini-optimization.md`, `paper-trading.md`.

Document what exists, changed, was implemented, remains experimental, is not
implemented, how to train/import/evaluate/recover/run.

## 26. Execution order

1. Audit and documentation.
2. Data pipeline hardening.
3. Market intelligence and features.
4. Regime detection.
5. Classical ML.
6. Neural network framework.
7. Lightweight neural architectures.
8. Transformer research.
9. RL experiments.
10. Ensemble and decision intelligence.
11. Autonomous learning and model management.
12. Risk and realistic backtesting.
13. Paper trading.
14. Dashboard and monitoring.
15. Full tests and documentation.

At each phase: explain changes, affected files, compatibility, RAM/CPU estimate,
implement, test, fix regressions, update docs, report actual results. Do not
skip tests, implement all architectures at once, create placeholder modules,
fake metrics, fabricate data, silently disable functionality, or push/modify
GitHub without explicit permission. Do not enable real-money trading.

## 27. Acceptance criteria

The project is complete only when it:

1. Works locally without external AI APIs.
2. Needs no local LLM.
3. Imports newly supplied datasets.
4. Validates and processes data correctly.
5. Trains classical ML.
6. Trains supported neural networks.
7. Experiments with Transformers.
8. Compares neural vs classical baselines.
9. Evaluates RL agents.
10. Detects regimes.
11. Evaluates strategies.
12. Supports multi-timeframe analysis.
13. Has independent risk controls.
14. Provides realistic backtesting.
15. Supports paper trading.
16. Keeps persistent model versions.
17. Trains candidates from new data.
18. Rejects inferior models.
19. Promotes validated improvements.
20. Rolls back unsuccessful versions.
21. Fits Mac mini M2 limits.
22. Supports every claim with actual tests.

Listing an architecture is not implementing it. Implementing it is not training
it. Training it is not proving profitable performance. Report those states
separately.

## Status legend used by the audit

- **DONE** — implemented, exercised by tests or a recorded run.
- **PARTIAL** — some pieces exist, but the requirement is not met end to end.
- **MISSING** — not implemented.
- **EXPERIMENTAL** — implemented but not validated out of sample.


---

# AUTONOMOUS SELF-LEARNING MISSION — OVERRIDES AND ADDITIONS

The following sections are first-class requirements of this project. They clarify
and, where stated, override earlier wording. The product goal is not a bot that
executes a human-authored strategy menu. The goal is a locally operating,
autonomous research-and-trading system that discovers its own strategies from
market evidence, learns through controlled experiments, manages its own
training/research workflow, and can trade autonomously in paper mode. No
particular trading strategy, named methodology, indicator combination, or
strategy supplied by the owner is a prerequisite.

## 28. Primary objective: strategy-independent autonomous intelligence

The system MUST NOT depend on the owner supplying entry rules, exit rules,
indicator combinations, ICT/SMC rules, named strategies, or a preferred trading
method. Existing strategies and human-supplied strategies may be imported as
optional baselines for comparison only. They must never constrain the search
space or be treated as the required answer.

The system must be able to:
- Observe raw and derived market data and discover predictive or tradable
  structure without being handed a strategy.
- Form its own measurable hypotheses from data.
- Generate candidate policies and strategies from a bounded, inspectable
  strategy representation.
- Discover entry, exit, position-management, timing, and NO-TRADE behaviour.
- Test candidates across symbols, timeframes, market regimes, and changing
  transaction costs where data permits.
- Learn from both simulated outcomes and paper-trading feedback.
- Reject candidates that fail independent validation, even if their training
  results look excellent.
- Continue researching when no candidate beats the current champion; it must
  be allowed to conclude that no trade or no improvement is the best decision.

The strategy search space must be generated from reusable primitives rather
than a fixed list of named strategies. Primitives may include causal price,
return, volatility, volume/flow (only when genuine), market-structure,
cross-timeframe, time/session, and execution-cost features; comparisons,
thresholds, boolean/conditional expressions, temporal windows, state
transitions, entry/exit rules, and position-management actions. The owner must
not have to author these combinations.

## 29. Autonomous Strategy Discovery Engine

Build a dedicated engine that creates, mutates, evaluates, and archives
strategies without human-written strategy rules.

Required capabilities:
1. A safe, versioned strategy DSL/AST representing observations, conditions,
   entry/exit decisions, holding logic, and sizing requests. No arbitrary code
   execution or eval of generated code.
2. Candidate generation using appropriate methods such as genetic programming,
   evolutionary search, grammar-based search, random search, and Bayesian
   optimisation. Select methods based on evidence and compute budget; do not
   implement every algorithm merely for its name.
3. Search over feature subsets, thresholds, lookback windows, entry/exit logic,
   regime conditioning, multi-timeframe relationships, and holding periods.
4. Optional policy discovery through RL, with an environment whose observations
   contain only information available at decision time and whose actions
   include BUY, SELL, HOLD, CLOSE, REDUCE and NO TRADE as appropriate.
5. Reward/objective design based on net outcomes after spread, commission,
   slippage and execution assumptions; penalise drawdown, tail risk,
   overtrading, unstable turnover and excessive complexity. Do not optimise
   win-rate or raw accuracy alone.
6. Candidate deduplication, complexity limits, search budgets, deterministic
   seeds, experiment lineage, reproducibility and cancellation.
7. A strategy archive containing source representation, parent/mutation
   lineage, parameters, dataset hashes, symbols, timeframes, costs, code
   version, experiment results and rejection reasons.
8. Strategy comparison against simple baselines and the current champion.
9. Regime-conditional and cross-symbol evaluation to detect strategies that
   only exploit one narrow historical condition.
10. A mandatory NO TRADE option when expected net edge is insufficient or
    uncertainty/cost/risk is too high.

No strategy may be selected using the final untouched test set. Use nested
chronological validation / walk-forward evaluation, purge and embargo where
labels overlap, a locked final test set, and a separate forward/paper period.
Control multiple-testing and data-snooping risk: record the number of trials,
penalise complexity, assess parameter sensitivity and stability, and do not
reuse the final holdout for repeated tuning. A large search producing one
profitable backtest is not evidence of a durable edge.

## 30. Knowledge Ingestion and Research Engine

The owner must be able to add research material after the application is
installed. Support local, offline ingestion where practical:
- Market datasets: CSV, Parquet, JSON and supported tick/bar formats.
- Research documents: PDF, TXT, Markdown, HTML and DOCX where lightweight,
  maintained parsers are available.
- Experiment and paper-trading logs, broker exports, strategy reports and
  model evaluation artifacts.

For every imported item:
1. Preserve the original file and provenance; compute a checksum and record
   source, import time, format, size and parser version.
2. Extract text/tables/metadata where supported; report extraction failures,
   scanned-PDF/OCR limitations and unsupported formats honestly.
3. Index extracted content locally with document/page/section references.
4. Separate factual claims, definitions, examples, opinions, hypotheses and
   empirical market data. Never treat a book's claim as proven market truth.
5. Convert potentially testable ideas into structured research hypotheses:
   what is observed, proposed condition, expected effect, market/timeframe
   scope, evidence source, assumptions and a falsification test.
6. Link every generated hypothesis to its source passages and later experiment
   IDs so the user can trace why it was proposed and what happened.
7. Send hypotheses to the Strategy Discovery Engine for quantitative testing;
   do not directly convert prose into live orders or trusted strategy rules.
8. Allow the system to learn from its own paper-trading and backtest reports.

Important capability boundary: document ingestion is not the same as training a
language model. Under the zero-cost/no-mandatory-LLM constraint, the system
should use deterministic parsers, metadata extraction, search/indexing,
lightweight NLP and structured hypothesis templates. It must not claim to
understand every book like a human or to have trained on a PDF. Optional
advanced language tooling may be proposed separately only if it is free,
offline, fits the hardware and is explicitly approved; it is not a dependency
for the core trading learner.

## 31. Autonomous Learning Orchestrator

Create one persistent orchestrator that coordinates the complete learning
lifecycle. It is a real executable subsystem, not just a scheduler label or
dashboard page.

It must:
- Detect newly imported data, documents, broker exports and completed paper
  sessions.
- Route each item to the correct ingestion/validation pipeline.
- Decide which research jobs are eligible based on data sufficiency, novelty,
  drift, previous results and resource budget.
- Create and queue dataset preparation, hypothesis generation, strategy
  search, ML/DL/RL training, evaluation, drift checks and paper-analysis jobs.
- Enforce dependency order, one heavy job at a time, RAM/CPU/disk budgets,
  priorities, cancellation, pause/resume, retry limits and crash recovery.
- Persist all jobs and states so restart does not lose experiment history.
- Avoid pointless retraining when there is no meaningful new data or evidence.
- Keep paper trading isolated from training failures and heavy training
  processes.
- Explain every automated action through an auditable event log.
- Support manual pause, safe mode, disable-autonomous-research and rollback
  controls without requiring the owner to design strategies.

Autonomy means the system can initiate and manage approved research tasks on
its own. It does not mean unlimited compute, uncontrolled self-modification,
or permission to alter risk limits.

## 32. Continuous learning, feedback and champion governance

Implement a closed learning loop:
market observation → hypothesis/candidate generation → simulation →
independent validation → paper trial → outcome analysis → model/strategy
update proposal → champion comparison → promotion or rejection → monitoring.

For every paper trade, persist the timestamp, data snapshot/version, symbol,
timeframe, observation/features, selected policy/strategy version, action,
confidence/uncertainty, expected outcome, risk decision, requested and filled
prices, spread, slippage, commission, exit reason, realised P&L/R, and relevant
market regime. Keep immutable historical records.

Learning from outcomes must account for delayed labels, overlapping trades,
position interactions and transaction costs. Do not update a model from an
outcome until the relevant outcome is fully known and correctly attributed.
Distinguish training feedback from independent evaluation data.

Maintain:
- Immutable experiment and model versions.
- A champion model/policy/strategy and separate candidate versions.
- Promotion rules with minimum sample sizes, OOS and walk-forward evidence,
  cost stress, regime coverage, stability and comparison against champion.
- Automatic rejection and rollback.
- Drift and degradation monitoring.
- A record of failed experiments and why they failed.
- The ability to keep the current champion or select NO TRADE if no candidate
  passes.

The system may automatically promote a candidate only inside the configured
paper-trading environment and only after all gates pass. Risk limits are
immutable to models and strategies. A strategy cannot raise its own limits,
disable the risk engine, or grant itself real-money access.

## 33. Autonomous trading and execution boundary

The finished system must be capable of making its own BUY, SELL, HOLD, CLOSE,
REDUCE or NO TRADE decisions and managing a complete simulated position
lifecycle without the owner supplying a strategy.

Required sequence:
1. Historical backtest and robust walk-forward validation.
2. Continuous/replay paper trading using new market data.
3. Optional broker demo adapter (for example, MT5 demo) behind explicit
   configuration, disabled by default and with no embedded credentials.
4. Human review of evidence before any future consideration of real capital.

For this project, real-money execution remains disabled and out of scope.
Autonomous paper trading is the required end state; demo connectivity is an
optional later stage. Never describe a backtest as live trading or a paper fill
as a real broker fill.

## 34. Scope discipline: build what serves the objective

The project is a self-learning trading research and paper-execution studio, not
a collection of fashionable model names.

- Do not implement every neural architecture, RL algorithm or data science
  package just because it appears in a list. Choose the smallest set that
  answers a measurable research question and fits the M2.
- Keep LightGBM/XGBoost and compact GRU/TCN/PPO as candidate tools, not
  mandatory winners. Add Transformers, SAC/TD3/DQN, autoencoders, Mamba, GNN or
  other methods only when an experiment justifies them.
- Remove duplicate modules, dead code, placeholder interfaces, unused
  dependencies, redundant dashboards and hard-coded strategy assumptions when
  an audit proves they do not serve the objective. Preserve working behaviour
  and tests; do not delete functionality merely to simplify.
- Do not require the owner to choose indicators, write strategies, label every
  trade manually, or select the winning model. The owner supplies data,
  documents, constraints and approval for external actions; the system
  performs the research.
- Keep the implementation local-first, offline-capable after data/dependencies
  are available, zero-cost in mandatory services, and resource-aware.
- Do not fabricate data, metrics, training runs, test results or profitability.
  Report separately: implemented, tested, trained, independently validated,
  paper-traded and profitable (profitability must never be guaranteed).

## 35. Updated execution priorities

The existing phases remain, but the autonomous objective is not postponed to a
cosmetic final feature. After data integrity, leakage-safe datasets and the
minimum usable ML/RL evaluation foundations are reliable, prioritise in this
order:

1. Complete the autonomous learning registry, experiment lineage, scheduler,
   champion/candidate gate and rollback.
2. Build the strategy DSL and autonomous strategy discovery/search engine.
3. Build document/data knowledge ingestion and hypothesis traceability.
4. Connect the Learning Orchestrator to new data, new documents, strategy
   discovery, model training, evaluation and paper feedback.
5. Complete independent risk and realistic execution simulation.
6. Run the full closed loop in paper trading and expose its state in the UI.
7. Expand model families only when measured evidence supports the added cost.

Do not block the strategy-independent discovery engine on implementing every
listed neural architecture. Do not treat a manual strategy library as the
strategy discovery engine.

## 36. Additional acceptance criteria for the actual autonomous objective

The system is not accepted as autonomous self-learning until all of these
end-to-end tests pass:

1. Import a previously unseen market dataset; validate provenance and quality;
   build a leakage-safe dataset without manual feature/strategy rules.
2. Generate multiple new strategy candidates from the strategy grammar/search
   space without the owner providing entry/exit rules.
3. Reproduce a candidate experiment from its saved data hash, code version,
   configuration and random seed.
4. Reject deliberately overfit candidates on unseen chronological data and
   transaction-cost stress tests.
5. Demonstrate that the search can select NO TRADE and can conclude that no
   candidate beats the champion.
6. Import a sample PDF and dataset; preserve source references; produce
   traceable test hypotheses without claiming the PDF itself trained a model.
7. Detect new data or completed paper-trade outcomes and automatically queue
   eligible research jobs.
8. Train a candidate, compare it with the champion, and demonstrate promotion,
   rejection and rollback with an audit trail.
9. Run paper trading autonomously without a human selecting the strategy.
10. Show that risk controls cannot be bypassed by generated strategies, RL
    policies, models or the orchestrator.
11. Recover correctly after application restart, interrupted training and
    cancelled jobs.
12. Operate within measured Mac mini M2 resource limits.
13. Demonstrate the complete loop:
    import → validate → learn/research → generate strategy → backtest →
    walk-forward/OOS validation → paper trade → analyse feedback → candidate
    update → champion decision → monitor/rollback.

A module name, prompt instruction, architecture diagram or placeholder does
not satisfy an acceptance criterion. Each item requires code, tests and a
recorded reproducible run. Even passing these criteria proves system
functionality, not future profitability.
