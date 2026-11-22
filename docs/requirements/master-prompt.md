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
