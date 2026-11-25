# Final report — XAUUSD autonomous research system

**Report date:** 2026-11-25  
**Commit:** `50fbca6856f7b6df13fe2a6cea223bf06d93c2f7` (`main`, base of `feat/finalreport`)  
**Scope:** claim audit at this commit. This document separates what is implemented, tested, trained, validated, paper-traded and profitable. It is not a roadmap and it does not restate the product vision as if it were evidence.

**Verdict in one paragraph.** The repository is a substantial, tested research and paper-execution *infrastructure*. It is not a validated autonomous trader, it has never paper-traded a governed champion, and it is not profitable. The full CI suite is green on this commit, the fail-closed evidence and promotion machinery exists, and real-data smoke runs behave correctly by returning NO TRADE. No independent locked out-of-sample result exists, and the only forward sample is 24 trades from a legacy hand-written card with a confidence interval that crosses zero.

## 0. Claim matrix

| Claim | Verdict at `50fbca6` | What backs it |
|---|---|---|
| **Implemented** | **Yes** | Code exists across data, features, DSL discovery, evidence, holdout, promotion, paper, orchestrator, API and UI; backend/frontend CI is green. |
| **Tested** | **Yes, as software** | CI run `45550807353`: backend `2320 passed, 3 skipped`; frontend `263 passed`; CodeQL success. This is not market validation. |
| **Trained** | **Only local baseline experiments** | Five classical classifiers were fitted on 2026-11-22 and their artifacts exist under gitignored `runs/ml/`. All lose to the prior baseline. Twenty-one legacy PPO run directories exist. No trained artifact is committed, and no model is a governed champion. |
| **Validated** | **No** | No real locked holdout has been spent, no evidence bundle carries a real `locked_oos` section, and no independent chronological OOS result exists. |
| **Paper-traded** | **No** | No governed champion exists and no `paper.sqlite` ledger with a champion session was found. `paper.advance` correctly refuses with `NoChampionError`. |
| **Profitable** | **No, and not claimed** | The only forward sample is 24 trades on a legacy card, `+0.502 R`, 95% CI `-0.04 … +1.04`, which includes zero. |
| **Real money enabled** | **No** | The MT5 adapter is a refused-by-construction skeleton; the paper package imports no broker or network client. |

## 1. Implemented

“Implemented” means code exists on this commit and the relevant tests pass. It does not mean the subsystem has produced market evidence. Legacy components are labelled as such.

| Area | Where it lives | Tests |
|---|---|---|
| Data store, validation, quality, registry, resampling | `backend/src/scalping/data/` (`store.py`, `validate.py`, `quality.py`, `provenance.py`, `registry.py`, `resample.py`) | [test_store.py](../backend/tests/test_store.py), [test_validate.py](../backend/tests/test_validate.py), [test_quality.py](../backend/tests/test_quality.py), [test_provenance.py](../backend/tests/test_provenance.py), [test_registry.py](../backend/tests/test_registry.py) |
| Causal features, feature cache, multi-timeframe context | `backend/src/scalping/features/` (`build.py`, `registry.py`, `cache.py`, `context.py`, `availability.py`) | [test_features.py](../backend/tests/test_features.py), [test_feature_cache.py](../backend/tests/test_feature_cache.py), [test_mtf_alignment.py](../backend/tests/test_mtf_alignment.py), [test_mtf_context_features.py](../backend/tests/test_mtf_context_features.py) |
| Dataset manifests and artifacts | `backend/src/scalping/datasets/` (`builder.py`, `artifact.py`) | [test_dataset_builder.py](../backend/tests/test_dataset_builder.py), [test_dataset_artifact.py](../backend/tests/test_dataset_artifact.py) |
| Strategy DSL/AST, search, evaluation, selection | `backend/src/scalping/strategy/dsl/`, `backend/src/scalping/discovery/` (`search.py`, `run.py`, `evaluate.py`, `selection.py`, `archive.py`) | [test_strategy_dsl.py](../backend/tests/test_strategy_dsl.py), [test_discovery_search.py](../backend/tests/test_discovery_search.py), [test_discovery_run.py](../backend/tests/test_discovery_run.py), [test_discovery_evaluator.py](../backend/tests/test_discovery_evaluator.py), [test_discovery_selection.py](../backend/tests/test_discovery_selection.py) |
| Evidence bundles, one-shot holdout, promotion, reproduction, champion provenance | `backend/src/scalping/discovery/` (`evidence.py`, `persist.py`, `holdout.py`, `promotion.py`, `reproduce.py`, `champion.py`) | [test_discovery_evidence.py](../backend/tests/test_discovery_evidence.py), [test_discovery_persist.py](../backend/tests/test_discovery_persist.py), [test_locked_holdout.py](../backend/tests/test_locked_holdout.py), [test_promotion_policy.py](../backend/tests/test_promotion_policy.py), [test_discovery_reproduce.py](../backend/tests/test_discovery_reproduce.py), [test_evidence_adversarial.py](../backend/tests/test_evidence_adversarial.py) |
| Classical ML: train, CV, evaluation, calibration, ablation, anomaly, incremental SGD | `backend/src/scalping/ml/` | [test_ml.py](../backend/tests/test_ml.py), [test_ml_cv.py](../backend/tests/test_ml_cv.py), [test_ml_phase5.py](../backend/tests/test_ml_phase5.py), [test_ml_ablation.py](../backend/tests/test_ml_ablation.py), [test_calibration.py](../backend/tests/test_calibration.py), [test_ml_temperature.py](../backend/tests/test_ml_temperature.py) |
| Compact NN/Transformer research tooling | `backend/src/scalping/nn/` (`models.py`, `train.py`, `evaluate.py`, `benchmark.py`, `transformers.py`) | [test_nn_models.py](../backend/tests/test_nn_models.py), [test_nn_train.py](../backend/tests/test_nn_train.py), [test_nn_benchmark.py](../backend/tests/test_nn_benchmark.py), [test_transformers.py](../backend/tests/test_transformers.py) |
| PPO / trading environment / action masks | `backend/src/scalping/rl/`, `backend/src/scalping/env/` | [test_train_smoke.py](../backend/tests/test_train_smoke.py), [test_env.py](../backend/tests/test_env.py), [test_action_masks.py](../backend/tests/test_action_masks.py) |
| Ensemble and regime-dependent weights | `backend/src/scalping/ensemble/` | [test_ensemble.py](../backend/tests/test_ensemble.py), [test_ensemble_regime.py](../backend/tests/test_ensemble_regime.py) |
| Paper ledger, DSL runner, continuous runner, mature outcomes, disabled MT5 skeleton | `backend/src/scalping/paper/` (`ledger.py`, `runner.py`, `autonomous.py`, `continuous.py`, `mt5.py`) | [test_paper_ledger.py](../backend/tests/test_paper_ledger.py), [test_paper_autonomous.py](../backend/tests/test_paper_autonomous.py), [test_paper_continuous.py](../backend/tests/test_paper_continuous.py), [test_paper_gaps.py](../backend/tests/test_paper_gaps.py), [test_paper_mt5.py](../backend/tests/test_paper_mt5.py) |
| Persistent orchestrator, handlers, daemon, resource guard, paper triggers | `backend/src/scalping/orchestrator/` | [test_orchestrator_core.py](../backend/tests/test_orchestrator_core.py), [test_orchestrator_handlers.py](../backend/tests/test_orchestrator_handlers.py), [test_orchestrator_daemon.py](../backend/tests/test_orchestrator_daemon.py), [test_orchestrator_resources.py](../backend/tests/test_orchestrator_resources.py), [test_paper_outcome_triggers.py](../backend/tests/test_paper_outcome_triggers.py) |
| Knowledge ingestion, extraction, index, hypotheses, proposal bridge | `backend/src/scalping/knowledge/`, `backend/src/scalping/orchestrator/proposal_handlers.py` | [test_knowledge_ingest.py](../backend/tests/test_knowledge_ingest.py), [test_knowledge_extract.py](../backend/tests/test_knowledge_extract.py), [test_hypothesis_routing.py](../backend/tests/test_hypothesis_routing.py), [test_proposal_bridge.py](../backend/tests/test_proposal_bridge.py) |
| Risk engine and non-bypass checks | `backend/src/scalping/env/risk.py`, `masks.py` | [test_risk.py](../backend/tests/test_risk.py), [test_risk_limits.py](../backend/tests/test_risk_limits.py), [test_risk_acceptance.py](../backend/tests/test_risk_acceptance.py) |
| HTTP API and UI pages | `backend/src/scalping/api/`, `frontend/src/app/` (`paper`, `autonomy`, `data`, `training`, `strategies`, `performance`, `intelligence`, `labs/ml`) | [test_api_contracts.py](../backend/tests/test_api_contracts.py), [test_api_paper.py](../backend/tests/test_api_paper.py), [test_api_security_surface.py](../backend/tests/test_api_security_surface.py), frontend `page.test.tsx` files |
| Synthetic end-to-end acceptance harness | `backend/src/scalping/acceptance/run.py` | [test_acceptance.py](../backend/tests/test_acceptance.py) |

Legacy classification is unchanged: `flow_momentum`/`FlowMomentumRule`/`PolicyCard` is the incumbent compatibility path; `strategy/signals.py` plus `strategy/bandit.py` is a finite-menu baseline; PPO and behaviour cloning are experimental baselines. None of them is the autonomous discovery engine.

## 2. Tested

The exact counts come from the latest CI run on this SHA, not from memory or an estimate.

| Check | Exact result | Source |
|---|---|---|
| Backend pytest | **2,320 passed, 3 skipped, 0 failed**, 40 warnings, 328.52 s | CI workflow `45550807353`, backend job `138991347449` |
| Backend lint | `All checks passed!` | same job, `uv run ruff check src tests scripts` |
| Backend types | `Success: no issues found in 149 source files` | same job, `uv run mypy src` |
| Frontend tests | **263 passed** in 31 files, 11.58 s | CI workflow `45550807353`, frontend job `138991346992` |
| Frontend build | `Compiled successfully` | same job |
| CodeQL | Python and JavaScript/TypeScript analysis success | CodeQL workflow `45550807354` |
| Test collection | **2,323 tests in 116 files** | `cd backend && uv run pytest --collect-only -q` |

The count is internally consistent: `2,320 passed + 3 skipped = 2,323 collected`. The run URL is `https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/runs/45550807353`.

I also ran the two secret-hygiene tests at this commit: `uv run pytest tests/test_no_committed_secrets.py tests/test_gitignore_hygiene.py -q` → **4 passed**. The full suite count above is the CI run; I did not substitute a local run for it.

**What these tests do not establish.** They do not establish a market edge, an independent OOS result, a paper-traded champion or profitability. The acceptance harness itself writes `validated: false`, `paper_traded: false` and `profitable: false` in its report ([acceptance-harness.md](acceptance-harness.md)).

## 3. Trained

“Trained” means a model was actually fitted and a reproducible run artifact exists. At this commit there are **local fits with artifacts, but no committed trained artifact and no governed model**.

### 3.1 Classical classifiers

Five scikit-learn/XGBoost/LightGBM runs were fitted on 2026-11-22. Artifacts are in the parent workspace under gitignored `runs/ml/`, outside the git tree:

| Run | Model | Validation accuracy | Validation log loss | Test accuracy | Artifact |
|---|---|---:|---:|---:|---|
| `ml-logistic-316312dff683` | logistic | 0.5162 | 0.6921 | not scored in `metrics.json` | `model.joblib` + `metrics.json` |
| `ml-random_forest-316312dff683` | random forest | 0.5053 | 0.6937 | 0.5064 | same |
| `ml-hist_gradient-316312dff683` | histogram gradient | 0.5147 | 0.6972 | not scored in `metrics.json` | same; this artifact is the earlier **3-class** run |
| `ml-lightgbm-316312dff683` | LightGBM | 0.5193 | 0.7073 | 0.5100 | same |
| `ml-xgboost-316312dff683` | XGBoost | 0.5184 | 0.7154 | 0.5073 | same |

Dataset id: `XAUUSD:H1:h5:316312dff683`; split counts around 20,654 train / 4,427 validation / 4,408 test, with 20 rows marked purged. The majority/prior baseline is **54.3% accuracy / 0.6919 log loss**. Every fitted model is at or below that baseline; the repository document states it plainly: “No model, including LightGBM and XGBoost, beats the prior baseline; none is promoted” ([classical-ml.md](classical-ml.md)). The validation numbers were cross-checked against each `metrics.json`.

Two caveats:

- The artifacts carry a `dataset_id` but no `source`/`synthetic` flag, so the real-versus-synthetic provenance of the training data is **unverified from the artifact alone** (provenance audit G15). The document calls it the “first real H1 run”, but that is a prose claim, not an artifact field.
- The logistic and histogram-gradient test metrics shown in `classical-ml.md` are not present in their `metrics.json`; those test cells are **doc-only and unverified by the artifact**.

### 3.2 PPO runs

Twenty-one run directories exist in the parent workspace under `backend/runs/*/manifest.json`, dated 2026-11-21/22. All 21 report `status: completed`. Sixteen are flagged `degenerate: true`, four are flagged `false`, and one (`ca81edf4cfed`) has no flag and no trades. These are the legacy experimental PPO path, not the autonomous discovery engine. Training uses `stable_baselines3.PPO`; masking exists only in evaluation. There is **no MaskablePPO training** ([reinforcement-learning.md](reinforcement-learning.md)).

These runs have `data_hash` values but no `source`/`synthetic` field. They are documented as an experimental baseline, and no run is used as validation, promotion or paper evidence.

### 3.3 What was not trained

- No GRU/TCN/Transformer checkpoint: no `runs/nn/` directory and no `.pt` checkpoint was found in the workspace.
- No discovery model: DSL candidates are searched and evaluated, not fitted. The equivalent record is an evaluated trial in the discovery archive, not a model artifact.
- No committed trained artifact of any kind. `docs/methodology.md` is correct on this point: run directories and weights are gitignored.

## 4. Validated

**No.** The project’s own definition requires independent chronological out-of-sample evidence **and** realistic cost stress, with a one-shot holdout that has actually been spent ([methodology.md](methodology.md)). That artifact does not exist here.

What does exist, and why it does not count:

- **Research smoke on real XAUUSD H1 (2026-11-24).** 29,586 bars, dataset `XAUUSD:H1:h1:439ad79658c7`; 25 attempts, 22 evaluated, 3 skipped. The best-by-net-profit candidate closed **0 trades** and the decision was `no_trade`. A separate high-trade candidate had **-0.0926 R** expectancy, **0.861** profit factor and **0/3** positive folds. Folds are chronological but live inside train+validation; the test split was never scored. This is a wiring smoke, not validation ([research-smoke-2026-11-24.md](research-smoke-2026-11-24.md)).
- **One uncommitted adjacent acceptance attempt.** Under `.worktrees/realaccept/backend/runs/realaccept/` there is a real-data run dated 2026-11-25: 22 trials, **0 accepted/tradable candidates**, decision `no_trade`. Its evidence bundle has `locked_oos: null`; an acceptance-test policy evaluation rejects it on `min_positive_fold_fraction`, `min_expectancy_r`, `min_profit_factor`, `cost_stress` and `stress_degradation`; the paper gate then refuses with `NoChampionError`. The rejection policy is not `PROMOTION_V1`, because a bundle without `locked_oos` cannot reach that policy. This proves fail-closed behaviour, not validation. It ran on code version `05c600129db13136`, **not** `50fbca6`, so it is unverified against this exact SHA.
- **The one-shot holdout is unspent.** No `holdout*.sqlite` and no evidence bundle with a non-null `locked_oos` section was found. The production policy sets `require_locked_oos=True`, and [backtesting-methodology.md](backtesting-methodology.md) states: “no real XAUUSD holdout has been spent.”

Two things that explicitly do **not** count:

1. A **validation-slice winner** is a candidate, never a validated strategy. Search fitness is not promotion evidence.
2. A **one-shot holdout that was never spent** is a protocol, not a result. An unspent holdout and an empty `locked_oos` section are not validation.

## 5. Paper-traded

**No.** No exact validated strategy version ran in the paper engine with immutable attribution.

- No `paper.sqlite` ledger was found anywhere in the workspace (`find … -name paper.sqlite`), and no champion archive with an evidence-mode champion was found.
- The paper gate refuses correctly: `NoChampionError: no champion in scope 'XAUUSD:H1'; promote a validated candidate first` (adjacent real-acceptance run, `07-paper-gate.json`).
- The synthetic acceptance harness does drive `AutonomousPaperRunner` and writes a real local ledger, but on synthetic bars for a few dozen bars. Its own report says `paper_traded: false`; it is an engine check, not a paper-trading trial ([acceptance-harness.md](acceptance-harness.md)).
- What is implemented and tested is the machinery: `PaperLedger`, `PaperSession`, `AutonomousPaperRunner`, `ContinuousPaperRunner`, canonical spec-hash attribution, append-only continuation, gap policy, matured outcomes and `paper.advance` with champion governance. Tests prove determinism and restart safety. They do not create a paper-traded record.

## 6. Profitable

**Not claimed.** There is no matured, attributed, non-synthetic paper-trading sample, so the word does not apply.

The only forward sample in the repository is a **legacy `flow_momentum` card**, not the autonomous DSL system, and it is a replay simulation rather than a paper ledger ([forward-test.md](forward-test.md)). The committed document and `backend/runs/forward_history.jsonl` agree on these numbers:

| Field | Forward sample (2026-06-01 → 2026-11-21) |
|---|---:|
| Trades | **24** |
| Expectancy | **+0.502 R** per trade |
| Profit factor | **2.15** |
| Net | **+$481** |
| Win rate | **58.3%** |
| Max drawdown | **2.4%** |
| Cost stress ×2 | +0.496 R, PF 2.14 |
| t-statistic | **1.83** |
| 95% confidence interval | **-0.04 … +1.04** (includes zero) |
| Bootstrap P(true mean ≤ 0) | **≈ 3.0%** |

The caveat is the sample itself: 24 trades is too small, the interval crosses zero, the rule is a hand-written legacy card, and the run is a simulator replay on held-out dates rather than an immutable paper ledger. The document’s own conclusion is that it is still early and the live card does not change. “First forward test positive and survived a cost stress” is defensible; “profitable” is not.

## 7. Security and risk posture

| Control | State at `50fbca6` | Evidence |
|---|---|---|
| Real-money execution | **Disabled and out of scope** | `Mt5DemoAdapter` is a skeleton: constructing it without `enabled=True` raises, and every order/account method raises `NotImplementedError`; no credentials, no network call ([mt5.py](../backend/src/scalping/paper/mt5.py)). Acceptance claims carry `real_money: false`. |
| Paper only | **Yes** | The paper package writes to a local SQLite ledger and calls the simulator; it imports no broker or exchange client. `test_paper_mt5.py` scans the source for network and credential markers. |
| Locked OOS | **Protocol implemented, unspent** | One-shot `HoldoutLedger` with append-only access log and immutable result; `PROMOTION_V1` cannot disable `require_locked_oos`; no real holdout has been spent. |
| Provenance | **Partial, with named gaps** | The store sidecar records `source`, `synthetic`, checksum and dataset id, and the continuous paper path carries the pair into the session and matured outcomes. The dataset manifest, evidence bundle, holdout, promotion, ML/NN run metadata and `AutonomousPaperRunner` remain provenance-blind; `data derive` still defaults derived timeframes to `synthetic=False`. These are G1–G6, G11 and G15 in [provenance-audit-2026-11-24.md](provenance-audit-2026-11-24.md), re-checked against the current code. |
| Secrets | **Clean** | The 2026-11-24 audit found no committed credential in the working tree or full git history (gitleaks plus manual and entropy passes). At this commit `test_no_committed_secrets.py` and `test_gitignore_hygiene.py` pass (4 tests). The only tracked env file is `frontend/.env.example` with a localhost URL. |
| API exposure | **Loopback by default, no auth** | The API binds `127.0.0.1` by default and has no authentication layer; CORS is an explicit localhost allowlist. Three low-severity information leaks were fixed; absolute report paths, corrupt-ledger 500s and no-auth are accepted and documented in [security-review-2026-11-24.md](security-review-2026-11-24.md). Do not expose the port. |
| Risk limits | **Independent and tested** | DSL spec, raw simulator action and orchestrator job payload all reach the same frozen `RiskEngine`; specs cannot raise limits or disable gates ([test_risk_acceptance.py](../backend/tests/test_risk_acceptance.py)). |
| Resource budget | **Not accepted on target hardware** | The H1 smoke measured 229–266 MiB peak RSS and ~43 s total, but no full target-M2 8 GB end-to-end acceptance was run on this commit. The one-heavy-job semaphore and measured guard are implemented and tested. |

## 8. Remaining blockers

1. **No validated candidate.** The one-shot holdout is unspent, no evidence bundle carries `locked_oos`, and the production policy therefore rejects every bundle. The real-data runs that exist returned NO TRADE, which is a valid result, not a validation pass.
2. **No runtime holdout/promotion command.** `discovery evidence` persists a bundle, and the holdout/promotion functions are tested, but there is no CLI or orchestrator handler that spends a real holdout and promotes the resulting bundle. The holdout remains library-only. The roadmap still says “#136 open” but the evidence producer is now merged; the remaining gap is the holdout/promotion operator path.
3. **Provenance cannot gate promotion.** Dataset manifests, evidence, holdout, promotion, ML/NN metadata and the autonomous paper runner have no `source`/`synthetic` field. A synthetic dataset can still produce a verified evidence bundle. This is the most important safety gap because the master prompt requires fail-closed provenance.
4. **No governed paper champion.** `paper.advance` is wired to `ChampionProvider` and fails closed with NO TRADE, but no evidence-mode champion exists. The ledger also does not record the authorising promotion id; the job result does.
5. **No model edge.** Every classical model is at or below the prior baseline; the PPO runs are experimental and mostly degenerate; no neural checkpoint exists. More model families are not the next step.
6. **No target-hardware acceptance.** The measured H1 smoke is small. The full import → discover → evidence → holdout → promote → paper loop has not been run on the target M2 8 GB machine with the user’s data.
7. **Documentation drift.** `STATUS.md` and the roadmap contain older snapshots. This report supersedes them for claims; they should not be quoted as current evidence without checking the commit.

**Single next action.** Run one frozen, real-data experiment end to end on this SHA: `discovery run` with the tradable-candidate gate at a documented trial budget → persist the evidence bundle → if and only if a tradable candidate survives, attach the one-shot locked holdout and evaluate `PROMOTION_V1` once. If no candidate survives, record NO TRADE and do not spend the holdout. Publish the artifacts either way. Do not tune against the holdout and do not add model families before this run.

## 9. What NOT to say

Do not say, write in a README, or imply in a demo:

- **“Autonomous profitable AI.”** No validated candidate, no paper-traded champion, no profitable forward record.
- **“The system understands books / trained on PDFs.”** Ingestion uses deterministic parsers and heuristic hypothesis extraction. It does not comprehend a book, and a document never becomes a trading rule or an order.
- **“MaskablePPO training.”** Training is plain PPO. Action masking exists in evaluation only.
- **“Validated on out-of-sample data.”** No real locked holdout has been spent. A validation-slice winner and an unspent one-shot holdout are not validation.
- **“Paper-traded.”** No champion ran in the paper engine. The synthetic harness paper stage is an engine check.
- **“Profitable.”** The only forward sample is 24 legacy-card trades with a CI crossing zero.
- **“Real-money ready.”** Real-money routing is disabled and out of scope; the MT5 adapter refuses every operation.
- **“The system learns from its paper outcomes.”** The outcome trigger and matured-outcome batch exist, but no real non-synthetic outcome batch has been produced.
- **“Green CI proves an edge.”** CI proves software behaviour. It does not prove market behaviour.
- **“22 trials found the best strategy.”** In the real-data smoke the best candidate was flat and the decision was NO TRADE.

## Appendix A — Number provenance

| Number | Source | Status |
|---|---|---|
| `2,320 passed, 3 skipped` | CI run `45550807353`, backend log | Verified by fetched log |
| `2,323 collected / 116 files` | `uv run pytest --collect-only -q` at this SHA | Verified locally |
| `263 frontend tests / 31 files` | CI run `45550807353`, frontend log | Verified by fetched log |
| `149 source files` (mypy) | CI backend log | Verified by fetched log |
| `29,586 H1 bars`, `22 evaluated`, `-0.0926 R`, `0/3 folds` | `docs/evidence/discovery-smoke-2026-11-24.json` and [research-smoke-2026-11-24.md](research-smoke-2026-11-24.md) | Verified; committed report of a run |
| `22 trials, 0 tradable, no_trade` (adjacent) | `.worktrees/realaccept/backend/runs/realaccept/03-discovery.json` | Verified on disk; **not this SHA** (`05c6001`) |
| Five classifier metrics | `runs/ml/*/metrics.json` (parent worktree) | Verified from artifacts; provenance flag absent |
| `21 PPO runs, 16 degenerate` | `backend/runs/*/manifest.json` (parent worktree) | Verified from manifests; legacy path |
| `24 trades, +0.502 R, PF 2.15, +$481, CI -0.04…+1.04` | [forward-test.md](forward-test.md) and `backend/runs/forward_history.jsonl` | Both agree; legacy card, replay |
| “No paper.sqlite” | Workspace-wide `find` | Not found; scoped to this workspace |
| “No holdout spent” | No `holdout*.sqlite`; no bundle with `locked_oos`; [methodology.md](methodology.md) | Verified by search and documentation |

Unverified items are marked where they appear: the real/synthetic provenance of the ML dataset, the adjacent real-acceptance run relative to this SHA, and the target-M2 acceptance.

## Appendix B — Reproduce the key checks

```bash
# Exact CI counts for this SHA (read-only)
gh run view 45550807353 --json status,conclusion,jobs
gh run view --job 138991347449 --log | grep -E "passed|skipped|Success: no issues"

# Collection count
cd backend
uv run pytest --collect-only -q

# Secret-hygiene tests at this commit
uv run pytest tests/test_no_committed_secrets.py tests/test_gitignore_hygiene.py -q
```

```bash
# Fail-closed policy, one-shot holdout, reproduction, risk non-bypass
cd backend
uv run pytest tests/test_promotion_policy.py tests/test_locked_holdout.py \
  tests/test_discovery_reproduce.py tests/test_risk_acceptance.py -q
```

```bash
# Real-data discovery and persisted evidence (writes a decision and a bundle,
# not a locked OOS result)
cd backend
uv run python -m scalping.cli discovery run \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --max-trials 64 --seed 7 --out runs/discovery/final-report.run.json

uv run python -m scalping.cli discovery evidence \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --spec runs/discovery/spec.json --out runs/discovery/final-report.evidence.json
```

The second command needs a concrete spec sidecar and a tradable candidate; it will not manufacture one. If the search returns NO TRADE, stop there.

## Appendix C — Document status

- `docs/methodology.md` is the clearest statement of the evidence ladder and matches the code’s fail-closed behaviour.
- `docs/provenance-audit-2026-11-24.md` is the gap report for provenance; its findings still apply at this commit.
- `docs/secret-audit-2026-11-24.md` and `docs/security-review-2026-11-24.md` are the security gap reports.
- `docs/autonomous-system-audit-2026-11-24.md` and `docs/current-state-addendum-2026-11-24.md` are historical snapshots; they predate the evidence producer and the green CI. Do not quote their “#136 open” or “main is red” statements as current.
- `STATUS.md` was minimally corrected in this change: the verified `main` hash and CI line now name `50fbca6` and the green runs, and the section that listed five merged PRs as “open work” now says they merged before this commit.

## Addendum — 2026-11-25 (late): runtime chain complete, not validated

**Commit:** `54eef5e7a25854c78f2bef2d26e4c42d50f3ab5e` (#220). This addendum records what merged after the report body was written at `50fbca6`: #215 optional PDF extraction, #216 provenance fail-closed, #217 read-only discovery API, #218 paper scheduler, #219 M30/H4 resample tests, #220 runtime promotion command and #178 audited gap policy.

**The claim matrix has not changed.** Every verdict in §0 still stands:

- **Validated: No.** No locked holdout has been spent and no bundle carries a real `locked_oos` section.
- **Paper-traded: No.** No evidence champion exists, so no paper session has run one.
- **Profitable: No**, and not claimed. The forward sample is still 24 legacy-card trades with a CI crossing zero.
- **Trained: unchanged.** The same local fits exist; no committed artifact and no governed model.

**What is complete now is the runtime chain, not the evidence.** Every link from a stored dataset to a scheduled paper advance has a runtime call site:

| Link | At `50fbca6` | At `54eef5e` |
|---|---|---|
| Provenance | Store sidecar only; dataset/evidence blind | `source`/`synthetic` carried through dataset artifact, evidence schema v4, holdout and promotion; `PROMOTION_V1` refuses `provenance_unknown` and `synthetic_data` (#216) |
| Promotion | Library-only, no call site | `discovery promote` and `promote_evidence`: identity check, dry run, one-shot holdout spend behind a non-empty owner token, second-spend refusal, decision record to `--out` (#220) |
| Paper scheduling | Human/caller enqueued every advance | `orchestrator run --schedule-paper` enqueues one `paper.advance` per evidence champion when a new decidable stored bar exists (#218) |
| Archive review | `/api/autonomy/*` status only | `GET /api/discovery/summary|candidates|candidates/{id}|trials`, read-only with `mode=ro` (#217) |
| PDF | `.pdf` refused | Optional `pypdf` extra, one section per page with page references; core install never imports it (#215) |
| M30/H4 | Alignment/context tests | Resample bucket alignment, left-closed boundaries and stored-derivation parity pinned in `test_store.py` (#219) |

“Complete” here means implemented and tested end to end in code. It does not mean the chain has produced market evidence. Two links still require a human: promotion is an explicit owner act, and the holdout spend is one-shot with no retry.

**Verification at `54eef5e`.** CI run `45627617157`: backend **2,435 passed, 8 skipped** in 386.52 s, lint `All checks passed!`, mypy `153 source files`; frontend **287 passed** in 34 files, build compiled; CodeQL `45627617898` success. The counts quoted in §2 of this report are at `50fbca6` and are historical. Those tests prove software behaviour, not a market edge.

**The single next action is unchanged, and it is now executable end to end:** run one frozen real-data experiment → `discovery evidence` → `discovery promote --dry-run` → if and only if a tradable candidate survives every non-holdout check, an owner-approved `discovery promote --spend-holdout --approval-token-env …` → `orchestrator run --schedule-paper`. If no candidate survives, record NO TRADE and do not spend the holdout. The exact commands and the one-shot/no-token-logging warnings are in `docs/runbook.md` §2.6.

**What not to say after this addendum:** that the system is validated, paper-traded, profitable or “runtime-proven”. A complete call graph and a green suite are `implemented` evidence. They are not `validated` evidence.
