# Autonomous system audit — 2026-11-24

> **Current-state notice (2026-11-24):** This document preserves its original
> historical code/PR audit snapshot. Its PR-state and main-commit references are
> not the latest verified snapshot. For current mainline status and corrected
> priorities, read `docs/current-state-addendum-2026-11-24.md` and
> `STATUS.md`. The latest verified main commit for this refresh is
> `54bcf38abe7ed654e99935360fb6527d23127493`; #107 and #110 are merged.
>
> Do not use historical statements below that say #107 is open, needs rebasing,
> or that #109 is still an open PR as current status. Technical limitations
> identified in the historical code review remain relevant unless the
> addendum explicitly updates them.
>
## Scope

Repository: resulbegmyradow33-sys/ai-scalping-studio  
Audited code baseline: main at 22cecdf5e2c6634b59433925b836bda6dea33199.\nDocumentation update: main advanced to e90121aa844e0665ac448e3f9986e87631bed56f after #109 and #105 merged on 2026-11-24; current PR state below is refreshed to that point.  
Default branch: protected main. Repository tree: 397 entries.

This is a source/document/PR/branch/CI audit. It is not a local execution of
private market-data stores. No claim is made that code has been trained or
validated on data unavailable in the repository.

## Executive verdict

**The project is moving in the intended strategy-independent direction, but
it is not yet an autonomous self-learning trading system.**

A safe DSL, bounded candidate search, cost-aware simulator evaluator, recorded
discovery decision, local research ingestion, persistent orchestration
infrastructure and DSL paper runner exist. The critical integration and
evidence loop is still missing: discovery uses one validation slice; champion
governance is not an independent OOS gate; orchestrator handlers/scheduler are
not connected; paper runner receives a spec instead of selecting a governed
champion; matured paper outcomes do not trigger the next research cycle. The
legacy rule path remains the current panel runtime.

## Mainline and PR state

| Change | Snapshot state | Meaning |
|---|---|---|
| #104 | Merged to main | One-command validation-slice discovery, bounded DSL search, simulator/cost evaluation, hold/ret_5 comparison, archive decision |
| #106 | Merged to main | Deterministic source-linked hypothesis drafts from text patterns |
| #108 | Merged to main | Risk/session gates exposed through action masks and enforced in evaluation |
| #107 | Open; 3 commits behind/diverged | Sentence-classification heuristics; rebase required after #106 merged |
| #109 | Merged to main | Champion history/provider/rollback; the PR explicitly says validation-only metrics and no OOS/walk-forward |
| #105 | Merged to main | Transformer benchmark/evaluator round-trip; useful research infrastructure, not a trading edge |
| #107 | Open; still based on pre-#109 main and stacked on #106 | Sentence-classification heuristics; must rebase before merge |

The latest mainline CI/CodeQL/Dependency Review records observed for #108 were
successful. PR descriptions report targeted tests passing. These prove code
quality at those commits, not market edge, OOS validity or profitability.
Branches must be rebased before merge. Main is protected.

## What is genuinely implemented

### Strategy-independent discovery foundation

- strategy/dsl/ defines a typed, bounded, versioned AST and closed interpreter.
  Generated strategies are not arbitrary Python.
- discovery/search.py implements bounded evolutionary search with seeds,
  candidate deduplication, mutation/crossover, attempt budgets and lineage.
- discovery/evaluate.py runs DSL candidates through the existing simulator
  and records net-of-cost trial metrics. It refuses dataset ids naming the
  final test/holdout split.
- discovery/run.py wires stored bars → cached causal features → chronological
  dataset geometry → validation slice → hold/ret_5 baselines → search → archive
  decision. It can end with no_trade, no_change or rejected.
- Discovery never promotes a candidate. This separation is correct.

### Knowledge foundation

- knowledge/ingest.py preserves originals/provenance and extracts TXT,
  Markdown, HTML and DOCX.
- knowledge/index.py provides a local searchable index.
- #106 creates deterministic sentence-pattern hypothesis drafts with exact
  source/section references and deduplication.
- #107 proposes deterministic sentence classification. It is not on main at
  this snapshot; it is heuristic triage, not semantic reading.

### Orchestrator foundation

- Persistent SQLite job store, append-only events, dependencies, priorities,
  retries, cancellation, pause/resume/safe-mode and crash recovery exist.
- File trigger infrastructure exists.
- The core is synchronous (run_once/run_until_idle) and explicitly has no
  scheduler thread in the current slice.
- Real handlers for ingestion, discovery, training, promotion, paper sessions
  and outcome analysis are not registered as one production pipeline.
- Default AllowAllGuard is not a measured RAM/CPU/disk resource guard.

### Paper foundation

- AutonomousPaperRunner executes a validated DSL StrategySpec through the
  existing PaperSession/Simulator/RiskEngine and stores strategy hashes.
- It has no broker/network/real-money path, correctly.
- It does not select a spec, read a champion automatically, watch a continuous
  feed or enqueue research from paper outcomes.
- ChampionProvider in open PR #109 is not integrated into main's paper lifecycle.

### Risk/evaluation foundation

- Simulator owns independent position sizing and risk gates.
- #108 adds shared entry-gate/action-mask logic for backtest evaluation.
- PPO training still uses plain PPO; do not claim MaskablePPO training.
- Walk-forward, purged CV, cost sensitivity and Monte Carlo tools exist, but
  discovery/promotion does not compose them into one evidence bundle.

## Critical gaps

### A. Discovery overfit / multiple testing — BLOCKING

run_discovery scores all candidates on one validation window. DslEvaluator is
one-candidate/one-frame and has no nested walk-forward. Selecting the best of
many candidates on one validation window is exposed to winner's curse and
data-snooping bias. The final holdout is protected from direct evaluator access,
but there is no complete promotion-grade independent evidence path.

Required: chronological walk-forward folds with purge/embargo; locked final
OOS; attempt counts and complexity penalties; parameter sensitivity; cost
stress; regime stability; confidence intervals; and a statistically justified
multiple-testing policy.

### B. Champion gate defaults are permissive — BLOCKING

GateCriteria defaults to zero/empty, disabling thresholds and accepting
candidates in exploratory search mode. This must never be a production
promotion policy. Promotion must require a named, versioned, non-empty policy
and complete evidence; missing OOS, cost, sample or regime data must reject.

PR #109 is now merged and adds champion governance and rollback, but its own
description says metrics are validation-only, with no OOS/walk-forward. It is
not yet a production trading promotion gate. A metric-only strict comparison
is insufficient.

### C. No closed autonomous orchestration — BLOCKING

Queue substrate exists; production handlers and long-running scheduler do
not. Need trigger→eligibility→job dependency routing, measured resource guards,
retry/cancel semantics across real jobs, and isolation so training failure
cannot stop paper replay.

### D. Research material is not a semantic research engine — HIGH

Current extraction is deterministic pattern matching. PDF is unsupported;
scanned PDFs need OCR; tables are not structurally ingested. Hypotheses remain
proposed drafts and are not automatically routed to discovery. Do not describe
ingestion as LLM training or human-level book comprehension.

### E. Legacy strategy path remains active — HIGH

The current UI/API simulation runner still loads PolicyCard and FlowMomentumRule;
absent/invalid cards can fall back to the legacy M1 momentum rule. The nightly
calibrator selects a fixed grid of flow-momentum parameters and can write the
legacy card. This is not the new DSL learner.

The five fixed strategy families in strategy/signals.py and LinUCB in
strategy/bandit.py are a finite, pre-authored strategy menu. They may remain
as baselines/ablation research, but must not define the discovery search space.

Do not delete them yet. Preserve reproducibility, inventory every consumer,
then migrate the UI/API to a validated DSL champion. After acceptance, retire
legacy auto-promotion/fallback in a separate reviewed change. If no DSL
champion exists, the autonomous system must report NO TRADE rather than
silently selecting the old rule.

### F. Paper feedback and continuous data — HIGH

The paper runner consumes a stored slice. It needs appended-bar continuation,
lifecycle scheduling, matured-outcome analysis and feedback handoff to
discovery. Do not train from a paper result until the trade is fully closed
and correctly attributed.

### G. Resource and data limits — HIGH

Target M2 has 8 GB unified memory. Resource hook is not measurement. Chunked
ingestion/features, disk quotas, actual process RSS and a global heavy-job
semaphore must be proven on target hardware. Real data beyond XAUUSD/XAGUSD
is incomplete; data is local and must be versioned by manifest/checksum.

## Legacy component classification

| Component/path | Status | Keep/use policy |
|---|---|---|
| strategy/rule.py, flow.py, card.py, calibrate.py | Legacy flow-momentum rule | Freeze as incumbent benchmark/compatibility until DSL champion migration is proven |
| backend/config/strategy_card.json, docs/cards/* | Historical seed/evidence | Preserve provenance; never constrain autonomous search |
| scripts/nightly_calibrate.py, API strategy calibration route | Legacy fixed-grid calibration | Do not call it the new learning loop; retire automatic promotion after DSL path replaces it |
| api/runner.py card/rule fallback | Active legacy UI runtime | Replace only after no-fallback DSL champion lifecycle and regression tests |
| strategy/signals.py five families + strategy/bandit.py LinUCB | Legacy finite-menu selector | Baseline/ablation only; do not add arms as autonomous engine |
| PPO/BC + nightly_rl.py | Experimental baseline | Retain for comparison; no privileged promotion |
| hold, ret_5, optional flow rule | Baselines | Identical timestamps/costs; never constrain generation |

## Ordered implementation plan

### P0 — Align active work

1. Rebase #107, #109 and #105 onto latest main; all branches are diverged and
   behind at this audit.
2. Review #107 as deterministic heuristic triage only.
3. Keep #109 from being treated as a production promotion path until evidence
   requirements below are implemented. Champion storage/rollback can be
   reviewed as a foundation.
4. #105 is optional; do not let it displace core work.

### P1 — Promotion-grade evidence pipeline

1. Define immutable CandidateEvidenceBundle: dataset ids/checksums,
   symbol/timeframe, feature version, code/config/seed, fold boundaries,
   purge/embargo, per-fold trades/metrics, cost stress, regime coverage,
   confidence intervals and trial-count/search-budget context.
2. Implement several non-overlapping chronological walk-forward folds. Keep
   final OOS inaccessible to candidate generation, HPO and repeated promotion.
3. Add multiple-testing/data-snooping controls; report search exposure, penalise
   complexity and unstable parameters, test parameter neighbourhoods and costs.
4. Compare every candidate with hold, ret_5, compatible legacy incumbent and
   current champion on identical timestamps and costs.
5. Fail closed if evidence is missing, stale or mismatched.

### P2 — Champion governance

1. Separate exploratory search criteria from promotion policy types.
2. Promotion requires an explicit named/versioned policy; empty criteria cannot
   promote.
3. Compare candidate vs champion across identical walk-forward/OOS evidence,
   not one validation scalar.
4. Preserve atomic history, rejection reason, prior champion and rollback.
   Disable autonomous force overrides.
5. Test missing evidence, leakage, stale datasets, scope mismatch,
   non-regression and rollback.

### P3 — Real orchestrator

1. Register handlers for data detection/validation, document ingest,
   hypothesis extraction, dataset build, discovery, walk-forward evaluation,
   champion review/promotion, paper start/advance, outcome analysis and drift.
2. Add long-running scheduler/daemon and triggers for new data, documents,
   closed paper trades and periodic health checks.
3. Add measured RAM/CPU/disk guard, one-heavy-job semaphore, cancellation,
   retries, restart recovery and dependency deduplication.
4. Keep paper execution in its own failure domain; training failures cannot
   pause or corrupt paper ledger.

### P4 — Knowledge-to-experiment bridge

1. Add optional pinned offline PDF text extraction with page refs; optional OCR
   only for scanned files and within resource limits. Preserve parser version.
2. Preserve table structure where possible; classify claims/observations/
   examples/opinions as heuristic metadata with confidence and source refs.
3. Route proposed hypotheses into quantitative test proposals. A hypothesis is
   not an order, strategy or truth claim.
4. Ingest backtest/paper/broker exports as source-linked evidence; prevent
   matured-outcome leakage.

### P5 — Autonomous paper loop

1. ChampionProvider selects only a validated DSL champion for exact scope.
2. No/invalid champion means NO TRADE with auditable reason; no silent
   flow_momentum fallback.
3. Continue on appended market bars and resume after process restart.
4. Record immutable spec hash, data hash, risk decision, action, fills, costs,
   exit and outcome; analyse only matured outcomes.
5. Enqueue next eligible research cycle without training on locked OOS or
   incomplete outcomes.

### P6 — M2 acceptance and monitoring

1. Measure peak RSS, CPU, disk and elapsed time for ingest, discovery,
   walk-forward, training and paper operation on M2 8 GB.
2. Enforce chunking, one heavy job, bounded trials, disk quotas,
   checkpoint/resume and graceful cancellation.
3. Run full import→validate→discover→walk-forward/OOS→champion decision→
   paper→matured feedback→next cycle on real XAUUSD data.
4. Report code/test/training/validation/paper evidence separately. Do not claim
   profitability from a backtest or small forward sample.

## Definition of done

The project is not autonomous until a reproducible acceptance run proves:

new data/document → validate/provenance → hypothesis/candidate generation →
bounded discovery → walk-forward + locked OOS → fail-closed champion decision →
autonomous DSL paper run → matured outcome analysis → next eligible research
cycle → drift/rollback.

It must run without owner-authored entry/exit rules, allow NO TRADE, stay within
measured M2 limits, preserve a full audit trail and keep real-money execution
disabled. Passing proves engineering behaviour, not future profitability.
