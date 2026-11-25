# §36 acceptance gap — 2026-11-25

Read-only assessment of the master-prompt §36 criteria at `main` base
`50fbca6` (`feat/assess1`, no commits ahead of base). This report adds no code.
It separates what exists from what §36 actually requires: code, tests **and a
recorded reproducible run** for each item.

**Verdict: not accepted.** 4 of 13 criteria are met, 7 are partial, 2 are not
met. The blocker is not missing modules. It is that the two artifacts §36
requires for the autonomous claim — a spent locked out-of-sample evaluation and
a promotion derived from it — do not exist at runtime, so no candidate becomes
a champion and the shipped paper loop has nothing to advance.

## Scorecard

| # | §36 criterion | Status | What exists | Missing artifact |
|---|---|---|---|---|
| 1 | Import unseen data, validate provenance/quality, leakage-safe dataset | **partial** | Validator, quality report, provenance sidecar, chronological purged builder, real H1 store with `synthetic:false` | No recorded import + quality report of a *new* dataset; dataset identity carries no `source`/`synthetic` |
| 2 | Generate multiple candidates without owner entry/exit rules | **met** | Grammar search `random_spec`/`evolve`; real H1 run evaluated 11 candidates, 5 tradable | — (grammar primitives/features are human-authored; rules are not) |
| 3 | Reproduce candidate from data hash, code version, config, seed | **met** | `discovery evidence` producer; `discovery reproduce`; checksum match verified on real H1 | — |
| 4 | Reject overfit candidates on unseen chronological data + cost stress | **partial** | 3 chronological folds, purge/embargo, cost stress; real bundle rejected on cost and fold metrics | No locked final-OOS rejection; no deliberately overfit candidate experiment |
| 5 | Select NO TRADE / conclude no candidate beats champion | **partial** | `no_trade`/`no_change` decisions; real H1 run `no_change`; champion comparison code + tests | No real champion to compare against; champion half only unit-tested |
| 6 | Import sample PDF + dataset, source refs, traceable hypotheses | **partial** | TXT/MD/HTML/DOCX ingest, FTS index, source refs, deterministic hypothesis drafts | PDF extraction is explicitly unsupported (`knowledge/ingest.py:492`); no recorded PDF→hypothesis run |
| 7 | Detect new data / matured outcomes, auto-queue research | **partial** | Data-file trigger → `dataset.build`; paper-outcome trigger → proposal; daemon scans | Matured outcomes stop at a proposal; discovery enqueue needs human approval; no recorded daemon run on real data |
| 8 | Train candidate, compare champion, promote/reject/rollback, audit | **partial** | Fail-closed `PROMOTION_V1`; rollback/history; real candidate rejected with codes | No promoted record, no `promote_with_evidence` runtime call site; accepting-path tests use a test policy |
| 9 | Paper trade autonomously without a human selecting the strategy | **not met** | `paper.advance` resolves an evidence champion; integration test seeds one | No evidence-mode champion exists; the acceptance paper stage bypasses the promotion decision (it ran a candidate the gate rejected); no trigger enqueues `paper.advance` |
| 10 | Risk controls cannot be bypassed by spec/RL/model/orchestrator | **met** | Frozen `RiskEngine` in the simulator; 40 non-bypass tests incl. job payload and kill switch | — (no real forward run, but structural impossibility is tested) |
| 11 | Recover after restart, interrupted training, cancelled jobs | **met** | Orchestrator recovery/cancel tests, paper restart/gap tests, NN atomic checkpoints + stop/pause tests | — |
| 12 | Operate within measured M2 limits | **partial** | Real M2 measurements (peak RSS 438–884 MiB), measured guard, `m2_8gb` daemon profile, 51 guard tests | Full pipeline not measured: paper stage failed on both real stores; no walk-forward; bare orchestrator still `AllowAllGuard` |
| 13 | Demonstrate the complete loop | **not met** | Synthetic acceptance harness runs 8 stages and reports honest false claims | No real import→discover→locked OOS→promote→paper→feedback artifact; holdout skipped; no promotion |

## Criterion detail

### 1. Unseen data import, provenance/quality, leakage-safe dataset — partial

Evidence:

- Structural validation: `backend/src/scalping/data/validate.py`; quality
  reporting: `data/quality.py`; provenance sidecar with a content checksum:
  `data/provenance.py:31,50,87,141`.
- Chronological, purged manifest: `datasets/builder.py:57` (`DatasetManifest`),
  with `test_dataset_builder.py` (5 tests), `test_validate.py` (5),
  `test_provenance.py` (4).
- The real H1 store used below has a sidecar with `"source": "csv:data-real-csv"`,
  `"synthetic": false`, `"checksum": "e48cd48d6aa1cdfb8071a511f0a2464a"`.
- My real run built `validation:XAUUSD:H1:h1:b189a7fdbad6` (15,003 validation
  rows) from 100,083 stored H1 bars.

Missing:

- No committed, recorded import of a *previously unseen* dataset followed by a
  quality/provenance report. The acceptance `import_validate` stage is synthetic
  (`acceptance/run.py:446`).
- Provenance stops at the store sidecar. `DatasetManifest`
  (`datasets/builder.py:57-95`) has no `source`, `synthetic` or provenance
  checksum, and `CandidateEvidence` (`discovery/evidence.py:489`) has none
  either. The evidence bundle's dataset id `XAUUSD:H1:h1:b189…` cannot be joined
  to the sidecar id `XAUUSD:H1:e48cd…`; there is no field that says "this
  evidence came from real bars". This is the same gap documented in
  `docs/provenance-audit-2026-11-24.md` G1/G2 and still present at this commit.
- The feature matrix is a fixed 59-column human-curated registry
  (`features/registry.py`). The owner supplies no per-run rules, but the feature
  vocabulary is not discovered either. This is a scope caveat, not the blocker.

### 2. Generate multiple candidates without owner rules — met

Evidence:

- `strategy/dsl/mutate.py:230` `random_spec` draws entries, exits, an optional
  no-trade condition, holding limit and sizing from the typed grammar;
  `StrategyFactory` at `:514`.
- `discovery/search.py:345` `evolve` runs the bounded population loop with
  dedupe, attempt budget and lineage.
- Real-data artifact: `docs/evidence/discovery-smoke-2026-11-24.json` (#143)
  evaluated 22 candidates on 29,586 XAUUSD H1 bars and recorded `no_trade`.
- My independent run (below) evaluated 11 candidates, 5 with closed trades, and
  produced full random trees (for example candidate `70fe1963…` with
  `cross_below`, `between`, `change` and feature operands).
- Tests: `test_discovery_search.py` (24), `test_strategy_dsl.py` (45).

The owner did not supply the generated rules. The grammar operators and feature
pool are code, not owner input.

### 3. Reproduce from data hash, code version, config, seed — met

Evidence:

- Producer: `discovery persist` / `persist.py:546 load_evidence_bundle`;
  `cli.py:1804` `discovery evidence` writes the dataset artifact, evidence JSON
  and spec sidecar.
- Replayer: `discovery/reproduce.py`; `cli.py:1779` `discovery reproduce`.
- Identity in the bundle: dataset id/checksum, code version (digest of the
  arithmetic modules), settings digest, seed, fold geometry, trial context.
- Verified here on real H1 data:

  ```text
  evidence checksum 211f48b19a70ed436a2b9cb02a5709f313251a21aae1b1533d82f0acbe08ed99
  discovery reproduce -> {"reproduced": true, "checksum": "211f48b1...", "mismatches": []}
  exit code 0
  ```

  This was run with the default Polars thread pool; #169 fixed the earlier
  thread-dependent regime sum. The temporary dataset artifact written into the
  store for this check was removed afterwards so the input store is unchanged.
- Tests: `test_discovery_reproduce.py` (23), `test_discovery_persist.py` (20).

### 4. Reject overfit candidates on unseen chronological data + cost stress — partial

Evidence:

- Three expanding, non-overlapping folds cut only from train + validation, with
  purge/embargo (`discovery/evidence.py:1-22`); the final test split is never
  sliced.
- Cost-stress rows at 1.5× and 2.0× plus bootstrap intervals and regime
  coverage in the bundle.
- Real rejection: I built evidence for the best H1 candidate and evaluated it
  with `PROMOTION_V1`. Result: `accepted=False`, codes
  `min_positive_fold_fraction`, `min_expectancy_r`, `min_profit_factor`,
  `min_ci_lower_r`, `cost_stress`, `stress_degradation`, `missing_locked_oos`.
- Tests: `test_discovery_evidence.py` (41), `test_evidence_adversarial.py` (40).

Missing:

- The locked final out-of-sample split is never scored. Every real bundle
  carries `locked_oos: null`; `PROMOTION_V1` correctly rejects with
  `missing_locked_oos` (`discovery/promotion.py:351-369`).
- No recorded experiment takes a deliberately overfit candidate and shows the
  untouched chronological OOS rejecting it. The existing folds are internal
  test blocks, not the locked holdout.
- The multiple-testing screener (`discovery/selection.py`) advises but is not
  consumed by the promotion decision (`docs/methodology.md` §3.4).

### 5. NO TRADE / no candidate beats the champion — partial

Evidence:

- Discovery's decision vocabulary is `no_trade`, `no_change`, `rejected`;
  `promoted` is never written (`discovery/run.py:21-40`).
- Artifact #143 recorded `no_trade` on real H1 data.
- My run recorded `no_change`: "candidate 70fe1963823c net_profit -424.274 does
  not beat the best baseline (hold net_profit 0) after 11 trial(s)".
- Tests: `test_discovery_run.py:285` (gate rejects everything → `no_trade`),
  `:367` (nothing beats the baseline → `no_change`).
- Champion comparison exists (`promotion.py:1523 _check_champion`) and is
  covered by `test_promotion_policy.py:1007-1242`.

Missing:

- A real champion. The "no candidate beats the champion" half is only exercised
  with synthetic evidence and a test policy; no run has a stored champion to
  compare against.

### 6. PDF + dataset + source refs + traceable hypotheses — partial

Evidence:

- Ingest, provenance, index, classification, hypothesis drafts and audited
  routing exist: `knowledge/{ingest,index,classify,extract,hypothesis,routing}.py`.
- Source references are preserved and hypotheses remain `proposed` until a
  human moves them; prose cannot create a strategy or order.
- Tests: `test_knowledge_ingest.py` (16), `test_knowledge_extract.py` (5),
  `test_knowledge_classify.py` (11), `test_knowledge_tables.py` (36),
  `test_knowledge_index.py` (16), `test_hypothesis_routing.py` (31).

Missing:

- PDF is explicitly unsupported: `knowledge/ingest.py:492-497` raises "PDF text
  extraction needs an optional parser". `test_knowledge_ingest.py:195` asserts
  the refusal rather than an extraction. There is no OCR.
- No recorded PDF → preserved page/section refs → hypothesis run exists. As
  written, criterion 6 fails on its first verb.

### 7. Detect new data / matured outcomes, auto-queue research — partial

Evidence:

- `orchestrator/data_triggers.py` turns a new/changed file under a watched data
  root into exactly one `dataset.build` job, with content hashing and a durable
  cursor.
- `orchestrator/paper_triggers.py` scans the paper ledger, emits one
  `paper.outcomes` trigger per matured-outcome fingerprint, refuses synthetic
  batches by default, and dedupes across restart.
- Daemon CLI supports `--watch-data-dir` and `--watch-paper-ledger`
  (`cli.py` orchestrator parser).
- Tests: `test_data_triggers.py` (17), `test_paper_outcome_triggers.py` (28),
  `test_paper_outcome_cli.py` (6), `test_orchestrator_integration.py` (6).

Missing:

- A matured batch only produces a **proposal** with `executed:false`
  (`paper_triggers.py:16`, `:615`). Enqueueing `discovery.run` requires a human
  `ProposalStore.approve`; the proposal bridge is deliberate
  (`proposal_handlers.py:1-30`). There is no unattended eligibility policy.
- A data trigger enqueues `dataset.build`, not discovery.
- No recorded daemon run on real data showing new data → research job without a
  human.

### 8. Train candidate, compare champion, promote/reject/rollback — partial

Evidence:

- `PROMOTION_V1` is named, versioned, non-empty, has no force flag and cannot
  disable `require_locked_oos` (`promotion.py:149-369`).
- `promote_with_evidence` writes a sealed archive decision and champion row;
  history and rollback exist in `discovery/archive.py` and `champion.py`.
- Rejection is real: the H1 bundle above was rejected with seven codes.
- Rollback and champion identity are tested:
  `test_discovery_champion.py:346`, `test_promotion_policy.py:1681`,
  `test_promotion_policy.py:1007-1242`.

Missing:

- No merged CLI or orchestrator handler calls `evaluate_promotion` /
  `promote_with_evidence` outside tests and the acceptance harness.
  `orchestrator/handlers.py:26-28` states no handler promotes. The CLI exposes
  `discovery run`, `reproduce` and `evidence` only; there is no `promote` and no
  holdout command.
- No committed promotion record. `champion_promotions` is empty in any archive
  produced by the shipped commands.
- The public `promote_with_evidence` has no successful test. The only calls in
  tests assert refusals (`test_promotion_policy.py:625`,
  `test_evidence_adversarial.py:525,545`); successful promotion goes through the
  private `_promote_with_evidence` with `require_production=False`
  (`tests/promotion_support.py:30`), i.e. a test backdoor.
- `evaluate_promotion(PROMOTION_V1)` *can* accept a bundle with an attached
  synthetic holdout (`test_locked_holdout.py:793`), but that is a unit test with
  hand-built evidence, not a runtime promotion or a real-data holdout.

### 9. Paper trade autonomously without a human selecting — not met

Evidence:

- `paper.advance` reads a governed champion through `ChampionProvider`
  (`orchestrator/handlers.py:953-1010`); a missing, exploratory, manual or
  invalid champion is an auditable NO TRADE, never a legacy fallback.
- `test_orchestrator_handlers.py:498` advances a champion that was seeded with a
  test-only promotion; restart, append and gap behaviour are covered.

Missing:

- No evidence-mode champion exists at this commit. Without a promotion record,
  the shipped `paper.advance` handler has no champion and returns NO TRADE.
- Nothing enqueues `paper.advance` periodically. The daemon scans data files
  (→ `dataset.build`) and paper ledgers (→ `paper.outcomes` proposal) only; the
  handler exists, but a caller must create the advance job. There is no paper
  trading clock or scheduler.
- The only recorded end-to-end paper run is the acceptance harness. Its paper
  stage uses whatever spec `_stage_evidence` selected (`acceptance/run.py:797`):
  in my run that was the search's best generated spec (`771333f3…`,
  `candidate_source=discovery_best`), which the promotion stage had just
  rejected (`accepted=false`). When the candidate is evidence-ineligible, the
  stage falls back to `acceptance_spec()` (`:146`), a fixed long/holding-5
  fixture. Either way, the paper run is outside the governed champion path and
  no promotion is recorded.
- `AutonomousPaperRunner` writes no provenance and no dataset id in the session
  config (`paper/autonomous.py:190-260`), and its outcomes classify as
  `synthetic=None`; the provenance audit flags this as G3.
- `paper.advance` accepts only `symbol`, `timeframe`, `bars`
  (`handlers.py:627-641`), and the continuous runner defaults to strict gaps
  (`paper/continuous.py:172`). The measured paper stage failed on both real
  stores with `BarGapError` (`docs/m2-measurements-2026-11-24.md:158-168`).
  There is no recorded real-store paper advance with a gap policy.

What is **not proven**: that a candidate generated by the search is ever
selected, promoted and traded without a human. What is proven: the paper engine
and the governed-champion reader work when a champion is inserted by test code.

### 10. Risk controls cannot be bypassed — met

Evidence:

- The simulator binds a frozen `RiskEngine` (`env/simulator.py:266`) and asks it
  before every entry, stop and kill switch.
- `tests/test_risk_acceptance.py` (40 tests) covers: a spec cannot carry a risk
  field (`:259`), no override parameter exists (`:579`), a risk key in a job
  payload is a permanent failure (`:666`), the kill switch blocks the same entry
  in simulator and evaluator (`:302`), and sizing/notional caps hold (`:240`).
- `tests/test_risk_limits.py` (50) pins the immutable limit semantics.
- `paper/mt5.py` is a disabled adapter that refuses every operation.

Caveat: this is structural impossibility demonstrated by tests, not a real
forward run. A passing test is not a market result.

### 11. Recovery after restart / interruption / cancellation — met

Evidence:

- Orchestrator: `test_orchestrator_core.py::test_start_recovers_interrupted_jobs`,
  `::test_controls_survive_a_restart`;
  `test_orchestrator_integration.py::test_restart_keeps_queued_jobs_requeues_running_and_never_repeats`;
  `test_orchestrator_cancel.py` (14 tests).
- Paper: `test_paper_continuous.py::test_restart_midway_resumes_without_duplicates`,
  `test_paper_gaps.py` (29), ledger resume tests
  (`test_paper_ledger.py::test_runner_resumes_after_reopen_without_duplicating_history`).
- Training: atomic checkpoints in `nn/train.py`;
  `test_train_smoke.py::test_stop_event_interrupts_and_leaves_a_loadable_run`
  and `::test_pause_event_freezes_then_resumes`.

All listed files passed locally at this commit.

### 12. Measured M2 limits — partial

Evidence:

- `docs/m2-measurements-2026-11-24.md` and
  `docs/evidence/bench-2026-11-24.json`: Mac14,3 M2 8 GB, peak RSS 839.6 MiB
  (M1 20k), 883.9 MiB (M1 200k), 438.8 MiB (H1 whole store).
- `orchestrator/resources.py` is a measured fail-closed guard; the daemon uses
  the `m2_8gb` profile by default; `test_orchestrator_resources.py` (51 tests).

Missing:

- The paper stage of the measurement suite **failed** on both real stores with
  `BarGapError`; 4 of 5 stages ran. No walk-forward stage was measured. The M1
  run covered 11.3% of the store. The bare `Orchestrator` still defaults to
  `AllowAllGuard`, so a caller can bypass measurement.
- No signed M2 acceptance protocol or SLA. The document itself says
  `m2_acceptance=false`.

### 13. Complete loop — not met

Evidence:

- The synthetic acceptance harness runs import → validate → dataset →
  discovery → evidence → promotion decision → optional holdout → paper →
  matured outcomes (`acceptance/run.py`), and its report honestly keeps
  `validated`, `paper_traded` and `profitable` false.
- My run of the exact documented command exited 0 with all stages passed except
  `holdout: skipped`.

Why this is not the §36 loop:

- The holdout was skipped because the pre-holdout policy rejected; no locked OOS
  was spent.
- The paper stage bypassed the promotion decision: it ran the discovery best
  spec (`771333f3…`) although the promotion stage had rejected it
  (`accepted=false`), and it recorded no promoted champion. There is no
  promotion-gated paper run.
- The data was synthetic. There is no real-data artifact, no promotion, no
  governed paper session, and no matured outcome that triggered the next cycle.

## “Autonomous, no owner-authored strategy” — what is and is not proven

Proven:

1. The DSL grammar produces novel candidate trees without owner entry/exit
   rules, and the bounded search evaluates them on stored bars.
2. The search can record NO TRADE (`no_trade`) or "no change"
   (`no_change`) and refuses to promote.
3. A chosen candidate's experiment can be reproduced from its saved identity,
   and promotion-grade evidence can be built and rejected fail-closed.

Not proven:

1. That any autonomously generated candidate is ever **chosen for evidence
   build** by the system. `discovery evidence` requires a human-supplied
   `--spec` (`cli.py:1809`); the run report exposes the best candidate, but
   nothing pipes it into the evidence producer.
2. That the locked one-shot holdout is ever spent. `HoldoutLedger` has no CLI or
   orchestrator caller; spending it requires ad-hoc library code.
3. That any candidate is ever **promoted**. No shipped command or handler calls
   `promote_with_evidence`, the public API has no successful test, and the only
   successful path is the private test helper with `require_production=False`.
4. That paper trading ever runs on a **promoted** strategy. The governed path is
   tested with a seeded champion; the recorded paper run used a generated spec
   that the promotion stage had rejected, not a champion.
5. That matured paper outcomes feed the next discovery cycle automatically. They
   produce a proposal that waits for human approval.

In short: the system can **discover and reject**. It has not demonstrated that
it can **select, promote, trade and learn** without a human in the loop.

## Single biggest blocker

**There is no runtime path that spends the one-shot locked holdout and promotes
the resulting candidate.** The evidence builder, holdout ledger and promotion
policy are separate libraries with no production call site (the synthetic
acceptance harness evaluates a decision but never promotes). Because no
evidence-mode champion can exist, the shipped `paper.advance` has no champion to
advance, and criteria 4, 5 (champion half), 8, 9 and 13 cannot be met. A second
runtime link is also missing for criterion 9: no scheduler or trigger enqueues
`paper.advance`, so even a manually promoted champion would wait for a caller.

The immediate second blocker is provenance: `DatasetManifest` and
`CandidateEvidence` carry no `source`/`synthetic`, so even a wired promotion
cannot prove the bars were real rather than synthetic. Fixing the call site
without fixing provenance would create a promotion that looks governed but
cannot fail closed on the master prompt's §38 provenance rule.

## Five-step path to acceptance

Each step is ordered by dependency. Commands assume `cd backend` and the local
real store at `../data-real-store` (adjust to the owner's store). Steps 2–5
describe the target commands; they do not exist yet.

### Step 1 — Make provenance fail closed through dataset → evidence → promotion

Add `source`, `synthetic` and `provenance_checksum` to `DatasetManifest` and the
dataset artifact; carry them into `EvidenceConfig`, `CandidateEvidence` and
`LockedOOSEvidence`; make `PROMOTION_V1` reject `synthetic is not False` or
unknown provenance. Decode old bundles as unknown, never as real.

```bash
cd backend
uv run pytest tests/test_provenance.py tests/test_dataset_builder.py \
  tests/test_dataset_artifact.py tests/test_promotion_policy.py -q
```

Acceptance for this step: a synthetic bundle is rejected by `PROMOTION_V1` with
a provenance code; a real bundle without provenance is also rejected. No
promotion is possible until this passes.

### Step 2 — Add the runtime holdout + promotion command

Implement one reviewed command (CLI and/or orchestrator handler) that:

1. loads a bundle and checks identity;
2. evaluates an explicit pre-holdout policy with `require_locked_oos=False`;
3. seals and unlocks the `HoldoutLedger` for that dataset build;
4. spends the holdout once, attaches the `locked_oos` section;
5. re-evaluates `PROMOTION_V1` and refuses on any missing code;
6. calls `promote_with_evidence` and writes the archive/champion plus a report;
7. refuses a second spend (`holdout_spent`).

```bash
# target interface, implemented in this step
cd backend
uv run python -m scalping.cli discovery promote \
  --bundle ../runs/discovery/XAUUSD-H1/evidence.json \
  --store-root ../data-real-store \
  --policy promotion-v1 \
  --out ../runs/discovery/XAUUSD-H1/promotion.json

uv run pytest tests/test_locked_holdout.py tests/test_promotion_policy.py \
  tests/test_discovery_champion.py -q
```

Acceptance: a second invocation on the same dataset build returns
`holdout_spent`; a rejected candidate leaves the champion unchanged and records
the reason.

### Step 3 — Record a real candidate experiment and a negative control

Run a real search with a defensible budget, export the best spec, build evidence,
and record both a rejection and (if it passes) a promotion. Also build a
deliberately overfit candidate and show it is rejected on the held-out folds and
the locked OOS.

```bash
cd backend
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery run \
  --root ../data-real-store --symbol XAUUSD --timeframe H1 \
  --max-trials 200 --population 16 --generations 8 --seed 7 \
  --archive ../runs/discovery/xauusd-h1.sqlite \
  --out ../runs/discovery/xauusd-h1.run.json --no-cache

# export best.canonical to spec.json, then
POLARS_MAX_THREADS=1 uv run python -m scalping.cli discovery evidence \
  --root ../data-real-store --symbol XAUUSD --timeframe H1 \
  --spec ../runs/discovery/spec.json --folds 3 --stress 1.5,2.0 \
  --trials 200 --search-budget 200 \
  --out ../runs/discovery/xauusd-h1.evidence.json
```

Acceptance: the run report, bundle checksum, fold metrics, rejection codes and
reproduction (`discovery reproduce` exit 0) are committed as one artifact. A
NO TRADE outcome is acceptable; a fabricated winner is not.

### Step 4 — Run the governed paper loop and close the feedback edge

Only if step 2 produced an evidence-mode champion. Add a paper-advance trigger
(or a scheduled job enqueuer) so the daemon, not a human, creates
`paper.advance` when new bars are available. Advance on a gap-free real slice,
or implement `skip_with_report` in the handler; verify the ledger records the
champion spec hash at session and signal level, then let the matured-outcome
trigger emit a proposal and approve one to enqueue `discovery.run`.

```bash
cd backend
# target: paper.advance is enqueued by a trigger, not a caller; gap policy and
# promotion provenance are part of the job payload
uv run python -m scalping.cli orchestrator run \
  --forever --profile m2_8gb \
  --root ../data-real-store \
  --runs-root ../runs \
  --archive-root ../runs \
  --db ../runs/orchestrator.sqlite3 \
  --watch-data-dir ../data-real-store \
  --watch-paper-ledger ../runs/paper.sqlite

uv run pytest tests/test_paper_continuous.py tests/test_paper_gaps.py \
  tests/test_paper_outcome_triggers.py tests/test_orchestrator_handlers.py -q
```

Acceptance: a paper session exists whose `model_version` equals the promoted
spec hash and whose job result carries the `promotion_id`; the matured batch
fingerprint appears once; the next discovery job is attributable to an approved
proposal.

### Step 5 — Close PDF and M2, then publish one acceptance report

Add an optional pinned offline PDF parser (no OCR required for a text PDF),
record a PDF → page/section refs → hypothesis run, and re-run the full pipeline
measurement including the paper stage and walk-forward. Publish one report that
uses the evidence ladder without promotion of lower rungs.

```bash
cd backend
uv run pytest tests/test_knowledge_ingest.py tests/test_knowledge_extract.py \
  tests/test_hypothesis_routing.py -q

POLARS_MAX_THREADS=1 uv run python scripts/bench_pipeline.py --real-store \
  --store-root ../data-real-store --timeframe H1 --suite full \
  --out ../runs/bench/h1-full.json

uv run python -m scalping.cli acceptance run --out ../runs/acceptance
```

Acceptance: the PDF path has a recorded extraction and source refs; the M2
report has every stage (including paper) with peak RSS/CPU/disk and identifies
the machine; the final acceptance report cites the real-data run, the holdout
spend, the promotion record and the governed paper session, or states plainly
which of them is still missing.

## Verification log for this report

Environment: `uv sync --frozen --offline` succeeded; Python 3.12.13,
pytest 9.2.0. All commands from `backend/`.

| Command | Result |
|---|---|
| `uv run pytest tests/test_acceptance.py tests/test_promotion_policy.py tests/test_locked_holdout.py tests/test_discovery_reproduce.py` | 187 passed (exit 0) |
| `uv run pytest tests/test_discovery_run.py tests/test_discovery_evidence.py tests/test_discovery_champion.py tests/test_discovery_selection.py tests/test_discovery_search.py tests/test_discovery_evaluator.py tests/test_evidence_adversarial.py tests/test_candidate_gate.py` | 221 passed (exit 0) |
| `uv run pytest tests/test_paper_autonomous.py tests/test_paper_continuous.py tests/test_paper_ledger.py tests/test_paper_outcome_triggers.py tests/test_paper_outcome_cli.py` | 97 passed (exit 0) |
| `uv run pytest tests/test_orchestrator_core.py tests/test_orchestrator_handlers.py tests/test_orchestrator_integration.py tests/test_orchestrator_daemon.py tests/test_orchestrator_resources.py tests/test_orchestrator_cancel.py tests/test_data_triggers.py` | 177 passed (exit 0) |
| `uv run pytest tests/test_knowledge_ingest.py tests/test_knowledge_extract.py tests/test_knowledge_classify.py tests/test_knowledge_tables.py tests/test_knowledge_index.py tests/test_hypothesis_routing.py tests/test_proposal_bridge.py` | 129 passed (exit 0) |
| `uv run pytest tests/test_risk_acceptance.py tests/test_risk.py tests/test_risk_limits.py tests/test_dataset_builder.py tests/test_dataset_artifact.py tests/test_provenance.py tests/test_validate.py tests/test_strategy_dsl.py tests/test_mtf_alignment.py tests/test_bench_pipeline.py` | 223 passed (exit 0) |
| `uv run pytest tests/test_discovery_persist.py tests/test_discovery_selection.py tests/test_discovery_archive.py tests/test_dataset_artifact.py` | 90 passed (exit 0) |
| `uv run pytest tests/test_cli.py tests/test_paper_gaps.py tests/test_nn_models.py tests/test_nn_data.py tests/test_nn_evaluate.py` | 110 passed (exit 0) |
| `uv run pytest tests/test_api_autonomy.py` | 25 passed (exit 0) — the historical red test is fixed at this commit |
| `uv run python -m scalping.cli acceptance run --out /tmp/assess1-acceptance` | exit 0; stages: import_validate, dataset, discovery (`no_change`), evidence (`candidate_source=discovery_best`, spec `771333f3…`), promotion (`accepted=false`, policy `acceptance-test-policy`), **holdout `skipped`**, paper_advance (same rejected spec; 8 trades, synthetic), matured_outcomes; claims `validated=false`, `paper_traded=false` |
| `discovery run` on real XAUUSD H1 (`data-real-store`, 100,083 bars, 12-trial cap, seed 7, `POLARS_MAX_THREADS=1`) | exit 0; 11 evaluated, 5 tradable, decision `no_change`, best `70fe1963…` net −424.274 vs hold 0 |
| `discovery evidence` on that candidate (3 folds, stress 1.5/2.0, trials 11/12) | exit 0; checksum `211f48b1…`, `locked_oos: null` |
| `evaluate_promotion(bundle, PROMOTION_V1)` | `accepted=False`; codes: `min_positive_fold_fraction`, `min_expectancy_r`, `min_profit_factor`, `min_ci_lower_r`, `cost_stress`, `stress_degradation`, `missing_locked_oos` |
| `discovery reproduce` on that bundle | exit 0; checksum reproduced exactly, no mismatches |

The full-suite result is recorded below. It is not used to claim market
evidence, and a green suite does not change any status above.

**Full local suite:** `uv run pytest -q` on this worktree exited `0` after
collecting 2,323 tests; pytest's `-q` mode suppressed the summary line, but the
exit code and the absence of any `F`/`E` marker in the progress output confirm
no failures. This is a local run, not a GitHub CI verification, and it is not
market evidence. The targeted batches in the table cover 1,232 distinct tests
(one file appears in two batches); the full run also covers the remaining
API/UI/ML/legacy modules.

## What this report does not claim

- It does not claim the project is close to profitability. No profitability
  artifact exists.
- It does not treat passing tests, a merged PR, a synthetic acceptance run or a
  green CI as validation.
- It does not claim a real holdout was spent, a champion was promoted, or a
  governed paper session ran. Those are precisely the missing artifacts.
- It does not claim the DSL search is free of human design: the grammar and
  feature registry are code written by humans. The owner supplies no entry/exit
  rules, which is the §36 requirement.
