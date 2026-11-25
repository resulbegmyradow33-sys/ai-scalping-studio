# Current-state addendum — 2026-11-24

## Snapshot and method

Repository: `resulbegmyradow33-sys/ai-scalping-studio`  
Verified main commit: `54bcf38abe7ed654e99935360fb6527d23127493`.

This addendum refreshes the mainline facts in the historical audit
`docs/autonomous-system-audit-2026-11-24.md`. The original audit's code
inspection baseline is retained for traceability; it must not be mistaken for
the current PR/mainline snapshot.

The refresh used GitHub main/commit comparison, current source files,
task-board/roadmap contents and PR search. It did not execute repository tests,
inspect private local market data, or benchmark the Mac mini. No CI-green,
market-edge, OOS-validation or profitability claim is made here.

## Mainline alignment

The following work is present on main in the verified snapshot:

| PR | Mainline status | Correct interpretation |
|---|---|---|
| #104 | Merged | Bounded validation-slice DSL discovery; not walk-forward promotion |
| #105 | Merged | Optional Transformer benchmark/evaluator research |
| #106 | Merged | Deterministic source-linked hypothesis drafts |
| #107 | Merged | Deterministic sentence classification; heuristic triage only |
| #108 | Merged | Risk gates/action masks enforced in simulator evaluation |
| #109 | Merged | Champion history/provider/rollback foundation; not production OOS gate |
| #110 | Merged | Shared repository instructions, ownership and agent workflow |

The previous audit and task-board references describing #107 as open or needing
rebase are stale. This does not change the technical limitation: #107's
classifier is pattern-based and is not a semantic research model.

## Current engineering verdict

**No strategic direction change is visible. The project remains aligned with
the strategy-independent, local-first, paper-only objective. It is not yet a
closed-loop autonomous learning system.**

Implemented foundations:
- bounded typed DSL/AST and safe interpreter;
- candidate search and cost-aware validation-slice simulation;
- baseline comparison and candidate lineage;
- local source-indexing and deterministic hypothesis drafts/classification;
- persistent orchestrator queue, controls and recovery;
- DSL paper runner with persistent paper ledger;
- risk-aware action masking in simulator evaluation;
- champion history/provider/rollback foundation.

Still blocking acceptance:
1. Candidate discovery needs chronological walk-forward folds with purge/
   embargo and a genuinely locked untouched OOS dataset that search/tuning
   cannot inspect.
2. Candidate attempt counts, multiple-testing/data-snooping controls,
   complexity/stability penalties, uncertainty and cost/regime stress must be
   part of an immutable evidence bundle.
3. Champion promotion must fail closed under a named/versioned non-empty
   policy. Missing, stale, mismatched or incomplete evidence must reject.
   Candidate and incumbent must be compared on identical folds/timestamps,
   costs and regime coverage.
4. The persistent orchestrator needs production handlers, a scheduler/daemon,
   triggers, idempotency, restart-safe job semantics and measured RAM/CPU/disk
   resource guards.
5. Source-linked hypotheses need a controlled bridge into quantitative
   experiment proposals. They must never directly create orders or bypass
   validation.
6. Paper execution needs automatic loading of only a validated DSL champion,
   appended-data continuation, immutable trade attribution, matured-outcome
   analysis and feedback into the next eligible research cycle. No/invalid
   champion must produce auditable NO TRADE, never silent legacy fallback.
7. Full acceptance must be run on real XAUUSD data and the target Mac mini M2
   8 GB environment. This refresh did not run that acceptance.

## Priority order

P0 — Promotion-grade evidence:
- immutable dataset/fold manifests and strict chronological boundaries;
- purge/embargo at every fold;
- locked OOS access separation;
- candidate attempt accounting and multiple-testing safeguards;
- realistic spread/commission/slippage and cost stress;
- regime/stability/sensitivity/uncertainty reporting.

P1 — Fail-closed champion governance:
- separate exploratory search criteria from production promotion policy;
- require explicit policy id/version and complete evidence;
- compare candidate and incumbent on identical evidence;
- preserve rejection reasons, lineage and rollback;
- test missing evidence, stale data, scope mismatch and attempted override.

P2 — Orchestrator integration:
- real handlers for data validation, ingestion, hypothesis routing, discovery,
  evaluation, champion review, paper advance and matured outcome analysis;
- scheduler/daemon and deterministic triggers;
- measured RAM/CPU/disk budgets, one-heavy-job semaphore and paper/training
  failure isolation.

P3 — Autonomous paper feedback:
- load only validated DSL champion;
- NO TRADE on missing/invalid champion;
- continue on appended bars and recover after restart;
- analyse only fully matured, immutable-attributed outcomes;
- enqueue next eligible research cycle without OOS leakage.

P4 — Target-hardware acceptance:
- real XAUUSD dataset manifests/checksums;
- end-to-end replay from new input to paper feedback;
- measured peak RSS, CPU, disk and elapsed time on M2 8 GB;
- verify real-money execution remains impossible/disabled.

## Safety and scope invariants

- Real-money execution remains out of scope and disabled.
- No mandatory paid API, paid data, cloud inference or LLM.
- Generated strategy logic remains bounded data-only DSL/AST; never arbitrary
  generated Python or shell execution.
- Existing fixed strategies remain optional baselines, not the search menu.
- Validation-slice winners are candidates only, never production champions.
- NO TRADE and no champion change are valid outcomes.
- A module, PR, passing unit test or green CI does not prove end-to-end autonomy,
  OOS validity or profitability.
- All repository changes remain on reviewed branches; main is not edited
  directly. Resul retains final project acceptance authority.

---

## Second refresh — 2026-11-24 (main `693f8a37`)

Verified main commit: `693f8a37ed5f5fdfbabb7bbbed0bd468468c9875`
(merge of #121). The sections above are left as written; where they conflict
with this note, this note is current. Method: `git log`/`git show`, current
source files, `gh pr`/`gh run` and targeted pytest (the five claim files passed;
`test_api_autonomy.py` failed). No market data was used, no real holdout was
spent, no M2 benchmark was run.

### What landed since the first refresh

| PR | State | Correct interpretation |
|---|---|---|
| #112 | Merged | Measured RSS/load/disk guard, fail-closed probe |
| #114 | Merged | Chronological-fold evidence bundle (schema v2) |
| #115 | Merged | Production orchestrator handlers |
| #116 | Merged | Continuous paper operation, matured-outcome batches |
| #117 | Merged | `PROMOTION_V1`: named, versioned, fail-closed policy; no empty policy, no force flag, locked OOS required |
| #118 | Merged | Orchestrator daemon; `m2_8gb` guard profile by default |
| #119 | **Open PR** | Knowledge table structure; not on main |
| #121 | Merged | One-shot locked holdout with append-only access ledger |
| #122 | Merged | Matured paper-outcome trigger; enqueues a proposal, not a promotion |
| #123 / #132 | Merged | Hypothesis → audited experiment proposal; provenance/audit hardening |
| #124 | Merged | Repeatable M2 pipeline measurement harness |
| #128 | Merged | `/paper` page over the real `/api/paper` ledger |
| #129 | Merged | Orchestrator end-to-end integration coverage |
| #130 | Merged | `discovery reproduce` from bundle identity |
| #131 | Merged | Paper-outcome trigger hardening |
| #133 | Merged | MTF alignment semantics and feature-naming pins |
| #134 | Merged | Read-only `/api/autonomy` status endpoints |
| #135 | Merged | Phase 11 non-bypass risk acceptance suite |

### Corrections to earlier wording

- **Promotion is fail-closed, not a proposal.** #117 rejects an empty policy, a
  version below 1, a disabled production `require_locked_oos`, a missing or
  fabricated locked-OOS section, stale identity, insufficient folds/trades,
  low expectancy/PF, deep drawdown, poor regime coverage, a CI touching zero,
  missing cost stress, complexity over cap and trial overrun. It has **no
  runtime call site**, so no production promotion exists.
- **Reproduction has a consumer, not a producer.** #130 rebuilds a bundle from
  its identity and refuses drift with typed codes. Nothing on main writes a
  bundle or its dataset artifact; that is issue #136, still open. Treat
  reproduce as implemented and tested, end-to-end unreachable.
- **Autonomy API is read-only.** Every `/api/autonomy` route is `GET`; SQLite is
  opened `mode=ro` with `query_only=ON`; a missing store answers
  `configured:false`, an unreadable one `readable:false`. No dashboard page
  consumes it.
- **Paper page is read-only.** #128 renders `/api/paper`; it may close a running
  session and does nothing else — no session creation, no runner step, no broker.
- **Risk tests are acceptance, not smoke.** #135 runs the same bars through a
  DSL policy, a raw action and a job payload against one frozen `RiskEngine`.
- **MTF is tested but not a feature input.** M30/H4 alignment and chained
  resampling are pinned; the feature layer still consumes M5/M15.
- **Knowledge tables are open PR #119.** The task-board/audit lines that imply
  table extraction is missing "and no PR exists" are stale; the line that says
  the capability is absent on main is correct.

### Known red on main

`backend/tests/test_api_autonomy.py` fails with
`AttributeError: 'DiscoveryArchive' object has no attribute 'promote'` in 4
tests: #117 replaced the archive `promote` with the sealed evidence path and
#134's tests still call the removed method. CI run `45424959773` (main at
`6113197`) ended `4 failed, 1816 passed, 2 skipped`. The newest green main CI
run found is `45422130310` at `2bbb312`. Earlier pushes after that were
cancelled or failed, and the run at `693f8a3` was still in progress at refresh
time. Do not cite a green CI for the current tree.

### Reproduce the claims

```bash
cd backend
uv run pytest tests/test_promotion_policy.py    # fail-closed policy + sealed decisions
uv run pytest tests/test_locked_holdout.py      # seal, refusal, one spend
uv run pytest tests/test_discovery_reproduce.py # checksum round-trip, typed refusals
uv run pytest tests/test_risk_acceptance.py     # no bypass by spec/action/job
uv run pytest tests/test_mtf_alignment.py       # HTF visibility after bucket close
uv run pytest tests/test_api_autonomy.py        # read-only API (red on main)
```

```bash
# One validation-slice discovery run; writes a decision, not a bundle
uv run python -m scalping.cli discovery run \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe M5 \
  --max-trials 64 --seed 0 --out runs/discovery/report.json

# Replay a stored bundle once a producer exists (#136)
uv run python -m scalping.cli discovery reproduce \
  --bundle runs/discovery/candidate.evidence.json \
  --store-root ../data-ticks5-store
```

### Still unproven

No candidate has been validated on an independent chronological OOS, no real
holdout has been spent, no promotion decision has been produced by a runtime, no
paper session has been selected by a promotion decision, and nothing is
paper-traded or profitable. The evidence vocabulary in `AGENTS.md` still applies unchanged:
implemented ≠ validated ≠ paper-traded ≠ profitable.
