# STATUS — autonomous system status

## Current refresh — 2026-11-25 (supersedes every snapshot below)

Verified `main`: `06c853c88b83b29c31b7478702fc9bb848f2e840`
(`docs: real-data acceptance attempt — chain stops at promotion (no edge) (#191)`).

**CI:** the latest verified run is `45550807353` at `50fbca6` — backend
`2,320 passed, 3 skipped`, frontend `263 passed`, CodeQL green. `06c853c` adds
two files (`docs/acceptance-attempt-2026-11-25.md`,
`docs/evidence/acceptance-attempt-2026-11-25.json`, +520 lines) and changes no
code or test, so those counts describe the same code. Do not call `06c853c`
"red"; it is green at `50fbca6` plus a documentation-only commit.

Canonical goal: `docs/requirements/master-prompt.md`.
Task-level status: `docs/task-board.md`.
Dependency order: `docs/implementation-roadmap.md`.
Mandatory contributor/AI-agent rules: `AGENTS.md`.

### Honest summary

The repository has real, tested infrastructure for discovery, evidence,
promotion and paper execution, and the evidence/holdout/promotion slice is
merged. It is **not** an accepted closed-loop self-learning trading system. The
system can discover candidates and refuse them — including on real XAUUSD H1
bars — but it cannot yet spend a real locked holdout, promote a candidate, or
trade a governed champion at runtime.

The merged real-data attempt (#191) is the clearest statement of where the chain
stops: validate → quality → dataset → discovery → evidence → reproduce →
promotion decision all ran; the decision was `no_trade`; `PROMOTION_V1` refused
the bundle with seven codes including `missing_locked_oos`; paper was refused
with `NoChampionError`. No holdout was sealed or spent and no promotion exists.

### Changes now present on main

- **#159 — evidence producer:** discovery atomically persists dataset artifacts,
  evidence bundles and spec sidecars, then verifies them by read-back and
  reproduction. The old "#136 has no producer" claim is closed.
- **#117 `PROMOTION_V1`:** named, versioned, non-empty, no force flag,
  `require_locked_oos` cannot be disabled.
- **#121 one-shot holdout:** `HoldoutLedger`, `evaluate_holdout`,
  `attach_holdout`; one spend per `(dataset_id, dataset_checksum)`.
- **#130/#169 reproduction:** `discovery reproduce` replays a bundle from
  identity and refuses drift; checksums are thread-independent.
- **#149 selection control:** adjusted screening and bootstrap/complexity advice.
  It advises; promotion does not consume it.
- **#151 synthetic acceptance harness:** eight stages with honest
  `validated=false`, `paper_traded=false`.
- **#153 approved proposal bridge:** only approved, scope-checked, deduplicated
  proposals may enqueue `discovery.run`.
- **#156/#164 paper governance:** `paper.advance` uses the continuous runner, a
  provenance-aware `ChampionProvider`, append-only continuation and a
  configurable gap policy.
- **#189 paper-completion trigger:** the daemon can now bind
  `--watch-paper-ledger` and record a matured-outcome proposal. It does not
  enqueue `paper.advance`.
- **#174 tradable-best gate:** flat or thin candidates no longer win a search by
  default (`--min-best-trades`).
- **#175/#177/#186/#187/#188/#190:** audited local orchestrator controls,
  fail-closed legacy UI fallback, slippage cap + data-quality entry block,
  opt-in M30/H4 context with closed-bucket causality, feature-group ablation
  runner, regime-dependent ensemble weights from OOF only.
- **#184:** incremental SGD, isolation forest and feature importances.
- **#191:** a real XAUUSD H1 acceptance attempt, committed with its JSON record,
  that stops at promotion with NO TRADE.

### In progress / not on `06c853c`

- **Provenance fail-closed** (acceptance step 1): `DatasetManifest`,
  `CandidateEvidence`, the holdout and `promotion.py` still carry no
  `source`/`synthetic` field. No committed work on `06c853c`.
- **PDF ingestion** (acceptance step 5): `knowledge/ingest.py:492-497` raises
  `UnsupportedFormat` for `.pdf`; the optional parser is not installed or
  merged. TXT/MD/HTML/DOCX and table structure (#119) work.
- **Discovery archive read API:** #134 serves read-only autonomy status, jobs,
  events and champions from the archive. A broader candidate/holdout archive
  read API is not on `06c853c`.

The three 2026-11-25 assessment reports (`docs/acceptance-gap-2026-11-25.md`,
`docs/audit-gap-2026-11-25.md`, `docs/final-report-2026-11-25.md`) are prepared
on unmerged branches. Cite them as evidence, not as repository files.

### Current blockers

1. **No validated candidate.** The locked holdout is unspent, no evidence
   bundle carries a real `locked_oos` section, and `PROMOTION_V1` therefore
   rejects every bundle. The real-data runs returned NO TRADE; that is a valid
   result, not a validation pass.
2. **No runtime holdout/promotion command.** The CLI exposes `discovery
   run|reproduce|evidence` only; there is no `discovery promote`, and no
   orchestrator handler runs the holdout or promotes. `docs/runbook.md` says the
   same: no CLI and no HTTP endpoint for DSL evidence promotion. The holdout and
   promotion modules remain library-only; the only successful promotion path in
   tests is a helper with `require_production=False`.
3. **Provenance cannot gate promotion.** Dataset manifests, evidence bundles,
   holdout, promotion, ML/NN metadata and `AutonomousPaperRunner` have no
   `source`/`synthetic` field. A synthetic dataset can satisfy every policy
   check. This is the most important safety gap because master-prompt §38
   requires fail-closed provenance.
4. **No governed paper champion and no paper scheduler.** No evidence-mode
   champion exists, so `paper.advance` correctly returns NO TRADE. Nothing
   enqueues `paper.advance` on a clock or on new bars: the daemon scans data
   files (→ `dataset.build`) and paper ledgers (→ `paper.outcomes` proposal)
   only. Even a manually promoted champion would wait for a caller.
5. **No model edge.** Every classical model is at or below the prior baseline;
   the PPO runs are experimental and most are degenerate; no committed neural
   checkpoint exists. More model families are not the next step.
6. **No target-hardware acceptance.** The measured H1 smoke is small, the bench
   `paper_advance` stage failed on real stores with `BarGapError`, and no
   walk-forward stage was measured. The full loop has not run on the target M2
   8 GB machine.
7. **Documentation drift.** Older STATUS/roadmap snapshots and a few
   task-board checkboxes lag the merged code. This refresh corrects the ones
   above; treat any undated claim as historical.

### Status by area

| Area | State | Evidence / gap |
|---|---|---|
| Local/offline/zero mandatory API | Implemented foundation | No mandatory LLM/API; local ML/DL tools exist |
| XAUUSD data and provenance | Usable, store-side only | Store sidecar has `source`/`synthetic`/checksum; dataset and evidence do not |
| Causal features and datasets | Implemented foundation | 59 causal features, chronological splits, purge/embargo; M30/H4 context opt-in (#187) |
| Classical ML / compact DL | Research tools | #184/#188 add incremental SGD, isolation forest, importances and ablation; metrics are not a trading edge |
| PPO / RL | Experimental | Plain PPO; masking in evaluation only; no #181 commit on `06c853c` |
| Legacy flow_momentum card | Active compatibility path | Still used by the legacy panel; not the autonomous learner |
| Safe strategy DSL/AST | Implemented | Typed bounded AST; no generated arbitrary code |
| Candidate discovery | Implemented, limited | Bounded validation-slice search; tradable-best gate #174; no edge found on real H1 |
| Evidence bundle producer | Implemented | #159 persists and verifies dataset artifact, bundle and spec |
| Promotion policy | Implemented and tested, no call site | `PROMOTION_V1` fail-closed; no CLI/handler invokes it |
| Locked holdout | Implemented and tested, unspent | #121 one-shot ledger; no real spend |
| Evidence reproduction | Implemented | #130/#169 rebuild and compare; drift refuses |
| Champion governance | Partial foundation | History/provider/rollback; paper refuses non-evidence champions |
| Knowledge ingestion | Partial | TXT/MD/HTML/DOCX + tables (#119); PDF optional parser absent |
| Hypothesis routing | Heuristic foundation | #106/#107/#123/#132/#153; proposals need approval, never auto-promote |
| Orchestrator | Infrastructure | Five handlers (#115), guard (#112), daemon `m2_8gb` (#118), data trigger (#158), paper-completion trigger (#189); no evidence/promotion/holdout job |
| Autonomy status API | Implemented, read-only | #134 GET-only `/api/autonomy/*`; autonomy page reads it |
| DSL paper runner | Implemented component | #156 continuous runner + champion governance; no scheduler and no real session |
| Risk layer | Implemented and tested | #135 non-bypass suite; #186 slippage cap and data-quality block |
| Closed-loop autonomy | Not accepted | No validated holdout, no promotion, no governed paper session |
| Real-money execution | Disabled | Must remain disabled and out of scope |

## Reproduce the key claims

All commands run from `backend/`; no network is required.

```bash
# Fail-closed promotion policy: no empty policy, no force flag, PROMOTION_V1
# rejects any bundle without a locked OOS section.
uv run pytest tests/test_promotion_policy.py

# One-shot holdout: seal, refusal, single spend, no recompute
uv run pytest tests/test_locked_holdout.py

# Bundle production and reproduction: round-trip, typed refusals, CLI output
uv run pytest tests/test_discovery_persist.py tests/test_discovery_reproduce.py

# Provenance-aware paper champion and continuous runner
uv run pytest tests/test_paper_continuous.py tests/test_paper_gaps.py \
  tests/test_orchestrator_handlers.py

# Risk limits cannot be bypassed by a spec, action or job payload
uv run pytest tests/test_risk_acceptance.py

# Read-only autonomy API (green at 50fbca6; the historical failure is fixed)
uv run pytest tests/test_api_autonomy.py
```

```bash
# One discovery run on stored bars (validation slice only; writes a decision,
# not a promotion). --max-trials is an attempt cap; a real run needs more.
uv run python -m scalping.cli discovery run \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --max-trials 64 --seed 7 --out runs/discovery/report.json

# Persist evidence for an exported spec (needs the spec sidecar)
uv run python -m scalping.cli discovery evidence \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --spec runs/discovery/spec.json --out runs/discovery/evidence.json

# Replay the bundle from identity
uv run python -m scalping.cli discovery reproduce \
  --bundle runs/discovery/evidence.json --store-root ../data-ticks5-store
```

There is no `discovery promote` command. A promotion currently requires library
code in a Python session; see `docs/runbook.md` for the exact call and its
`non_production_policy`/`missing_locked_oos` refusals.

## Reporting vocabulary

- **Implemented:** code exists and relevant tests pass.
- **Trained:** a model was fitted and a reproducible run/artifact is recorded.
- **Validated:** independent chronological OOS and realistic cost-stress evidence exists.
- **Paper-traded:** exact version ran in paper mode with immutable attribution.
- **Profitable:** cautious claim supported by adequate forward evidence; never guaranteed.

A green CI run proves neither a trading edge nor profitability.

## Historical notes (superseded)

Earlier snapshots (2026-11-24, `693f8a3`) are not reproduced here. Claims that
mattered and their current status: the "known red on main" note referred to
`6113197` and was fixed before `50fbca6`; the PRs it listed as open (#174, #175,
#177, #170, #171) all merged by `06c853c`; "no producer" (#136) is stale because
#159 merged the producer; "`paper.advance` lacks promotion provenance" is stale
because #156/#164 route it through `ChampionProvider` and the continuous runner.
The remaining gap is a champion and a scheduler, not the governance code.
