# STATUS — autonomous system status

## Current refresh — 2026-11-25 (supersedes every snapshot below)

Verified `main`: `54eef5e7a25854c78f2bef2d26e4c42d50f3ab5e`
(`feat(discovery): runtime promotion command with one-shot owner approval (#220)`).

**CI:** green at `54eef5e` — workflow `45627617157`: backend `2,435 passed,
8 skipped`, lint `All checks passed!`, mypy `153 source files`; frontend
`287 passed` in 34 files, build compiled; CodeQL `45627617898` success. These
counts describe this exact SHA. The previous snapshot's counts (`2,320 passed,
3 skipped` at `50fbca6`) are historical and do not describe this code.

Canonical goal: `docs/requirements/master-prompt.md`.
Task-level status: `docs/task-board.md`.
Dependency order: `docs/implementation-roadmap.md`.
Mandatory contributor/AI-agent rules: `AGENTS.md`.

### Honest summary

The repository has real, tested infrastructure for discovery, provenance,
evidence, promotion and paper execution, and the runtime chain is now **complete
end to end in code**: dataset → discovery → evidence → one-shot holdout →
`PROMOTION_V1` → governed champion → scheduler-driven `paper.advance`. It is
**not** an accepted closed-loop self-learning trading system. The chain has
never been validated: the locked holdout is unspent, no evidence bundle carries
a real `locked_oos` section, no governed champion exists, and no paper session
has run a champion.

The merged real-data attempt (#191) is still the clearest statement of what the
market evidence says: validate → quality → dataset → discovery → evidence →
reproduce → promotion decision all ran; the decision was `no_trade`;
`PROMOTION_V1` refused the bundle with seven codes including `missing_locked_oos`;
paper was refused with `NoChampionError`. Later merges (#216–#220) added the
missing runtime call sites and safety gates; they did not manufacture a
candidate or spend the holdout.

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
  configurable gap policy. #178 exposes the audited gap policy through the
  orchestrator payload and event.
- **#189 paper-completion trigger:** the daemon can bind
  `--watch-paper-ledger` and record a matured-outcome proposal.
- **#174 tradable-best gate:** flat or thin candidates no longer win a search by
  default (`--min-best-trades`).
- **#175/#177/#186/#187/#188/#190:** audited local orchestrator controls,
  fail-closed legacy UI fallback, slippage cap + data-quality entry block,
  opt-in M30/H4 context with closed-bucket causality, feature-group ablation
  runner, regime-dependent ensemble weights from OOF only.
- **#184:** incremental SGD, isolation forest and feature importances.
- **#191:** a real XAUUSD H1 acceptance attempt, committed with its JSON record,
  that stops at promotion with NO TRADE.
- **#215 — optional PDF extraction:** `.pdf` ingestion through an optional
  `pypdf` extra (`uv sync --extra pdf`), one section per page with page
  references; the original file is preserved and the core install never imports
  `pypdf`. A missing extra produces an install hint, not a silent skip. No OCR.
- **#216 — provenance fail-closed:** the dataset manifest/artifact, evidence
  bundle (schema v4), holdout record, promotion policy and reproduce path now
  carry `source`/`synthetic`. `PROMOTION_V1` refuses `provenance_unknown` and
  `synthetic_data`; candidate and incumbent must agree (`champion_provenance_mismatch`);
  config/manifest disagreement is `provenance_mismatch` at evidence build.
- **#217 — read-only discovery API:** `GET /api/discovery/summary`,
  `/candidates`, `/candidates/{id}`, `/trials` serve the archive with `mode=ro`
  + `PRAGMA query_only`, honest `configured`/`readable` states and no writes,
  migrations or jobs.
- **#218 — paper scheduler:** `orchestrator run --schedule-paper` binds
  `PaperAdvanceScheduler`, which reads the discovery archive and stored bars and
  enqueues one `paper.advance` per evidence champion for a newly decidable bar
  (the dedupe key includes the bar timestamp; the ledger is the idempotency
  cursor). Exploratory/manual champions are refused; restart cannot
  double-advance.
- **#219 — M30/H4 resample tests:** bucket alignment, left-closed boundaries
  and stored-derivation parity pinned in `test_store.py`.
- **#220 — runtime promotion command:** `discovery promote` and
  `scalping.discovery.promote.promote_evidence` are the only runtime call site.
  `--dry-run` evaluates `PROMOTION_V1` and writes nothing; a real spend requires
  `--spend-holdout` plus a non-empty owner token from `--approval-token-file` or
  `--approval-token-env`. The token is never stored, logged or echoed.

### Not yet exercised (code merged, act pending)

- **The one-shot holdout has still not been spent on real data.** The command
  exists and is tested, but spending is an explicit owner-approved act and no
  owner approval has been recorded. Until then every bundle still lacks
  `locked_oos`, so `PROMOTION_V1` correctly rejects.
- **The paper scheduler has no champion to schedule.** `--schedule-paper` is
  wired and tested, but without an evidence champion every scope is a
  `no_champion`/`no_evidence_champion` refusal and no `paper.advance` job is
  created.

### Current blockers

1. **No validated candidate.** The locked holdout is unspent, no evidence
   bundle carries a real `locked_oos` section, and `PROMOTION_V1` therefore
   rejects every bundle. The real-data runs returned NO TRADE; that is a valid
   result, not a validation pass.
2. **The holdout spend is unexercised.** `discovery promote` exists and is
   tested, including `holdout_spent` on a second attempt, but no owner-approved
   spend has been executed. This is the one remaining PARTIAL in the promotion
   path and it requires owner approval, not more code.
3. **No governed paper champion and no paper run.** The scheduler
   (`--schedule-paper`) and the continuous runner are merged; with no evidence
   champion every scope is refused and `paper.advance` still returns NO TRADE.
   No real paper session with immutable attribution exists.
4. **No model edge.** Every classical model is at or below the prior baseline;
   the PPO runs are experimental and most are degenerate; no committed neural
   checkpoint exists. More model families are not the next step.
5. **No target-hardware acceptance.** The measured H1 smoke is small, the bench
   `paper_advance` stage failed on real stores with `BarGapError`, and no
   walk-forward stage was measured. The full loop has not run on the target M2
   8 GB machine.
6. **Documentation drift.** Older snapshots and a few task-board checkboxes lag
   the merged code. This refresh corrects the ones above; treat any undated
   claim as historical.

### Status by area

| Area | State | Evidence / gap |
|---|---|---|
| Local/offline/zero mandatory API | Implemented foundation | No mandatory LLM/API; local ML/DL tools exist |
| XAUUSD data and provenance | Fail-closed end to end | Store sidecar + dataset artifact + evidence bundle (schema v4) carry `source`/`synthetic`; promotion refuses unknown/synthetic (#216) |
| Causal features and datasets | Implemented foundation | 59 causal features, chronological splits, purge/embargo; M30/H4 context opt-in (#187), resample boundaries pinned (#219) |
| Classical ML / compact DL | Research tools | #184/#188 add incremental SGD, isolation forest, importances and ablation; metrics are not a trading edge |
| PPO / RL | Experimental | Plain PPO; masking in evaluation only; no #181 commit on `54eef5e` |
| Legacy flow_momentum card | Active compatibility path | Still used by the legacy panel; not the autonomous learner |
| Safe strategy DSL/AST | Implemented | Typed bounded AST; no generated arbitrary code |
| Candidate discovery | Implemented, limited | Bounded validation-slice search; tradable-best gate #174; no edge found on real H1 |
| Evidence bundle producer | Implemented | #159 persists and verifies dataset artifact, bundle and spec; #216 adds provenance identity |
| Promotion policy | Implemented, runtime call site merged | `PROMOTION_V1` fail-closed; `discovery promote` is the CLI call site (#220) |
| Locked holdout | Implemented, one-shot, unspent | #121 ledger; #216 carries provenance; #220 spends only with owner approval. No real spend |
| Evidence reproduction | Implemented | #130/#169 rebuild and compare; #216 verifies provenance; drift refuses |
| Champion governance | Partial foundation | History/provider/rollback; paper refuses non-evidence champions; no real champion |
| Knowledge ingestion | Implemented with optional PDF | TXT/MD/HTML/DOCX + tables (#119); PDF via optional `pypdf` extra with page refs (#215); no OCR |
| Hypothesis routing | Heuristic foundation | #106/#107/#123/#132/#153; proposals need approval, never auto-promote |
| Orchestrator | Infrastructure | Five handlers (#115), guard (#112), daemon `m2_8gb` (#118), data trigger (#158), paper-completion trigger (#189), paper scheduler (#218); no automatic promotion/holdout job |
| Discovery read API | Implemented, read-only | #217 `/api/discovery/*` with `mode=ro`; #134 `/api/autonomy/*` |
| DSL paper runner | Implemented component | #156 continuous runner + champion governance; #218 scheduler; no real session |
| Risk layer | Implemented and tested | #135 non-bypass suite; #186 slippage cap and data-quality block |
| Closed-loop autonomy | Runtime chain complete, not validated | All call sites exist; no holdout spend, no champion, no paper session |
| Real-money execution | Disabled | Must remain disabled and out of scope |

## Reproduce the key claims

All commands run from `backend/`; no network is required.

```bash
# Fail-closed promotion policy: no empty policy, no force flag, PROMOTION_V1
# rejects any bundle without a locked OOS section.
uv run pytest tests/test_promotion_policy.py

# One-shot holdout: seal, refusal, single spend, no recompute
uv run pytest tests/test_locked_holdout.py

# Provenance fail-closed through dataset -> evidence -> promotion (#216)
uv run pytest tests/test_discovery_provenance.py

# Runtime promotion command and one-shot owner approval (#220)
uv run pytest tests/test_discovery_promote.py

# Bundle production and reproduction: round-trip, typed refusals, CLI output
uv run pytest tests/test_discovery_persist.py tests/test_discovery_reproduce.py

# Provenance-aware paper champion, continuous runner and scheduler
uv run pytest tests/test_paper_continuous.py tests/test_paper_gaps.py \
  tests/test_paper_scheduler.py tests/test_orchestrator_handlers.py

# Read-only discovery archive API (#217)
uv run pytest tests/test_api_discovery.py

# Risk limits cannot be bypassed by a spec, action or job payload
uv run pytest tests/test_risk_acceptance.py

# Read-only autonomy API (green at 54eef5e; the historical failure is fixed)
uv run pytest tests/test_api_autonomy.py
```

```bash
# One discovery run on stored bars (validation slice only; writes a decision,
# not a promotion). --max-trials is an attempt cap; a real run needs more.
uv run python -m scalping.cli discovery run \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --max-trials 64 --seed 7 --out runs/discovery/report.json

# Persist evidence for an exported spec (needs the spec sidecar). The bundle
# now records dataset provenance; an unknown or synthetic dataset is refused.
uv run python -m scalping.cli discovery evidence \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe H1 \
  --spec runs/discovery/spec.json --out runs/discovery/evidence.json

# Evaluate the production policy without spending the holdout or writing.
uv run python -m scalping.cli discovery promote \
  --archive runs/discovery/discovery.sqlite --candidate "<id>" \
  --evidence runs/discovery/evidence.json --data-root ../data-ticks5-store \
  --reason "promotion-v1 dry run" --dry-run --out runs/discovery/promote.dryrun.json

# Replay the bundle from identity
uv run python -m scalping.cli discovery reproduce \
  --bundle runs/discovery/evidence.json --store-root ../data-ticks5-store
```

A real promotion is one command further: `--spend-holdout` plus an owner token
(`--approval-token-env NAME` or `--approval-token-file PATH`). It is a one-shot,
owner-approved act; do not run it to "see what happens". The full sequence and
safety warnings are in `docs/runbook.md` §2.6. The paper scheduler is the next
link and only acts after a champion exists:

```bash
uv run python -m scalping.cli orchestrator run --forever --schedule-paper \
  --db runs/orchestrator.sqlite3 --root ../data-ticks5-store \
  --runs-root runs --archive-root ../data-ticks5-store
```

With no evidence champion, `--schedule-paper` records a `paper.advance_refused`
event and enqueues nothing.

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
#177, #170, #171) all merged; "no producer" (#136) is stale because #159 merged
the producer; "`paper.advance` lacks promotion provenance" is stale because
#156/#164 route it through `ChampionProvider` and the continuous runner. The
scheduler and runtime promotion gaps were closed by #218 and #220. The remaining
gaps are evidence, not call sites: no spent holdout, no champion, no paper run.
