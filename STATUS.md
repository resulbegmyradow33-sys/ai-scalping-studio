# STATUS — autonomous system status

Snapshot: **2026-11-24 (second refresh)**, verified `main` at
`693f8a37ed5f5fdfbabb7bbbed0bd468468c9875` (merge of #121).

Canonical goal: `docs/requirements/master-prompt.md`.
Current code audit and priority addendum: `docs/current-state-addendum-2026-11-24.md`.
Historical detailed audit: `docs/autonomous-system-audit-2026-11-24.md` (its original
code/PR snapshot is preserved; consult the addendum for current mainline status).
Task-level status: `docs/task-board.md`.
Dependency order: `docs/implementation-roadmap.md`.
Mandatory contributor/AI-agent rules: `AGENTS.md`.

## Honest summary

The repository contains a strategy-independent DSL discovery foundation,
cost-aware simulator evaluation, a promotion-grade evidence bundle, a
**fail-closed promotion policy**, a **one-shot locked-holdout protocol**, a
bundle **reproducer**, source-linked knowledge extraction, persistent
orchestration with a daemon and measured guard, and a DSL paper runner with a
read-only UI page.

It is **not yet an accepted closed-loop self-learning trading system**. The
promotion policy is library-only: no merged command or orchestrator handler
builds and persists an evidence bundle from a discovery run, so `reproduce` has
a consumer but no producer (issue #136 is open). `paper.advance` still does not
require promotion-policy provenance. No chronological OOS, cost-stress,
paper-traded or profitability claim is made here.

## Current state

| Area | State | Evidence / gap |
|---|---|---|
| Local/offline/zero mandatory API | Implemented foundation | No mandatory LLM/API; local ML/DL tools exist |
| XAUUSD data and provenance | Partial/usable | Historical/broker/forward data are local; no private-data acceptance run was performed in this refresh |
| Multi-symbol data | Partial | Registry/contract metadata exists; real feed coverage remains limited |
| Causal features and datasets | Implemented foundation | 59 causal features, chronological splits, purge/embargo; M30/H4 alignment and feature naming now pinned by tests (#133); ablation and genuine order-book coverage remain |
| Classical ML / compact DL | Research tools | Prediction metrics are not evidence of a trading edge |
| Transformer benchmark | Research tool | #105 merged; benchmark/evaluator work is not trading-edge evidence |
| PPO / RL | Experimental | #108 adds risk-aware evaluation action masking; training remains plain PPO, not MaskablePPO |
| Legacy strategy family + LinUCB | Legacy baseline | Finite pre-authored families are optional comparisons, not autonomous discovery |
| Legacy flow_momentum card | Active compatibility path | Still used by legacy panel/runtime; migration to governed DSL champion remains |
| Safe strategy DSL/AST | Implemented | Typed bounded AST; no generated arbitrary Python |
| Candidate discovery | Implemented, limited | #104 bounded validation-slice search; #114 adds a chronological fold evidence bundle, but folds stay inside train+validation and the final test is not scored inside discovery |
| Evidence promotion policy | Implemented and tested, not wired | #117 `PROMOTION_V1` is named, versioned, fail-closed and cannot disable `require_locked_oos`; no CLI/orchestrator call site, so no runtime promotion happens |
| Locked holdout | Implemented and tested, unspent | #121 one-shot `HoldoutLedger` + `evaluate_holdout` + `attach_holdout`; no real holdout has been spent on market data |
| Evidence reproduction | Implemented, no producer | #130 `discovery reproduce` rebuilds a bundle from identity and refuses drift; nothing on main writes a bundle (#136 open) |
| Champion governance | Partial foundation | #109 history/provider/rollback; exploratory archive promotion is separate from `promote_with_evidence`; paper does not enforce evidence-mode provenance |
| Knowledge ingestion | Partial | TXT/MD/HTML/DOCX; table structure extraction is open PR #119; PDF/OCR absent |
| Hypothesis extraction and routing | Heuristic foundation | #106 drafts, #107 classification, #123 audited experiment proposals (#132 hardening); proposals are not auto-converted into discovery runs |
| Orchestrator | Infrastructure | Queue/control/recovery, five production handlers (#115), measured guard (#112), daemon with `m2_8gb` default profile (#118), paper-outcome trigger as a proposal (#122); the bare `Orchestrator` still defaults to `AllowAllGuard` |
| Autonomy status API | Implemented, read-only | #134 GET-only `/api/autonomy/{status,jobs,events,champions}`; no autonomy dashboard page |
| DSL paper runner | Implemented component | #103 runs a supplied spec; #116 adds append-only continuation and matured outcomes; #128 adds a `/paper` page over `/api/paper`; `paper.advance` still uses the slice runner without promotion provenance |
| Risk layer | Implemented and tested | #135 acceptance suite: DSL spec, simulator action and orchestrator job all reach the same frozen `RiskEngine`; no slippage cap or rejected-order model |
| Closed-loop autonomy | Not accepted | No proven import → discover → independent OOS → promote → paper → matured feedback cycle |
| Real-money execution | Disabled | Must remain disabled and out of scope |

## Verified recent mainline changes

- #112: measured resource guard (RSS/load/disk, fail-closed probe).
- #113: current-state addendum and corrected autonomy priorities.
- #114: promotion-grade candidate evidence bundle (schema v2, settings/dataset identity).
- #115: production orchestrator handlers.
- #116: continuous paper operation and matured-outcome batches.
- #117: fail-closed `PROMOTION_V1` policy over evidence bundles (no empty
  policy, no force flag, locked OOS required).
- #118: orchestrator daemon with the `m2_8gb` guard profile by default.
- #119: knowledge table structure — **open PR**, not merged.
- #120: docs sync of STATUS/board with the merged evidence/handler/paper slices.
- #121: one-shot locked holdout with an append-only access ledger.
- #122: matured paper-outcome trigger (proposal only; no promotion).
- #123: hypothesis → audited experiment proposal; #132 hardens provenance/audit.
- #124: repeatable M2 pipeline measurement harness.
- #128: `/paper` page on the real `/api/paper` ledger.
- #129: orchestrator end-to-end integration coverage.
- #130: `discovery reproduce` from a stored bundle identity.
- #131: paper-outcome trigger hardening.
- #133: MTF alignment semantics and feature-naming pins.
- #134: read-only autonomy status endpoints.
- #135: Phase 11 non-bypass risk acceptance suite.

## Known red on `main` (2026-11-24)

`backend/tests/test_api_autonomy.py` fails on current `main`:
`AttributeError: 'DiscoveryArchive' object has no attribute 'promote'` in 4
tests. #117 replaced the old archive `promote` with the sealed evidence path, and
#134's tests still call the removed method. CI run `45424959773` (main at
`6113197`) finished `4 failed, 1816 passed, 2 skipped`. The newest green main CI
run found is `45422130310` at `2bbb312` (#117); pushes after it were cancelled,
failed or still in progress. Do not cite a green CI as proof of the current
tree.

## Reproduce the key claims

All commands run from `backend/`; no network is required.

```bash
# Full suite (add -q for less noise)
uv run pytest

# Fail-closed promotion policy: no empty policy, no force flag, PROMOTION_V1
# rejects any bundle without a locked OOS section. Tests register sealed
# test-only policies; the production policy keeps require_locked_oos=True.
uv run pytest tests/test_promotion_policy.py

# One-shot holdout: seal, refusal, single spend, no recompute
uv run pytest tests/test_locked_holdout.py

# Bundle reproduction: checksum round-trip, typed refusals, CLI one-line output
uv run pytest tests/test_discovery_reproduce.py

# Risk limits cannot be bypassed by a spec, action or job payload
uv run pytest tests/test_risk_acceptance.py

# MTF alignment (higher timeframe only after its bucket closes)
uv run pytest tests/test_mtf_alignment.py

# Read-only autonomy API (currently red on main — see above)
uv run pytest tests/test_api_autonomy.py
```

```bash
# One discovery run on stored bars (validation slice only; writes a decision,
# not an evidence bundle). --max-trials is an attempt cap; a real run needs more.
uv run python -m scalping.cli discovery run \
  --root ../data-ticks5-store --symbol XAUUSD --timeframe M5 \
  --max-trials 64 --seed 0 --out runs/discovery/report.json

# Replay a stored bundle (requires a bundle file and a dataset artifact; the
# producer that writes both is issue #136, not merged)
uv run python -m scalping.cli discovery reproduce \
  --bundle runs/discovery/candidate.evidence.json \
  --store-root ../data-ticks5-store
```

## Critical path — do not reorder

1. Persist the evidence bundle and dataset artifact from a discovery run, then
   wire `evaluate_promotion` / `promote_with_evidence` into a real command or
   handler (issue #136). Until then reproduce and promotion are unreachable in
   normal operation.
2. Fix the `test_api_autonomy.py` regression so main is green again.
3. Enforce promotion provenance in `paper.advance`: only a sealed, accepted
   decision may select the paper champion; missing/invalid champion is auditable
   NO TRADE, never legacy fallback.
4. Attach candidate-vs-incumbent comparison and attempt accounting to the same
   immutable evidence, with multiple-testing controls.
5. Route source-linked proposals into discovery experiments without letting
   prose become a strategy or an order.
6. Run reproducible end-to-end acceptance on real XAUUSD data within measured
   M2 8 GB limits.

Do not expand the model zoo or dashboard ahead of these blockers unless a
measured experiment demonstrates that expansion is necessary.

## Reporting vocabulary

- **Implemented:** code exists and relevant tests pass.
- **Trained:** a model was fitted and a reproducible run/artifact is recorded.
- **Validated:** independent chronological OOS and realistic cost-stress evidence exists.
- **Paper-traded:** exact version ran in paper mode with immutable attribution.
- **Profitable:** cautious claim supported by adequate forward evidence; never guaranteed.

A green CI run proves neither a trading edge nor profitability. This refresh ran
documentation checks plus the targeted pytest files listed above on 2026-11-25:
`test_promotion_policy.py`, `test_locked_holdout.py`, `test_discovery_reproduce.py`,
`test_risk_acceptance.py` and `test_mtf_alignment.py` passed; `test_api_autonomy.py`
failed as described. It did not run the full suite, use market data, spend a real
holdout, or benchmark the M2.
