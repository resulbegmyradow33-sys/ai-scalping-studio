# STATUS — autonomous system status

Snapshot: **2026-11-24**, verified `main` at
`54bcf38abe7ed654e99935360fb6527d23127493`.

Canonical goal: `docs/requirements/master-prompt.md`.
Current code audit and priority addendum: `docs/current-state-addendum-2026-11-24.md`.
Historical detailed audit: `docs/autonomous-system-audit-2026-11-24.md` (its original
code/PR snapshot is preserved; consult the addendum for current mainline status).
Task-level status: `docs/task-board.md`.
Dependency order: `docs/implementation-roadmap.md`.
Mandatory contributor/AI-agent rules: `AGENTS.md`.

## Honest summary

The repository contains a strategy-independent DSL discovery foundation,
cost-aware simulator evaluation, source-linked heuristic research extraction,
persistent orchestration infrastructure and a DSL paper runner. It is **not
yet an accepted closed-loop self-learning trading system**. Promotion-grade
walk-forward/OOS evidence, fail-closed champion governance, production
orchestrator wiring, automatic champion-to-paper lifecycle and matured-outcome
feedback remain incomplete.

## Current state

| Area | State | Evidence / gap |
|---|---|---|
| Local/offline/zero mandatory API | Implemented foundation | No mandatory LLM/API; local ML/DL tools exist |
| XAUUSD data and provenance | Partial/usable | Historical/broker/forward data are local; no private-data acceptance run was performed in this audit |
| Multi-symbol data | Partial | Registry/contract metadata exists; real feed coverage remains limited |
| Causal features and datasets | Implemented foundation | 59 causal features, chronological splits, purge/embargo; MTF alignment, ablation and genuine order-book coverage remain |
| Classical ML / compact DL | Research tools | Prediction metrics are not evidence of a trading edge |
| Transformer benchmark | Research tool | #105 merged; benchmark/evaluator work is not trading-edge evidence |
| PPO / RL | Experimental | #108 adds risk-aware evaluation action masking; training remains plain PPO, not MaskablePPO |
| Legacy strategy family + LinUCB | Legacy baseline | Finite pre-authored families are optional comparisons, not autonomous discovery |
| Legacy flow_momentum card | Active compatibility path | Still used by legacy panel/runtime; migration to governed DSL champion remains |
| Safe strategy DSL/AST | Implemented | Typed bounded AST; no generated arbitrary Python |
| Candidate discovery | Implemented, limited | #104 merged; bounded search on a validation slice, not walk-forward/OOS |
| Champion governance | Partial foundation | #109 merged; history/provider/rollback exist, but promotion-grade evidence is incomplete |
| Knowledge ingestion | Partial | TXT/MD/HTML/DOCX; PDF/OCR and structured tables remain absent |
| Hypothesis extraction | Heuristic foundation | #106 creates source-linked drafts; not semantic book understanding or proven alpha |
| Sentence classification | Merged heuristic | #107 is on main; deterministic triage, not semantic understanding |
| Orchestrator | Infrastructure only | Persistent queue/control/recovery; production handlers, scheduler and measured resource guard remain |
| DSL paper runner | Implemented component | #103 runs a supplied validated spec over a stored slice; it does not autonomously select/advance a champion |
| Closed-loop autonomy | Not accepted | No proven import → discover → independent OOS → promote → paper → matured feedback cycle |
| Real-money execution | Disabled | Must remain disabled and out of scope |

## Verified recent mainline changes

- #104: validation-slice strategy discovery and baseline decision.
- #105: optional Transformer benchmark/evaluator work.
- #106: deterministic source-linked hypothesis drafts.
- #107: deterministic research sentence classification.
- #108: risk gates included in action masks and evaluation.
- #109: champion history/provider/rollback foundation.
- #110: repository-wide contributor/AI-agent contract and team workflow.

The current main commit includes #107. The old audit/task-board text that says
#107 is open or needs rebasing was stale and is being corrected.

## Critical path — do not reorder

1. Promotion-grade chronological walk-forward evaluation with purge/embargo,
   locked untouched OOS, cost stress, regime/stability evidence and
   multiple-testing controls.
2. Separate exploratory scoring from production promotion; require a named,
   versioned, non-empty fail-closed policy and candidate-vs-champion comparison
   on identical evidence.
3. Register real orchestrator handlers; add scheduler/daemon, triggers,
   idempotency, measured RAM/CPU/disk limits and paper/training isolation.
4. Route source-linked hypotheses into quantitative experiment proposals.
5. Connect validated DSL champion → paper continuation → immutable attribution
   → matured-outcome analysis → next eligible research cycle.
6. Run reproducible end-to-end acceptance on real XAUUSD data and target M2
   8 GB hardware.

Do not expand the model zoo or dashboard ahead of these blockers unless a
measured experiment demonstrates that expansion is necessary.

## Reporting vocabulary

- **Implemented:** code exists and relevant tests pass.
- **Trained:** a model was fitted and a reproducible run/artifact is recorded.
- **Validated:** independent chronological OOS and realistic cost-stress evidence exists.
- **Paper-traded:** exact version ran in paper mode with immutable attribution.
- **Profitable:** cautious claim supported by adequate forward evidence; never guaranteed.

A green CI run proves neither a trading edge nor profitability. No local market
data, M2 resource benchmark or end-to-end acceptance was executed during this
documentation refresh.
