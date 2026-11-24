# STATUS — autonomous system status

Snapshot: **2026-11-24**, main at
22cecdf5e2c6634b59433925b836bda6dea33199.

Canonical goal: docs/requirements/master-prompt.md.
Current evidence and gaps: docs/autonomous-system-audit-2026-11-24.md.
Ordered tasks: docs/task-board.md and docs/implementation-roadmap.md.
Mandatory contributor/AI-agent rules: AGENTS.md.

## Honest summary

The repository has a tested local XAUUSD research stack and now contains a
strategy-independent DSL discovery path. It is **not yet a closed-loop
self-learning trading AI**. Discovery, champion governance, orchestration,
knowledge research and autonomous paper execution are separate foundations;
the production connections and independent evidence gates remain incomplete.

## Current state

| Area | State | Evidence / gap |
|---|---|---|
| Local/offline/zero mandatory API | Implemented | Python/ML/DL run locally; no paid LLM/API dependency |
| XAUUSD data and provenance | Partial/usable | Historical tick-derived, broker M1 and forward datasets documented; data files are local, not committed |
| Multi-symbol data | Partial | Registry/contract metadata exists; real feed coverage remains limited |
| Causal features and datasets | Implemented foundation | 59 causal features, chronological splits, purge/embargo; MTF alignment, ablation and true order-book coverage remain |
| Classical ML / compact DL | Implemented research tools | Baselines and shared evaluation exist; model prediction metrics are not strategy proof |
| Transformer benchmark | Open PR | #105; optional, behind main, not core-path priority |
| PPO / RL | Experimental | PPO/BC exists; #108 adds evaluation action masking; training is not MaskablePPO |
| Legacy strategy family + LinUCB | Legacy research baseline | Five fixed arms; not autonomous open-ended discovery |
| Legacy flow_momentum card | Active compatibility incumbent | Current panel path; historical forward sample is small and not proof of the new learner |
| Safe strategy DSL/AST | Implemented | Typed bounded AST; no generated arbitrary Python |
| Candidate discovery | Implemented, limited | #104 merged; evolutionary search + cost-aware simulator on one validation slice; no walk-forward/OOS |
| Champion governance | In progress | #109 open; validation-only and no OOS/walk-forward in its stated limits |
| Knowledge ingestion | Partial | TXT/MD/HTML/DOCX; #106 source-linked regex hypothesis drafts; PDF/OCR absent |
| Sentence classification | Open PR | #107; heuristic classification, not semantic understanding |
| Orchestrator | Infrastructure only | Persistent queue/control/recovery; real handlers, scheduler and measured resource guard absent |
| DSL paper runner | Implemented component | #103 executes a supplied spec over a stored slice; no automatic champion selection/continuous feedback |
| Closed-loop autonomy | Not accepted | No proven import→discover→OOS→promote→paper→feedback cycle |
| Real-money execution | Disabled | Must remain disabled |

## Recent mainline changes

- #104 merged: end-to-end validation-slice strategy discovery and baseline decision.
- #106 merged: deterministic, source-linked hypothesis drafts.
- #108 merged: risk gates included in action masks and backtest evaluation.
- #107 open: sentence classification; rebase required.
- #109 open: champion governance/rollback foundation; do not treat as production promotion until independent evidence is mandatory.
- #105 open: transformer benchmark; non-blocking for autonomous completion.

## Immediate critical path

1. Walk-forward discovery with purge/embargo, locked untouched OOS and multiple-testing controls.
2. Fail-closed champion gate; compare candidate vs incumbent on identical evidence.
3. Real orchestrator handlers, scheduler/daemon, triggers and measured resource guard.
4. Source-linked hypothesis routing into quantitative discovery.
5. DSL champion → autonomous paper runner → matured outcome analysis → next research cycle.
6. Reproducible real-XAUUSD acceptance within measured M2 8 GB limits.

## Reporting vocabulary

- **Implemented:** code exists and tests pass.
- **Trained:** a model was actually fitted and its reproducible run/artifact is recorded.
- **Validated:** independent chronological OOS and realistic cost-stress evidence exists.
- **Paper-traded:** exact version ran in the paper engine with immutable attribution.
- **Profitable:** only a cautious statement supported by sufficiently large forward evidence; never guaranteed.

A green CI run proves neither a trading edge nor profitability.
