# GitHub Copilot instructions

All repository changes must follow the root AGENTS.md. Read it before
suggesting or editing code. The canonical product goal is
docs/requirements/master-prompt.md; the current state is
docs/autonomous-system-audit-2026-11-24.md.

Do not add owner-authored strategy recipes, fixed strategy menus, unnecessary
model families, paid/cloud/LLM dependencies or real-money execution. Keep legacy
strategies as explicitly labelled baselines until a reviewed migration proves
they can be retired. Search winners on validation data are not champions:
promotion requires fail-closed walk-forward, locked OOS, cost, sample, regime,
stability and provenance evidence. Missing champion means NO TRADE, never a
silent legacy fallback. Every change needs tests, docs, an issue/PR and an
honest statement of what remains unproven.
