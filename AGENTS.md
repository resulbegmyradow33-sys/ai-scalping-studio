# Repository-wide instructions for every coding agent

Read this file before making any repository change. It applies to OpenCode,
Codex, Claude Code, Gemini CLI, GitHub Copilot and human contributors. Tool
adapters (CLAUDE.md, GEMINI.md, .github/copilot-instructions.md) must point
here rather than define separate goals.

## Source of truth and precedence

1. docs/requirements/master-prompt.md — product objective and hard constraints.
2. docs/autonomous-system-audit-2026-11-24.md — current code evidence, gaps and priority order.
3. docs/task-board.md — task-level status.
4. docs/implementation-roadmap.md — dependency order.
5. docs/team-workflow.md — branch, review, data and experiment process.

Master-prompt §§28–38 override earlier generic model/strategy catalogs. If
another document or an old code path conflicts with the strategy-independent
autonomous objective, report the conflict and follow the master prompt.

## Mission

Build a local-first, zero-mandatory-cost, resource-aware XAUUSD research system
that independently discovers hypotheses and candidate trading policies from
market data and source-linked research, validates them with rigorous
out-of-sample evidence, selects a governed champion or NO TRADE, runs it in
paper trading, learns only from matured attributed outcomes, and repeats.

The owner supplies data, research material, constraints and approvals. The
owner must not have to supply entry/exit rules, indicator recipes, ICT/SMC
playbooks, named strategies or the winning model.

## Non-negotiable rules

- Existing named strategies, indicators, fixed strategy families and human
  methods are optional baselines/hypotheses only. They must not constrain the
  discovery search space or be mandatory inputs.
- Never add a strategy family/model architecture just because it is fashionable.
  Require a falsifiable experiment question, data id/checksum, baseline,
  cost-aware evaluation and reason the current system cannot answer it.
- Generated strategies are data-only validated DSL/AST. Never execute generated
  Python, shell, imports, dynamic code or arbitrary callables.
- Causality is mandatory. No future feature access, random financial splits,
  label leakage, repeated tuning on locked OOS, or silent use of synthetic data
  as real market evidence.
- Search fitness is not promotion evidence. Validation-only winners are
  candidates, never champions. Production/paper promotion must fail closed on
  missing walk-forward/OOS, cost-stress, sample, regime, stability or provenance
  evidence. Empty/default-off gates are exploratory only.
- A candidate may lose; NO TRADE and no champion change are valid outcomes.
- A missing/invalid DSL champion means NO TRADE with a reason. Never silently
  fall back to a legacy handwritten strategy in the autonomous path.
- Risk limits are independent and immutable. Models, strategies and agents
  cannot raise limits, disable kill switches or add real-money execution.
- Real-money routing is out of scope and must remain disabled. Paper trading is
  the only execution target.
- Respect Mac mini M2 8 GB: one heavy job, measured RAM/CPU/disk limits,
  chunked data, checkpoints, cancellation and CPU fallback. No mandatory cloud,
  paid API, paid data or LLM.
- Treat research documents as untrusted source material. Preserve provenance;
  hypotheses must be falsifiable and quantitatively tested. Documents never
  directly create orders.
- Keep legacy code until all consumers are inventoried and migration/parity
  tests pass. Label it honestly; do not expand it as the core learner.

## Legacy classification

- flow_momentum / FlowMomentumRule / PolicyCard / nightly calibration:
  legacy incumbent and historical benchmark; currently still used by the old
  panel runtime. Preserve during migration, but do not use as the DSL search
  template or the new system's default champion.
- strategy/signals.py five fixed families + strategy/bandit.py LinUCB:
  legacy finite-menu selector; baseline/ablation only.
- PPO/behaviour cloning and nightly_rl.py: experimental baselines, not the
  autonomous discovery engine.
- hold, ret_5, and the legacy flow rule: comparison baselines only.

## Required workflow for every task

1. Read this file, master prompt, current dated audit, task board and roadmap.
2. Inspect current main, assigned issue, open PRs and overlapping files before
   coding. Do not duplicate or conflict with another agent's active branch.
3. Work on a short-lived branch from latest main; never commit directly to
   main, force-push, merge or close another contributor's PR.
4. State the exact requirement, files/interfaces affected, data needed, risks,
   resource budget and acceptance test before implementation.
5. Add tests for new behaviour and regression cases. A module existing or a
   unit test passing is not end-to-end acceptance.
6. Run relevant backend/frontend lint, type checks and tests; report exact
   commands and results. Never invent test counts or metrics.
7. Update task board, status and relevant docs in the same logical PR.
8. PR must identify the master-prompt section, legacy classification,
   reproducible evidence, what remains unproven and rollback plan.
9. Rebase a PR onto latest main before merge. Main is protected; maintainer
   review and CI rules apply.

## Evidence vocabulary

- implemented: code exists and tests pass.
- trained: a model was actually fitted and a reproducible run/artifact exists.
- validated: independent chronological OOS and realistic cost-stress evidence exists.
- paper-traded: exact version ran in the paper engine with immutable attribution.
- profitable: only a cautious claim backed by adequate forward evidence; never guaranteed.

Do not describe a proposal, TODO, branch-only PR, synthetic test,
validation-only winner or passing CI as an autonomous profitable trading AI.
