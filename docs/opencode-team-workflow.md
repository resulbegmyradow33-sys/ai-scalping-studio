# OpenCode team operating guide

This file configures how the owner uses OpenCode Teams in this repository. It
does not change the trading application's runtime architecture.

## Owner's environment

- Primary IDE/agent host: OpenCode Desktop on Mac mini M2, 8 GB unified memory.
- Agent model source: cloud models selected in the OpenCode session/provider.
- The project runtime remains local-first and must not depend on cloud LLM
  inference, paid AI APIs, paid market data or subscription services.
- Child agents inherit the current session model unless a model is explicitly
  configured. No provider/model is hard-coded here, so the owner can change
  the cloud model in OpenCode without editing this repository.
- OpenCode's cloud model usage is coding/research assistance, not a dependency
  of the trading application.

## What this repository configures

- `opencode.jsonc`: makes `autonomy-lead` the default primary agent, enables
  OpenCode's task list, and places new Git worktrees in a sibling directory
  outside the repository.
- `.opencode/agents/`: role-specific primary/subagents.
- `.opencode/commands/`: repeatable team audit, planning and review commands.
- Root `AGENTS.md`: common project contract for OpenCode and all other agents.

OpenCode discovers project agents under `.opencode/agents/` and commands
under `.opencode/commands/`. See the official
[Agents](https://opencode.ai/docs/agents/),
[Config](https://opencode.ai/docs/config/) and
[Commands](https://opencode.ai/docs/commands/) documentation.

## Team roles

| Agent | Mode | Write access | Responsibility |
|---|---|---:|---|
| `autonomy-lead` | primary | No | Owns task decomposition, delegation, conflict resolution and final evidence report |
| `repo-scout` | subagent | No | Maps source code, call paths, PRs, tests and legacy consumers |
| `quant-researcher` | subagent | No | Reviews causal evaluation, OOS, costs, leakage and selection bias |
| `knowledge-analyst` | subagent | No | Reviews document ingestion, provenance and hypothesis-to-experiment flow |
| `implementation-engineer` | subagent | Yes, assigned files only | Implements one bounded task in an isolated worktree |
| `test-engineer` | subagent | Yes, tests only unless reassigned | Designs/runs bounded tests and regression checks |
| `independent-reviewer` | subagent | No | Reviews implementation independently |
| `risk-safety-auditor` | subagent | No | Audits paper-only execution, risk gates, promotion and NO TRADE |

The lead may launch only the named subagents above. Read-only agents have
editing and shell execution denied. Implementers cannot push or merge.

## Standard cycle

1. **Reconnaissance:** lead asks repo-scout, quant-researcher and
   knowledge-analyst for independent read-only reports. For a change review,
   also use independent-reviewer and risk-safety-auditor.
2. **Plan:** lead reconciles evidence against source code and writes a
   dependency-ordered plan with task IDs, file ownership, dependencies,
   acceptance tests, resource budget and rollback.
3. **Isolated implementation:** each parallel writer gets a separate OpenCode
   worktree/branch. Assign one writer per file. If isolation is unavailable,
   serialize implementation; never concurrently edit one shared working tree.
4. **Verification:** test-engineer runs bounded targeted tests. Ask the owner
   before full suites, training, large backtests or resource-heavy jobs.
5. **Independent review:** reviewer and safety auditor inspect the resulting
   diff and callers, not just the implementer's summary.
6. **Integration:** lead resolves findings, verifies exact test output and
   updates docs/task status. The owner/maintainer reviews and merges the PR.

## Concurrency and M2 resource budget

- Parallelize read-only analysis freely when useful.
- Parallelize coding only across separate worktrees and non-overlapping files.
- One heavy training/backtest job at a time on the 8 GB M2.
- Keep ordinary code/test tasks small; use fixtures, chunked data, checkpoints
  and cancellation for expensive work.
- Cloud model concurrency does not mean local training concurrency. Do not
  launch several local training jobs because several agents are active.
- Do not run a full test suite from every agent. Assign one verification owner.

## Required task handoff

Every delegated task must include:
- exact goal and master-prompt section;
- evidence and current behavior;
- allowed files and files owned by others;
- dependencies and branch/worktree;
- acceptance tests and exact commands;
- RAM/CPU/disk budget;
- safety constraints and rollback;
- expected report format.

Every agent report must distinguish completed work from assumptions and
unproven items. The lead verifies reports against the repository.

## Slash commands

- `/team-audit [scope]` — coordinated read-only system audit.
- `/team-plan [goal]` — dependency-ordered multi-agent plan, no code changes.
- `/team-review [PR, branch or diff]` — independent code, quantitative and
  safety review.

## Safety and project invariants

All agents obey root `AGENTS.md` and the canonical master prompt:
- no owner-authored strategy recipe or mandatory named-strategy menu;
- generated strategies are bounded data-only DSL/AST, never arbitrary code;
- validation search fitness is not promotion evidence;
- missing champion means NO TRADE, never silent legacy fallback;
- independent risk gates cannot be relaxed by a model or agent;
- real-money execution remains out of scope;
- research documents never directly create orders;
- preserve legacy code until consumers and migration tests prove it safe to retire.

## First-run check

After pulling the branch and reopening the repository in OpenCode Desktop:

1. Confirm the active project root is the repository root.
2. Confirm `autonomy-lead` appears as the selected/default agent.
3. Confirm the seven named subagents appear in the agent picker.
4. Try `/team-plan walk-forward evidence pipeline` and confirm it produces a
   plan without editing source files.
5. Try `/team-audit current autonomous loop` and confirm it returns evidence
   from independent read-only roles.
6. Create a disposable worktree and confirm it appears in the configured
   sibling directory. Remove the disposable worktree only after confirming
   there is no user work in it.

If an agent is missing, inspect OpenCode's config/agent loading diagnostics;
do not copy these project agents into global configuration as a workaround.
