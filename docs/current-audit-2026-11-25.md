# Current repository audit — 2026-11-25

## Scope and evidence

Read-only review of `main` at `a369adaff8fb66e7b9490f65a3792ce5d966d525` (563 tree entries), the current task board and master requirements, source modules, committed data-quality/model-comparison/paper-gap evidence, open issues, open PRs, and GitHub Actions checks.

Repository: https://github.com/resulbegmyradow33-sys/ai-scalping-studio

Main-branch CI checks on the audited commit: `frontend`, `backend`, `analyze (python)`, and `analyze (javascript-typescript)` all completed successfully. This establishes build/test health for that commit, not trading performance.

## Executive verdict

The project is materially closer to the intended research platform: causal features and datasets, LSTM/GRU/TCN and Transformer implementations, PPO/RL, regime models, bounded DSL candidate discovery, immutable evidence bundles, locked-holdout primitives, promotion policy, champion governance, a continuous paper runner, an orchestrator, and research/backtest UI are present.

It is **not** a completed autonomous self-learning trading system. There is no validated production candidate, no real locked holdout spend, no evidence-mode champion, no sustained champion paper session, and no demonstrated trading edge. The key missing work is trustworthy data/provenance and connecting existing lifecycle components—not adding more model names.

## Verified findings

### 1. Data quality is the immediate research blocker

The committed multi-store audit reports 15 slices, 11,502,309 physical rows across 1,309 Parquet partitions, 0 unreadable partitions, but **0 clean slices, 9 warning slices, and 6 failed slices**.

- Only XAUUSD is present among the ten registered instruments; no L2/order-book data exists.
- The 17-year `data-real-store` history has null `ask_close` and `spread` on every row and zero volume throughout. Its current validator labels all five timeframes failed. This is bid-only data, not full quote/order-flow data.
- `data-ticks5-store` H1/M1 histories are structurally cleaner but end 2026-05-22 (about 187 days before this audit).
- `data-real-store` ends 2026-01-09; `data-forward-store` ends 2026-11-15 (about 9.2 days old at audit time).
- The flat `data-flow-store` layout is read differently by `ParquetStore.read` and `flow_series()`; the latter sees 1,785,951 rows and 12,620 duplicate timestamps while the former sees zero bar rows.
- Nine of fifteen slices lack provenance sidecars. The two M1 histories are not independent in the checked month.
- The gap rule flags D1 weekends; M1 outlier threshold is too noisy for the observed tick-size scale.

Source: `docs/data-quality-audit-2026-11-25.md` and `docs/evidence/data-quality-2026-11-25.json`. Issue #202 is the active data-contract work item.

### 2. Current model comparison is a negative result

The real XAUUSD H1 comparison used 29,586 bars, 59 features and a fixed chronological split. Logistic regression, LightGBM, MLP and the `ret_5` rule did not beat the train-prior baseline on the reported prediction metrics. All tested arms lost money at base, 2x and 3x costs. No arm is suitable for promotion.

Source: `docs/model-comparison-2026-11-25.md` and `docs/evidence/model-comparison-2026-11-25.json`.

Model-family implementation status:
- LSTM, GRU, CNN-LSTM, causal TCN, Transformer, causal Transformer, PatchTST and iTransformer builders exist.
- PPO/RL and regime detection code exist.
- Presence of builders/tests does not mean trained checkpoints, out-of-sample edge, or promotion-grade evidence. The latest status reports no committed neural checkpoint and PPO remains experimental.

### 3. Discovery → evidence → holdout → promotion still stops

The merged real XAUUSD H1 acceptance attempt ran validation, quality, dataset construction, discovery, evidence persistence, reproduction and promotion evaluation. Discovery returned `no_trade`; `PROMOTION_V1` rejected the candidate, including `missing_locked_oos`; paper correctly refused with `NoChampionError`. No holdout was sealed or spent and no promotion occurred.

Source: `docs/evidence/acceptance-attempt-2026-11-25.json` and the acceptance report.

The modules exist, but the following links remain missing:
- Dataset/store provenance (`source`, `synthetic`, lineage/checksum) is not carried through `DatasetManifest` → `CandidateEvidence` → holdout → promotion. The promotion policy therefore cannot independently fail closed on synthetic/unknown provenance.
- No ordinary supported CLI/API/orchestrator path performs owner-approved one-shot holdout evaluation and promotion. The CLI has discovery run/reproduce/evidence, not discovery promote.
- The orchestrator's data trigger can build datasets, but does not autonomously complete evidence → holdout → promotion. Matured paper outcomes create proposals; they do not automatically advance paper.
- No clock/new-bar scheduler drives `paper.advance`. The continuous runner is caller-driven.
- The paper-gap real-data run used an explicit test-only constant DSL spec. It proves gap-handling plumbing and append-only gap audit only; it is not an evidence champion and is not strategy validation.

### 4. Validation and target hardware

The paper-gap acceptance measured a small bounded M1/H1 run (about 0.58s / 0.12s median, 116MB / 107MB peak RSS) with a test-only spec. It does not establish a full research loop or long-running process on the target Mac mini M2 8GB.

The full target-hardware acceptance, full walk-forward acceptance, and sustained champion paper operation remain unproven. Keep the 8GB one-heavy-job limit and offline/local-first constraints.

### 5. UI and PR status

The main commit's frontend/backend and Python/TypeScript analysis checks are green. Two useful open PRs also have successful checks on their heads:
- #203 adds interaction-level Playwright coverage and fixes a paper-form focus/submit defect. It is still open and its branch is behind main.
- #204 derives trial exposure from the discovery archive rather than trusting a caller-declared trial count. It is still open; its checks are green.

Other open work:
- #181 addresses a real stale-state bug in RL action masking, but GitHub currently reports the PR branch as dirty/conflicted against main.
- #182 adds optional next-bar entry delay and depends on #181.
- #178 wires audited gap policy through the orchestrator, but its branch is also dirty/conflicted against main.
- #202 is the active multi-store data-contract issue; #195 is the dependency-ordered master backlog.

No PR was merged during this audit. Project governance in #195 says PRs are proposals and the owner is the final authority; do not self-merge.

## Priority order

1. **P0 — Data contract (#202):** define bid-only quote semantics and expose capabilities; unify flow-store access/deduplication; classify session gaps; calibrate outliers; add freshness/coverage and provenance gates. Do not silently reinterpret missing quotes or invent volume/order flow.
2. **P0 — End-to-end provenance:** carry immutable source/synthetic/checksum lineage from store through dataset, evidence, holdout, promotion, champion and paper ledger; reject missing/unknown/synthetic provenance for production promotion.
3. **P0 — Statistical exposure:** review and land #204 only after it is rebased/rechecked; archive-derived trial exposure must be used by evidence/promotion, not remain an isolated helper.
4. **P1 — Runtime lifecycle:** implement owner-authorized holdout/promotion entry point with one-shot enforcement and durable rejection audit; preserve locked OOS isolation.
5. **P1 — Orchestration and paper loop:** connect data → dataset → discovery → evidence/review; add an explicit safe scheduler for paper advance after evidence promotion; matured outcomes should create experiments, never auto-promote.
6. **P1 — RL correctness:** rebase and independently review #181; only then consider #182. Rebase #178 and keep `strict` gap handling as default.
7. **P1 — Acceptance:** repeat a reproducible real-data walk-forward and end-to-end paper test on the M2 8GB target, with memory/disk/runtime records.
8. **P2 — Model expansion:** only after data and loop gates pass, train/compare small LSTM/GRU/TCN/Transformer/PPO variants against strong simple baselines. Do not add a larger model zoo as a substitute for edge.

## Evidence vocabulary

- **Implemented:** code exists and relevant tests pass.
- **Trained:** a reproducible fitted model/checkpoint and run record exist.
- **Validated:** independent chronological OOS plus realistic costs/stress exists.
- **Paper-traded:** the exact governed version ran in paper mode with immutable attribution.
- **Profitable:** only a cautious claim after adequate sustained forward evidence net of costs.

Green CI means the code checks passed. It does not mean a model is trained, validated, paper-traded, or profitable.
