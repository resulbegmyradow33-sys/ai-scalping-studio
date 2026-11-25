# AI Scalping Studio

[![CI](https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/workflows/ci.yml)
[![CodeQL](https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/workflows/codeql.yml)

Local-first, zero-mandatory-cost XAUUSD research and paper-trading studio.
Real-money execution is disabled.

The target is a strategy-independent autonomous research learner: discover
hypotheses from market data and source-linked research, generate candidate
policies without owner-authored entry/exit rules, test them with realistic costs
and independent validation, and paper-trade only a governed DSL champion.
That complete closed loop is not yet finished; see the dated audit and task
board before interpreting any module as autonomous.

The existing flow_momentum card is a legacy incumbent with historical
forward-test evidence. The current panel still contains that older runtime
path; it is retained for compatibility and baseline comparison, not as the
strategy-discovery engine. PPO/behaviour cloning and the fixed five-family
LinUCB selector are research baselines, not the required autonomous learner.

## Project status

- [`docs/runbook.md`](docs/runbook.md) — operator runbook: install, daily flows,
  orchestrator, audited local controls, legacy-fallback switch, troubleshooting
  and safety rules.
- [`STATUS.md`](STATUS.md) — one-page matrix: done / partial / missing, with
  the exact commands that reproduce each claim.
- [docs/current-state-addendum-2026-11-24.md](docs/current-state-addendum-2026-11-24.md) — current mainline facts and corrections.
- [docs/autonomous-system-audit-2026-11-24.md](docs/autonomous-system-audit-2026-11-24.md) — historical autonomous-system audit and ordered gaps.
- [AGENTS.md](AGENTS.md) — mandatory instructions for every human/AI contributor.
- [docs/architecture-audit.md](docs/architecture-audit.md) — historical 2026-11-22 snapshot.
- [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md) — phased plan.
- [`docs/task-board.md`](docs/task-board.md) — every task, checked or open.
- [`docs/requirements/master-prompt.md`](docs/requirements/master-prompt.md) — target brief.
- [`docs/language-decision.md`](docs/language-decision.md) — why Python/TypeScript stay; when Rust would be justified.
- [`docs/api-contracts.md`](docs/api-contracts.md) — public API schema, drift guard and how to update the snapshot.

## Team

**New here? Start with [`START_HERE.md`](START_HERE.md).**

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — clone, branch, test, push, PR.
- [`docs/team-workflow.md`](docs/team-workflow.md) — roles, reviews, data/model rules.
- [`TEAM.md`](TEAM.md) — members, roles and permissions.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — expected behaviour.
- [`SECURITY.md`](SECURITY.md) — private vulnerability reporting.
- [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE) — bug, feature, data, experiment forms.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed.

`main` is the protected integration branch: pull request, one review,
CODEOWNERS, `backend` and `frontend` CI checks, up-to-date branch, linear
history, conversation resolution, no force push or deletion. CodeQL and secret
scanning run on the public repository. `make hooks` adds a local pre-push guard
on top of the server rules.

## Legacy incumbent evidence (not the autonomous learner)

flow_momentum, H1, card backend/config/strategy_card.json:

- momentum signal (EMA20 + 5-bar displacement + ATR expansion),
- entry only when `|flow|` exceeds the causal trailing 15th-percentile threshold
  over 480 bars, where `flow = (ask_volume - bid_volume) / (ask_volume + bid_volume)`,
- exit at 3×ATR stop, 6×ATR target, or 48 bars.

| Slice | Trades | Expectancy | PF | Max DD |
|---|---|---|---|---|
| Historical holdout (2025-05→2026-05) | 215 | +0.159 R | 1.32 | 8.1% |
| Forward, Jun–Nov 2026 (card unchanged) | 24 | +0.502 R | 2.15 | 2.4% |
| Forward, momentum only (ablation) | 58 | −0.128 R | 0.79 | 6.5% |

The forward sample is small (95% CI still includes zero); it is historical
baseline evidence, not a guarantee and not evidence that the new DSL discovery
system has found a profitable strategy. The ablation is useful evidence about
this specific legacy rule only.

RL result: imitation + PPO is stable but weaker than the rule (+0.012 R and
+0.020 R on two splits). See `docs/rl-imitation.md`.

## Layout

```
backend/   FastAPI, Parquet store, features, simulator, PPO, strategy card
frontend/  Next.js 15 panel (simulator, training, performance, data, paper)
docs/      design, live strategy, forward test, RL notes
```

The backend is Python 3.12 managed by `uv`; the frontend is Next.js + Tailwind.

## Quick start (synthetic data)

```bash
cd backend
uv sync
uv run pytest                       # full backend suite

uv run python -m scalping.cli data synth \
  --root ../demo-data --symbol XAUUSD --start 2024-10-01 --end 2024-10-10
uv run python -m scalping.cli serve --root ../demo-data --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev -- -p 3001
```

Open `http://127.0.0.1:3001`. Without a strategy card the panel replays the
legacy M1 momentum rule by default during migration. Open PR #177 adds
`--no-legacy-fallback` for fail-closed UI testing; it does not change the
autonomous paper path. See the runbook for details and merge status.

To use the validated rule, copy the seed card and start with a flow store:

```bash
cd backend
mkdir -p runs
cp config/strategy_card.json runs/strategy_card.json
uv run python -m scalping.cli serve \
  --root ../data-ticks5-store --flow-root ../data-flow-store --port 8000
```

`runs/strategy_card.json` is runtime state and is gitignored; the tracked seed
lives in `backend/config/`.

## Verify the current claims

`STATUS.md` lists the pytest files and CLI invocations behind each status row.
Two examples:

```bash
cd backend
uv run pytest tests/test_promotion_policy.py tests/test_locked_holdout.py
uv run python -m scalping.cli discovery run --root ../data-ticks5-store \
  --symbol XAUUSD --timeframe M5 --max-trials 64 --seed 0
```

Main is red on `tests/test_api_autonomy.py` as of 2026-11-24; the cause and the
latest green CI run id are recorded in `STATUS.md`.

## Data

No market data is committed (the stores are hundreds of MB to several GB).

| Dataset | Source | Script |
|---|---|---|
| 5y tick-derived XAUUSD M1/M5/M15/H1/D1 + flow | HuggingFace `CarlosSilva1/xauusd-ticks` | `scripts/aggregate_ticks_v2.py`, `scripts/aggregate_flow.py` |
| 17y broker M1 (no volume) | MetaTrader CSV export | `scripts/ingest_broker_csv.py` |
| 2026 forward ticks + flow | GitHub `Touhid-HZ/xauusd-tick-data-2026` | `scripts/ingest_2026_ticks.py` |

Build the forward store and run the card unchanged on unseen data:

```bash
cd backend
make forward-update      # download new weeks, rebuild, test, record history
make forward-test        # rerun the current forward store only
```

See `docs/forward-test.md` for coverage, results and caveats.

## Legacy experiment loops (not the autonomous orchestrator)

These older jobs remain for reproducibility and comparison; they are not yet
wired into the new persistent orchestrator:

- `scripts/nightly_calibrate.py` re-selects the rule's free parameters on the
  training slice, gates the winner on a holdout, and replaces the card only
  when it passes. It never sees the forward store.
- `scripts/nightly_rl.py` retries imitation + PPO only when the historical data
  digest changes, and records `beats_rule` without promoting anything.

The forward store is deliberately separate: feeding it into selection would
destroy the only clean forward test.

## Commands

```bash
cd backend
uv run python -m scalping.cli serve --root ../data-ticks5-store --flow-root ../data-flow-store
uv run python -m scalping.cli ml train --help
uv run python -m scalping.cli ml evaluate --help
uv run python -m scripts.nightly_calibrate --help
uv run python -m scripts.nightly_rl --help
uv run python -m scripts.forward_test --help
uv run python -m scripts.train_flow --help
```

```bash
cd frontend
npm test
npm run build
npm run e2e        # Playwright, needs the backend on :8000
```

## Safety

- Demo/paper only. No broker credentials, no order routing, no real money.
- No secrets are committed; API keys are not used.
- The card is a research artifact, not investment advice.
