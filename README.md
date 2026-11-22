# AI Scalping Studio

[![CI](https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/resulbegmyradow33-sys/ai-scalping-studio/actions/workflows/ci.yml)

Local research and paper-trading studio for XAUUSD. Everything runs on one
machine: no paid AI APIs, no cloud services, no real money.

The project grew out of a simple question: can a reinforcement-learning agent
learn a gold scalping edge from tick data? The honest answer so far is **no, but
a small rule with signed order-flow does have a measured edge**. The live panel
therefore replays that rule; PPO stays in the research lab.

## Project status

- [`STATUS.md`](STATUS.md) — one-page matrix: done / partial / missing.
- [`docs/architecture-audit.md`](docs/architecture-audit.md) — evidence-based audit.
- [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md) — phased plan.
- [`docs/requirements/master-prompt.md`](docs/requirements/master-prompt.md) — target brief.

## Team

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — clone, branch, test, push, PR.
- [`docs/team-workflow.md`](docs/team-workflow.md) — roles, reviews, data/model rules.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — expected behaviour.
- [`SECURITY.md`](SECURITY.md) — private vulnerability reporting.
- [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE) — bug, feature, data, experiment forms.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed.

`main` is the protected integration branch. CI runs `backend` and `frontend`
on every PR. On the private GitHub Free plan, server-side branch protection is
unavailable, so `make hooks` installs a local pre-push guard and the team
follows the PR rule; `scripts/setup_github.sh` applies labels, milestones and
all settings the plan allows. Making the repository public (or upgrading to
Pro) enables server-side protection at no cost only in the public case.

## What is validated

`flow_momentum`, H1, card `backend/config/strategy_card.json`:

- momentum signal (EMA20 + 5-bar displacement + ATR expansion),
- entry only when `|flow|` exceeds the causal trailing 15th-percentile threshold
  over 480 bars, where `flow = (ask_volume - bid_volume) / (ask_volume + bid_volume)`,
- exit at 3×ATR stop, 6×ATR target, or 48 bars.

| Slice | Trades | Expectancy | PF | Max DD |
|---|---|---|---|---|
| Historical holdout (2025-05→2026-05) | 215 | +0.159 R | 1.32 | 8.1% |
| Forward, Jun–Nov 2026 (card unchanged) | 24 | +0.502 R | 2.15 | 2.4% |
| Forward, momentum only (ablation) | 58 | −0.128 R | 0.79 | 6.5% |

The forward sample is small (95% CI still includes zero). The ablation is the
useful part: without the signed-flow gate the same momentum signal loses money.

RL result: imitation + PPO is stable but weaker than the rule (+0.012 R and
+0.020 R on two splits). See `docs/rl-imitation.md`.

## Layout

```
backend/   FastAPI, Parquet store, features, simulator, PPO, strategy card
frontend/  Next.js 15 panel (simulator, training, performance, data)
docs/      design, live strategy, forward test, RL notes
```

The backend is Python 3.12 managed by `uv`; the frontend is Next.js + Tailwind.

## Quick start (synthetic data)

```bash
cd backend
uv sync
uv run pytest                       # ~500 tests

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
legacy M1 momentum rule; the data screen can generate synthetic bars.

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

## The learning loop

Two background jobs keep the system honest:

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
