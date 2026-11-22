.PHONY: test lint check setup-github dev e2e-backend
test:
	cd backend && uv run pytest
lint:
	cd backend && uv run ruff check src tests scripts && uv run mypy src

# check: exactly what CI runs, for a local pre-push gate.
check:
	cd backend && uv run ruff check src tests scripts
	cd backend && uv run mypy src
	cd backend && uv run pytest
	cd frontend && npx tsc --noEmit
	cd frontend && npm test
	cd frontend && npm run build

# setup-github: labels, milestones, merge settings and main protection.
setup-github:
	./scripts/setup_github.sh
dev:
	cd backend && uv run uvicorn scalping.api.app:app --reload --port 8000

# serve-live: the validated flow-momentum card on the 5-year tick-derived store.
serve-live:
	cd backend && uv run python -m scalping.cli serve --root ../data-ticks5-store --flow-root ../data-flow-store --port 8000

# train-bc: imitate the live card's rule, then fine-tune PPO (experimental).
train-bc:
	cd backend && uv run python -m scripts.train_flow --tf H1 --train-end 2024-01-01 --steps 300000 --envs 8 --bc-steps 100000 --bc-epochs 4 --bc-lr 3e-4 --out-dir runs

# forward-crawl: Dukascopy ticks (bars + bid/ask flow) for the unseen window.
forward-crawl:
	cd backend && uv run python -m scripts.fetch_dukascopy_forward --start 2026-06-01 --end 2026-11-21 --root ../data-forward-store --flow-root ../data-forward-flow --cache ../data-forward-cache --hour-sleep 0.8

# forward-update: fetch new weekly ticks, rebuild, test, append to history.
forward-update:
	cd backend && uv run python -m scripts.update_forward

# forward-test: run the promoted card unchanged on the forward store.
forward-test:
	cd backend && uv run python -m scripts.forward_test

# rl-lab: retry imitation+PPO only when the data digest changed; never promotes.
rl-lab:
	cd backend && uv run python -m scripts.nightly_rl

# calibrate: re-select the card on the training slice and gate it on the holdout.
calibrate:
	cd backend && uv run python -m scripts.nightly_calibrate --store ../data-ticks5-store --flow-root ../data-flow-store --card runs/strategy_card.json --history runs/strategy_history.jsonl

# e2e-backend: reseed ./e2e-data with 10 days of synthetic XAUUSD M1, then serve it on :8000 for the Playwright suite.
e2e-backend:
	rm -rf e2e-data
	cd backend && uv run python -m scalping.cli data synth --root ../e2e-data --symbol XAUUSD --start 2024-10-01 --end 2024-10-10 --seed 7
	cd backend && uv run python -m scalping.cli serve --root ../e2e-data --port 8000
