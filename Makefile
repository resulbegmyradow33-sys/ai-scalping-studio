.PHONY: test lint dev e2e-backend
test:
	cd backend && uv run pytest
lint:
	cd backend && uv run ruff check . && uv run mypy src
dev:
	cd backend && uv run uvicorn scalping.api.app:app --reload --port 8000

# serve-live: the validated flow-momentum card on the 5-year tick-derived store.
serve-live:
	cd backend && uv run python -m scalping.cli serve --root ../data-ticks5-store --flow-root ../data-flow-store --port 8000

# calibrate: re-select the card on the training slice and gate it on the holdout.
calibrate:
	cd backend && uv run python -m scripts.nightly_calibrate --store ../data-ticks5-store --flow-root ../data-flow-store --card runs/strategy_card.json --history runs/strategy_history.jsonl

# e2e-backend: reseed ./e2e-data with 10 days of synthetic XAUUSD M1, then serve it on :8000 for the Playwright suite.
e2e-backend:
	rm -rf e2e-data
	cd backend && uv run python -m scalping.cli data synth --root ../e2e-data --symbol XAUUSD --start 2024-10-01 --end 2024-10-10 --seed 7
	cd backend && uv run python -m scalping.cli serve --root ../e2e-data --port 8000
