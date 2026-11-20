.PHONY: test lint dev
test:
	cd backend && uv run pytest
lint:
	cd backend && uv run ruff check . && uv run mypy src
dev:
	cd backend && uv run uvicorn scalping.api.app:app --reload --port 8000
