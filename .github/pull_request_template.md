## Summary

What changed and why. Link the issue or phase.

## Phase / area

- [ ] Data pipeline
- [ ] Features / datasets
- [ ] Simulator / risk / backtest
- [ ] Strategy / calibration
- [ ] Classical ML
- [ ] Neural networks / RL
- [ ] API / frontend
- [ ] Docs / infrastructure

## Evidence

Commands actually run and their result. Do not paste invented metrics.

```text
cd backend && uv run pytest
cd frontend && npm test
```

## Data and model state

- Dataset id / checksum (if a model was trained):
- Model run id:
- Out-of-sample metric:
- Cost stress:
- Promotion / rejection decision:

## Checklist

- [ ] Tests added or updated for the change.
- [ ] `uv run ruff check src tests scripts` passes.
- [ ] `uv run mypy src` passes.
- [ ] `npm test` and `npx tsc --noEmit` pass for frontend changes.
- [ ] No data stores, `runs/`, model weights, secrets or `.env` committed.
- [ ] Docs updated (`STATUS.md`, relevant `docs/*`).
- [ ] No real-money execution path enabled.
- [ ] Claimed results are reproducible from the stated commit and dataset.

## Risk and rollback

What can break, and how to revert it. A failed experiment stays in the log;
it is not deleted.
