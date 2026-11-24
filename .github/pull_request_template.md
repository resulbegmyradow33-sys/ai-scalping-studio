## Summary

What changed and why. Link the issue or phase.

## Master objective alignment

- Master-prompt section(s):
- Does this advance strategy-independent discovery / evidence / closed-loop paper learning?
- Does it preserve the rule that owner-supplied strategies are optional baselines only?
- Which existing interfaces and open PRs were checked for overlap?
- Legacy components touched (if any) and why they remain necessary:
- Acceptance test proving the end-to-end behaviour (not just a unit placeholder):

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
- [ ] No hard-coded strategy menu or owner-authored entry/exit recipe was added to the core learner.
- [ ] Missing OOS/cost/regime/provenance evidence fails closed; no validation-only champion promotion.
- [ ] Root AGENTS.md and canonical master prompt were followed.
- [ ] Claimed results are reproducible from the stated commit and dataset.

## Risk and rollback

What can break, and how to revert it. A failed experiment stays in the log;
it is not deleted.
