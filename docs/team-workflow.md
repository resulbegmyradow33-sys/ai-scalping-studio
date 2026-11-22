# Team workflow

How work moves from an idea to `main` without breaking the validated system.
The repository is public; write access is granted per person, not per team (a
personal GitHub account has collaborators, not org teams).

## Roles

| Role | Responsibility |
|---|---|
| Maintainer | merges to `main`, branch protection, releases, security reports |
| Contributor | feature/fix/experiment branches, tests, PR evidence |
| Reviewer | at least one approval on code PRs; checks evidence, not style only |
| Data owner | source, licence, coverage, provenance and checksum for a dataset |
| Model owner | dataset id, config, run id, out-of-sample and cost-stress evidence |

One person may hold several roles, but a model is never promoted on the
author's claim alone.

## Branch model

- `main` is the only long-lived branch and the default.
- Work happens on short-lived branches:
  - `feat/<scope>` — user-visible feature
  - `fix/<scope>` — bug fix
  - `data/<source>-<symbol>` — dataset import
  - `exp/<model>-<dataset>` — experiment
  - `docs/<scope>` — documentation
- Branch from `main`, rebase on `main` before opening/updating a PR.
- Squash merge only; the branch is deleted after merge.
- Direct pushes to `main` are blocked for non-admins. The maintainer can bypass
  in an emergency, and must say so in the PR/issue.

## Pull request lifecycle

1. Open an issue if the work is not trivial. Use the templates.
2. Create the branch and keep the diff focused. One logical change per PR.
3. Add or update tests. A feature without a test is not done.
4. Run locally:

   ```bash
   cd backend
   uv run ruff check src tests scripts
   uv run mypy src
   uv run pytest

   cd ../frontend
   npx tsc --noEmit
   npm test
   npm run build
   ```

5. Fill the PR template. State what was actually run and the result.
6. CI must be green (`backend`, `frontend`) and the branch up to date.
7. One approving review, conversations resolved, then squash merge.
8. Delete the branch. Update `STATUS.md` and `CHANGELOG.md` when the phase
   status changes.

Review SLA: best effort within one business day. A review checks correctness,
tests, data/model claims and risk, not formatting preferences.

## Issues, labels and milestones

Use an issue for every non-trivial task. Labels:

`bug`, `enhancement`, `data`, `experiment`, `research`, `docs`, `infra`,
`security`, `triage`, `blocked`, `good first issue`, `help wanted`,
`priority:high`, `priority:normal`, `priority:low`, `promotion`, `rollback`.

Milestones are the roadmap phases (`Phase 2` … `Phase 15`). An issue belongs
to exactly one phase. A phase closes only when its exit criteria in
`docs/implementation-roadmap.md` are met.

Use Discussions for design questions and result interpretation; use Issues for
actionable work; use PRs for code.

### Project board

The Projects tab is enabled. Create one board manually (the CLI token does not
carry the `project` scope):

1. Repository → Projects → New project → Board.
2. Columns: **Backlog**, **Ready**, **In progress**, **Review**, **Done**.
3. Add open issues; keep at most three in **In progress** per person.
4. Link the phase milestone in each issue; a phase is done only when its exit
   criteria and all issues are closed.

## Data rules

- Never commit `data-*` stores, `runs/`, `*.parquet`, model weights or `.env`.
- Every imported dataset gets:
  - source URL/provider and licence,
  - symbol, timeframe, UTC coverage, row count,
  - checksum and provenance sidecar (`_meta/<symbol>/<timeframe>.json`),
  - a note when synthetic or gap-filled.
- A dataset used by a model is identified by its `dataset_id` in the PR and
  run config.
- Do not overwrite a dataset silently. A new source is a new dataset id.

## Experiment protocol

A model experiment PR/issue must state:

1. **Hypothesis** — one falsifiable sentence.
2. **Dataset id / checksum** — what was trained on.
3. **Split** — chronological ranges, purge/embargo, and the untouched test
   window.
4. **Baselines** — majority, rule, hold, previous champion.
5. **Results** — prediction metrics, simulator P&L, trade count, drawdown,
   cost stress and regime breakdown.
6. **Decision** — promote, reject, needs more data, or needs review.
7. **Rollback** — the previous champion and how to restore it.

Promotion is never automatic. The candidate must beat the champion on the same
windows and pass the risk/cost gates in `docs/model-validation.md`.

Use these words precisely:

- **implemented** — code exists and is tested;
- **trained** — a model was fitted and artifacts exist;
- **validated** — out-of-sample + cost-stress evidence exists;
- **profitable** — not claimed without forward evidence.

## Definition of done

- Code and tests merged through a PR.
- Existing tests still pass; new behavior has a test.
- Docs/status updated.
- No data, secrets or run artifacts committed.
- The claim in the PR is reproducible from the stated commit and dataset.
- For a model: run id, dataset id, cost stress, and promotion/rejection record
  exist.

## Releases

Tags are `v0.x.y` on `main`. A release notes file is generated from
`CHANGELOG.md`. No release enables real-money execution.

## Onboarding checklist for a new contributor

- [ ] Invited as a collaborator with write access (or fork + PR).
- [ ] `gh auth login` and `gh auth setup-git` done.
- [ ] Repository cloned; `uv sync`, `npm install` done.
- [ ] Backend and frontend tests pass locally.
- [ ] Read `README.md`, `STATUS.md`, `CONTRIBUTING.md`, this file and
      `docs/architecture-audit.md`.
- [ ] Picked an issue from a phase milestone and commented on it.
- [ ] Opened the first PR with the template filled.

## GitHub settings

`scripts/setup_github.sh` applies what the repository owner can change with the
`gh` CLI:

- description, topics, issues/projects/discussions;
- squash-only merges, auto-merge, delete branch on merge;
- labels and phase milestones;
- vulnerability alerts and Dependabot security fixes;
- `main` branch protection **when the plan allows it**.

### Public repository

The repository is public, so GitHub Free includes branch protection, rulesets,
secret scanning, push protection and CodeQL. `scripts/setup_github.sh` applies:

- `main` protection: PR required, one approval, CODEOWNERS review,
  `backend` + `frontend` checks, up-to-date branch, linear history,
  conversation resolution, no force push/delete;
- secret scanning and push protection;
- vulnerability alerts and Dependabot security fixes.

The local `pre-push` hook remains as defense in depth. If the repository ever
becomes private again, server-side protection disappears on the Free plan and
the hook becomes the only local gate.
