# Contributing / collaborator guide

The repository is **public**. Anyone can read it; write access still requires an
invitation (or a fork + pull request).

## 1. Get access

Owner adds the collaborator (write permission):

```bash
gh api -X PUT repos/resulbegmyradow33-sys/ai-scalping-studio/collaborators/USERNAME \
  -f permission=push
```

Or GitHub → repository → **Settings → Collaborators → Add people**.

Without write access, fork the repository and open a pull request from the
fork. The rest of this guide assumes write access.

## 2. Authenticate

HTTPS (simplest):

```bash
gh auth login          # GitHub.com → HTTPS → browser/token
gh auth setup-git      # makes git use the gh credential helper
```

A normal GitHub password does not work for git. Use a **personal access token**
with `repo` scope if not using `gh`.

SSH (optional, if the machine has a key added to GitHub):

```bash
ssh -T git@github.com
git clone git@github.com:resulbegmyradow33-sys/ai-scalping-studio.git
```

## 3. Clone and install

```bash
git clone https://github.com/resulbegmyradow33-sys/ai-scalping-studio.git
cd ai-scalping-studio

cd backend
uv sync
uv run pytest

cd ../frontend
npm install
npm test
```

Install the local guard (defense in depth; the server also protects `main`):

```bash
make hooks     # blocks direct pushes to main unless ALLOW_MAIN_PUSH=1
```

## 4. Data is not in the repository

Stores are hundreds of MB to several GB and are gitignored on purpose:

- `data-*`, `runs/`, `*.parquet`, model weights, `.env`.

Each developer keeps their own local data. Minimum workflow without real data:

```bash
cd backend
uv run python -m scalping.cli data synth \
  --root ../demo-data --symbol XAUUSD --start 2024-10-01 --end 2024-10-10
uv run python -m scalping.cli serve --root ../demo-data --port 8000
```

Real data sources and scripts are listed in `README.md` and
`docs/forward-test.md`. Never commit data, credentials or generated run
directories.

## 5. Issues and milestones

Non-trivial work starts as an issue (bug, feature, data import or experiment
template). Pick the phase milestone; keep at most three issues in progress per
person. The repository board uses Backlog / Ready / In progress / Review / Done.

## 6. Daily workflow

Always start from the latest branch:

```bash
git switch main
git pull --rebase origin main
```

Create a topic branch for each change:

```bash
git switch -c feat/symbol-registry
# edit, test
git add backend/src/scalping/data/registry.py backend/tests/test_registry.py
git commit -m "feat(data): add symbol registry"
git push -u origin feat/symbol-registry
```

Then open a pull request into `main`:

```bash
gh pr create --base main --head feat/symbol-registry \
  --title "feat(data): symbol registry" --body "What changed and what was tested."
```

Every change goes through a pull request. `main` requires green CI and one
review, so even a documentation-only change gets the same gate.

## 7. Receiving someone else's changes

```bash
git switch main
git pull --rebase origin main
```

If you have uncommitted work, commit or stash it first. Rebase keeps the
history linear; resolve conflicts file by file, then:

```bash
git add <resolved files>
git rebase --continue
```

If a rebase goes wrong:

```bash
git rebase --abort
```

## 8. Before every push

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

Do not push with failing tests, unformatted code or committed data. If a test
is too slow for a quick loop, run the relevant file first and the full suite
before opening the PR.

## 9. What not to commit

- `backend/runs/` and `runs/` (cards, histories, checkpoints)
- `data-*` stores, CSV/Parquet/tick caches
- `.env`, tokens, broker credentials
- Large model weights (`*.zip`, `*.pt`, `*.pkl`, `*.npz`)
- `node_modules/`, `.next/`

`git status` must be clean of those before a push. If one was staged by
accident:

```bash
git restore --staged <path>
```

## 10. Commit style

Short imperative subject, scope when useful:

```
feat(data): add ten-symbol registry
fix(risk): block entries above max spread
docs(nn): document TCN receptive field
test(api): cover provenance endpoint
```

Keep one logical change per commit. Do not mix refactors with behavior changes.

## 11. Reporting state honestly

The project separates four states; use them in PRs and reports:

- **implemented** — code exists and is tested;
- **trained** — a model was actually fitted and artifacts exist;
- **validated** — out-of-sample and cost-stress evidence exists;
- **profitable** — not claimed unless the forward test supports it.

Never describe a listed architecture as implemented, or a trained model as
profitable.
