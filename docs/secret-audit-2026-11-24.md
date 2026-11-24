# Secret and credential hygiene audit — 2026-11-24

Scope: the whole repository at `feat/sec2` (base `main` = `693f8a3`), working
tree plus full git history. Read-only scan; this document and one regression
test are the only changes. No history was rewritten, nothing was pushed and no
dependency was added.

Result: **no credential, provider key, private key, broker login or `.env`
content is committed, and none was found in history.** The two remaining issues
are structural, not leaks: `.gitignore` is narrower than the secret surface, and
CI has no secret gate of its own. Both are low severity and listed below with
a recommendation each.

## What was run

Tools used as already installed locally; nothing installed for this audit.

```
gitleaks 8.30.1

# working tree (all files on disk, 4.61 MB)
gitleaks detect --source . --no-git \
  --report-format json --report-path /tmp/sec2-tree.json --exit-code 0

# full history, every ref
gitleaks git --report-format json --report-path /tmp/sec2-hist.json \
  --exit-code 0 --log-opts="--all"

# manual passes gitleaks cannot judge
git log --all -p --no-color                    # inspected for added secret-shaped lines
git log --all --pretty=format: --name-only --diff-filter=A | sort -u   # added filenames
rg -n -i 'api[_-]?key|secret[_-]?key|access[_-]?token|private[_-]?key|
  BEGIN .*PRIVATE KEY|password\s*[:=]|passwd|bearer .{12,}|sk-.{16,}|
  ghp_.{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-' .
python3 entropy sweep over tracked text files (Shannon > 4.2, len >= 24)
git check-ignore against .env, .env.local, .env.production, id_rsa, key.pem,
  secrets.json, credentials.yml, .npmrc
cd backend && uv run pytest tests/test_no_committed_secrets.py
```

## Coverage and result

| Surface | Scanned | Result |
| --- | --- | --- |
| Working tree | 409 tracked files, 4.61 MB | clean |
| Git history | 259 non-merge commits of 287 (28 merges carry no unique patch) | clean |
| Commit authors/dates | 2026-11-20 → 2026-11-24 | clean |
| Files ever added with secret-like names (`.env`, `*.pem`, `*.key`, `id_rsa`, `credentials*`, `.npmrc`, `.netrc`) | none | clean |
| High-entropy sweep | 126 candidates | all test identifiers and lockfile hashes, no credential |
| `backend/src/scalping/paper/mt5.py` | full module | no login, no credential, no network import |
| Configs (`backend/config/*.json`, `*.yml`, `*.toml`, `opencode.jsonc`) | full | no secret keys; `strategy_card.json` holds only strategy params and holdout metrics |
| CI and hooks (`.github/workflows/*`, `.githooks/pre-push`) | full | no embedded token; no secret step exists |
| Docs and fixtures (`docs/**`, `backend/tests/fixtures/**`, `conftest.py`) | full | only prose, synthetic CSV, and one generated YAML/CSV fixture; no credential |
| `frontend/.env.example` | single line | `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`, public by design |
| Scripts (`scripts/*`, `backend/scripts/*`) | full | use `gh` auth and CLI args only; no hardcoded token |

There is therefore no finding to attach a commit hash and date to: an exhaustive
history scan with two independent methods (gitleaks rules plus a manual added-line
and entropy pass) returned nothing.

## Controls the audit verified on purpose

These are the places a leak usually hides, and each is already handled:

- **MT5 adapter has no credential.** `backend/src/scalping/paper/mt5.py`
  (introduced `f1e53b1`, 2026-11-23) is a refused-by-construction skeleton: no
  login, no account number, no server name, no network client, and every order
  method raises. `backend/tests/test_paper_mt5.py` asserts the source contains
  no `password`/`api_key`/`access_token` marker and no network import.
- **Runtime tokens are generated, not committed.** The locked-holdout unlock
  token (`693f8a3`, 2026-11-24) is minted with `secrets.token_urlsafe(32)` and
  only its SHA-256 digest is persisted; the plaintext lives in memory for one
  call. That is the correct pattern and should stay the template for any future
  token.
- **No secret ships to the browser.** The only client-exposed variable is
  `NEXT_PUBLIC_API_BASE` pointing at localhost (`frontend/src/lib/api.ts`,
  `frontend/.env.example`, `9360365`). Nothing secret sits behind a
  `NEXT_PUBLIC_`/`VITE_` prefix.
- **Settings hold no secret.** `backend/src/scalping/config.py` loads contract,
  risk and cost blocks only; there is no credential field and no `SecretStr`
  path that could be misconfigured.
- **`SECURITY.md` states the policy correctly**: rotate first, then remove from
  history. That matches the response order this audit assumes.

## Findings

### F-1 — No live or historical secret (informational)

Nothing to rotate and nothing to scrub. The exposure window is empty. No
provider-side revocation is required.

### F-2 — `.gitignore` covers `.env` but not the rest of the secret surface (Low, structural)

Checked 2026-11-24 against `.gitignore` (`e7eb53c`, 2026-11-22), which has one
pattern on line 15: `.env`.

```
$ git check-ignore -q <path>
IGNORED   .env
NOT-IGN   .env.local
NOT-IGN   .env.production
NOT-IGN   id_rsa
NOT-IGN   key.pem
NOT-IGN   secrets.json
NOT-IGN   credentials.yml
NOT-IGN   .npmrc
```

This is a future-leak vector, not a leak. The realistic path is a developer
copying `.env` to `.env.local`, or a tool dropping a `*.pem` into the tree, then
`git add -A` sweeping it in. No such file exists today; `git status` is clean and
the tree tracks only `frontend/.env.example`.

Recommended ignore additions (owner approval needed, not applied here):

```
.env.*
!.env.example
*.pem
*.key
*.p12
*.pfx
id_rsa
id_ed25519
.npmrc
.netrc
```

The existing `.env` rule should stay; `!.env.example` keeps the one safe example
file tracked.

### F-3 — No automated secret gate in CI or hooks (Low, structural)

`.github/workflows/ci.yml` runs lint, types, tests and build, with no secret
scan. `.githooks/pre-push` only blocks a direct push to `main`. Protection today
is therefore: GitHub secret scanning and push protection, enabled server-side by
`scripts/setup_github.sh` (`--enable-secret-scanning`,
`--enable-secret-scanning-push-protection`) on the now-public repository. That
is real coverage, but it is a repository setting that a fork, a plan change or a
renamed repo silently loses, and it cannot see a leak in a branch that was never
pushed.

What this audit added instead of a CI workflow (see next section): a pytest
regression test that fails the existing `backend` CI job when a provider key,
private-key header or credentialed URL is committed. This is the repo-local
equivalent, with no workflow file and no new dependency.

### F-4 — `backend/config/strategy_card.json` is a committed generated artifact (Info)

It holds timestamps, a digest, strategy parameters and holdout metrics — no
secret. Noted only because it is generated output living in `config/`; if a
future generator ever writes account identifiers there, it would become a leak
path. Keep credentials out of that file by construction.

## Recommended gate (implemented as a test, not a workflow)

`backend/tests/test_no_committed_secrets.py` scans **only git-tracked files**, so
a developer's untracked `.env` or local key never fails the suite. Patterns are
high-signal (AWS access key, GitHub token, OpenAI/Stripe/Slack/Google prefixes,
private-key header, `scheme://user:pass@host`) and a finding prints the file,
line and rule — never the value. Generic entropy stays in gitleaks, where a
low-precision rule cannot train people to bypass the check.

Validation:

```
$ cd backend && uv run pytest tests/test_no_committed_secrets.py -q
1 passed

$ uv run ruff check tests/test_no_committed_secrets.py
All checks passed!

# negative control: add a tracked file containing a fake AWS key
FAILED ... AssertionError: committed secret-like values (value redacted):
  backend/tests/_tmp_scan_probe.py:1 [aws-access-key]
```

The probe file was removed before commit and the tree is clean.

## Rotation, if a secret is ever found here

Order matters and code removal is not the fix:

1. Rotate and revoke at the provider (AWS IAM, GitHub, Stripe, OpenAI, broker).
   Issue the replacement with the shortest viable TTL and least privilege.
2. Replace the value in code with a reference to a secret store or an
   OIDC/workload-identity path; deploy.
3. Purge from history (below) and coordinate the rewrite with anyone holding a
   clone.
4. Audit provider usage between the commit timestamp and the revocation
   timestamp; widen the response if the credential was used.
5. Add the missed pattern to the gate and make the safe path easier.

A secret is compromised at commit time, not discovery time. Treat the commit
date as the start of the exposure window.

## History-scrubbing options (not performed)

No rewrite is needed today. If one becomes necessary:

- `git filter-repo --path <file> --invert-paths` for a whole leaked file, or
  `--replace-text` with a mapping to replace values in place. Requires
  `git-filter-repo` installed locally; it is not in this repo's toolchain.
- BFG Repo-Cleaner as the alternative for large histories.
- Either way: every clone and every fork keeps the old object until recloned;
  open PRs and tags need re-pointing; the value must already be revoked at the
  provider before the rewrite, because the rewrite does not un-leak it.
- Coordinate with the owner first. This branch did not touch history.

## Bottom line

Three independent methods (gitleaks working tree, gitleaks full history, manual
added-line plus entropy sweep) agree: the repository has never committed a
credential. The MT5 path is a disabled skeleton with no login, tokens are
generated at runtime and stored hashed, and the only public env variable points
at localhost. Close the two structural gaps — widen `.gitignore`, keep the new
pytest gate in CI — and the next leak fails locally instead of shipping.
