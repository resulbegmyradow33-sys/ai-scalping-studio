# API security review — 2026-11-24

> **Scope notice.** Review of the HTTP surface at branch `feat/apisec`, base
> `main` `6113197` (`feat(api): read-only autonomy status endpoints (#134)`).
> This is a code and request-level review, not a penetration test of a deployed
> host. The API is a local, single-user research panel: every claim below assumes
> the process is bound to loopback, as the CLI does by default
> (`backend/src/scalping/cli.py:1487`, `--host` default `127.0.0.1`). What
> changes if that assumption breaks is section 6.

Audited surface:

- `backend/src/scalping/api/routes.py` — health, registry, data status/validate/
  download, regime, train/backtest/evaluate, runs, strategy, simulation, WS.
- `backend/src/scalping/api/routes_paper.py` — `/api/paper/*` (new).
- `backend/src/scalping/api/routes_ml.py`, `ml_runner.py` — `/api/ml/*` (new).
- `backend/src/scalping/api/routes_autonomy.py` — `/api/autonomy/*` (new).
- `backend/src/scalping/api/schemas.py`, `train_runner.py`, `app.py`.
- Reached data layer: `paper/ledger.py`, `rl/registry.py`,
  `rl/policy.py`, `data/store.py`, `data/csv_provider.py`,
  `knowledge/ingest.py` (checked for HTTP reachability only).

## Verdict

**No path traversal, no SQL injection and no read-only violation were found.**
The autonomy GET surface is genuinely read-only, including against a seeded
database; the identifiers that reach a filesystem are either opaque tokens,
allowlist lookups or constrained patterns.

Three low-severity information/robustness defects were **proven with requests
and fixed**, each with a regression test:

1. an unbounded training run id reached `os.stat`, producing a 500
   (`ENAMETOOLONG`) and putting the absolute run path into the traceback;
2. loader failures returned absolute artefact paths inside 409 bodies
   (`/api/evaluate`, `/api/ml/evaluate`);
3. the CSV-import 422 echoed `<data_root>/imports` verbatim.

Six observations are documented and **accepted** (no code change), with the
reasoning and, where relevant, a recommendation. Nothing here rates High or
Critical: the exploitable preconditions are a local process, a misconfigured
store or an operator exposing a loopback service. Inflating those to Critical
would train readers to ignore this document.

## Findings

| ID | Severity | Finding | Status |
|----|----------|---------|--------|
| F-01 | Low | Unbounded train `run_id` → `OSError`/500 and absolute path in the traceback | **Fixed** |
| F-02 | Low | 409 loader errors embedded absolute artefact paths (train and ML) | **Fixed** |
| F-03 | Low | CSV-import 422 body embedded the absolute imports directory | **Fixed** |
| F-04 | Low | `report_path` in backtest/evaluate responses is an absolute path | Accepted |
| F-05 | Low | `GET /api/paper/*` answers 500 on a corrupt ledger while autonomy degrades to `readable:false` | Accepted |
| F-06 | Info | No authentication, authorization or CSRF token; local-only by design | Accepted, documented |
| F-07 | Info | Resource-abuse surface: no request-body cap, unbounded training knobs, full-slice scans per GET | Accepted, documented |
| F-08 | Info | Knowledge ingestion is not HTTP-reachable; DOCX/XML bomb controls verified, plain-text readers unbounded | Accepted, documented |
| F-09 | Info | ML run `error` field relays exception text; known path-bearing sites now scrubbed | Accepted |

---

## F-01 — Unbounded train run id reaches the filesystem (Low, fixed)

**Where.** `api/schemas.py` (old `BacktestRequest.run_id`, `EvaluateRequest.run_id`
were `str = Field(min_length=1)`), `api/routes.py` (`/api/runs/{run_id}`,
`/api/runs/{run_id}/control` took a bare `str`), `api/train_runner.py:612`
(`_run_dir` rejected separators but not length).

**PoC (before the fix).**

```
POST /api/backtest  {"run_id": "<5000 a's>", "symbol": "XAUUSD"}   → 500
GET  /api/runs/<5000 a's>                                          → 500

Traceback (server log):
  File "scalping/rl/registry.py", line 197, in load_run
    if not path.is_file():
OSError: [Errno 63] File name too long: '/…/runs/aaaa…/manifest.json'
```

`Path.is_file()` swallows `ValueError` and the "missing" `OSError` codes, but
re-raises the rest; `ENAMETOOLONG` is one of them, so `load_run`'s existence
probe becomes a 500. The response is generic, but the exception text —
absolute path included — reaches the server log and any debug handler added
later. It is also an unauthenticated way to make the server raise.

**Fix.** A single constrained type, `TrainRunId` (`api/schemas.py:180-188`):
one path component, `^[A-Za-z0-9][A-Za-z0-9._-]*$`, max 128 characters. Applied
to both request bodies (`schemas.py:391,401`) and both path parameters
(`routes.py:383,392`). `TrainRunner._run_dir` also refuses ids longer than 128
as a second line of defence for non-HTTP callers (`train_runner.py:612`).

**Evidence.** `tests/test_api_security_surface.py::test_run_id_length_is_rejected_before_the_filesystem`
(422, not 500, no path in the body),
`::test_train_run_ids_cannot_escape_the_runs_root`,
`::test_the_runner_guard_is_a_second_line_of_defence`.

---

## F-02 — 409 loader errors embedded absolute artefact paths (Low, fixed)

**Where.** `api/train_runner.py:701,796,872`, `api/ml_runner.py:815,861`,
`api/routes.py:310`. Each built `detail` from `str(exc)`; `load_policy`
(`rl/policy.py:131-143`) and `load_result` raise messages that contain the run
directory, and `OSError` strings contain the failed path.

**PoC (before the fix).**

```
# runs/abc123/manifest.json exists, no model.zip
POST /api/evaluate {"run_id": "abc123", "symbol": "XAUUSD"}
→ 409 {"detail": "run abc123 cannot be loaded: no model.zip in /var/folders/…/runs/abc123"}

# runs/ml/abc123/meta.json + a corrupt model.joblib, no metrics.json
POST /api/ml/evaluate {"run_id": "abc123"}
→ 409 {"detail": "run abc123 cannot be loaded: [Errno 2] No such file or directory: '/var/folders/…/runs/ml/abc123/metrics.json'"}
```

A partial or failed run is enough: no attacker-supplied file is required, so a
network peer (section 6) learns the OS user name and the run layout from an
ordinary error.

**Fix.** `api/errors.py::public_error_text` replaces the configured run root
with `<root>` (exact, handles spaces) and any remaining absolute path with
`<path>` (regex, two or more segments so a `6/8` ratio is untouched). Applied at
the six loader/translator sites and to the download `OSError` handler
(`routes.py:310`). The full exception is still logged where it occurs.
Two `{exc}` interpolations remain (`train_runner.py:363,734`); both are
bandit-fit and backtest-replay errors that do not open a file.

**Evidence.** `::test_loader_errors_do_not_embed_absolute_paths`,
`::test_public_error_text_scrubs_paths_and_keeps_the_reason`.

---

## F-03 — CSV-import 422 embedded the absolute imports directory (Low, fixed)

**Where.** `api/routes.py:251` (`_import_file`), detail string interpolated
`imports_dir`.

**PoC (before the fix).**

```
POST /api/data/download {"provider":"csv","symbol":"XAUUSD","path":"missing.csv",…}
→ 422 {"detail":"no csv file named 'missing.csv' in /var/folders/…/data/imports; …"}
```

The allowlist itself is correct (the name is compared against `iterdir()`
entries; symlinks and non-files are rejected). Only the message leaked.

**Fix.** The body now says `<data_root>/imports` (`routes.py:267-271`). The
request's own value is still echoed (bounded to 255 by `ImportFileName`), which
is what a user needs to fix the request.

**Evidence.** `::test_csv_import_error_names_no_absolute_path`; the pre-existing
`tests/test_api_security.py::test_csv_import_rejects_a_symlinked_file_that_escapes`
still passes.

---

## F-04 — `report_path` is an absolute path (Low, accepted)

**Where.** `api/train_runner.py:762` and `:924` return
`{"report_path": str(run_dir / …)}`; the schema declares `report_path: str`
(`api/schemas.py`, `BacktestResponse`/`EvaluateResponse`).

A successful `POST /api/backtest` or `/api/evaluate` therefore answers with
`/Users/<name>/…/runs/<id>/backtest-XAUUSD.json`. Confirmed by inspection; the
runtime PoC needs a finished run, and the repository's own tests consume the
value (`backend/tests/test_api_train.py:260,324,516`), so it is a deliberate
contract.

**Why accepted.** The response also contains the metrics themselves; the path is
the only reliable way a local script finds the artefact, and the endpoint is a
local research tool. It is not fixed here because changing the field is an API
contract change, which is outside a security-review patch. If the API is ever
exposed beyond loopback, return `f"{run_id}/{filename}"` instead.

---

## F-05 — Corrupt ledger → 500 on paper GETs (Low, accepted)

**Where.** `api/routes_paper.py:72-88`: `_reader` opens the ledger and calls
`has_schema()` outside a `sqlite3.Error` guard, so a non-database file raises
`DatabaseError("file is not a database")` and FastAPI answers 500. The autonomy
reader wraps the same call (`routes_autonomy.py:156-180`) and reports
`configured=true, readable=false`.

**PoC.**

```
printf 'not a database' > data/paper.sqlite
GET /api/paper/sessions        → 500 "Internal Server Error"
GET /api/autonomy/status       → paper: {"configured": true, "readable": false}
```

**Why accepted.** It needs a local operator to put a corrupt file where the
ledger belongs; there is no request that can create that state, and the 500 body
carries no stack trace or path (pinned by
`::test_unhandled_errors_do_not_publish_a_traceback`). The honest fix is a
follow-up: mirror autonomy's `configured/readable` answer on the paper routes.
Recorded so the inconsistency is not rediscovered as a "new" bug.

---

## F-06 — No authentication or authorization; local-only by design (Info, documented)

There is no API key, session, CSRF token or `dependencies=[...]` security layer
on any route; `app.openapi()` declares no `securitySchemes`
(`::test_no_authentication_layer_is_declared`). CORS is an explicit allowlist,
not `*` (`api/app.py:61-67,103-105`):

- allowed: `http://localhost:3000|3001`, `http://127.0.0.1:3000|3001`;
- `allow_methods=["*"]`, `allow_headers=["*"]`, no `allow_credentials`.

**What the browser model does protect.** A cross-origin JSON `POST` needs a
preflight; a foreign origin gets `400` and no `Access-Control-Allow-Origin`
(`::test_cors_allows_only_the_local_panel_origins`). Cross-origin `GET`s cannot
be read by script. So while the allowlist holds, drive-by CSRF from an arbitrary
page is not practical.

**What it does not protect.** CORS is not authentication. If the server is bound
to a non-loopback interface (`--host 0.0.0.0`, a tunnel, a reverse proxy), any
peer that can reach the port can, without credentials:

- read the paper ledger (sessions, positions, orders, trades, equity), queue
  counts, audit events, champions and run metrics;
- open/close paper sessions (write rows);
- start training/ML jobs and simulations (CPU/RAM), run calibration (rewrites
  `runs/strategy_card.json`), trigger data downloads and file writes;
- read `/api/strategy/card` and `/api/ml/runs/<id>`.

There is no destructive file primitive (no delete/upload endpoint), and
generated strategies are data-only, so this is exposure and resource abuse, not
RCE. Browser CSRF becomes possible too if a malicious page is served from one of
the allowed origins — note `3000`/`3001` are *trusted by origin*, so a local
process that squats either port inherits that trust.

**Recommendation.** Keep `--host 127.0.0.1`; if remote access is ever needed,
add a shared-secret header check plus a `TrustedHostMiddleware`-style host check
and tighten CORS to the exact panel origin, in that order. Do not bolt auth onto
this patch — there is no existing pattern to follow, and the owner must decide
the deployment model first.

---

## F-07 — Resource-abuse surface (Info, documented)

No request is authenticated and Starlette/uvicorn impose no default body-size
limit. Reachable amplification, all bounded by "one local user" today:

| Sink | Bound in code | Residual abuse |
|------|---------------|----------------|
| `POST /api/paper/sessions` `config` | none (`dict[str, Any]`) | a multi-MB JSON body is parsed and stored in the ledger row |
| `POST /api/train` `total_steps`, `n_envs` | `ge=1` only | arbitrarily long CPU thread; cancellation exists but is best-effort |
| `POST /api/ml/train` `horizon`, `purge_bars`, `embargo_bars` | ≤1000, ≤100k, ≤100k | large feature matrices; `MAX_BARS` row guard exists |
| `GET /api/regime`, `/api/data/validate` | `limit` ≤5000 (timeline only) | every request re-reads and re-classifies the full slice; no cache |
| `GET /api/data/status` | none | full parquet read per poll |
| `POST /api/data/download` synthetic range | none (`start`, `end` are any datetimes) | one synchronous request can materialise a century of M1 bars (~1,440 rows/day) in RAM before the 200; `end < start` is a cheap empty frame |
| CSV import | chunked at 100k rows (`csv_provider.py:64,170,194`) | all chunk frames are concatenated in memory; the file must be placed locally in `<data_root>/imports` |
| `GET /api/paper/*`, `/api/autonomy/*` | `limit` ≤1000 / ≤500, integer-typed | an unbounded `offset` is a cheap keyset walk, not a scan of the response |

The paper and autonomy page sizes are correct and regression-tested
(`::test_pagination_parameters_are_bounded_integers`). If this ever leaves
loopback, add a body-size middleware and per-route budgets before anything else.

---

## F-08 — Knowledge ingestion and bombs (Info, not HTTP)

`grep` shows no route imports `scalping.knowledge`. Ingestion is reached from
the CLI watcher, which enqueues `knowledge.ingest` jobs (`cli.py:1528`), and the
orchestrator runs them; `/api/autonomy` is read-only and cannot enqueue. So
ingestion is outside this HTTP surface. Checked anyway because the brief names
it:

- DOCX (`knowledge/ingest.py:384-410`) reads only `word/document.xml`, refuses
  the archive when the declared uncompressed size exceeds
  `MAX_DOCX_XML_BYTES = 32 MiB` (`:52,398`), and Python's `ZipExtFile` stops at
  that declared size and verifies the CRC, so a lying header truncates rather
  than expands.
- XML entity expansion: on the project interpreter (CPython 3.12.13) a
  billion-laughs document raises
  `ParseError: limit on input amplification factor (from DTD and entities) breached`
  before materialising text — verified directly with `xml.etree.ElementTree`.
- Plain text/Markdown/HTML extractors (`:211-247,364`) call `read_text()` with
  no size cap. A local file is required, so this is a "don't ingest a 10 GB
  file" hazard, not a remote bomb. If ingestion ever gains an upload route, add
  a size check before `read_text` and keep the DOCX cap.

---

## F-09 — ML `error` field relays exception text (Info, accepted)

`api/ml_runner.py:735` returns `meta["error"]`/`report["error"]`, which is
`str(exc)` from a failed fit (`_finish`). Training failures we exercised carry
no path, and the two loader paths that did are now scrubbed (F-02). A future
failure mode could still include one. Accepted: the field is diagnostic and
local-only; if it is ever rendered to an untrusted viewer, run it through
`public_error_text`.

---

## Area audit

### 1. Path traversal — no finding

Every request value that could become a path segment:

| Request value | Reaches | Control |
|---------------|---------|---------|
| `symbol` (query and body) | `<data_root>/<symbol>/<tf>/…` | `SymbolId` / `is_symbol_id`: single segment, `^[A-Za-z0-9][A-Za-z0-9._-]*$`, ≤32 (`schemas.py:106-119`); query routes call `_checked_ids` (`routes.py:131`) |
| `timeframe` | same | `Literal[...]` on bodies; `TIMEFRAMES` membership on query (`routes.py:133`) |
| CSV `path` | `open()` | name-only; looked up in `iterdir()`, symlinks and non-files rejected, never joined (`routes.py:251-272`) |
| paper `session_id` | SQL only | `PaperSessionId`: `^[A-Za-z0-9][A-Za-z0-9_-]*$`, ≤64 (`schemas.py:140-147`); never a path |
| ML `run_id` | `runs/ml/<id>` | `MLRunId` + `is_ml_run_id` at the runner (`schemas.py:157-166`, `ml_runner.py:738`) |
| train `run_id` | `runs/<id>` | `TrainRunId` + `_run_dir` (`F-01`) |
| ledger / orchestrator / archive paths | SQLite files | `app.state` configuration from the factory; no request field can move them (`app.py:86-123`) |

Tests: `::test_train_run_ids_cannot_escape_the_runs_root` (planted decoy
`manifest.json` outside the root is never read),
`::test_ml_and_paper_ids_cannot_escape_their_stores`, plus the existing
`tests/test_api_security.py` CSV and symbol cases. Symlinks inside `runs/` or
`<data_root>/imports` are rejected or ignored; creating one requires local
write access, i.e. a different threat model (LPE, not API).

### 2. Read-only guarantee — holds

Mechanisms: autonomy opens `mode=ro` + `PRAGMA query_only=ON` and never
constructs `DiscoveryArchive`/`PaperLedger`/`JobStore` writers
(`routes_autonomy.py:101-128`); paper GETs use `PaperLedger.open_reader`
(`mode=ro`, no `init_schema`, `paper/ledger.py:559-580`); the data section only
stats and globs (`routes_autonomy.py:379-397`).

Evidence: `::test_paper_reads_leave_a_seeded_ledger_byte_identical` (all six
paper GETs plus `/api/autonomy/status`, sha256 unchanged, table list unchanged),
`::test_autonomy_gets_leave_seeded_stores_byte_identical`,
`::test_get_surface_creates_no_store_when_the_ledger_is_absent`, and the existing
`tests/test_api_autonomy.py` byte-identical/WAL test. Caveat kept in the module
docstring: SQLite may create `-wal`/`-shm` sidecars for a WAL database in a
writable directory; the database bytes are unchanged, which is what the tests
assert.

### 3. Information disclosure — fixed the reachable cases

Fixed: F-01, F-02, F-03. Accepted: F-04, F-05. Guard test:
`::test_error_bodies_do_not_expose_the_machine_layout` walks 404/409/422 bodies
(run, ML, paper, data) and asserts no configured root, no `Traceback`, no
`File "`. Rule for future code: never interpolate `str(exc)` or a `Path` into a
`detail` — use `public_error_text` or a fixed message.

### 4. Injection — no finding

SQL reachable from HTTP is inventoried and parameterised:

- `paper/ledger.py:183-202` — static `_PAGE_SQL`/`_COUNT_SQL`, every value bound
  with `?` (`_page`, `:1238-1256`); `session()` and `count_trades()` use `?`.
- `routes_autonomy.py` — `_flag` uses `?` (`:194`); jobs/events `LIMIT ?`;
  `_columns` interpolates a table name that is an internal literal from a
  `frozenset`, never a request value.
- No `LIKE`, FTS `MATCH` or any other pattern-matching SQL exists in `api/` or
  `paper/ledger.py`, so the "escape LIKE/FTS" case has no instance to test.

Tests: `::test_sql_metacharacters_in_ids_are_rejected_and_stores_are_untouched`
(`' OR 1=1--`, `x'; DROP TABLE sessions;--`, `UNION SELECT` against paper, train
and ML ids → 4xx, ledger hash and table list unchanged),
`::test_pagination_parameters_are_bounded_integers`.

### 5. Resource abuse — see F-07

Every list page is bounded and integer-typed; the remaining amplification needs
either a local caller or a public bind.

### 6. Auth / CSRF — see F-06

No auth layer exists to test; the local-only assumption, the CORS allowlist and
the consequences of `0.0.0.0` are documented and pinned by
`::test_cors_allows_only_the_local_panel_origins` and
`::test_no_authentication_layer_is_declared`.

---

## Verification

Commands run from `backend/` with the worktree's own `.venv`, matching CI:

```
uv sync --frozen
.venv/bin/ruff check src tests scripts      # All checks passed!
.venv/bin/mypy src                          # Success: no issues found in 135 source files
.venv/bin/python -m pytest tests/test_api_security_surface.py -q   # 24 passed
.venv/bin/python -m pytest tests/test_api_security.py tests/test_api_data.py \
    tests/test_api_paper.py tests/test_api_train.py tests/test_api_ml.py \
    tests/test_api_security_surface.py -q   # 139 passed
.venv/bin/python -m pytest -q   # 1842 passed, 4 failed (the pre-existing four below)
```

The new file `backend/tests/test_api_security_surface.py` is dependency-free in
the sense that it adds no package: pytest, the FastAPI test client, and the
project's own `PaperLedger`/`JobStore` helpers.

### Pre-existing failure (not caused by this review)

`tests/test_api_autonomy.py` has four failures on the base commit itself: its
seeding helper calls `archive.promote(...)`, but `DiscoveryArchive` exposes
`promote_exploratory`, `_promote_evidence` and `_promote_manual`
(`backend/src/scalping/discovery/archive.py:1453,1654,2014`). Verified against
`git show 6113197:…` for both files: the mismatch exists in base, not in this
branch. Fixing it means changing the archive fixture (or the public API), which
is outside a security-review patch — flagged for the owner.

## Residual risk

1. **Deployment is the whole security model.** Loopback binding is the only
   thing standing between the network and F-06. Any `--host` change, tunnel or
   proxy should be treated as a security change.
2. **Origin trust is port-based.** A local process on `:3000`/`:3001` is
   trusted by CORS.
3. **Two accepted leaks remain** (F-04 absolute report path, F-05 corrupt-ledger
   500) and one diagnostic field (F-09); none is remotely reachable while the
   bind stays loopback.
4. **No automated secret scanning or SAST gate was added.** The vulnerabilities
   found here were cheap to fix once proven; the durable fix is a CI check
   (CodeQL `py/path-injection`/`py/sql-injection` already matches this code's
   style) plus the `public_error_text` rule in review.
