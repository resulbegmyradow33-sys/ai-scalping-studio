# API contracts and the schema drift guard

The HTTP surface is the contract between `backend/src/scalping/api/` and the
Next.js panel in `frontend/`. `backend/tests/test_openapi_snapshot.py` fails when
that contract changes without a reviewed snapshot update, so a renamed field or
a dropped enum value cannot reach `main` silently.

## What the snapshot contains

`backend/tests/snapshots/openapi.json` is derived from the app's own
`/openapi.json`, normalized to the parts a client depends on:

| Guarded | Examples |
|---|---|
| Paths and methods | `GET /api/health`, `POST /api/paper/sessions` |
| Request models | `PaperSessionCreate` |
| Response models per status | `201 -> PaperSessionCreated`, `422 -> HTTPValidationError` |
| Query parameters | `symbol: string!` (the `!` marks required), `limit: integer` |
| Required fields | `PaperSessionSummary.required: session_id, status, ...` |
| Field types | `balance: number \| null`, `items: array<PaperOrder>` |
| Enums | `timeframe: M1, M5, M15, M30, H1, H4, D1` |

Dropped on purpose, because they change between FastAPI/Pydantic patch releases
without changing the contract: titles, descriptions, operation ids, tags,
examples, server lists, `summary` text and JSON-schema `format` hints.

Endpoints that return an untyped `dict` (for example `/api/data/validate` and
`/api/strategy/card`) normalize to `object`. The guard proves the path exists and
keeps its method, query parameters and status codes, but it cannot see inside an
untyped body; type those handlers with a response model when feasible.

## Normal update (intentional API change)

1. Change the route, schema or model.
2. Rewrite the snapshot:

   ```bash
   cd backend
   UPDATE_OPENAPI=1 uv run pytest tests/test_openapi_snapshot.py
   ```

   The non-pytest equivalent is:

   ```bash
   cd backend
   uv run python scripts/update_openapi_snapshot.py
   ```

3. Review the change like code, because it is the review record:

   ```bash
   git diff backend/tests/snapshots/openapi.json
   ```

   The diff lists only what moved: added/removed operations, changed request or
   response models, added/removed/changed query parameters, added/removed
   required fields, and added/removed enum values.

4. Update the frontend call sites and the API table of the owning document (for
   example `docs/paper-trading.md`) in the same PR.

`scripts/update_openapi_snapshot.py --check` prints the drift and exits `1`
without writing, which is handy before a commit.

## Reading a failure

```
operations: +0 -1 ~1
  - GET /api/symbols
  ~ POST /api/train
      ~ response 202: TrainResponse -> TrainAccepted
models: +1 -1 ~2
  + TrainAccepted
  - TrainResponse
  ~ ControlSetup
      - enum action: reset
  ~ RunSummary
      - field reward_mean: number | null
      - required reward_mean
```

Uncommitted drift means one of two things: the code changed and the snapshot
needs the reviewed update above, or an unintended change slipped in and should
be reverted. The snapshot is never regenerated automatically by the suite.

## Properties of the guard

- Deterministic: keys are sorted, lists are sorted, and the snapshot has no
  timestamps, paths, hostnames or environment values, so two app factories
  produce byte-identical JSON (asserted by a test).
- No network or data access: the test builds the app with a temporary data root
  and reads the in-process schema; `create_app()` opens no store and starts no
  thread.
- Fail-closed: a reference to a missing model raises instead of writing an empty
  signature, so a broken schema cannot be normalized into a passing snapshot.

The guard covers shape, not behaviour. Status-specific payload correctness,
validation messages and runtime errors remain the job of
`tests/test_api*.py` and the paper-trading tests.
