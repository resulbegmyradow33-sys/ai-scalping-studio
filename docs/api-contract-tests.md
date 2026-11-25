# API contract tests

`backend/tests/test_api_contracts.py` pins the **wire shape** of the merged HTTP
surface: `/api/paper`, `/api/ml`, `/api/autonomy`, `/api/orchestrator`,
`/api/data/status`, `/api/regime` and `/api/strategy/card`.

These tests are shape guards, not behaviour proofs. A green run means "the
response envelope did not drift"; it says nothing about whether a strategy,
metric or paper session is correct. Behaviour, persistence, concurrency and
security are covered by the per-route suites:

| Area | Behaviour suite |
|---|---|
| Paper ledger reads and writes | `test_api_paper.py` |
| ML run lifecycle and evaluation | `test_api_ml.py` |
| Autonomy stores and read-only GETs | `test_api_autonomy.py` |
| Orchestrator owner controls | `test_api_orchestrator_control.py` |
| Data download, provenance, ids | `test_api_data.py`, `test_api_security.py` |
| Regime labels | `test_api_regime.py` |
| Strategy card | `test_api_strategy.py` |

## What is asserted

- Exact response key sets for every documented envelope and item: `items`,
  `total`, `limit`, `offset`, the ML run detail, autonomy status sections,
  champions, jobs, events, card and regime objects.
- JSON types for the fields a client reads (ints are not bools, floats are not
  strings, timestamps are strings, nullable fields are `null`).
- Status codes: `201`/`200` success, `404` unknown ids, `409` conflicts,
  `422` malformed bodies, ids and pagination bounds.
- Pagination inclusiveness: paper `limit` 1..1000 and autonomy `limit` 1..500
  are accepted, while `0`, a value above the ceiling and a negative value are
  `422`.
- The `configured`/`readable` split in `/api/autonomy`: a missing store is
  `configured: false`; an existing but unreadable file is
  `configured: true, readable: false` and carries **no** items, counts or
  flags. Unreadable is never reported as empty.
- No success body carries a filesystem path, a stack/traceback, a `detail`
  error envelope or the orchestrator's free-form event `detail`. The helper
  `_assert_no_leak` also scans the serialised JSON for the test's temporary
  root.

## What a failure means

A failure is a contract change. Either the response model changed on purpose —
update the expected key set and the docs/UI consumer in the same change — or an
accidental field appeared. Check the direction before editing the test: a
missing key usually breaks the panel, an extra key is usually a leak.

Do not weaken these tests to make a behaviour fix pass. Fix the response model
or the handler; keep this file about shape.

## Running

```bash
cd backend && uv run pytest tests/test_api_contracts.py -q
```

Fixtures write only into pytest's `tmp_path`; the ML and autonomy cases use
small synthetic stores (900 M1 bars for ML) and never touch the real data root.
