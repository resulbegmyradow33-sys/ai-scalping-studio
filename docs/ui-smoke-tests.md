# UI smoke tests (Playwright)

`frontend/e2e/pages.spec.ts` is a browser smoke suite for the ten top-level
screens: `/`, `/data`, `/intelligence`, `/training`, `/performance`,
`/backtesting`, `/paper`, `/labs/ml`, `/autonomy` and `/strategies`.

It runs **without a backend**. Every `/api/*` call is answered by a fixture in
the spec and the simulator WebSocket is routed without a server, so results do
not depend on a seeded store, network access or another process.

For each page the suite asserts:

- the document title and the level-1 heading;
- the honest empty or unreadable branch the page should show with no data
  (for example `data-empty`, `regime-empty`, `paper-sessions-empty`, the
  backtesting 404 run list, or the autonomy "store yapılandırılmadı" /
  "okunamıyor" notices);
- no visible `NaN` or `undefined` text — unread numbers must render as `—`;
- zero console errors, zero uncaught page errors, and zero requests to a path
  the fixture table does not cover.

The last check is what keeps the suite honest: an unexpected endpoint fails the
test instead of silently reaching `127.0.0.1:8000`. A dedicated test fetches an
unlisted path and asserts the harness answers `500` and records it, so the
guard itself is covered. Fixtures may be keyed by `"METHOD /path"` to answer a
write differently from the read of the same path, and may carry a status
(`reply(body, 404)`) for the unreadable branches.

## Interaction coverage

Beyond the empty renders, the suite exercises the interactions that can lie
about data:

- **`/paper` equity**: a ledger page with three marks draws the SVG chart and a
  text summary, including the window delta (`+$12,50`) and its percentage. When
  the ledger returns no marks, the chart stays absent, `paper-equity-chart-empty`
  says why, and balance/equity render `—` — never a fabricated `$0`.
- **`/paper` start form**: an invalid symbol plus a blank strategy blocks the
  request locally, marks both fields `aria-invalid`, focuses the first one and
  lists both errors. A valid submit posts exactly `symbol`, `timeframe`,
  `strategy` and `model_version: null`, then shows the server's session id and
  the settled detail. The form never starts a runner or selects a champion.
- **`/backtesting`**: a missing run list is a `404` notice with submission
  disabled; an invalid symbol is rejected before any request; a completed run
  posts exactly `{run_id, symbol}`. A response with metrics but no `equity`
  array renders the metrics and an explicit "equity eğrisi döndürülmüyor"
  message instead of a reconstructed curve.
- **`/strategies`**: the card is classified `legacy` or `fallback`; a readable
  archive with no champion is "kayıtlı şampiyon yok", while an unreadable
  champion table and an absent archive are distinct notices. Regime shares
  render without `NaN`.
- **`/autonomy`**: champion modes are labelled `evidence`, `exploratory`,
  `manual` and `kanıtsız`; anything that is not `evidence` is marked
  "paper dışı". A failed status read is an alert, not stale zeroes.

## Commands

```bash
cd frontend
npm ci                 # once
npm run e2e:install    # once: downloads the pinned Chromium build
npm run e2e            # smoke project, headless, no backend
```

`npm run e2e` uses the Playwright config's `smoke` project only. It starts the
Next dev server on port 3000 (`PLAYWRIGHT_PORT` overrides) and stops it when the
run ends. Failure artifacts (trace, error context) land in
`frontend/test-results/`.

### Live API suite (opt-in)

`frontend/e2e/simulator.spec.ts` exercises the real simulator stream and must
not run in the smoke lane. Seed and serve the API in another terminal, then:

```bash
make e2e-backend       # reseeds ./e2e-data and serves :8000
cd frontend
npm run e2e:live       # E2E_LIVE_API=1 playwright test
```

`npm run e2e` does not include this file, so a checkout with no backend still
has a green, deterministic lane.

### When Chromium is absent

Playwright fails before the first test with
`browserType.launch: Executable doesn't exist`. Install the browser once with
`npm run e2e:install`; the tests never download a browser themselves. If the
machine must stay offline and the browser cannot be installed, skip the suite
and record that UI smoke was not run — do not report it as passing.

## CI

`.github/workflows/ci.yml` does not run Playwright (types, lint, Vitest and
build only). This suite is intentionally not added there: it needs a browser
download and a Next dev server, and neither belongs in the current fast
frontend job. Run it locally before a UI change and paste the command/output in
the PR.

## What this does not prove

The suite checks rendering and client-side failure handling against fixed
payloads. It does not validate API schemas, real simulator sequencing, styling,
chart canvases or accessibility; those have their own tests
(`src/**/*.test.tsx`, `e2e/simulator.spec.ts`, `docs/ui-accessibility-audit-2026-11-24.md`).
