import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests. Two projects, one server:
 *
 * - `smoke` (default, `npm run e2e`) needs no backend. `pages.spec.ts` mocks
 *   `/api/*` and the simulator socket in the browser.
 * - `live-api` (`E2E_LIVE_API=1 npm run e2e`, or `npm run e2e:live`) runs
 *   `simulator.spec.ts` against a real API. Run `make e2e-backend` first, which
 *   reseeds `./e2e-data` with synthetic XAUUSD M1 and serves it on :8000.
 *
 * Playwright only starts the Next dev server. The API allows CORS from :3000
 * only, so changing `PLAYWRIGHT_PORT` also needs that origin on the backend.
 */
const PORT = process.env.PLAYWRIGHT_PORT ?? "3000";
const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? `http://localhost:${PORT}`;
const LIVE_API = process.env.E2E_LIVE_API === "1";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  // One worker: the live simulator is process-wide state on the single API
  // worker, and a single Next dev compiler avoids per-page cold-build races.
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "smoke",
      testMatch: /pages\.spec\.ts/,
      use: { ...devices["Desktop Chrome"] },
    },
    // Opt-in: the live suite needs `make e2e-backend` in another terminal.
    ...(LIVE_API
      ? [
          {
            name: "live-api",
            testMatch: /simulator\.spec\.ts/,
            use: { ...devices["Desktop Chrome"] },
          },
        ]
      : []),
  ],
  webServer: {
    command: `npm run dev -- -p ${PORT}`,
    url: BASE_URL,
    // Never adopt a stranger's process on the port: specs then pass against
    // the wrong app while the real dev server never started.
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
