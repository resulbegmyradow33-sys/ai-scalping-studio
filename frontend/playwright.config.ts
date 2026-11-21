import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end smoke tests.
 *
 * The API is not managed here: run `make e2e-backend` first, which reseeds
 * `./e2e-data` with synthetic XAUUSD M1 and serves it on :8000. Playwright only
 * starts the Next dev server. The API allows CORS from :3000 only, so changing
 * `PLAYWRIGHT_PORT` also needs that origin on the backend.
 */
const PORT = process.env.PLAYWRIGHT_PORT ?? "3000";
const BASE_URL = process.env.PLAYWRIGHT_BASE_URL ?? `http://localhost:${PORT}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  // One worker: the simulator is process-wide state on the single API worker.
  workers: 1,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run dev -- -p ${PORT}`,
    url: BASE_URL,
    // Never adopt a stranger's process on the port: specs then pass against
    // the wrong app while the real dev server never started.
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
