import { expect, test } from "@playwright/test";

/** `make e2e-backend` serves the API here; the browser talks to it directly. */
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? "http://127.0.0.1:8000";

test.beforeEach(async ({ request }) => {
  // The API owns one simulator per process. Resetting before each test makes a
  // second `npx playwright test` run behave like the first one.
  const response = await request.post(`${BACKEND_URL}/api/sim/control`, {
    data: { action: "reset" },
  });
  expect(response.ok()).toBeTruthy();
});

test("simulator streams bars and shows the data warning", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText(/veri kaynağı: sentetik/i)).toBeVisible();
  await page.getByRole("button", { name: "Başlat" }).click();
  await expect(page.locator("[data-testid=bar-count]")).not.toHaveText("0", {
    timeout: 15_000,
  });
  await expect(page.locator("[data-testid=last-price]")).toContainText(/2\.\d/);
});

test("data page shows the empty state for an unused symbol", async ({ page }) => {
  await page.goto("/data");
  await expect(page.getByRole("heading", { name: "Veri", exact: true })).toBeVisible();

  // The seeded backend makes XAUUSD non-empty, so query a symbol it cannot
  // hold: this exercises the real empty branch instead of a zero-looking
  // provider card. "0 bar" also proves inactive providers stay at zero.
  await page.getByLabel("Sembol").fill("NOPE");
  await expect(page.getByTestId("data-empty")).toBeVisible();
  await expect(page.getByTestId("provider-rows-csv")).toHaveText("0 bar");
});
