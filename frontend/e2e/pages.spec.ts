import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

/**
 * Browser smoke for every top-level screen, with no backend process.
 *
 * `/api/*` is answered from the deterministic fixtures below and the simulator
 * socket is routed without a server, so each page can be checked for: the
 * document title, the level-1 heading, the honest empty/unreadable branch, and
 * clean console/page error logs. Numbers that the UI has not read must render
 * as an em dash; `NaN` or `undefined` in visible text is a failure.
 */

const HEALTH = { status: "ok", version: "e2e", device: "cpu" };

const EMPTY_DATA_STATUS = {
  symbol: "XAUUSD",
  rows: 0,
  first_ts: null,
  last_ts: null,
  providers: [],
  provenance: null,
};

/** An un-promoted card: the panel keeps its legacy M1 replay branch. */
const INACTIVE_CARD = { active: false, strategy: "flow_momentum" };

const TIMEFRAMES = [{ timeframe: "M1", rule: "1m" }];

const SYMBOLS = [{ symbol: "XAUUSD", description: "Gold", asset_class: "metal" }];

const ML_MODELS = ["logistic"];

const EMPTY_REGIME = {
  symbol: "XAUUSD",
  timeframe: "M1",
  rows: 0,
  current: null,
  counts: [],
  series: [],
};

/** A bounded ledger page with no rows. */
function emptyPage(limit = 20) {
  return { items: [], total: 0, limit, offset: 0 };
}

const EMPTY_AUTONOMY = {
  paper: { configured: true, readable: true, sessions: 0, open_positions: 0, closed_trades: 0 },
  orchestrator: {
    configured: true,
    readable: true,
    queued: 0,
    running: 0,
    failed: 0,
    paused: false,
    safe_mode: true,
    research_disabled: false,
    last_event: null,
  },
  champions: [],
  champions_readable: true,
  discoveries: { configured: true, readable: true, runs: 0, decisions: 0 },
  data: { configured: true, readable: true, symbols: [] },
};

const EMPTY_AUTONOMY_JOBS = { configured: true, readable: true, items: [], limit: 25 };
const EMPTY_AUTONOMY_EVENTS = { configured: true, readable: true, items: [], limit: 25 };

/** Every store is absent: the page must say so instead of printing zeros. */
const UNREADABLE_AUTONOMY = {
  paper: { configured: false, readable: false },
  orchestrator: { configured: false, readable: false },
  champions: [],
  champions_readable: false,
  discoveries: { configured: false, readable: false },
  data: { configured: false, readable: false },
};

const UNREADABLE_AUTONOMY_PAGE = { configured: false, readable: false, limit: 25 };

type Fixture = unknown;

/** Pathname → body; query strings are ignored because no fixture depends on them. */
const FIXTURES: Record<string, Fixture> = {
  "/api/health": HEALTH,
  "/api/data/status": EMPTY_DATA_STATUS,
  "/api/strategy/card": INACTIVE_CARD,
  "/api/timeframes": TIMEFRAMES,
  "/api/symbols": SYMBOLS,
  "/api/regime": EMPTY_REGIME,
  "/api/runs": [],
  "/api/ml/models": ML_MODELS,
  "/api/ml/runs": [],
  "/api/paper/sessions": emptyPage(),
  "/api/autonomy/status": EMPTY_AUTONOMY,
  "/api/autonomy/jobs": EMPTY_AUTONOMY_JOBS,
  "/api/autonomy/events": EMPTY_AUTONOMY_EVENTS,
};

interface Harness {
  /** `METHOD path` for any request the fixture table did not cover. */
  unexpected: string[];
  consoleErrors: string[];
  pageErrors: string[];
}

/**
 * Install the API/socket doubles and start collecting browser-side failures.
 * Must run before `page.goto` so no request escapes to `API_ORIGIN`.
 */
async function harness(page: Page, overrides: Record<string, Fixture> = {}): Promise<Harness> {
  const result: Harness = { unexpected: [], consoleErrors: [], pageErrors: [] };

  page.on("console", (message) => {
    if (message.type() === "error") result.consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => result.pageErrors.push(error.message));

  // The simulator socket is process-wide on the real backend; this suite has
  // none. Routing opens it inside the page and never connects to a server, so
  // the idle screen is exercised without a connection-refused console error.
  await page.routeWebSocket(/\/ws\/sim$/, () => {
    // Deliberately silent: no session, no frames.
  });

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    const fixture = pathname in overrides ? overrides[pathname] : FIXTURES[pathname];
    if (fixture === undefined) {
      result.unexpected.push(`${request.method()} ${pathname}`);
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: `e2e fixture missing for ${pathname}` }),
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(fixture),
    });
  });

  return result;
}

async function open(page: Page, path: string, title = "AI Scalping Studio"): Promise<void> {
  await page.goto(path);
  await expect(page).toHaveTitle(title);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
}

/** No unread number may leak into visible copy. */
async function expectNoUnreadableText(page: Page): Promise<void> {
  const text = await page.locator("main").innerText();
  expect(text).not.toMatch(/\bNaN\b/);
  expect(text).not.toMatch(/\bundefined\b/);
}

function expectCleanBrowser(state: Harness): void {
  expect(state.unexpected, "unmocked API requests").toEqual([]);
  expect(state.consoleErrors, "console errors").toEqual([]);
  expect(state.pageErrors, "uncaught page errors").toEqual([]);
}

test("simulator renders its idle board without a backend", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/");

  await expect(page.getByRole("heading", { level: 1, name: "Simülatör" })).toBeVisible();
  await expect(page.getByTestId("status-chip")).toHaveText("Hazır");
  await expect(page.getByTestId("bar-count")).toHaveText("0");
  await expect(page.getByTestId("last-price")).toHaveText("—");
  // Health answered `ok`; the offline banner and a start error must not appear.
  await expect(page.getByText(/Backend.*ulaşılamıyor/)).toHaveCount(0);
  await expect(page.getByTestId("strategy-card-state")).toHaveText(/kart yok/);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("data page shows the empty store and the download form", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/data");

  await expect(page.getByRole("heading", { level: 1, name: "Veri" })).toBeVisible();
  await expect(page.getByTestId("data-empty")).toContainText("Henüz veri yok");
  await expect(page.getByTestId("download-form")).toBeVisible();
  await expect(page.getByTestId("status-error")).toHaveCount(0);
  // An empty store has no provenance to describe.
  await expect(page.getByRole("heading", { name: "Veri kökeni" })).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("intelligence page shows the no-data branch, not a guessed regime", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/intelligence");

  await expect(page.getByRole("heading", { level: 1, name: "Piyasa Zekâsı" })).toBeVisible();
  await expect(page.getByTestId("regime-empty")).toContainText("veri yok");
  await expect(page.getByTestId("regime-current")).toHaveCount(0);
  await expect(page.getByTestId("regime-error")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("training page shows the empty state with no run list", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/training");

  await expect(page.getByRole("heading", { level: 1, name: "Eğitim" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Henüz eğitim yok" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Veri ekranına git" })).toHaveAttribute(
    "href",
    "/data",
  );

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("performance page shows the empty state with no backtest", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/performance");

  await expect(page.getByRole("heading", { level: 1, name: "Performans" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Henüz backtest yok" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Simülatöre git" })).toHaveAttribute("href", "/");

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("paper page shows the paper-only notice and an empty ledger", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/paper", "Paper Trading — AI Scalping Studio");

  await expect(page.getByRole("heading", { level: 1, name: "Paper Trading" })).toBeVisible();
  await expect(page.getByTestId("paper-notice")).toContainText("gerçek para yok");
  await expect(page.getByTestId("paper-sessions-empty")).toContainText(
    "Henüz paper oturumu yok",
  );
  // No session was invented, so no balance/equity panel exists.
  await expect(page.getByTestId("paper-session-facts")).toHaveCount(0);

  const text = await page.locator("main").innerText();
  expect(text).not.toContain("$");
  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("ML lab loads its registries and says the run list is empty", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/labs/ml", "ML Laboratuvarı — AI Scalping Studio");

  await expect(page.getByRole("heading", { level: 1, name: "ML Laboratuvarı" })).toBeVisible();
  await expect(page.getByTestId("ml-registries-error")).toHaveCount(0);
  await expect(page.getByLabel("Sembol")).toHaveValue("XAUUSD");
  await expect(page.getByLabel("Model")).toHaveValue("logistic");
  await expect(page.getByTestId("ml-runs-empty")).toContainText("Henüz ML koşusu yok");

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("autonomy page renders readable-but-empty sections", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/autonomy", "Otonomi — AI Scalping Studio");

  await expect(page.getByRole("heading", { level: 1, name: "Otonomi" })).toBeVisible();
  await expect(page.getByTestId("autonomy-paper-sessions")).toHaveText("0");
  await expect(page.getByTestId("autonomy-orchestrator-queued")).toHaveText("0");
  await expect(page.getByTestId("autonomy-champions-empty")).toContainText(
    "Kayıtlı şampiyon yok",
  );
  await expect(page.getByTestId("autonomy-data-empty")).toContainText("sembol yok");
  await expect(page.getByTestId("autonomy-jobs-empty")).toBeVisible();
  await expect(page.getByTestId("autonomy-events-empty")).toBeVisible();

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("autonomy page names absent stores instead of showing zeroes", async ({ page }) => {
  const state = await harness(page, {
    "/api/autonomy/status": UNREADABLE_AUTONOMY,
    "/api/autonomy/jobs": UNREADABLE_AUTONOMY_PAGE,
    "/api/autonomy/events": UNREADABLE_AUTONOMY_PAGE,
  });

  await open(page, "/autonomy", "Otonomi — AI Scalping Studio");

  await expect(page.getByTestId("autonomy-paper-state")).toContainText("yapılandırılmadı");
  await expect(page.getByTestId("autonomy-orchestrator-state")).toContainText(
    "yapılandırılmadı",
  );
  await expect(page.getByTestId("autonomy-champions-state")).toContainText("yapılandırılmadı");
  await expect(page.getByTestId("autonomy-data-state")).toContainText("yapılandırılmadı");
  await expect(page.getByTestId("autonomy-jobs-state")).toContainText("yapılandırılmadı");
  await expect(page.getByTestId("autonomy-events-state")).toContainText("yapılandırılmadı");
  // A missing store has no counts to show.
  await expect(page.getByTestId("autonomy-paper-sessions")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});
