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

/** An empty slice: every validator count is `null`, not a clean zero. */
const UNCONFIGURED_QUALITY = {
  symbol: "XAUUSD",
  timeframe: "M1",
  configured: false,
  status: "ok",
  status_rule: "unconfigured",
  status_detail: "Bu dilimde doğrulanacak bar yok; hiçbir kontrol çalışmadı.",
  rows: 0,
  first_ts: null,
  last_ts: null,
  duplicates: null,
  gaps: null,
  ohlc_violations: null,
  crossed_quotes: null,
  negative_spread: null,
  negative_volume: null,
  outliers: null,
  outliers_checked: false,
  provenance: null,
  thresholds: {
    gap_threshold_minutes: 5,
    outlier_return_z: 6,
    outlier_volume_z: 6,
    outlier_min_rows: 100,
  },
  issues: [],
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

/** `GET /api/autonomy/champions`: the archive is readable and holds no row. */
const EMPTY_CHAMPIONS = { configured: true, readable: true, items: [] };

// --- paper ledger fixtures ---------------------------------------------------

const PAPER_SESSION_ID = "sess-e2e-1";

const PAPER_SESSION_ITEM = {
  session_id: PAPER_SESSION_ID,
  symbol: "XAUUSD",
  timeframe: "M1",
  strategy: "manual",
  model_version: null,
  status: "closed",
  started_at: "2026-11-24T08:00:00Z",
  ended_at: "2026-11-24T10:00:00Z",
};

const PAPER_SESSION_SUMMARY = {
  ...PAPER_SESSION_ITEM,
  balance: 10000,
  equity: 10012.5,
  open_position: null,
  trade_count: 0,
  last_bar_ts: "2026-11-24T09:59:00Z",
  config: { origin: "api", extra: { dataset_id: "ds-xauusd-m1" } },
};

/** Three marks; the window delta is +12.50 and must be labelled as a delta. */
const PAPER_EQUITY = [
  { ts: "2026-11-24T08:05:00Z", balance: 10000, equity: 10000 },
  { ts: "2026-11-24T09:05:00Z", balance: 10000, equity: 10006.25 },
  { ts: "2026-11-24T09:55:00Z", balance: 10000, equity: 10012.5 },
];

function sessionsPage(items: readonly unknown[], total = items.length) {
  return { items, total, limit: 20, offset: 0 };
}

/** Every read one selected session makes, so no request leaves the fixture set. */
function paperDetail(
  sessionId: string,
  summary: unknown = PAPER_SESSION_SUMMARY,
  equity: unknown = emptyPage(),
): Record<string, unknown> {
  return {
    [`/api/paper/sessions/${sessionId}`]: summary,
    [`/api/paper/sessions/${sessionId}/positions`]: emptyPage(),
    [`/api/paper/sessions/${sessionId}/orders`]: emptyPage(),
    [`/api/paper/sessions/${sessionId}/history`]: emptyPage(),
    [`/api/paper/sessions/${sessionId}/equity`]: equity,
  };
}

const COMPLETED_RUN = {
  run_id: "run-done",
  status: "completed",
  created_at: "2026-11-20T10:00:00Z",
  steps: 1200,
  reward_mean: 0.42,
};

/** The nine headline metrics; non-finite wire values otherwise arrive as null. */
const BACKTEST_METRICS = {
  net_profit: 1234.56,
  profit_factor: 1.42,
  sharpe: 0.88,
  max_drawdown: 0.12,
  win_rate: 0.54,
  expectancy_r: 0.21,
  trades: 37,
  total_costs: 210.5,
  cost_ratio: 0.09,
  avg_bars: 42,
};

type Fixture = unknown;

/** A fixture plus a non-200 status; keyed replies can also be method-specific. */
const REPLY = Symbol("e2e-reply");

interface StubReply {
  readonly [REPLY]: true;
  readonly status: number;
  readonly body: Fixture;
}

function reply(body: Fixture, status = 200): StubReply {
  return { [REPLY]: true, status, body };
}

function isReply(value: unknown): value is StubReply {
  return typeof value === "object" && value !== null && (value as StubReply)[REPLY] === true;
}

/**
 * Pathname → body; query strings are ignored because no fixture depends on them.
 * Overrides may key on `"METHOD /path"` to answer a write differently from the
 * read of the same path, and may carry a `reply(body, status)`.
 */
const FIXTURES: Record<string, Fixture> = {
  "/api/health": HEALTH,
  "/api/data/status": EMPTY_DATA_STATUS,
  "/api/data/quality": UNCONFIGURED_QUALITY,
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
  "/api/autonomy/champions": EMPTY_CHAMPIONS,
};

interface Harness {
  /** `METHOD path` for every `/api/*` request the browser made. */
  requests: string[];
  /** `METHOD path` for any request the fixture table did not cover. */
  unexpected: string[];
  /** Parsed JSON bodies the page posted, in request order. */
  posts: { request: string; body: unknown }[];
  consoleErrors: string[];
  pageErrors: string[];
}

function fixtureFor(
  overrides: Record<string, Fixture>,
  method: string,
  pathname: string,
): Fixture | undefined {
  const keyed = `${method} ${pathname}`;
  if (Object.hasOwn(overrides, keyed)) return overrides[keyed];
  if (Object.hasOwn(overrides, pathname)) return overrides[pathname];
  return FIXTURES[pathname];
}

/**
 * Install the API/socket doubles and start collecting browser-side failures.
 * Must run before `page.goto` so no request escapes to `API_ORIGIN`.
 */
async function harness(page: Page, overrides: Record<string, Fixture> = {}): Promise<Harness> {
  const result: Harness = {
    requests: [],
    unexpected: [],
    posts: [],
    consoleErrors: [],
    pageErrors: [],
  };

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
    const label = `${request.method()} ${pathname}`;
    result.requests.push(label);
    if (request.method() !== "GET" && request.postData() !== null) {
      result.posts.push({ request: label, body: request.postDataJSON() });
    }

    const fixture = fixtureFor(overrides, request.method(), pathname);
    if (fixture === undefined) {
      result.unexpected.push(label);
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: `e2e fixture missing for ${pathname}` }),
      });
      return;
    }

    const status = isReply(fixture) ? fixture.status : 200;
    const body = isReply(fixture) ? fixture.body : fixture;
    await route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
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

/**
 * Chromium reports an intentional non-2xx fixture as a resource console error.
 * Only a test that asks for that exact status may whitelist it.
 */
const RESOURCE_404 = /Failed to load resource: .*status of 404 /;
const RESOURCE_500 = /Failed to load resource: .*status of 500 /;

function expectCleanBrowser(state: Harness, expectedResourceErrors: RegExp[] = []): void {
  expect(state.unexpected, "unmocked API requests").toEqual([]);
  const unexpectedConsole = state.consoleErrors.filter(
    (text) => !expectedResourceErrors.some((pattern) => pattern.test(text)),
  );
  expect(unexpectedConsole, "console errors").toEqual([]);
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

test("backtesting page shows the empty state with no runs", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/backtesting", "Backtesting — AI Scalping Studio");

  await expect(page.getByRole("heading", { level: 1, name: "Backtesting" })).toBeVisible();
  await expect(page.getByTestId("backtest-runs-empty")).toContainText("Henüz koşu yok");
  await expect(page.getByRole("button", { name: "Backtest çalıştır" })).toBeDisabled();
  // No response was received, so no metric card or equity chart may exist.
  await expect(page.getByTestId("backtest-result")).toHaveCount(0);
  await expect(page.getByTestId("metric-trades")).toHaveCount(0);

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

// --- backtesting -------------------------------------------------------------

test("backtesting page refuses to replay when the run list is missing", async ({ page }) => {
  const state = await harness(page, {
    "/api/runs": reply({ detail: "run index absent" }, 404),
  });

  await open(page, "/backtesting", "Backtesting — AI Scalping Studio");

  await expect(page.getByTestId("backtest-runs-missing")).toContainText("(404)");
  await expect(page.getByTestId("backtest-runs-missing")).toContainText("metrik uydurmaz");
  // Without a run there is nothing to select and nothing to submit.
  await expect(page.locator("#backtest-run")).toBeDisabled();
  await expect(page.getByRole("button", { name: "Backtest çalıştır" })).toBeDisabled();
  await expect(page.getByTestId("backtest-result")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state, [RESOURCE_404]);
});

test("backtesting posts only run_id and symbol and never invents an equity curve", async ({
  page,
}) => {
  const state = await harness(page, {
    "/api/runs": [COMPLETED_RUN],
    // The current backend returns metrics and a report path, no `equity` array.
    "POST /api/backtest": {
      report_path: "runs/run-done/backtest-XAUUSD.json",
      metrics: BACKTEST_METRICS,
    },
  });

  await open(page, "/backtesting", "Backtesting — AI Scalping Studio");

  await expect(page.locator("#backtest-run")).toHaveValue("run-done");
  await expect(page.getByTestId("backtest-runs-empty")).toHaveCount(0);
  await page.getByRole("button", { name: "Backtest çalıştır" }).click();

  await expect(page.getByTestId("backtest-result")).toBeVisible();
  await expect(page.getByTestId("metric-trades")).toHaveText("37");
  // A metrics object without `equity` keeps the chart absent and says why.
  await expect(page.getByTestId("backtest-equity")).toHaveCount(0);
  await expect(page.getByTestId("backtest-equity-absent")).toContainText("uydurulmadı");

  const posted = state.posts.filter((entry) => entry.request === "POST /api/backtest");
  expect(posted).toHaveLength(1);
  expect(posted[0].body).toEqual({ run_id: "run-done", symbol: "XAUUSD" });

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("backtesting validates the symbol locally before any request", async ({ page }) => {
  const state = await harness(page, { "/api/runs": [COMPLETED_RUN] });

  await open(page, "/backtesting", "Backtesting — AI Scalping Studio");

  const symbol = page.locator("#backtest-symbol");
  await symbol.fill("../etc");
  await page.getByRole("button", { name: "Backtest çalıştır" }).click();

  await expect(page.locator("#backtest-symbol-error")).toContainText(
    "Sembol harf veya rakamla başlamalı",
  );
  await expect(symbol).toHaveAttribute("aria-invalid", "true");
  await expect(symbol).toBeFocused();
  expect(state.posts).toEqual([]);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

// --- strategy intelligence ---------------------------------------------------

test("strategies page shows the empty regime and an unclassified card", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/strategies");

  await expect(page.getByRole("heading", { level: 1, name: "Strateji Zekâsı" })).toBeVisible();
  await expect(page.getByTestId("classification-card")).toBeVisible();
  await expect(page.getByTestId("classification-fallback")).toHaveText("fallback");
  await expect(page.getByTestId("classification-card")).toContainText("flow_momentum");
  // A readable archive with no row is "no champion", not "unreadable".
  await expect(page.getByTestId("champions-empty")).toContainText("kayıtlı şampiyon yok");
  await expect(page.getByTestId("champions-unreadable")).toHaveCount(0);
  await expect(page.getByTestId("regime-empty")).toContainText("veri yok");
  await expect(page.getByTestId("strategy-intel-status")).toContainText("rejim verisi yok");
  await expect(page.getByTestId("strategy-regime-unavailable")).toContainText("açık değil");
  await expect(page.getByTestId("regime-error")).toHaveCount(0);
  await expect(page.getByTestId("champions-error")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("strategies page renders read regime shares and champion rows without NaN", async ({
  page,
}) => {
  const state = await harness(page, {
    "/api/strategy/card": {
      active: true,
      strategy: "flow_momentum",
      timeframe: "M1",
      created_at: "2026-11-20T10:00:00Z",
    },
    "/api/regime": {
      symbol: "XAUUSD",
      timeframe: "M1",
      rows: 10,
      current: null,
      counts: [
        { regime: "trend_up", bars: 7, share: 0.7 },
        { regime: "chop", bars: 3, share: 0.3 },
      ],
      series: [],
    },
    "/api/autonomy/champions": {
      configured: true,
      readable: true,
      items: [
        {
          scope: "XAUUSD:M1",
          candidate_id: "cand-0000000000000001",
          mode: "evidence",
          promoted_at: "2026-11-20T10:00:00Z",
          policy_id: "pol-0000000000000001",
          policy_digest: "digest-1",
        },
        {
          scope: "XAUUSD:H1",
          candidate_id: "cand-2",
          mode: null,
          promoted_at: null,
        },
      ],
    },
  });

  await open(page, "/strategies");

  await expect(page.getByTestId("classification-legacy")).toHaveText("legacy");
  await expect(page.getByTestId("regime-count-trend_up")).toContainText("70,0%");
  await expect(page.getByTestId("regime-count-chop")).toContainText("30,0%");
  await expect(page.getByTestId("strategy-intel-status")).toContainText(
    "10 bar sınıflandırıldı, 2 rejim",
  );
  await expect(page.getByTestId("classification-dsl")).toHaveText("dsl");
  await expect(page.getByTestId("champion-row")).toHaveCount(2);
  await expect(page.getByTestId("champions-list")).toContainText("2 kapsamda şampiyon");
  await expect(page.getByTestId("champions-empty")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("strategies page separates an unreadable champion table from an absent archive", async ({
  page,
}) => {
  const unreadable = await harness(page, {
    "/api/autonomy/champions": { configured: true, readable: false },
  });

  await open(page, "/strategies");
  await expect(page.getByTestId("champions-unreadable")).toContainText(
    "şampiyon tablosu okunamadı",
  );
  await expect(page.getByTestId("champions-unreadable")).toContainText(
    "şampiyon yok",
  );
  await expect(page.getByTestId("champions-empty")).toHaveCount(0);
  await expectNoUnreadableText(page);
  expectCleanBrowser(unreadable);

  const absent = await harness(page, {
    "/api/autonomy/champions": { configured: false, readable: false },
  });

  await open(page, "/strategies");
  await expect(page.getByTestId("champions-unconfigured")).toContainText(
    "Keşif arşivi yapılandırılmadı",
  );
  await expect(page.getByTestId("champions-empty")).toHaveCount(0);
  await expectNoUnreadableText(page);
  expectCleanBrowser(absent);
});

// --- autonomy champion states ------------------------------------------------

test("autonomy page labels champion modes and marks non-evidence rows paper-dışı", async ({
  page,
}) => {
  const state = await harness(page, {
    "/api/autonomy/status": {
      ...EMPTY_AUTONOMY,
      champions: [
        { scope: "XAUUSD:M1", candidate_id: "c-evidence", mode: "evidence" },
        { scope: "XAUUSD:H1", candidate_id: "c-exploratory", mode: "exploratory" },
        { scope: "EURUSD:M5", candidate_id: "c-manual", mode: "manual" },
        { scope: "BTCUSD:M1", candidate_id: "c-unknown", mode: "weird" },
        { scope: "XAGUSD:M1", candidate_id: "c-absent", mode: null },
      ],
    },
  });

  await open(page, "/autonomy", "Otonomi — AI Scalping Studio");

  await expect(page.getByTestId("autonomy-champion-row-XAUUSD:M1")).toBeVisible();
  await expect(page.getByTestId("autonomy-champion-mode-XAUUSD:M1")).toHaveText(
    "kanıt · evidence",
  );
  await expect(page.getByTestId("autonomy-champion-mode-XAUUSD:H1")).toHaveText(
    "keşif · exploratory",
  );
  await expect(page.getByTestId("autonomy-champion-mode-EURUSD:M5")).toHaveText(
    "manuel · manual",
  );
  // An unknown string is not upgraded to evidence; a missing mode is not either.
  await expect(page.getByTestId("autonomy-champion-mode-BTCUSD:M1")).toHaveText(
    "weird · kanıtsız",
  );
  await expect(page.getByTestId("autonomy-champion-mode-XAGUSD:M1")).toHaveText(
    "mod yok · kanıtsız",
  );
  await expect(page.getByText("paper dışı", { exact: true })).toHaveCount(4);
  await expect(page.getByTestId("autonomy-champions-empty")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("autonomy page says the champion table is unreadable, not empty", async ({ page }) => {
  const state = await harness(page, {
    "/api/autonomy/status": {
      ...EMPTY_AUTONOMY,
      champions: [],
      champions_readable: false,
    },
  });

  await open(page, "/autonomy", "Otonomi — AI Scalping Studio");

  await expect(page.getByTestId("autonomy-champions-state")).toContainText("okunamıyor");
  await expect(page.getByTestId("autonomy-champions-state")).toContainText(
    "şampiyon listesi gösterilmiyor",
  );
  await expect(page.getByTestId("autonomy-champions-empty")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("autonomy page surfaces a status failure instead of showing stale zeroes", async ({
  page,
}) => {
  const state = await harness(page, {
    "/api/autonomy/status": reply({ detail: "ledger unavailable" }, 500),
  });

  await open(page, "/autonomy", "Otonomi — AI Scalping Studio");

  await expect(page.getByTestId("autonomy-status-error")).toContainText("Durum okunamadı");
  await expect(page.getByTestId("autonomy-status-error")).toContainText("ledger unavailable");
  // No section can render from a status that never arrived.
  await expect(page.getByTestId("autonomy-paper-state")).toHaveCount(0);
  await expect(page.getByTestId("autonomy-paper-sessions")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state, [RESOURCE_500]);
});

// --- paper ledger: equity and the start form ---------------------------------

test("paper page draws the ledger equity window with a text alternative", async ({ page }) => {
  const state = await harness(page, {
    "/api/paper/sessions": sessionsPage([PAPER_SESSION_ITEM]),
    ...paperDetail(PAPER_SESSION_ID, PAPER_SESSION_SUMMARY, {
      items: PAPER_EQUITY,
      total: PAPER_EQUITY.length,
      limit: 20,
      offset: 0,
    }),
  });

  await open(page, "/paper", "Paper Trading — AI Scalping Studio");

  await expect(page.getByTestId("paper-session-facts")).toBeVisible();
  await expect(page.getByTestId("paper-equity-figure")).toBeVisible();
  await expect(page.getByTestId("paper-equity-figure").getByRole("img")).toBeVisible();
  await expect(page.getByTestId("paper-equity-points")).toHaveText("3");
  await expect(page.getByTestId("paper-equity-first")).toHaveText("$10.000,00");
  await expect(page.getByTestId("paper-equity-last")).toHaveText("$10.012,50");
  // The window delta is +12.50, not the account's total P&L.
  await expect(page.getByTestId("paper-equity-change")).toHaveText("+$12,50");
  await expect(page.getByTestId("paper-equity-change-pct")).toHaveText("+%0,13");
  await expect(page.getByTestId("paper-equity-chart-empty")).toHaveCount(0);
  await expect(page.getByTestId("paper-equity-chart-unread")).toHaveCount(0);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("paper page does not invent equity when the ledger has no marks", async ({ page }) => {
  const state = await harness(page, {
    "/api/paper/sessions": sessionsPage([PAPER_SESSION_ITEM]),
    ...paperDetail(
      PAPER_SESSION_ID,
      { ...PAPER_SESSION_SUMMARY, balance: null, equity: null },
      emptyPage(),
    ),
  });

  await open(page, "/paper", "Paper Trading — AI Scalping Studio");

  await expect(page.getByTestId("paper-balance")).toHaveText("—");
  await expect(page.getByTestId("paper-equity")).toHaveText("—");
  await expect(page.getByTestId("paper-equity-chart-empty")).toContainText(
    "özkaynak işareti yok",
  );
  await expect(page.getByTestId("paper-equity-summary")).toHaveCount(0);
  await expect(page.getByTestId("paper-equity-figure").getByRole("img")).toHaveCount(0);

  const figureText = await page.getByTestId("paper-equity-figure").innerText();
  expect(figureText).not.toContain("$");
  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("paper start form blocks an invalid submission locally", async ({ page }) => {
  const state = await harness(page);

  await open(page, "/paper", "Paper Trading — AI Scalping Studio");

  const form = page.getByTestId("paper-start-form");
  await form.getByLabel("Sembol").fill("../etc");
  await page.getByRole("button", { name: "Oturumu kaydet" }).click();

  await expect(page.getByTestId("paper-start-errors")).toContainText("2 alan düzeltilmeli");
  await expect(page.locator("#paper-start-symbol-error")).toContainText(
    "Sembol harf veya rakamla başlamalı",
  );
  await expect(page.locator("#paper-start-strategy-error")).toContainText("Strateji gerekli");
  await expect(form.getByLabel("Sembol")).toHaveAttribute("aria-invalid", "true");
  await expect(form.getByLabel("Sembol")).toBeFocused();
  // The page is the only writer of a session; an invalid form must not call it.
  expect(state.posts).toEqual([]);

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

test("paper start form records a session and shows the server's reply", async ({ page }) => {
  const createdId = "new-session-1";
  const createdItem = {
    ...PAPER_SESSION_ITEM,
    session_id: createdId,
    status: "running",
    ended_at: null,
  };
  const state = await harness(page, {
    // The ledger is empty before the write; only the POST creates the row.
    "POST /api/paper/sessions": reply(
      { session_id: createdId, status: "running", started_at: "2026-11-24T11:00:00Z" },
      201,
    ),
    ...paperDetail(createdId, { ...PAPER_SESSION_SUMMARY, ...createdItem }),
  });

  await open(page, "/paper", "Paper Trading — AI Scalping Studio");
  await expect(page.getByTestId("paper-sessions-empty")).toBeVisible();

  const form = page.getByTestId("paper-start-form");
  await expect(form).toContainText("runner başlatmaz");
  await form.getByLabel("Strateji").fill("manual");
  await page.getByRole("button", { name: "Oturumu kaydet" }).click();

  await expect(page.getByTestId("paper-start-notice")).toContainText(
    `Oturum kaydedildi: ${createdId}`,
  );
  await expect(page.getByTestId("paper-start-notice")).toContainText(
    "sunucu yalnızca gönderilen alanları kaydetti",
  );
  await expect(page.getByTestId("paper-detail-status")).toHaveText("Çalışıyor");
  await expect(page.getByTestId("paper-start-error")).toHaveCount(0);
  await expect(page.getByTestId("paper-start-conflict")).toHaveCount(0);

  const posted = state.posts.filter((entry) => entry.request === "POST /api/paper/sessions");
  expect(posted).toHaveLength(1);
  // An empty optional field is null on the wire, never an invented value.
  expect(posted[0].body).toEqual({
    symbol: "XAUUSD",
    timeframe: "M1",
    strategy: "manual",
    model_version: null,
  });

  await expectNoUnreadableText(page);
  expectCleanBrowser(state);
});

// --- the fixture guard itself ------------------------------------------------

test("an unmocked API request fails closed instead of reaching a backend", async ({ page }) => {
  const state = await harness(page);

  await page.goto("/data");
  // Any `/api/*` path outside the table is a 500 and is recorded by the guard.
  const status = await page.evaluate(async () => {
    const response = await fetch("/api/e2e-unmocked-probe");
    return response.status;
  });

  expect(status).toBe(500);
  expect(state.unexpected).toEqual(["GET /api/e2e-unmocked-probe"]);
  expect(state.pageErrors).toEqual([]);
});
