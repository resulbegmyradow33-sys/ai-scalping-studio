import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  PaperEquityPoint,
  PaperOrder,
  PaperPage,
  PaperPosition,
  PaperSessionItem,
  PaperSessionSummary,
  PaperTrade,
} from "@/lib/types";

const { default: PaperTradingPage } = await import("./page");

const RUNNING = "abc123";
const STARTED = "2026-11-24T08:00:00Z";

interface Call {
  path: string;
  search: string;
  method: string;
  body?: string;
}

type Handler = () => Response | Promise<Response>;

let routes: Record<string, Handler>;
let calls: Call[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function page<T>(items: T[], total = items.length, offset = 0, limit = 20): PaperPage<T> {
  return { items, total, limit, offset };
}

function sessionItem(overrides: Partial<PaperSessionItem> = {}): PaperSessionItem {
  return {
    session_id: RUNNING,
    symbol: "XAUUSD",
    timeframe: "M1",
    strategy: "dsl",
    model_version: "hash-demo-001",
    status: "running",
    started_at: STARTED,
    ended_at: null,
    ...overrides,
  };
}

function summary(overrides: Partial<PaperSessionSummary> = {}): PaperSessionSummary {
  return {
    session_id: RUNNING,
    symbol: "XAUUSD",
    timeframe: "M1",
    strategy: "dsl",
    model_version: "hash-demo-001",
    status: "running",
    started_at: STARTED,
    ended_at: null,
    balance: 12345.67,
    equity: 12400.5,
    open_position: null,
    trade_count: 3,
    last_bar_ts: "2026-11-24T09:30:00Z",
    config: {
      origin: "api",
      extra: {
        dsl: { spec_hash: "hash-demo-001", spec_version: 2 },
        dataset_id: "ds-xauusd-m1",
      },
    },
    ...overrides,
  };
}

const POSITION: PaperPosition = {
  position_id: 7,
  side: "long",
  lot: 0.1,
  entry_ts: "2026-11-24T08:05:00Z",
  entry_price: 2350.1,
  sl: 2349,
  tp: 2352,
  status: "open",
  closed_ts: null,
  closed_price: null,
};

const ORDER: PaperOrder = {
  order_id: 9,
  signal_id: 4,
  ts: "2026-11-24T08:05:00Z",
  side: "long",
  lot: 0.1,
  requested_price: 2350,
  status: "rejected",
  rejection_reason: "daily loss limit",
};

const TRADE: PaperTrade = {
  trade_id: 3,
  side: "long",
  entry_ts: "2026-11-24T08:05:00Z",
  entry_price: 2350.1,
  exit_ts: "2026-11-24T08:20:00Z",
  exit_price: 2352.1,
  lot: 0.1,
  pnl: 18,
  r_multiple: 1.4,
  exit_reason: "tp",
};

const EQUITY: PaperEquityPoint = {
  ts: "2026-11-24T08:05:00Z",
  balance: 10000,
  equity: 10012.5,
};

function routeSessions(
  items: PaperSessionItem[],
  total = items.length,
  offset = 0,
): void {
  routes[`/api/paper/sessions?limit=20&offset=${offset}`] = () => json(page(items, total, offset));
}

interface DetailOptions {
  summaryResponse?: Handler;
  summaryOverrides?: Partial<PaperSessionSummary>;
  positions?: PaperPosition[];
  positionsTotal?: number;
  orders?: PaperOrder[];
  history?: PaperTrade[];
  equity?: PaperEquityPoint[];
  equityTotal?: number;
}

function routeDetail(sessionId: string, options: DetailOptions = {}): void {
  routes[`/api/paper/sessions/${sessionId}`] =
    options.summaryResponse ??
    (() => json(summary({ session_id: sessionId, ...options.summaryOverrides })));

  const positions = options.positions ?? [];
  routes[`/api/paper/sessions/${sessionId}/positions?limit=20&offset=0`] = () =>
    json(page(positions, options.positionsTotal ?? positions.length));
  routes[`/api/paper/sessions/${sessionId}/orders?limit=20&offset=0`] = () =>
    json(page(options.orders ?? []));
  routes[`/api/paper/sessions/${sessionId}/history?limit=20&offset=0`] = () =>
    json(page(options.history ?? []));
  routes[`/api/paper/sessions/${sessionId}/equity?limit=20&offset=0`] = () =>
    json(page(options.equity ?? [], options.equityTotal ?? (options.equity ?? []).length));
}

function detailGetCount(): number {
  return calls.filter((call) => call.path === `/api/paper/sessions/${RUNNING}`).length;
}

/** Let queued fetch promises and their state updates settle inside `act`. */
async function flush(): Promise<void> {
  await act(async () => {
    for (let index = 0; index < 12; index += 1) await Promise.resolve();
  });
}

function renderPage() {
  return render(<PaperTradingPage />);
}

beforeEach(() => {
  vi.useFakeTimers();
  calls = [];
  routes = {};

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input));
      const key = `${url.pathname}${url.search}`;
      calls.push({
        path: url.pathname,
        search: url.search,
        method: init?.method ?? "GET",
        body: typeof init?.body === "string" ? init.body : undefined,
      });
      const handler = routes[key];
      if (handler === undefined) return json({ detail: `no route ${key}` }, 404);
      return handler();
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("PaperTradingPage", () => {
  it("shows the paper-only notice and an honest empty state, with no metrics", async () => {
    routeSessions([]);

    renderPage();
    await flush();

    expect(screen.getByRole("heading", { name: "Paper Trading" })).toBeInTheDocument();
    expect(screen.getByTestId("paper-notice")).toHaveTextContent("gerçek para yok");
    expect(screen.getByTestId("paper-sessions-empty")).toHaveTextContent(
      "Henüz paper oturumu yok",
    );
    expect(screen.getByText(/Ayrıntı için listeden bir oturum seçin/)).toBeInTheDocument();

    // No session row was invented, so no balance/equity/attribution exists.
    expect(calls).toEqual([
      expect.objectContaining({ path: "/api/paper/sessions", search: "?limit=20&offset=0" }),
    ]);
    expect(screen.queryByTestId("paper-session-facts")).toBeNull();
    expect(document.body.textContent).not.toContain("$");
  });

  it("marks legacy flow_momentum paper sessions", async () => {
    routeSessions([sessionItem({ strategy: "flow_momentum" })]);
    routeDetail(RUNNING, { summaryOverrides: { strategy: "flow_momentum" } });

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-row-legacy")).toHaveTextContent("legacy baseline");
    expect(screen.getByTestId("paper-detail-legacy")).toHaveTextContent("legacy baseline");
  });

  it("renders the loaded session from the API: balance, equity, counts and attribution", async () => {
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, {
      positions: [POSITION],
      orders: [ORDER],
      history: [TRADE],
      equity: [EQUITY],
    });

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-row-status")).toHaveTextContent("Çalışıyor");
    expect(screen.getByTestId("paper-detail-status")).toHaveTextContent("Çalışıyor");
    expect(screen.getByTestId("paper-balance")).toHaveTextContent("$12.345,67");
    expect(screen.getByTestId("paper-equity")).toHaveTextContent("$12.400,50");
    expect(screen.getByTestId("paper-trade-count")).toHaveTextContent("3");
    expect(screen.getByTestId("paper-last-bar")).not.toHaveTextContent("—");

    // Attribution comes from the ledger config, not from a hard-coded hash.
    expect(screen.getByTestId("paper-strategy")).toHaveTextContent("dsl");
    expect(screen.getByTestId("paper-model-version")).toHaveTextContent("hash-demo-001");
    expect(screen.getByTestId("paper-spec-hash")).toHaveTextContent("hash-demo-001");
    expect(screen.getByTestId("paper-spec-version")).toHaveTextContent("2");
    expect(screen.getByTestId("paper-dataset")).toHaveTextContent("ds-xauusd-m1");

    // A running session has no open position here; the page says so instead of
    // showing a zeroed position.
    expect(screen.getByTestId("paper-open-position")).toHaveTextContent("Açık pozisyon yok.");

    expect(screen.getByRole("rowheader", { name: "#7" })).toBeInTheDocument();
    expect(screen.getByText("daily loss limit")).toBeInTheDocument();
    expect(screen.getByText("tp")).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "24.11.2026 11:05" })).toBeInTheDocument();
  });

  it("treats a 404 session as missing, explains it and stops polling", async () => {
    routeSessions([sessionItem()]);
    routes[`/api/paper/sessions/${RUNNING}`] = () => json({ detail: "unknown paper session" }, 404);
    routes[`/api/paper/sessions/${RUNNING}/positions?limit=20&offset=0`] = () =>
      json({ detail: "unknown paper session" }, 404);
    routes[`/api/paper/sessions/${RUNNING}/orders?limit=20&offset=0`] = () =>
      json({ detail: "unknown paper session" }, 404);
    routes[`/api/paper/sessions/${RUNNING}/history?limit=20&offset=0`] = () =>
      json({ detail: "unknown paper session" }, 404);
    routes[`/api/paper/sessions/${RUNNING}/equity?limit=20&offset=0`] = () =>
      json({ detail: "unknown paper session" }, 404);

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-session-missing")).toHaveTextContent("(404)");
    expect(screen.getByTestId("paper-session-missing")).toHaveTextContent("değer uydurmaz");
    expect(screen.queryByTestId("paper-session-facts")).toBeNull();

    const before = detailGetCount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });
    expect(detailGetCount()).toBe(before);
  });

  it("closes a running session after confirmation and refreshes the record", async () => {
    let closed = false;
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, {
      summaryResponse: () => json(closed ? closedSummary() : summary()),
    });
    routes[`/api/paper/sessions/${RUNNING}/close`] = () => {
      closed = true;
      return json(closedSummary());
    };

    renderPage();
    await flush();

    fireEvent.click(screen.getByTestId("paper-close-button"));
    expect(screen.getByText("Bu oturum kapatılacak; kapanış geri alınamaz.")).toBeInTheDocument();
    expect(screen.getByTestId("paper-close-confirm")).toHaveFocus();
    expect(calls.some((call) => call.path.endsWith("/close"))).toBe(false);

    await act(async () => {
      fireEvent.click(screen.getByTestId("paper-close-confirm"));
    });
    await flush();

    const closeCalls = calls.filter((call) => call.path.endsWith("/close"));
    expect(closeCalls).toHaveLength(1);
    expect(closeCalls[0].method).toBe("POST");
    expect(screen.getByTestId("paper-close-notice")).toHaveTextContent("Oturum kapatıldı");
    expect(screen.getByTestId("paper-detail-status")).toHaveTextContent("Kapalı");
    expect(screen.getByTestId("paper-close-button")).toBeDisabled();
    // The confirm group is gone; focus lands on the section heading, not <body>.
    expect(screen.getByRole("heading", { name: "Oturum ayrıntısı" })).toHaveFocus();

    // The detail and the list are re-read after the write, not patched locally.
    expect(detailGetCount()).toBeGreaterThan(1);
  });

  it("handles a 409 as already closed: no error, status refreshed", async () => {
    let closed = false;
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, {
      summaryResponse: () => json(closed ? closedSummary() : summary()),
    });
    routes[`/api/paper/sessions/${RUNNING}/close`] = () => {
      closed = true;
      return json({ detail: `paper session ${RUNNING} is already closed` }, 409);
    };

    renderPage();
    await flush();
    const before = detailGetCount();

    fireEvent.click(screen.getByTestId("paper-close-button"));
    await act(async () => {
      fireEvent.click(screen.getByTestId("paper-close-confirm"));
    });
    await flush();

    expect(screen.getByTestId("paper-close-notice")).toHaveTextContent("zaten kapalı");
    expect(screen.queryByTestId("paper-close-error")).toBeNull();
    expect(screen.getByTestId("paper-detail-status")).toHaveTextContent("Kapalı");
    expect(detailGetCount()).toBeGreaterThan(before);
  });

  it("returns focus to the close button when the confirmation is cancelled", async () => {
    routeSessions([sessionItem()]);
    routeDetail(RUNNING);

    renderPage();
    await flush();

    fireEvent.click(screen.getByTestId("paper-close-button"));
    expect(screen.getByTestId("paper-close-confirm")).toHaveFocus();

    fireEvent.click(screen.getByRole("button", { name: "Vazgeç" }));
    expect(screen.getByTestId("paper-close-button")).toHaveFocus();
  });

  it("bounds session pagination to the real total", async () => {
    const first = Array.from({ length: 20 }, (_, index) =>
      sessionItem({ session_id: `sess-${index}`, status: "closed" }),
    );
    const last = Array.from({ length: 5 }, (_, index) =>
      sessionItem({ session_id: `sess-${index + 20}`, status: "closed" }),
    );
    routeSessions(first, 25, 0);
    routeSessions(last, 25, 20);
    routeDetail("sess-0", { summaryOverrides: { session_id: "sess-0", status: "closed" } });

    renderPage();
    await flush();

    const pager = screen.getByRole("group", { name: "Paper oturumları — en yeni önce sayfalama" });
    expect(pager).toHaveTextContent("1–20 / 25");
    // Paging changes the range without moving focus; announce it politely.
    expect(within(pager).getByText("1–20 / 25")).toHaveAttribute("aria-live", "polite");
    expect(within(pager).getByRole("button", { name: "Önceki" })).toBeDisabled();

    fireEvent.click(within(pager).getByRole("button", { name: "Sonraki" }));
    await flush();

    expect(calls.some((call) => call.search === "?limit=20&offset=20")).toBe(true);
    expect(pager).toHaveTextContent("21–25 / 25");
    expect(within(pager).getByRole("button", { name: "Sonraki" })).toBeDisabled();

    fireEvent.click(within(pager).getByRole("button", { name: "Önceki" }));
    await flush();

    const offsets = calls
      .filter((call) => call.path === "/api/paper/sessions")
      .map((call) => call.search);
    expect(offsets).toEqual(["?limit=20&offset=0", "?limit=20&offset=20", "?limit=20&offset=0"]);
  });

  it("pages the equity window backwards and never asks past the total", async () => {
    routeSessions([sessionItem({ status: "closed" })]);
    const newest = Array.from({ length: 20 }, (_, index) => ({
      ...EQUITY,
      ts: `2026-11-24T09:${String(index).padStart(2, "0")}:00Z`,
    }));
    const older = Array.from({ length: 10 }, (_, index) => ({
      ...EQUITY,
      ts: `2026-11-24T07:${String(index).padStart(2, "0")}:00Z`,
    }));

    routeDetail(RUNNING, {
      equity: newest,
      equityTotal: 30,
      summaryOverrides: { status: "closed", ended_at: "2026-11-24T10:00:00Z" },
    });
    routes[`/api/paper/sessions/${RUNNING}/equity?limit=20&offset=20`] = () =>
      json(page(older, 30, 20));

    renderPage();
    await flush();

    const pager = screen.getByRole("group", {
      name: "Özkaynak işaretleri — offset 0 en yeni pencere, artan zamanla sayfalama",
    });
    expect(within(pager).getByRole("button", { name: "Daha yeni" })).toBeDisabled();
    expect(within(pager).getByRole("button", { name: "Daha eski" })).toBeEnabled();

    fireEvent.click(within(pager).getByRole("button", { name: "Daha eski" }));
    await flush();

    expect(
      calls.some(
        (call) => call.path === `/api/paper/sessions/${RUNNING}/equity` && call.search === "?limit=20&offset=20",
      ),
    ).toBe(true);
    expect(within(pager).getByRole("button", { name: "Daha eski" })).toBeDisabled();
    expect(within(pager).getByRole("button", { name: "Daha yeni" })).toBeEnabled();

    fireEvent.click(within(pager).getByRole("button", { name: "Daha eski" }));
    await flush();
    const equityOffsets = calls
      .filter((call) => call.path === `/api/paper/sessions/${RUNNING}/equity`)
      .map((call) => call.search);
    expect(equityOffsets).toEqual(["?limit=20&offset=0", "?limit=20&offset=20"]);
  });

  it("shows the list error as an alert and the detail error with a working retry", async () => {
    routes["/api/paper/sessions?limit=20&offset=0"] = () => json({ detail: "store unavailable" }, 500);

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-sessions-error")).toHaveTextContent("store unavailable");
    expect(screen.queryByTestId("paper-session-facts")).toBeNull();

    cleanup();
    calls = [];
    routes = {};

    routeSessions([sessionItem()]);
    let attempts = 0;
    routes[`/api/paper/sessions/${RUNNING}`] = () => {
      attempts += 1;
      return attempts === 1
        ? json({ detail: "ledger busy" }, 500)
        : json(summary({ status: "closed" }));
    };
    routes[`/api/paper/sessions/${RUNNING}/positions?limit=20&offset=0`] = () => json(page([]));
    routes[`/api/paper/sessions/${RUNNING}/orders?limit=20&offset=0`] = () => json(page([]));
    routes[`/api/paper/sessions/${RUNNING}/history?limit=20&offset=0`] = () => json(page([]));
    routes[`/api/paper/sessions/${RUNNING}/equity?limit=20&offset=0`] = () => json(page([]));

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-detail-error")).toHaveTextContent("ledger busy");
    fireEvent.click(screen.getByRole("button", { name: "Yenile" }));
    await flush();

    expect(screen.getByTestId("paper-balance")).toHaveTextContent("$12.345,67");
    expect(screen.queryByTestId("paper-detail-error")).toBeNull();
  });

  it("never invents a balance, equity, hash or position when the ledger has none", async () => {
    routeSessions([sessionItem({ status: "closed", model_version: null })]);
    routeDetail(RUNNING, {
      summaryOverrides: {
        status: "closed",
        ended_at: "2026-11-24T10:00:00Z",
        balance: null,
        equity: null,
        trade_count: 0,
        last_bar_ts: null,
        model_version: null,
        config: {},
      },
    });

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-balance")).toHaveTextContent("—");
    expect(screen.getByTestId("paper-equity")).toHaveTextContent("—");
    expect(screen.getByTestId("paper-last-bar")).toHaveTextContent("—");
    expect(screen.getByTestId("paper-model-version")).toHaveTextContent("—");
    expect(screen.getByTestId("paper-spec-hash")).toHaveTextContent("—");
    expect(screen.getByTestId("paper-trade-count")).toHaveTextContent("0");
    expect(screen.getByTestId("paper-open-position")).toHaveTextContent("Açık pozisyon yok.");
    expect(screen.getByText(/DSL spec hash’i yok/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("$");
    expect(screen.queryByText("0,00")).toBeNull();
  });

  it("polls only while the session is running and cleans the timer on unmount", async () => {
    let status = "running";
    let detailCalls = 0;
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, {
      summaryResponse: () => {
        detailCalls += 1;
        return json(summary({ status }));
      },
    });

    const view = renderPage();
    await flush();

    expect(detailCalls).toBe(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_900);
    });
    expect(detailCalls).toBe(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(detailCalls).toBe(2);

    // Closed is terminal: no further poll, even after 30 s.
    status = "closed";
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_100);
    });
    const settled = detailCalls;
    expect(screen.getByTestId("paper-detail-status")).toHaveTextContent("Kapalı");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(detailCalls).toBe(settled);

    view.unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(detailCalls).toBe(settled);
  });

  it("draws the equity window with a text alternative and a window summary", async () => {
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, {
      equity: [
        { ts: "2026-11-24T08:05:00Z", balance: 10000, equity: 10000 },
        { ts: "2026-11-24T08:35:00Z", balance: 10020, equity: 10010 },
        { ts: "2026-11-24T09:05:00Z", balance: 10040, equity: 10030 },
      ],
    });

    renderPage();
    await flush();

    const figure = screen.getByTestId("paper-equity-figure");
    const chart = within(figure).getByRole("img", { name: /Özkaynak eğrisi/ });
    expect(chart.querySelector("polyline")?.getAttribute("points")?.trim().split(/\s+/)).toHaveLength(
      3,
    );
    // The svg title is the text alternative and names the window's change.
    expect(chart).toHaveAccessibleName(/pencerede değişim \+\$30,00/);

    expect(screen.getByTestId("paper-equity-first")).toHaveTextContent("$10.000,00");
    expect(screen.getByTestId("paper-equity-last")).toHaveTextContent("$10.030,00");
    expect(screen.getByTestId("paper-equity-change")).toHaveTextContent("+$30,00");
    expect(screen.getByTestId("paper-equity-change-pct")).toHaveTextContent("+%0,30");
    expect(screen.getByTestId("paper-equity-points")).toHaveTextContent("3");

    // The window delta is labelled as a delta, not as the account's total P&L.
    expect(figure).toHaveTextContent("toplam P&L değildir");
  });

  it("shows an explicit no-chart state when the ledger has no equity marks", async () => {
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, { equity: [] });

    renderPage();
    await flush();

    const figure = screen.getByTestId("paper-equity-figure");
    expect(figure.querySelector("svg")).toBeNull();
    expect(screen.getByTestId("paper-equity-chart-empty")).toHaveTextContent("eğri çizilmez");
    expect(screen.queryByTestId("paper-equity-summary")).toBeNull();
    // No first/last/change value is invented for an empty curve.
    expect(figure.textContent).not.toContain("$");
  });

  it("refuses to draw marks whose timestamp cannot be parsed", async () => {
    routeSessions([sessionItem()]);
    routeDetail(RUNNING, {
      equity: [{ ts: "not-a-timestamp", balance: 10000, equity: 10010 }],
    });

    renderPage();
    await flush();

    const figure = screen.getByTestId("paper-equity-figure");
    expect(figure.querySelector("svg")).toBeNull();
    expect(screen.getByTestId("paper-equity-chart-invalid")).toHaveTextContent(
      "eğri çizilmedi",
    );
    expect(screen.queryByTestId("paper-equity-summary")).toBeNull();
    expect(figure.textContent).not.toContain("$");
  });

  it("records a session from the form, refreshes the list and focuses the notice", async () => {
    const sessions: PaperSessionItem[] = [];
    routes["/api/paper/sessions?limit=20&offset=0"] = () => json(page(sessions));
    routes["/api/paper/sessions"] = () => {
      sessions.push(sessionItem({ session_id: "new-session-1" }));
      return json(
        { session_id: "new-session-1", status: "running", started_at: STARTED },
        201,
      );
    };
    routeDetail("new-session-1");

    renderPage();
    await flush();

    expect(screen.getByTestId("paper-sessions-empty")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Sembol/), { target: { value: " XAUUSD " } });
    fireEvent.change(screen.getByLabelText(/Zaman dilimi/), { target: { value: "M15" } });
    fireEvent.change(screen.getByLabelText(/Strateji/), { target: { value: "manual" } });
    fireEvent.change(screen.getByLabelText(/Yapılandırma/), {
      target: { value: '{"note":"x"}' },
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Oturumu kaydet" }));
    });
    await flush();

    const post = calls.find(
      (call) => call.method === "POST" && call.path === "/api/paper/sessions",
    );
    expect(post).toBeDefined();
    expect(JSON.parse(post?.body ?? "{}")).toEqual({
      symbol: "XAUUSD",
      timeframe: "M15",
      strategy: "manual",
      // An empty field is `null` on the wire, not an invented version string.
      model_version: null,
      config: { note: "x" },
    });

    // The list is re-read after the write, and the 201 session is selected.
    const listGets = calls.filter(
      (call) => call.method === "GET" && call.path === "/api/paper/sessions",
    );
    expect(listGets).toHaveLength(2);
    expect(screen.getByTestId("paper-detail-status")).toHaveTextContent("Çalışıyor");

    const notice = screen.getByTestId("paper-start-notice");
    expect(notice).toHaveTextContent("new-session-1");
    expect(notice).toHaveTextContent("yalnızca gönderilen alanları kaydetti");
    // Focus lands on the result, not on <body> or a stale control.
    expect(notice).toHaveFocus();

    // The form says the API stores the submission and picks no champion.
    expect(screen.getByTestId("paper-start-form")).toHaveTextContent("şampiyon seçilmez");
  });

  it("blocks an invalid start locally, announces it and focuses the first invalid field", async () => {
    routeSessions([]);

    renderPage();
    await flush();

    fireEvent.change(screen.getByLabelText(/Sembol/), { target: { value: "../etc" } });
    fireEvent.change(screen.getByLabelText(/Yapılandırma/), { target: { value: "{oops" } });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Oturumu kaydet" }));
    });
    await flush();

    expect(calls.some((call) => call.method === "POST")).toBe(false);
    expect(screen.getByTestId("paper-start-errors")).toHaveTextContent("3 alan");
    expect(document.getElementById("paper-start-symbol-error")).toHaveTextContent(
      "Sembol harf veya rakamla başlamalı",
    );
    expect(document.getElementById("paper-start-strategy-error")).toHaveTextContent(
      "Strateji gerekli",
    );
    expect(document.getElementById("paper-start-config-error")).toHaveTextContent(
      "geçerli JSON değil",
    );

    const symbol = screen.getByLabelText(/Sembol/);
    expect(symbol).toHaveAttribute("aria-invalid", "true");
    expect(symbol).toHaveFocus();
  });

  it("surfaces a 409 from the start endpoint without pretending a session exists", async () => {
    routeSessions([]);
    routes["/api/paper/sessions"] = () =>
      json({ detail: "duplicate paper session" }, 409);

    renderPage();
    await flush();

    fireEvent.change(screen.getByLabelText(/Strateji/), { target: { value: "manual" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Oturumu kaydet" }));
    });
    await flush();

    const conflict = screen.getByTestId("paper-start-conflict");
    expect(conflict).toHaveTextContent("Çakışma (409)");
    expect(conflict).toHaveTextContent("duplicate paper session");
    expect(screen.queryByTestId("paper-start-notice")).toBeNull();
    // Nothing was selected or refreshed as if the write had succeeded.
    expect(screen.getByTestId("paper-sessions-empty")).toBeInTheDocument();
    expect(
      calls.filter((call) => call.method === "GET" && call.path === "/api/paper/sessions"),
    ).toHaveLength(1);
  });

  it("surfaces the server's 422 message instead of a generic failure", async () => {
    routeSessions([]);
    routes["/api/paper/sessions"] = () =>
      json(
        {
          detail: [
            {
              loc: ["body", "timeframe"],
              msg: "Input should be 'M1', 'M5', 'M15', 'M30', 'H1', 'H4' or 'D1'",
              type: "literal_error",
            },
          ],
        },
        422,
      );

    renderPage();
    await flush();

    fireEvent.change(screen.getByLabelText(/Strateji/), { target: { value: "manual" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Oturumu kaydet" }));
    });
    await flush();

    const error = screen.getByTestId("paper-start-error");
    expect(error).toHaveTextContent("422");
    expect(error).toHaveTextContent("Input should be");
    expect(screen.queryByTestId("paper-start-notice")).toBeNull();
  });
});

function closedSummary(): PaperSessionSummary {
  return summary({ status: "closed", ended_at: "2026-11-24T10:00:00Z" });
}
