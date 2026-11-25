import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BacktestResponse, Metrics, RunSummary } from "@/lib/types";

const { default: BacktestingPage } = await import("./page");

const METRICS: Metrics = {
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

const RUNS: RunSummary[] = [
  {
    run_id: "run-done",
    status: "completed",
    created_at: "2026-11-20T10:00:00Z",
    steps: 1200,
    reward_mean: 0.42,
  },
  {
    run_id: "run-live",
    status: "running",
    created_at: "2026-11-21T10:00:00Z",
    steps: 100,
    reward_mean: null,
  },
  {
    run_id: "run-failed",
    status: "failed",
    created_at: "2026-11-19T10:00:00Z",
    steps: 800,
    reward_mean: null,
  },
];

interface Call {
  path: string;
  method: string;
  body: unknown;
}

type Handler = (init?: RequestInit) => Response | Promise<Response>;

let handlers: Record<string, Handler>;
let calls: Call[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function response(overrides: Partial<BacktestResponse> = {}): BacktestResponse {
  return {
    report_path: "runs/run-done/backtest-XAUUSD.json",
    metrics: METRICS,
    ...overrides,
  };
}

/** Let queued fetch promises and their state updates settle inside `act`. */
async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

function renderPage() {
  return render(<BacktestingPage />);
}

function submit(): void {
  fireEvent.click(screen.getByRole("button", { name: "Backtest çalıştır" }));
}

beforeEach(() => {
  calls = [];
  handlers = {
    "/api/runs": () => json(RUNS),
    "/api/backtest": () => json(response()),
  };

  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input)).pathname;
      const method = init?.method ?? "GET";
      calls.push({
        path,
        method,
        body: init?.body === undefined ? undefined : JSON.parse(String(init.body)),
      });
      const handler = handlers[path];
      if (handler === undefined) return json({ detail: `no route ${path}` }, 404);
      return handler(init);
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("BacktestingPage", () => {
  it("lists only completed runs and posts exactly run_id and symbol", async () => {
    renderPage();
    await flush();

    expect(screen.getByRole("heading", { level: 1, name: "Backtesting" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /run-done/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /run-live/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /run-failed/ })).not.toBeInTheDocument();
    expect(screen.getByLabelText("Koşu")).toHaveValue("run-done");

    // The available-run table carries a caption and column scopes.
    const table = screen.getByRole("table", {
      name: "Backtest için kullanılabilir tamamlanmış koşular",
    });
    expect(table).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader")).toHaveLength(5);
    expect(screen.getByRole("rowheader", { name: "run-done" })).toBeInTheDocument();

    submit();
    await flush();

    const post = calls.find((call) => call.path === "/api/backtest");
    expect(post?.method).toBe("POST");
    expect(post?.body).toEqual({ run_id: "run-done", symbol: "XAUUSD" });

    expect(screen.getByTestId("backtest-result")).toBeInTheDocument();
    expect(screen.getByTestId("metric-trades")).toHaveTextContent("37");
    expect(screen.getByTestId("metric-net-profit")).toHaveTextContent("1.234,56");
    expect(screen.getByTestId("metric-win-rate")).toHaveTextContent("%54,0");
    expect(screen.getByTestId("metric-expectancy")).toHaveTextContent("0,21");
    expect(screen.getByTestId("metric-max-drawdown")).toHaveTextContent("-%12,00");
    expect(screen.getByTestId("metric-total-costs")).toHaveTextContent("210,50");

    // The response has no equity array: the page says so instead of drawing one.
    expect(screen.getByTestId("backtest-equity-absent")).toHaveTextContent("Eğri uydurulmadı");
    expect(screen.queryByTestId("backtest-equity")).not.toBeInTheDocument();

    const text = document.body.textContent ?? "";
    expect(text).not.toContain("NaN");
    expect(text).not.toContain("undefined");
  });

  it("treats a missing run list as an explicit 404 state and cannot submit", async () => {
    handlers["/api/runs"] = () => json({ detail: "Not Found" }, 404);

    renderPage();
    await flush();

    expect(screen.getByTestId("backtest-runs-missing")).toHaveTextContent("404");
    expect(screen.getByTestId("backtest-runs-missing")).toHaveTextContent("metrik uydurmaz");
    expect(screen.getByRole("button", { name: "Backtest çalıştır" })).toBeDisabled();
    expect(calls.some((call) => call.path === "/api/backtest")).toBe(false);
    expect(screen.queryByTestId("backtest-result")).not.toBeInTheDocument();
  });

  it("shows a run-list API error and no invented metrics", async () => {
    handlers["/api/runs"] = () => json({ detail: "registry unavailable" }, 500);

    renderPage();
    await flush();

    expect(screen.getByTestId("backtest-runs-error")).toHaveTextContent(
      "Koşu listesi alınamadı: registry unavailable",
    );
    expect(screen.getByRole("button", { name: "Backtest çalıştır" })).toBeDisabled();
    expect(screen.queryByTestId("backtest-result")).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toContain("NaN");
  });

  it("surfaces a 404 run and renders no metrics", async () => {
    handlers["/api/backtest"] = () => json({ detail: "unknown run: run-done" }, 404);

    renderPage();
    await flush();
    submit();
    await flush();

    expect(screen.getByTestId("backtest-error")).toHaveTextContent("404");
    expect(screen.getByTestId("backtest-error")).toHaveTextContent("unknown run: run-done");
    expect(screen.queryByTestId("backtest-result")).not.toBeInTheDocument();
    expect(screen.queryByTestId("metric-trades")).not.toBeInTheDocument();

    const text = document.body.textContent ?? "";
    expect(text).not.toContain("NaN");
    expect(text).not.toContain("undefined");
  });

  it("surfaces a 409 for a run that is not replayable", async () => {
    handlers["/api/backtest"] = () =>
      json({ detail: "run run-done is not finished (status running)" }, 409);

    renderPage();
    await flush();
    submit();
    await flush();

    expect(screen.getByTestId("backtest-error")).toHaveTextContent("409");
    expect(screen.getByTestId("backtest-error")).toHaveTextContent("not finished");
    expect(screen.queryByTestId("metric-net-profit")).not.toBeInTheDocument();
  });

  it("does not render a metric grid when the response has no metrics object", async () => {
    handlers["/api/backtest"] = () => json({ report_path: "runs/run-done/backtest.json" });

    renderPage();
    await flush();
    submit();
    await flush();

    expect(screen.getByTestId("backtest-metrics-absent")).toHaveTextContent(
      "Hiçbir metrik gösterilmiyor",
    );
    expect(screen.queryByTestId("metric-trades")).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toContain("NaN");
  });

  it("renders an equity curve only when the response returns one", async () => {
    handlers["/api/backtest"] = () =>
      json(response({ equity: [10_000, 10_120, 9_980, 10_400] }));

    renderPage();
    await flush();
    submit();
    await flush();

    const chartRegion = screen.getByRole("region", { name: /Equity eğrisi/ });
    expect(chartRegion).toHaveAttribute("tabindex", "0");
    expect(screen.getByTestId("backtest-equity")).toBeInTheDocument();
    expect(screen.queryByTestId("backtest-equity-absent")).not.toBeInTheDocument();
    expect(screen.getByTestId("backtest-equity")).toHaveTextContent("4 işaret");
    expect(screen.getByTestId("backtest-equity")).toHaveTextContent("10.000,00");
    expect(document.body.textContent ?? "").not.toContain("NaN");
  });

  it("blocks an invalid symbol before any request", async () => {
    renderPage();
    await flush();

    fireEvent.change(screen.getByLabelText("Sembol"), { target: { value: "-XAU" } });
    submit();
    await flush();

    expect(screen.getByLabelText("Sembol")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText(/Sembol harf veya rakamla başlamalı/)).toBeInTheDocument();
    expect(calls.some((call) => call.path === "/api/backtest")).toBe(false);
  });

  it("states that Monte Carlo is CLI-only and never draws a fake distribution", async () => {
    renderPage();
    await flush();

    const scope = screen.getByRole("region", {
      name: "Bu sayfada olmayan analizler",
    });
    expect(scope).toHaveTextContent("Monte Carlo");
    expect(scope).toHaveTextContent("yalnızca CLI");
    expect(scope).toHaveTextContent("grafik veya dağılım göstermez");
    expect(scope).toHaveTextContent("/api/evaluate");
  });
});
