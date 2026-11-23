import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  MLEvaluateResponse,
  MLRunDetail,
  MLRunSummary,
  MLTradingMetrics,
} from "@/lib/types";

const { default: MLLabPage } = await import("./page");

const SYMBOLS = [
  { symbol: "XAUUSD", description: "Gold", asset_class: "metal" },
  { symbol: "EURUSD", description: "Euro", asset_class: "fx" },
];

const TIMEFRAMES = [
  { timeframe: "M1", rule: "1m" },
  { timeframe: "H1", rule: "60m" },
];

const MODELS = ["logistic", "random_forest", "extra_trees"];

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

const TRADING: MLTradingMetrics = {
  trades: 40,
  net_profit: -2002.34,
  profit_factor: 0.83,
  win_rate: 0.42,
  expectancy_r: -0.131,
  max_drawdown: 0.223,
  total_costs: 1503,
};

const HOLD: MLTradingMetrics = {
  trades: 0,
  net_profit: 0,
  profit_factor: null,
  win_rate: 0,
  expectancy_r: 0,
  max_drawdown: 0,
  total_costs: 0,
};

function detail(overrides: Partial<MLRunDetail> = {}): MLRunDetail {
  return {
    run_id: "mlrun123",
    status: "running",
    created_at: "2026-11-20T10:00:00Z",
    finished_at: null,
    model: "logistic",
    symbol: "XAUUSD",
    timeframe: "H1",
    dataset_id: "ds-abc123",
    config: {},
    spec: {
      model: "logistic",
      horizon: 5,
      train_fraction: 0.7,
      validation_fraction: 0.15,
      purge_bars: 5,
      embargo_bars: 5,
      include_flat: false,
      groups: ["price", "momentum"],
    },
    columns: ["f1", "f2", "f3"],
    validation: null,
    test: null,
    baseline: null,
    split_counts: {},
    calibration: null,
    error: null,
    ...overrides,
  };
}

function completed(overrides: Partial<MLRunDetail> = {}): MLRunDetail {
  return detail({
    status: "completed",
    finished_at: "2026-11-20T10:02:00Z",
    validation: {
      rows: 4427,
      accuracy: 0.516,
      balanced_accuracy: 0.518,
      log_loss: 0.6921,
      brier: 0.2504,
      calibration: { ece: 0.043, mce: 0.087 },
    },
    test: null,
    baseline: { accuracy: 0.543, log_loss: 0.6919 },
    split_counts: { train: 20654, validation: 4427, test: 4408, purged: 12, flat_dropped: 7 },
    calibration: { brier: 0.2504, log_loss: 0.6921, ece: 0.043, mce: 0.087 },
    ...overrides,
  });
}

function summary(overrides: Partial<MLRunSummary> = {}): MLRunSummary {
  return {
    run_id: "mlrun123",
    status: "completed",
    created_at: "2026-11-20T10:00:00Z",
    model: "logistic",
    symbol: "XAUUSD",
    timeframe: "H1",
    dataset_id: "ds-abc123",
    ...overrides,
  };
}

const EVALUATION: MLEvaluateResponse = {
  run_id: "mlrun123",
  symbol: "XAUUSD",
  timeframe: "H1",
  dataset_id: "ds-abc123",
  split: "validation",
  threshold: 0.55,
  exit_threshold: 0.55,
  metrics: TRADING,
  hold: HOLD,
  stress: {
    x2: { ...TRADING, trades: 22, net_profit: -2100.1 },
    x3: { ...TRADING, trades: 18, net_profit: -2300.5 },
  },
  regime: {
    rows: [
      {
        regime: "trend_up",
        trades: 12,
        win_rate: 0.5,
        expectancy_r: 0.1,
        total_r: 1.2,
        pnl: 120.5,
        profit_factor: 1.4,
      },
      {
        regime: "sideways",
        trades: 25,
        win_rate: 0.36,
        expectancy_r: -0.2,
        total_r: -5,
        pnl: -900,
        profit_factor: 0.6,
      },
    ],
    unmatched_trades: 3,
  },
};

/** Let queued fetch promises and their state updates settle inside `act`. */
async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

function renderPage() {
  return render(<MLLabPage />);
}

beforeEach(() => {
  vi.useFakeTimers();
  calls = [];
  handlers = {
    "/api/symbols": () => json(SYMBOLS),
    "/api/timeframes": () => json(TIMEFRAMES),
    "/api/ml/models": () => json(MODELS),
    "/api/ml/runs": () => json([]),
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
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("MLLabPage", () => {
  it("loads its lists from the API and preselects the first entries", async () => {
    renderPage();
    await flush();

    expect(screen.getByRole("heading", { name: "ML Laboratuvarı" })).toBeInTheDocument();
    expect(screen.getByLabelText("Sembol")).toHaveValue("XAUUSD");
    expect(screen.getByLabelText("Zaman dilimi")).toHaveValue("H1");
    expect(screen.getByLabelText("Model")).toHaveValue("logistic");
    expect(calls.map((call) => call.path)).toEqual(
      expect.arrayContaining(["/api/symbols", "/api/timeframes", "/api/ml/models", "/api/ml/runs"]),
    );
    // The options come from the server, not a hard-coded list.
    expect(screen.getByRole("option", { name: /EURUSD/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "extra_trees" })).toBeInTheDocument();
  });

  it("blocks an invalid form with a clear message and never posts", async () => {
    renderPage();
    await flush();

    fireEvent.change(screen.getByLabelText("Eğitim oranı"), { target: { value: "1.2" } });
    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();

    const summaryBox = screen.getByTestId("ml-form-errors");
    expect(summaryBox).toHaveTextContent("Form gönderilemedi");
    expect(summaryBox).toHaveTextContent("Eğitim oranı 0 ile 1 arasında olmalı");
    expect(screen.getByLabelText("Eğitim oranı")).toHaveAttribute("aria-invalid", "true");
    expect(calls.some((call) => call.path === "/api/ml/train")).toBe(false);
  });

  it("rejects fractions that leave no test rows", async () => {
    renderPage();
    await flush();

    fireEvent.change(screen.getByLabelText("Eğitim oranı"), { target: { value: "0.9" } });
    fireEvent.change(screen.getByLabelText("Doğrulama oranı"), {
      target: { value: "0.2" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();

    expect(screen.getByTestId("ml-form-errors")).toHaveTextContent(
      "Eğitim + doğrulama oranı 1'den küçük olmalı",
    );
    expect(calls.some((call) => call.path === "/api/ml/train")).toBe(false);
  });

  it("posts a 202, shows the run id, polls every 2 s and stops at completed", async () => {
    let detailCalls = 0;
    handlers["/api/ml/train"] = () => json({ run_id: "mlrun123" }, 202);
    handlers["/api/ml/runs/mlrun123"] = () => {
      detailCalls += 1;
      return json(detailCalls === 1 ? detail() : completed());
    };

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();

    const start = calls.find((call) => call.path === "/api/ml/train");
    expect(start).toBeDefined();
    expect(start?.method).toBe("POST");
    expect(start?.body).toMatchObject({
      symbol: "XAUUSD",
      timeframe: "H1",
      model: "logistic",
      horizon: 5,
      train_fraction: 0.7,
      validation_fraction: 0.15,
      purge_bars: 5,
      embargo_bars: 5,
      include_flat: false,
    });
    expect(screen.getByTestId("ml-run-id")).toHaveTextContent("mlrun123");
    expect(detailCalls).toBe(1);
    expect(screen.getByTestId("ml-run-running")).toBeInTheDocument();

    // The next poll is 2 s later, not a tight loop.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_900);
    });
    expect(detailCalls).toBe(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(detailCalls).toBe(2);
    expect(screen.getByTestId("ml-run-status")).toHaveTextContent("Bitti");

    // Completed is terminal: no further requests even after 10 s.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(detailCalls).toBe(2);
  });

  it("renders validation metrics, split counts and the not-a-trading-result note", async () => {
    handlers["/api/ml/runs"] = () => json([summary()]);
    handlers["/api/ml/runs/mlrun123"] = () => json(completed());

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "mlrun123" }));
    await flush();

    expect(screen.getByTestId("ml-validation-Doğruluk")).toHaveTextContent("%51,60");
    expect(screen.getByTestId("ml-validation-Dengeli doğruluk")).toHaveTextContent("%51,80");
    expect(screen.getByTestId("ml-validation-Log loss")).toHaveTextContent("0,6921");
    expect(screen.getByTestId("ml-validation-Brier")).toHaveTextContent("0,2504");
    expect(screen.getByTestId("ml-validation-ECE")).toHaveTextContent("0,0430");
    expect(screen.getByTestId("ml-validation-MCE")).toHaveTextContent("0,0870");
    expect(screen.getByTestId("ml-count-train")).toHaveTextContent("20.654");
    expect(screen.getByTestId("ml-count-flat_dropped")).toHaveTextContent("7");
    expect(screen.getByTestId("ml-baseline")).toHaveTextContent("%54,3");
    expect(screen.getByTestId("ml-run-facts")).toHaveTextContent("ds-abc123");
    expect(screen.getByTestId("ml-run-facts")).toHaveTextContent("3");
    expect(screen.getByTestId("ml-not-trading")).toHaveTextContent(
      "tahmin kalitesidir, alım satım sonucu değildir",
    );
    // evaluate_test was off, so the test column says so instead of guessing.
    expect(screen.getByTestId("ml-test-absent")).toBeInTheDocument();
    expect(screen.getByTestId("ml-test-Doğruluk")).toHaveTextContent("—");
  });

  it("explains a 409 busy slot instead of pretending the run started", async () => {
    handlers["/api/ml/train"] = () =>
      json({ detail: "another ML job is running (fit mlrun123)" }, 409);

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();

    expect(screen.getByTestId("ml-busy")).toHaveTextContent("Başka bir ML işi çalışıyor");
    expect(screen.getByTestId("ml-busy")).toHaveTextContent("kuyruğa alınmadı");
    expect(screen.queryByTestId("ml-run-id")).not.toBeInTheDocument();
  });

  it("shows a failed run's error and offers no metrics", async () => {
    handlers["/api/ml/train"] = () => json({ run_id: "mlrun123" }, 202);
    handlers["/api/ml/runs/mlrun123"] = () =>
      json(detail({ status: "failed", error: "boom", finished_at: "2026-11-20T10:01:00Z" }));

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();

    expect(screen.getByTestId("ml-run-failed")).toHaveTextContent("Koşu başarısız oldu");
    expect(screen.getByTestId("ml-run-failed")).toHaveTextContent("boom");
    expect(screen.queryByTestId("ml-not-trading")).not.toBeInTheDocument();
  });

  it("cancels a running run, shows cancelling and disables the button honestly", async () => {
    let status = "running";
    handlers["/api/ml/train"] = () => json({ run_id: "mlrun123" }, 202);
    handlers["/api/ml/runs/mlrun123"] = () => json(detail({ status }));
    handlers["/api/ml/runs/mlrun123/control"] = () => {
      status = "cancelling";
      return json({ ok: true });
    };

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "İptal et" }));
    await flush();

    const control = calls.find((call) => call.path === "/api/ml/runs/mlrun123/control");
    expect(control?.body).toEqual({ action: "cancel" });
    expect(screen.getByTestId("ml-run-status")).toHaveTextContent("İptal ediliyor");
    expect(screen.getByTestId("ml-run-cancelling")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "İptal et" })).toBeDisabled();
    expect(screen.getByText(/İptal zaten istendi/)).toBeInTheDocument();
  });

  it("evaluates a completed run and renders hold, stress and regime rows", async () => {
    handlers["/api/ml/runs"] = () => json([summary()]);
    handlers["/api/ml/runs/mlrun123"] = () => json(completed());
    handlers["/api/ml/evaluate"] = () => json({ ...EVALUATION, split: "test" });

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "mlrun123" }));
    await flush();

    fireEvent.change(screen.getByLabelText("Bölünme"), { target: { value: "test" } });
    fireEvent.change(screen.getByLabelText("Maliyet çarpanları"), {
      target: { value: "2, 3" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Değerlendir" }));
    await flush();

    const evaluate = calls.find((call) => call.path === "/api/ml/evaluate");
    expect(evaluate?.body).toEqual({
      run_id: "mlrun123",
      split: "test",
      stress: [2, 3],
      regime: true,
    });

    expect(screen.getByTestId("ml-eval-net_profit")).toHaveTextContent("-$2.002,34");
    expect(screen.getByTestId("ml-stress-x2")).toHaveTextContent("22");
    expect(screen.getByTestId("ml-stress-x3")).toHaveTextContent("18");
    expect(screen.getByTestId("ml-regime-table")).toHaveTextContent("trend_up");
    expect(screen.getByTestId("ml-regime-table")).toHaveTextContent("sideways");
    expect(screen.getByTestId("ml-regime-unmatched")).toHaveTextContent("3");
    expect(screen.getByTestId("ml-evaluation-caption")).toHaveTextContent("Test bölünmesi");
  });

  it("shows an explicit empty regime table instead of a fake bucket", async () => {
    handlers["/api/ml/runs"] = () => json([summary()]);
    handlers["/api/ml/runs/mlrun123"] = () => json(completed());
    handlers["/api/ml/evaluate"] = () =>
      json({ ...EVALUATION, stress: {}, regime: { rows: [], unmatched_trades: 0 } });

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "mlrun123" }));
    await flush();

    fireEvent.change(screen.getByLabelText("Maliyet çarpanları"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Değerlendir" }));
    await flush();

    expect(screen.getByTestId("ml-regime-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("ml-regime-table")).not.toBeInTheDocument();
    expect(screen.getByTestId("ml-stress-empty")).toBeInTheDocument();
  });

  it("rejects a bad stress multiplier before calling the API", async () => {
    handlers["/api/ml/runs"] = () => json([summary()]);
    handlers["/api/ml/runs/mlrun123"] = () => json(completed());

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "mlrun123" }));
    await flush();

    fireEvent.change(screen.getByLabelText("Maliyet çarpanları"), {
      target: { value: "2, 0" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Değerlendir" }));
    await flush();

    expect(screen.getByTestId("ml-stress-error")).toHaveTextContent("0'dan büyük");
    expect(calls.some((call) => call.path === "/api/ml/evaluate")).toBe(false);
  });

  it("explains why evaluate is disabled before a completed run is selected", async () => {
    renderPage();
    await flush();

    expect(screen.getByRole("button", { name: "Değerlendir" })).toBeDisabled();
    expect(screen.getByText("Önce bitmiş bir koşu seçin.")).toBeInTheDocument();
  });

  it("renders the empty runs state without inventing rows", async () => {
    renderPage();
    await flush();

    expect(screen.getByTestId("ml-runs-empty")).toHaveTextContent("Henüz ML koşusu yok");
    expect(screen.queryByRole("table", { name: /ML koşuları/ })).not.toBeInTheDocument();
  });

  it("treats a 404 runs endpoint as an explicit missing state", async () => {
    handlers["/api/ml/runs"] = () => json({ detail: "Not Found" }, 404);

    renderPage();
    await flush();

    expect(screen.getByTestId("ml-runs-missing")).toHaveTextContent(
      "Sunucuda ML koşu listesi yok (404)",
    );
    expect(screen.queryByTestId("ml-runs-empty")).not.toBeInTheDocument();
  });

  it("reports a run that disappeared as 404, not as empty metrics", async () => {
    handlers["/api/ml/runs"] = () => json([summary()]);
    handlers["/api/ml/runs/mlrun123"] = () => json({ detail: "unknown ML run: mlrun123" }, 404);

    renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "mlrun123" }));
    await flush();

    expect(screen.getByTestId("ml-run-missing")).toHaveTextContent("bulunamadı (404)");
    expect(screen.queryByTestId("ml-not-trading")).not.toBeInTheDocument();
  });

  it("stops polling on unmount and clears its timer", async () => {
    let detailCalls = 0;
    handlers["/api/ml/train"] = () => json({ run_id: "mlrun123" }, 202);
    handlers["/api/ml/runs/mlrun123"] = () => {
      detailCalls += 1;
      return json(detail());
    };

    const view = renderPage();
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Koşuyu başlat" }));
    await flush();
    expect(detailCalls).toBe(1);

    view.unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(detailCalls).toBe(1);
  });
});
