import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BarPayload, EquityPayload, SimState } from "@/lib/types";

/**
 * The page reads two hooks: the stream and backend health. Both are replaced
 * with plain data so the board can be asserted without a socket or a server.
 */
const state = vi.hoisted(() => ({
  simulation: {} as Record<string, unknown>,
  health: {} as Record<string, unknown>,
  strategy: {} as Record<string, unknown>,
}));

/**
 * The chart mounts against a mocked lightweight-charts so the timeframe test
 * can inspect the payload handed to `setData`; every other test keeps the
 * canvas-surface fallback because `getContext` returns null by default.
 */
const chartState = vi.hoisted(() => {
  const series = {
    setData: vi.fn(),
    update: vi.fn(),
    createPriceLine: vi.fn(),
    removePriceLine: vi.fn(),
  };
  const timeScale = { fitContent: vi.fn(), applyOptions: vi.fn() };
  const chart = {
    addSeries: vi.fn(),
    remove: vi.fn(),
    applyOptions: vi.fn(),
    timeScale: vi.fn(),
  };
  const markers = { setMarkers: vi.fn(), detach: vi.fn() };
  return { series, timeScale, chart, markers, createChart: vi.fn(), createSeriesMarkers: vi.fn() };
});

vi.mock("lightweight-charts", () => ({
  CandlestickSeries: { type: "Candlestick" },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createChart: chartState.createChart,
  createSeriesMarkers: chartState.createSeriesMarkers,
}));

const statusHook = vi.hoisted(() => ({ providers: ["synthetic"] as string[] }));

vi.mock("@/hooks/use-data-status", () => ({
  useDataStatus: () => ({
    data: { symbol: "XAUUSD", rows: 1, first_ts: null, last_ts: null, providers: statusHook.providers },
    error: undefined,
    isLoading: false,
    mutate: vi.fn(),
  }),
}));

vi.mock("@/hooks/use-simulation", () => ({ useSimulation: () => state.simulation }));
vi.mock("@/hooks/use-health", () => ({ useHealth: () => state.health }));
vi.mock("@/hooks/use-strategy-card", () => ({ useStrategyCard: () => state.strategy }));

const { default: SimulatorPage } = await import("./page");

function bar(close: number, minute: number): BarPayload {
  const ts = new Date(Date.UTC(2026, 0, 5, 10, minute, 0)).toISOString();
  return { ts, open: close - 1, high: close + 1, low: close - 2, close };
}

function equity(equityValue: number, balance = equityValue): EquityPayload {
  return { equity: equityValue, balance };
}

function simulation(overrides: Record<string, unknown> = {}) {
  return {
    status: "idle" as SimState,
    seq: 0,
    bars: [] as BarPayload[],
    position: null,
    equity: [] as EquityPayload[],
    equityAnchor: null as number | null,
    trades: [],
    lastDecision: null,
    riskBlocked: null,
    error: null,
    errorStatus: null as number | null,
    start: vi.fn(),
    control: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  state.simulation = simulation();
  state.health = { error: undefined, data: { status: "ok", version: "0.1.0", device: "cpu" } };
  state.strategy = {
    card: { active: false, strategy: "momentum" },
    error: undefined,
    isLoading: false,
    running: false,
    result: null,
    calibrateError: null,
    recalibrate: vi.fn(),
  };
  statusHook.providers = ["synthetic"];
  chartState.chart.addSeries.mockReturnValue(chartState.series);
  chartState.chart.timeScale.mockReturnValue(chartState.timeScale);
  chartState.createChart.mockReturnValue(chartState.chart);
  chartState.createSeriesMarkers.mockReturnValue(chartState.markers);
  chartState.series.createPriceLine.mockImplementation((options: unknown) => ({ options }));
});

afterEach(cleanup);

describe("SimulatorPage", () => {
  it("renders the bar count and last price from the stream", () => {
    state.simulation = simulation({ bars: [bar(2345.67, 0), bar(2341.2, 1)] });

    render(<SimulatorPage />);

    expect(screen.getByTestId("bar-count")).toHaveTextContent("2");
    expect(screen.getByTestId("last-price")).toHaveTextContent("2.341,20");
  });

  it("renders an empty board without data and without crashing", () => {
    render(<SimulatorPage />);

    expect(screen.getByTestId("bar-count")).toHaveTextContent("0");
    expect(screen.getByTestId("last-price")).toHaveTextContent("—");
    expect(screen.getByText("Pozisyon yok")).toBeInTheDocument();
    expect(screen.getByText("—", { selector: "[data-testid=risk-blocked-value]" })).toBeInTheDocument();
  });

  it("turns the equity curve into a signed, coloured session pnl", () => {
    const { rerender } = render(<SimulatorPage />);
    state.simulation = simulation({ equity: [equity(10_000), equity(10_250)], equityAnchor: 10_000 });
    rerender(<SimulatorPage />);

    const gain = screen.getByTestId("day-pnl-value");
    expect(gain).toHaveTextContent("+%2,50");
    expect(gain).toHaveClass("text-profit");

    state.simulation = simulation({ equity: [equity(10_000), equity(9_800)], equityAnchor: 10_000 });
    rerender(<SimulatorPage />);

    const loss = screen.getByTestId("day-pnl-value");
    expect(loss).toHaveTextContent("-%2,00");
    expect(loss).toHaveClass("text-loss");
  });

  it("shows the backend banner when health fails", () => {
    state.health = { error: new Error("down"), data: undefined };

    render(<SimulatorPage />);

    expect(
      screen.getByText("Backend'e ulaşılamıyor (127.0.0.1:8000)"),
    ).toBeInTheDocument();
  });

  it("maps a missing-data start failure to the data page hint", () => {
    state.simulation = simulation({
      error: "no data for XAUUSD (M1); download bars before starting",
    });

    render(<SimulatorPage />);

    expect(
      screen.getByText("Veri yok — Veri ekranından sentetik veri üret."),
    ).toBeInTheDocument();
  });

  it("anchors the pnl to the session's first equity, not the capped buffer head", () => {
    // The 5 000-entry buffer dropped the session's opening frame; `equity[0]`
    // is now mid-session and must not define the baseline.
    state.simulation = simulation({
      equity: [equity(11_000), equity(11_100)],
      equityAnchor: 10_000,
    });

    render(<SimulatorPage />);

    const pnl = screen.getByTestId("day-pnl-value");
    expect(pnl).toHaveTextContent("+%11,00");
    expect(pnl).toHaveClass("text-profit");
  });

  it("labels the session change as Seans PnL in both places", () => {
    render(<SimulatorPage />);

    expect(screen.getByText("Seans PnL", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByText("Seans PnL", { selector: "dt" })).toBeInTheDocument();
    expect(screen.queryByText("Günlük PnL")).not.toBeInTheDocument();
  });

  it("renders exactly 1234 as the bar count", () => {
    const bars = Array.from({ length: 1_234 }, (_, index) => bar(2_300 + index / 100, index));
    state.simulation = simulation({ bars });

    render(<SimulatorPage />);

    expect(screen.getByTestId("bar-count")).toHaveTextContent(/^1234$/);
  });

  it("shows the backend banner when health is not ok even without an error", () => {
    state.health = { error: undefined, data: { status: "degraded", version: "0.1.0", device: "cpu" } };

    render(<SimulatorPage />);

    expect(
      screen.getByText("Backend'e ulaşılamıyor (127.0.0.1:8000)"),
    ).toBeInTheDocument();
  });

  it("uses the 409 status, not the prose, for the missing-data hint", () => {
    state.simulation = simulation({
      error: "Veri yok ama İngilizce bir hata",
      errorStatus: 409,
    });

    render(<SimulatorPage />);

    expect(
      screen.getByText("Veri yok — Veri ekranından sentetik veri üret."),
    ).toBeInTheDocument();
  });

  it("maps another 4xx to a generic Turkish message instead of backend English", () => {
    state.simulation = simulation({
      error: "Unprocessable Entity: speed must be positive",
      errorStatus: 422,
    });

    render(<SimulatorPage />);

    expect(
      screen.getByText("İstek reddedildi — sunucu bu işlemi kabul etmedi."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/speed must be positive/)).not.toBeInTheDocument();
  });

  it("suppresses the start error banner while the health banner is shown", () => {
    state.health = { error: new Error("down"), data: undefined };
    state.simulation = simulation({ error: "fetch failed", errorStatus: 0 });

    render(<SimulatorPage />);

    expect(screen.getByText("Backend'e ulaşılamıyor (127.0.0.1:8000)")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("puts aria-live on the status chip instead of the streaming equity value", () => {
    state.simulation = simulation({ status: "running", equity: [equity(10_000), equity(10_050)] });

    const { rerender } = render(<SimulatorPage />);

    expect(screen.getByTestId("equity-value")).not.toHaveAttribute("aria-live");
    const chip = screen.getByTestId("status-chip");
    expect(chip).toHaveAttribute("aria-live", "polite");
    expect(chip).toHaveTextContent("Çalışıyor");

    state.simulation = simulation({ status: "paused" });
    rerender(<SimulatorPage />);

    expect(screen.getByTestId("status-chip")).toHaveTextContent("Duraklatıldı");
    expect(screen.getByTestId("candle-chart-surface")).not.toHaveAttribute("aria-live");
  });

  it("wires reset through the control channel when not running", () => {
    state.simulation = simulation({ status: "paused" });

    render(<SimulatorPage />);
    fireEvent.click(screen.getByRole("button", { name: "Sıfırla" }));

    expect(state.simulation.control).toHaveBeenCalledWith("reset");
  });
});

describe("SimulatorPage timeframe switch", () => {
  let disconnectSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    disconnectSpy = vi.fn();
    class FakeResizeObserver {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = disconnectSpy;
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({} as never);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const fiveMinuteBars: BarPayload[] = [
    { ts: "2026-01-05T10:00:00Z", open: 100, high: 102, low: 99, close: 101 },
    { ts: "2026-01-05T10:01:00Z", open: 101, high: 103, low: 100, close: 102 },
    { ts: "2026-01-05T10:02:00Z", open: 102, high: 104, low: 101, close: 103 },
    { ts: "2026-01-05T10:03:00Z", open: 103, high: 106, low: 102, close: 105 },
    { ts: "2026-01-05T10:04:00Z", open: 105, high: 107, low: 98, close: 104 },
  ];

  it("collapses five M1 bars in one bucket into a single M5 candle", () => {
    state.simulation = simulation({ bars: fiveMinuteBars });

    render(<SimulatorPage />);

    expect(screen.getByRole("button", { name: "M1" })).toHaveAttribute("aria-pressed", "true");
    expect(chartState.series.setData).toHaveBeenCalledTimes(1);
    expect(chartState.series.setData.mock.calls[0][0]).toHaveLength(5);

    fireEvent.click(screen.getByRole("button", { name: "M5" }));

    const payload = chartState.series.setData.mock.calls.at(-1)?.[0] as Array<{
      time: number;
      open: number;
      high: number;
      low: number;
      close: number;
    }>;

    expect(payload).toHaveLength(1);
    expect(payload[0]).toMatchObject({ open: 100, high: 107, low: 98, close: 104 });
    expect(payload[0].time).toBe(Date.UTC(2026, 0, 5, 10, 0, 0) / 1000);
    expect(screen.getByRole("button", { name: "M5" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "M1" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("heading", { name: "Fiyat (M5)" })).toBeInTheDocument();
  });

  it("labels the chart with the promoted card's timeframe", () => {
    state.strategy = {
      ...state.strategy,
      card: {
        active: true,
        strategy: "flow_momentum",
        params: { timeframe: "H1", flow_quantile: 0.85, flow_window: 480 },
      },
    };

    render(<SimulatorPage />);

    expect(screen.getByText("Fiyat (H1)")).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Mum periyodu" })).not.toBeInTheDocument();
  });

  it("offers all three timeframes", () => {
    render(<SimulatorPage />);

    for (const label of ["M1", "M5", "M15"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
  });
});
