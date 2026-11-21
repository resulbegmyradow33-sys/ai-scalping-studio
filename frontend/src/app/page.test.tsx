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
}));

vi.mock("@/hooks/use-simulation", () => ({ useSimulation: () => state.simulation }));
vi.mock("@/hooks/use-health", () => ({ useHealth: () => state.health }));

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
    trades: [],
    lastDecision: null,
    riskBlocked: null,
    error: null,
    start: vi.fn(),
    control: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  state.simulation = simulation();
  state.health = { error: undefined, data: { status: "ok", version: "0.1.0", device: "cpu" } };
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

  it("turns the equity curve into a signed, coloured day pnl", () => {
    const { rerender } = render(<SimulatorPage />);
    state.simulation = simulation({ equity: [equity(10_000), equity(10_250)] });
    rerender(<SimulatorPage />);

    const gain = screen.getByTestId("day-pnl-value");
    expect(gain).toHaveTextContent("+%2,50");
    expect(gain).toHaveClass("text-profit");

    state.simulation = simulation({ equity: [equity(10_000), equity(9_800)] });
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

  it("keeps aria-live on the equity value and off the chart", () => {
    state.simulation = simulation({ status: "running", equity: [equity(10_000), equity(10_050)] });

    render(<SimulatorPage />);

    expect(screen.getByTestId("equity-value")).toHaveAttribute("aria-live", "polite");
    expect(screen.getByTestId("candle-chart-surface")).not.toHaveAttribute("aria-live");
  });

  it("wires reset through the control channel", () => {
    state.simulation = simulation({ status: "running" });

    render(<SimulatorPage />);
    fireEvent.click(screen.getByRole("button", { name: "Sıfırla" }));

    expect(state.simulation.control).toHaveBeenCalledWith("reset");
  });
});
