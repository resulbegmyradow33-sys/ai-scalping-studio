import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";

/**
 * The page has one boundary: the regime hook. Mocking it keeps these tests
 * about rendering the API payload — current axes, real counts, timeline and the
 * empty/error states — instead of about SWR.
 */
const regimeHook = vi.hoisted(() => ({
  useRegime: vi.fn(),
  useTimeframes: vi.fn(),
}));

vi.mock("@/hooks/use-regime", () => regimeHook);

const { default: IntelligencePage } = await import("./page");

const CURRENT = {
  ts: "2026-01-07T23:58:00Z",
  trend: "strong_up",
  trend_strength: "strong",
  volatility: "high",
  expansion: "expanding",
  breakout: "none",
  unusual: false,
  regime: "up_high",
};

const PAYLOAD = {
  symbol: "XAUUSD",
  timeframe: "M1",
  rows: 100,
  current: CURRENT,
  counts: [
    { regime: "up_high", bars: 80, share: 0.8 },
    { regime: "sideways_normal", bars: 20, share: 0.2 },
  ],
  series: [
    { ts: "2026-01-07T23:57:00Z", regime: "sideways_normal", trend: "sideways", volatility: "normal" },
    { ts: "2026-01-07T23:58:00Z", regime: "up_high", trend: "strong_up", volatility: "high" },
  ],
};

function regimeState(overrides: Record<string, unknown> = {}) {
  return { data: PAYLOAD, error: undefined, isLoading: false, ...overrides } as never;
}

beforeEach(() => {
  regimeHook.useRegime.mockReset();
  regimeHook.useTimeframes.mockReset();
  regimeHook.useRegime.mockReturnValue(regimeState());
  regimeHook.useTimeframes.mockReturnValue({
    data: [
      { timeframe: "M1", rule: "1m" },
      { timeframe: "H1", rule: "1h" },
    ],
    error: undefined,
  });
});

afterEach(cleanup);

describe("IntelligencePage", () => {
  it("renders the current axes, real counts and the timeline", () => {
    render(<IntelligencePage />);

    expect(screen.getByRole("heading", { name: "Piyasa Zekâsı" })).toBeInTheDocument();
    expect(screen.getByTestId("current-regime")).toHaveTextContent("up_high");
    expect(screen.getByTestId("current-trend")).toHaveTextContent("strong_up (strong)");
    expect(screen.getByTestId("current-volatility")).toHaveTextContent("high");
    expect(screen.getByTestId("current-expansion")).toHaveTextContent("expanding");
    expect(screen.getByTestId("current-breakout")).toHaveTextContent("none");
    expect(screen.getByTestId("current-unusual")).toHaveTextContent("hayır");

    expect(screen.getByTestId("regime-count-up_high")).toHaveTextContent("80 bar");
    expect(screen.getByTestId("regime-count-up_high")).toHaveTextContent("80.0%");
    expect(screen.getByTestId("regime-count-sideways_normal")).toHaveTextContent("20 bar");

    expect(screen.getAllByTestId("regime-point")).toHaveLength(2);
    expect(screen.queryByTestId("regime-empty")).not.toBeInTheDocument();
    expect(screen.queryByTestId("regime-error")).not.toBeInTheDocument();
  });

  it("offers the timeframe ladder from the API and refetches on change", () => {
    render(<IntelligencePage />);

    const select = screen.getByLabelText("Zaman dilimi");
    expect(select).toHaveValue("M1");
    expect(screen.getByRole("option", { name: "H1" })).toBeInTheDocument();

    fireEvent.change(select, { target: { value: "H1" } });

    expect(regimeHook.useRegime).toHaveBeenLastCalledWith("XAUUSD", "H1");
  });

  it("shows the explicit empty state for a slice without bars", () => {
    regimeHook.useRegime.mockReturnValue(
      regimeState({
        data: {
          symbol: "XAUUSD",
          timeframe: "M1",
          rows: 0,
          current: null,
          counts: [],
          series: [],
        },
      }),
    );

    render(<IntelligencePage />);

    expect(screen.getByTestId("regime-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("regime-current")).not.toBeInTheDocument();
  });

  it("shows a backend failure without claiming the slice is empty", () => {
    regimeHook.useRegime.mockReturnValue(
      regimeState({ data: undefined, error: new ApiError(503, "backend down") }),
    );

    render(<IntelligencePage />);

    expect(screen.getByTestId("regime-error")).toHaveTextContent("backend down");
    expect(screen.queryByTestId("regime-empty")).not.toBeInTheDocument();
  });

  it("does not render data for an empty symbol", () => {
    render(<IntelligencePage />);

    fireEvent.change(screen.getByLabelText("Sembol"), { target: { value: "   " } });

    expect(screen.getByText("Sembol girerek rejimi görüntüle.")).toBeInTheDocument();
    expect(screen.queryByTestId("regime-current")).not.toBeInTheDocument();
  });
});
