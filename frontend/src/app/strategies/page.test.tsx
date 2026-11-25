import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";

/**
 * The page reads three boundaries: the regime hook, the strategy card hook and
 * the champion archive hook. Mocking them keeps the tests about rendering real
 * payloads — classification, distribution table, and the explicit gap where the
 * strategy × regime API does not exist yet.
 */
const hooks = vi.hoisted(() => ({
  useRegime: vi.fn(),
  useTimeframes: vi.fn(),
  useStrategyCard: vi.fn(),
  useAutonomyChampions: vi.fn(),
}));

vi.mock("@/hooks/use-regime", () => ({
  useRegime: hooks.useRegime,
  useTimeframes: hooks.useTimeframes,
}));
vi.mock("@/hooks/use-strategy-card", () => ({ useStrategyCard: hooks.useStrategyCard }));
vi.mock("@/hooks/use-autonomy-champions", () => ({
  useAutonomyChampions: hooks.useAutonomyChampions,
}));

const { default: StrategiesPage } = await import("./page");

const PAYLOAD = {
  symbol: "XAUUSD",
  timeframe: "M1",
  rows: 100,
  current: {
    ts: "2026-01-07T23:58:00Z",
    trend: "strong_up",
    trend_strength: "strong",
    volatility: "high",
    expansion: "expanding",
    breakout: "none",
    unusual: false,
    regime: "up_high",
  },
  counts: [
    { regime: "up_high", bars: 80, share: 0.8 },
    { regime: "sideways_normal", bars: 20, share: 0.2 },
  ],
  series: [],
};

const ACTIVE_CARD = {
  active: true,
  strategy: "flow_momentum",
  timeframe: "H1",
  symbol: "XAUUSD",
  params: { timeframe: "H1" },
  created_at: "2026-01-01T00:00:00Z",
};

const INACTIVE_CARD = { active: false, strategy: "flow_momentum", timeframe: "M1" };

const CHAMPIONS = {
  configured: true,
  readable: true,
  items: [
    {
      scope: "xauusd",
      candidate_id: "c0ffee1234567890abcdef",
      mode: "dsl",
      policy_id: "policy-42",
      promoted_at: "2026-01-02T00:00:00Z",
    },
  ],
};

function regimeState(overrides: Record<string, unknown> = {}) {
  return { data: PAYLOAD, error: undefined, isLoading: false, ...overrides } as never;
}

function cardState(overrides: Record<string, unknown> = {}) {
  return {
    card: ACTIVE_CARD,
    error: undefined,
    isLoading: false,
    running: false,
    result: null,
    calibrateError: null,
    recalibrate: vi.fn(),
    ...overrides,
  } as never;
}

function championsState(overrides: Record<string, unknown> = {}) {
  return { data: CHAMPIONS, error: undefined, isLoading: false, ...overrides } as never;
}

beforeEach(() => {
  for (const fn of Object.values(hooks)) fn.mockReset();
  hooks.useRegime.mockReturnValue(regimeState());
  hooks.useTimeframes.mockReturnValue({
    data: [
      { timeframe: "M1", rule: "1m" },
      { timeframe: "H1", rule: "1h" },
    ],
    error: undefined,
  });
  hooks.useStrategyCard.mockReturnValue(cardState());
  hooks.useAutonomyChampions.mockReturnValue(championsState());
});

afterEach(cleanup);

describe("StrategiesPage", () => {
  it("renders the regime distribution from the API as a captioned table", () => {
    render(<StrategiesPage />);

    expect(screen.getByRole("heading", { name: "Strateji Zekâsı" })).toBeInTheDocument();

    const table = screen.getByRole("table", { name: /XAUUSD M1/ });
    expect(table).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Rejim" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Bar" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Pay" })).toBeInTheDocument();

    expect(screen.getByTestId("regime-count-up_high")).toHaveTextContent("80");
    expect(screen.getByTestId("regime-count-up_high")).toHaveTextContent("80,0%");
    expect(screen.getByTestId("regime-count-sideways_normal")).toHaveTextContent("20");
    expect(screen.getByTestId("regime-count-sideways_normal")).toHaveTextContent("20,0%");

    expect(screen.getByTestId("strategy-intel-status")).toHaveTextContent(
      "100 bar sınıflandırıldı, 2 rejim.",
    );
    expect(screen.queryByTestId("regime-empty")).not.toBeInTheDocument();
  });

  it("badges an active card as legacy and the archived champion as dsl", () => {
    render(<StrategiesPage />);

    expect(screen.getByTestId("classification-legacy")).toHaveTextContent("legacy");
    expect(screen.getByTestId("classification-card")).toHaveTextContent("flow_momentum");
    expect(screen.getByTestId("classification-dsl")).toHaveTextContent("dsl");
    expect(screen.getByTestId("champion-row")).toHaveTextContent("xauusd");
    expect(screen.getByTestId("champion-row")).toHaveTextContent("policy-42");
  });

  it("badges a missing card as the legacy fallback", () => {
    hooks.useStrategyCard.mockReturnValue(cardState({ card: INACTIVE_CARD }));

    render(<StrategiesPage />);

    expect(screen.getByTestId("classification-fallback")).toHaveTextContent("fallback");
    expect(screen.getByTestId("classification-card")).toHaveTextContent("NO TRADE");
    expect(screen.queryByTestId("classification-legacy")).not.toBeInTheDocument();
  });

  it("states that strategy × regime is not exposed instead of inventing rows", () => {
    render(<StrategiesPage />);

    const gap = screen.getByTestId("strategy-regime-unavailable");
    expect(gap).toHaveTextContent("henüz API'de açık değil");
    expect(gap).toHaveTextContent("strategy_regime_table");
    expect(gap).toHaveTextContent("POST /api/ml/evaluate");
    expect(screen.queryByTestId("strategy-regime-row")).not.toBeInTheDocument();
  });

  it("shows the explicit empty state for a slice without bars", () => {
    hooks.useRegime.mockReturnValue(
      regimeState({
        data: { symbol: "XAUUSD", timeframe: "M1", rows: 0, current: null, counts: [], series: [] },
      }),
    );

    render(<StrategiesPage />);

    expect(screen.getByTestId("regime-empty")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.getByTestId("strategy-intel-status")).toHaveTextContent(
      "rejim verisi yok",
    );
    // The card classification is independent of the regime slice and still renders.
    expect(screen.getByTestId("classification-legacy")).toBeInTheDocument();
  });

  it("shows a backend failure without claiming the slice is empty", () => {
    hooks.useRegime.mockReturnValue(
      regimeState({ data: undefined, error: new ApiError(503, "backend down") }),
    );

    render(<StrategiesPage />);

    expect(screen.getByTestId("regime-error")).toHaveTextContent("backend down");
    expect(screen.queryByTestId("regime-empty")).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.getByTestId("strategy-intel-status")).toHaveTextContent("backend down");
  });

  it("reports an unreadable champion archive instead of zero champions", () => {
    hooks.useAutonomyChampions.mockReturnValue(
      championsState({
        data: { configured: true, readable: false, items: null },
      }),
    );

    render(<StrategiesPage />);

    expect(screen.getByTestId("champions-unreadable")).toHaveTextContent("okunamadı");
    expect(screen.queryByTestId("classification-dsl-missing")).not.toBeInTheDocument();
  });

  it("offers the timeframe ladder and refetches on change", () => {
    render(<StrategiesPage />);

    const select = screen.getByLabelText("Zaman dilimi");
    expect(select).toHaveValue("M1");
    expect(screen.getByRole("option", { name: "H1" })).toBeInTheDocument();

    fireEvent.change(select, { target: { value: "H1" } });

    expect(hooks.useRegime).toHaveBeenLastCalledWith("XAUUSD", "H1");
  });
});
