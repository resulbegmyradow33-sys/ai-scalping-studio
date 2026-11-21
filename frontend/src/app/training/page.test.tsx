import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { SWRConfig } from "swr";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { Metrics, RunDetail, RunSummary } from "@/lib/types";

const state = vi.hoisted(() => ({
  getRuns: vi.fn(),
  getRun: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    api: { ...actual.api, getRuns: state.getRuns, getRun: state.getRun },
  };
});

const { default: TrainingPage } = await import("./page");

const METRICS: Metrics = {
  net_profit: 0,
  profit_factor: null,
  sharpe: 0,
  max_drawdown: 0,
  win_rate: 0,
  expectancy_r: 0,
  trades: 0,
  total_costs: 0,
  cost_ratio: 0,
  avg_bars: 0,
};

function wrapper({ children }: { children: ReactNode }) {
  return (
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      {children}
    </SWRConfig>
  );
}

function run(runId: string, createdAt: string, status = "completed"): RunSummary {
  return { run_id: runId, status, created_at: createdAt, steps: 50, reward_mean: 1.2 };
}

function detail(overrides: Partial<RunDetail> = {}): RunDetail {
  return {
    config: { total_steps: 100 },
    metrics: METRICS,
    status: "completed",
    step: 50,
    total_steps: 100,
    series: {
      reward: [
        { step: 0, value: 0.1 },
        { step: 50, value: 1.2 },
      ],
      episode_length: [
        { step: 0, value: 100 },
        { step: 50, value: 140 },
      ],
      entropy: [
        { step: 0, value: 1.4 },
        { step: 50, value: 0.9 },
      ],
      loss: [
        { step: 0, value: 0.8 },
        { step: 50, value: 0.4 },
      ],
    },
    bandit: [
      { strategy: "momentum", weight: 0.4 },
      { strategy: "reversal", weight: 0.2 },
      { strategy: "sweep", weight: 0.15 },
      { strategy: "fvg", weight: 0.15 },
      { strategy: "trend", weight: 0.1 },
    ],
    ...overrides,
  };
}

beforeEach(() => {
  state.getRuns.mockReset();
  state.getRun.mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("TrainingPage", () => {
  it("treats a 404 from /api/runs as an empty list without retrying", async () => {
    vi.useFakeTimers();
    state.getRuns.mockRejectedValue(new ApiError(404, "Not Found"));

    render(<TrainingPage />, { wrapper });

    // SWR's default retry waits 5 s; ten seconds must still be one request.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    expect(screen.getByText("Henüz eğitim yok")).toBeInTheDocument();
    expect(
      screen.getByText("Eğitim başlatıldığında ölçümler burada görünecek."),
    ).toBeInTheDocument();
    expect(state.getRuns).toHaveBeenCalledTimes(1);
  });

  it("renders the newest run, four curves, five bandit rows and progress", async () => {
    state.getRuns.mockResolvedValue([
      run("a", "2026-01-01T00:00:00Z"),
      run("b", "2026-01-03T00:00:00Z", "running"),
    ]);
    state.getRun.mockResolvedValue(detail());

    const { container } = render(<TrainingPage />, { wrapper });

    expect(await screen.findByRole("button", { name: "b" })).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId("training-progress")).toHaveTextContent("50 / 100"),
    );

    expect(state.getRun).toHaveBeenCalledWith("b");
    expect(container.querySelectorAll("polyline")).toHaveLength(4);
    expect(screen.getByTestId("bandit-weights").querySelectorAll("tbody tr")).toHaveLength(5);

    const runsRegion = screen.getByRole("region", { name: "Eğitim koşuları" });
    expect(runsRegion).toHaveAttribute("tabindex", "0");

    const stop = screen.getByRole("button", { name: "Durdur" });
    expect(stop).toBeDisabled();
    expect(stop).toHaveAttribute("title", "Eğitim altyapısı henüz yok");
  });

  it("shows an unknown denominator and no completion claim without a total", async () => {
    state.getRuns.mockResolvedValue([run("a", "2026-01-01T00:00:00Z")]);
    state.getRun.mockResolvedValue(detail({ config: {}, total_steps: undefined, step: 50 }));

    render(<TrainingPage />, { wrapper });

    await waitFor(() =>
      expect(screen.getByTestId("training-progress")).toHaveTextContent("50 / —"),
    );

    const bar = screen.getByRole("progressbar", { name: "Eğitim ilerlemesi" });
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(bar).not.toHaveAttribute("aria-valuemax");
  });

  it("falls back to the config total when detail.total_steps is zero", async () => {
    state.getRuns.mockResolvedValue([run("a", "2026-01-01T00:00:00Z")]);
    state.getRun.mockResolvedValue(detail({ total_steps: 0, step: 50 }));

    render(<TrainingPage />, { wrapper });

    await waitFor(() =>
      expect(screen.getByTestId("training-progress")).toHaveTextContent("50 / 100"),
    );

    const bar = screen.getByRole("progressbar", { name: "Eğitim ilerlemesi" });
    expect(bar).toHaveAttribute("aria-valuenow", "50");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
  });

  it("clamps the progress bar when step exceeds total", async () => {
    state.getRuns.mockResolvedValue([run("a", "2026-01-01T00:00:00Z")]);
    state.getRun.mockResolvedValue(detail({ step: 150, total_steps: 100 }));

    render(<TrainingPage />, { wrapper });

    await waitFor(() =>
      expect(screen.getByTestId("training-progress")).toHaveTextContent("150 / 100"),
    );

    const bar = screen.getByRole("progressbar", { name: "Eğitim ilerlemesi" });
    expect(bar).toHaveAttribute("aria-valuenow", "100");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
  });

  it("disables resume for a completed run", async () => {
    state.getRuns.mockResolvedValue([run("a", "2026-01-01T00:00:00Z", "completed")]);
    state.getRun.mockResolvedValue(detail({ status: "completed" }));

    render(<TrainingPage />, { wrapper });

    expect(await screen.findByRole("button", { name: "a" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Devam" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Durdur" })).toBeDisabled();
  });
});
