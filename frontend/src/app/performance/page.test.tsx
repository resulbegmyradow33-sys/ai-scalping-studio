import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "@/lib/api";

/**
 * `/api/runs` belongs to the RL plan and is absent on the current backend.
 * The hook is replaced with the three states the page must survive: empty
 * data, a 404, and the normal loading hand-off.
 */
const swr = vi.hoisted(() => ({ useSWR: vi.fn() }));

vi.mock("swr", () => ({ default: swr.useSWR }));

const { default: PerformancePage } = await import("./page");

function runsState(
  overrides: Partial<{ data: unknown; error: unknown; isLoading: boolean }> = {},
) {
  return { data: undefined, error: undefined, isLoading: false, ...overrides };
}

beforeEach(() => {
  swr.useSWR.mockReturnValue(runsState());
});

afterEach(cleanup);

describe("PerformancePage", () => {
  it("renders the empty state when there are no runs", () => {
    swr.useSWR.mockReturnValue(runsState({ data: [] }));

    render(<PerformancePage />);

    expect(screen.getByText("Henüz backtest yok")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Simülatöre git" }),
    ).toHaveAttribute("href", "/");
    expect(screen.queryByText("Yükleniyor…")).not.toBeInTheDocument();
  });

  it("treats a 404 as the empty state instead of crashing", () => {
    swr.useSWR.mockReturnValue(runsState({ error: new ApiError(404, "Not Found") }));

    render(<PerformancePage />);

    expect(screen.getByText("Henüz backtest yok")).toBeInTheDocument();
  });

  it("disables SWR retries for the missing endpoint", () => {
    render(<PerformancePage />);

    expect(swr.useSWR).toHaveBeenCalledWith("api/runs", api.getRuns, {
      shouldRetryOnError: false,
    });
  });
});
