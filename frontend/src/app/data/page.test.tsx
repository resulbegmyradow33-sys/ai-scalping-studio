import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "@/lib/api";

/**
 * The status hook and the download call are the page's only two boundaries.
 * Replacing them keeps these tests about the page: empty store, coverage,
 * form validation, the success line and the error line.
 */
const statusHook = vi.hoisted(() => ({ useDataStatus: vi.fn() }));

vi.mock("@/hooks/use-data-status", () => ({
  useDataStatus: statusHook.useDataStatus,
}));

const { default: DataPage } = await import("./page");

const mutate = vi.fn();

const EMPTY = {
  symbol: "XAUUSD",
  rows: 0,
  first_ts: null,
  last_ts: null,
  providers: [] as string[],
};

const SEEDED = {
  symbol: "XAUUSD",
  rows: 14400,
  first_ts: "2024-10-01T00:00:00Z",
  last_ts: "2024-10-10T23:59:00Z",
  providers: ["synthetic"],
};

function statusState(overrides: Record<string, unknown> = {}) {
  return {
    data: EMPTY,
    error: undefined,
    isLoading: false,
    mutate,
    ...overrides,
  } as never;
}

beforeEach(() => {
  mutate.mockReset();
  statusHook.useDataStatus.mockReset();
  statusHook.useDataStatus.mockReturnValue(statusState());
});

afterEach(cleanup);

describe("DataPage", () => {
  it("renders a card per provider and the empty state", () => {
    render(<DataPage />);

    expect(screen.getByRole("heading", { name: "Veri" })).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-synthetic")).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-csv")).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-dukascopy")).toBeInTheDocument();
    expect(screen.getByText(/Henüz veri yok/)).toBeInTheDocument();
    expect(screen.getAllByText("0 bar")).toHaveLength(3);
  });

  it("shows the stored coverage and row count", () => {
    statusHook.useDataStatus.mockReturnValue(statusState({ data: SEEDED }));

    render(<DataPage />);

    expect(screen.getByTestId("provider-rows-synthetic")).toHaveTextContent("14.400 bar");
    expect(screen.getByTestId("provider-range-synthetic")).toHaveTextContent("01.10.2024");
    expect(screen.getByTestId("provider-dot-synthetic")).toHaveAttribute(
      "data-state",
      "ready",
    );
    expect(screen.getByTestId("provider-dot-csv")).toHaveAttribute("data-state", "empty");
    expect(screen.queryByText(/Henüz veri yok/)).not.toBeInTheDocument();
  });

  it("will not submit an empty symbol", () => {
    const download = vi.spyOn(api, "downloadData");
    render(<DataPage />);

    fireEvent.change(screen.getByLabelText("Sembol"), { target: { value: "   " } });
    fireEvent.submit(screen.getByTestId("download-form"));

    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();
    expect(download).not.toHaveBeenCalled();
  });

  it("downloads synthetic data, refreshes the status and reports the rows", async () => {
    const download = vi.spyOn(api, "downloadData").mockResolvedValue({ rows: 1440 });
    render(<DataPage />);

    fireEvent.click(screen.getByRole("button", { name: "İndir" }));

    await waitFor(() =>
      expect(screen.getByTestId("download-rows")).toHaveTextContent("1.440"),
    );
    expect(download).toHaveBeenCalledTimes(1);
    expect(download).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: "synthetic",
        symbol: "XAUUSD",
        seed: 7,
      }),
    );
    const body = download.mock.calls[0][0];
    expect(body.start).toMatch(/T00:00:00Z$/);
    expect(body.end).toMatch(/T23:59:00Z$/);
    expect(mutate).toHaveBeenCalledTimes(1);
  });

  it("disables submit while in flight and clears the previous success line", async () => {
    let resolveSecond: (value: { rows: number }) => void = () => {};
    const download = vi
      .spyOn(api, "downloadData")
      .mockResolvedValueOnce({ rows: 1440 })
      .mockReturnValueOnce(
        new Promise<{ rows: number }>((resolve) => {
          resolveSecond = resolve;
        }),
      );
    render(<DataPage />);

    fireEvent.click(screen.getByRole("button", { name: "İndir" }));
    await waitFor(() =>
      expect(screen.getByTestId("download-rows")).toHaveTextContent("1.440"),
    );

    fireEvent.click(screen.getByRole("button", { name: "İndir" }));

    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();
    expect(screen.queryByTestId("download-rows")).not.toBeInTheDocument();

    resolveSecond({ rows: 120 });
    await waitFor(() =>
      expect(screen.getByTestId("download-rows")).toHaveTextContent("120"),
    );
    expect(download).toHaveBeenCalledTimes(2);
  });

  it("shows the ApiError detail when the download fails", async () => {
    vi.spyOn(api, "downloadData").mockRejectedValue(
      new ApiError(400, "cannot read data source: missing.csv"),
    );
    render(<DataPage />);

    fireEvent.click(screen.getByRole("button", { name: "İndir" }));

    await waitFor(() =>
      expect(screen.getByTestId("download-error")).toHaveTextContent(
        "cannot read data source: missing.csv",
      ),
    );
    expect(screen.getByRole("button", { name: "İndir" })).toBeEnabled();
    expect(mutate).not.toHaveBeenCalled();
    expect(screen.queryByTestId("download-rows")).not.toBeInTheDocument();
  });

  it("swaps the seed field for a path field on the csv provider", () => {
    render(<DataPage />);

    fireEvent.change(screen.getByLabelText("Sağlayıcı"), { target: { value: "csv" } });

    expect(screen.getByLabelText("CSV dosyası")).toBeInTheDocument();
    expect(screen.queryByLabelText("Seed")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();

    fireEvent.change(screen.getByLabelText("CSV dosyası"), {
      target: { value: "/tmp/xauusd.csv" },
    });
    expect(screen.getByRole("button", { name: "İndir" })).toBeEnabled();
  });
});
