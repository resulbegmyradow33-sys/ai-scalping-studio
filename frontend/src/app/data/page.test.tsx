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
    expect(screen.getByTestId("data-empty")).toBeInTheDocument();
    expect(screen.getAllByText("0 bar")).toHaveLength(3);
  });

  it("shows the stored coverage, but not a per-provider row split", () => {
    statusHook.useDataStatus.mockReturnValue(statusState({ data: SEEDED }));

    render(<DataPage />);

    // The API has no per-provider breakdown, so an active card must not
    // present the symbol total as that provider's own row count.
    expect(screen.getByTestId("provider-rows-synthetic")).toHaveTextContent("—");
    expect(screen.getByTestId("provider-rows-csv")).toHaveTextContent("0 bar");
    expect(screen.getByText(/14\.400 bar/)).toBeInTheDocument();
    expect(screen.getByTestId("provider-range-synthetic")).toHaveTextContent("01.10.2024");
    expect(screen.getByTestId("provider-dot-synthetic")).toHaveAttribute(
      "data-state",
      "ready",
    );
    expect(screen.getByTestId("provider-dot-csv")).toHaveAttribute("data-state", "empty");
    expect(screen.queryByText(/Henüz veri yok/)).not.toBeInTheDocument();
  });

  it("keeps the last known coverage when a poll fails and shows the error as a note", () => {
    statusHook.useDataStatus.mockReturnValue(
      statusState({ data: SEEDED, error: new ApiError(503, "backend down") }),
    );

    render(<DataPage />);

    expect(screen.getByText("Veri durumu alınamadı (backend down).")).toBeInTheDocument();
    expect(screen.getByText(/14\.400 bar/)).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-synthetic")).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-csv")).toBeInTheDocument();
    expect(screen.getByTestId("provider-dot-synthetic")).toHaveAttribute("data-state", "ready");
    expect(screen.queryByTestId("data-empty")).not.toBeInTheDocument();
  });

  it("shows the poll error without claiming the store is empty", () => {
    statusHook.useDataStatus.mockReturnValue(
      statusState({ error: new ApiError(503, "backend down") }),
    );

    render(<DataPage />);

    expect(screen.getByText("Veri durumu alınamadı (backend down).")).toBeInTheDocument();
    expect(screen.getByTestId("provider-card-csv")).toBeInTheDocument();
    expect(screen.queryByTestId("data-empty")).not.toBeInTheDocument();
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

    // Changing the provider while the request is open must not clear the
    // pending guard: the form would otherwise fire a second concurrent call.
    fireEvent.change(screen.getByLabelText("Sağlayıcı"), { target: { value: "dukascopy" } });
    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();
    expect(screen.getByLabelText("Sağlayıcı")).toBeDisabled();

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

  it("requires a bare csv file name and sends it in the body", async () => {
    const download = vi.spyOn(api, "downloadData").mockResolvedValue({ rows: 5 });
    render(<DataPage />);

    fireEvent.change(screen.getByLabelText("Sağlayıcı"), { target: { value: "csv" } });

    const field = screen.getByLabelText("CSV dosya adı");
    expect(field).toBeInTheDocument();
    expect(screen.getByText(/imports\//)).toBeInTheDocument();
    expect(screen.queryByLabelText("Seed")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();

    // A path is rejected; the message waits for blur, not the keystroke.
    fireEvent.change(field, { target: { value: "relative/xauusd.csv" } });
    expect(screen.queryByTestId("path-error")).not.toBeInTheDocument();
    fireEvent.blur(field);
    expect(screen.getByTestId("path-error")).toHaveTextContent(/dosya adı/i);
    expect(field).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();

    fireEvent.change(field, { target: { value: "/tmp/xauusd.csv" } });
    fireEvent.blur(field);
    expect(screen.getByTestId("path-error")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "İndir" })).toBeDisabled();

    fireEvent.change(field, { target: { value: "xauusd_h1.csv" } });
    expect(screen.queryByTestId("path-error")).not.toBeInTheDocument();
    expect(field).toHaveAttribute("aria-invalid", "false");
    expect(screen.getByRole("button", { name: "İndir" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "İndir" }));
    await waitFor(() => expect(download).toHaveBeenCalledTimes(1));
    expect(download.mock.calls[0][0]).toEqual(
      expect.objectContaining({ provider: "csv", path: "xauusd_h1.csv", seed: null }),
    );
  });

  it("surfaces the backend 422 for an unknown csv file name", async () => {
    vi.spyOn(api, "downloadData").mockRejectedValue(
      new ApiError(
        422,
        "no csv file named 'absent.csv' in /srv/data/imports; place the export there and pass its bare file name",
      ),
    );
    render(<DataPage />);

    fireEvent.change(screen.getByLabelText("Sağlayıcı"), { target: { value: "csv" } });
    fireEvent.change(screen.getByLabelText("CSV dosya adı"), {
      target: { value: "absent.csv" },
    });
    fireEvent.click(screen.getByRole("button", { name: "İndir" }));

    await waitFor(() =>
      expect(screen.getByTestId("download-error")).toHaveTextContent(
        "no csv file named 'absent.csv'",
      ),
    );
    expect(screen.getByTestId("download-error")).toHaveTextContent("/srv/data/imports");
    expect(mutate).not.toHaveBeenCalled();
  });

  it("shows provenance fields verbatim for a synthetic dataset", () => {
    statusHook.useDataStatus.mockReturnValue(
      statusState({
        data: {
          ...SEEDED,
          provenance: {
            dataset_id: "XAUUSD:M1:0123456789ab",
            symbol: "XAUUSD",
            timeframe: "M1",
            source: "provider:synthetic",
            synthetic: true,
            rows: 14400,
            first_ts: "2024-10-01T00:00:00Z",
            last_ts: "2024-10-10T23:59:00Z",
            checksum: "0123456789abcdef0123456789abcdef",
            created_at: "2026-01-07T12:30:00Z",
            notes: "",
          },
        },
      }),
    );

    render(<DataPage />);

    expect(screen.getByTestId("provenance-kind")).toHaveTextContent("Sentetik");
    expect(screen.getByTestId("provenance-kind")).toHaveAttribute("data-kind", "synthetic");
    expect(screen.getByTestId("provenance-source")).toHaveTextContent("provider:synthetic");
    expect(screen.getByTestId("provenance-dataset-id")).toHaveTextContent(
      "XAUUSD:M1:0123456789ab",
    );
    expect(screen.getByTestId("provenance-checksum")).toHaveTextContent(
      "0123456789abcdef0123456789abcdef",
    );
    expect(screen.getByTestId("provenance-created-at")).toHaveTextContent(/07\.01\.2026/);
    expect(screen.queryByTestId("provenance-missing")).not.toBeInTheDocument();
  });

  it("marks a real dataset and shows its source", () => {
    statusHook.useDataStatus.mockReturnValue(
      statusState({
        data: {
          ...SEEDED,
          providers: ["dukascopy"],
          provenance: {
            dataset_id: "XAUUSD:M1:fedcba987654",
            symbol: "XAUUSD",
            timeframe: "M1",
            source: "provider:dukascopy",
            synthetic: false,
            rows: 14400,
            first_ts: "2024-10-01T00:00:00Z",
            last_ts: "2024-10-10T23:59:00Z",
            checksum: "fedcba9876543210fedcba9876543210",
            created_at: "2026-01-07T12:30:00Z",
            notes: "tick archive",
          },
        },
      }),
    );

    render(<DataPage />);

    expect(screen.getByTestId("provenance-kind")).toHaveTextContent("Gerçek");
    expect(screen.getByTestId("provenance-kind")).toHaveAttribute("data-kind", "real");
    expect(screen.getByTestId("provenance-source")).toHaveTextContent("provider:dukascopy");
    expect(screen.getByTestId("provenance-notes")).toHaveTextContent("tick archive");
  });

  it("says a store has no provenance instead of inventing one", () => {
    statusHook.useDataStatus.mockReturnValue(statusState({ data: SEEDED }));

    render(<DataPage />);

    expect(screen.getByTestId("provenance-missing")).toHaveTextContent(/köken kaydı yok/i);
    expect(screen.queryByTestId("provenance-dataset-id")).not.toBeInTheDocument();
    expect(screen.queryByTestId("provenance-checksum")).not.toBeInTheDocument();
  });

  it("shows no provenance panel for an empty store", () => {
    render(<DataPage />);

    expect(screen.queryByTestId("provenance-missing")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Veri kökeni" })).not.toBeInTheDocument();
  });
});
