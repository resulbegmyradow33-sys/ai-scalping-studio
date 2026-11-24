import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";

/**
 * The panel's only boundary is the quality hook; the timeframe ladder comes
 * from the shared timeframes hook. Mocking both keeps these tests about the
 * rendered report: verdicts, real counts, nulls that must not become zeros,
 * configured:false, provenance and the verbatim 422 detail.
 */
const qualityHook = vi.hoisted(() => ({ useDataQuality: vi.fn() }));
const regimeHook = vi.hoisted(() => ({ useTimeframes: vi.fn() }));

vi.mock("@/hooks/use-data-quality", () => qualityHook);
vi.mock("@/hooks/use-regime", () => regimeHook);

const { DataQualityPanel } = await import("./data-quality-panel");

const SYNTHETIC = {
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
};

const CLEAN = {
  symbol: "XAUUSD",
  timeframe: "M1",
  configured: true,
  status: "ok" as const,
  status_rule: "clean",
  status_detail: "no validation issues",
  rows: 14400,
  first_ts: "2024-10-01T00:00:00Z",
  last_ts: "2024-10-10T23:59:00Z",
  duplicates: 0,
  gaps: 0,
  ohlc_violations: 0,
  crossed_quotes: 0,
  negative_spread: 0,
  negative_volume: 0,
  outliers: 0,
  outliers_checked: true,
  provenance: SYNTHETIC,
  thresholds: {
    gap_threshold_minutes: 10,
    outlier_return_z: 8,
    outlier_volume_z: 8,
    outlier_min_rows: 240,
  },
  issues: [],
};

function qualityState(overrides: Record<string, unknown> = {}) {
  return { data: CLEAN, error: undefined, isLoading: false, mutate: vi.fn(), ...overrides } as never;
}

beforeEach(() => {
  qualityHook.useDataQuality.mockReset();
  regimeHook.useTimeframes.mockReset();
  qualityHook.useDataQuality.mockReturnValue(qualityState());
  regimeHook.useTimeframes.mockReturnValue({
    data: [
      { timeframe: "M1", rule: "1m" },
      { timeframe: "H1", rule: "1h" },
    ],
    error: undefined,
  });
});

afterEach(cleanup);

describe("DataQualityPanel", () => {
  it("renders an ok report with real counts, range, provenance and thresholds", () => {
    render(<DataQualityPanel symbol="XAUUSD" />);

    expect(screen.getByRole("heading", { name: "Veri kalitesi" })).toBeInTheDocument();
    expect(screen.getByTestId("quality-status")).toHaveTextContent("Sağlıklı");
    expect(screen.getByTestId("quality-status")).toHaveAttribute("data-status", "ok");
    expect(screen.getByTestId("quality-detail")).toHaveTextContent("no validation issues");
    expect(screen.getByTestId("quality-count-gaps")).toHaveTextContent("0");
    expect(screen.getByTestId("quality-count-outliers")).toHaveTextContent("0");
    expect(screen.getByTestId("quality-first-ts")).toHaveTextContent(/01\.10\.2024/);
    expect(screen.getByTestId("quality-last-ts")).toHaveTextContent(/11\.10\.2024/);
    expect(screen.getByTestId("quality-provenance")).toHaveTextContent("Sentetik");
    expect(screen.getByTestId("quality-provenance")).toHaveAttribute("data-kind", "synthetic");
    expect(screen.getByTestId("quality-source")).toHaveTextContent("provider:synthetic");
    expect(screen.getByTestId("quality-thresholds")).toHaveTextContent("boşluk ≥ 10 dk");
    expect(screen.getByTestId("quality-thresholds")).toHaveTextContent("outlier min 240 satır");
    expect(screen.queryByTestId("quality-null-note")).not.toBeInTheDocument();

    // The counters are a real table with a caption and scoped headers.
    const table = screen.getByRole("table", { name: /Doğrulama sayaçları — XAUUSD M1/ });
    expect(table).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader")).toHaveLength(2);
    expect(screen.getByRole("rowheader", { name: "Zaman boşlukları" })).toBeInTheDocument();
    expect(screen.getByTestId("quality-load")).toHaveAttribute("role", "status");
    expect(screen.getByTestId("quality-load")).toHaveAttribute("aria-live", "polite");
    expect(screen.getByTestId("quality-load")).toHaveTextContent("Güncel");
  });

  it("shows a warn verdict and the deciding rule", () => {
    qualityHook.useDataQuality.mockReturnValue(
      qualityState({
        data: { ...CLEAN, status: "warn", status_rule: "time_gaps", gaps: 3, outliers: 2 },
      }),
    );

    render(<DataQualityPanel symbol="XAUUSD" />);

    expect(screen.getByTestId("quality-status")).toHaveTextContent("Uyarı");
    expect(screen.getByTestId("quality-status")).toHaveAttribute("data-status", "warn");
    expect(screen.getByTestId("quality-rule")).toHaveTextContent("time_gaps");
    expect(screen.getByTestId("quality-count-gaps")).toHaveTextContent("3");
    expect(screen.getByTestId("quality-count-outliers")).toHaveTextContent("2");
  });

  it("shows a fail verdict for a structural error", () => {
    qualityHook.useDataQuality.mockReturnValue(
      qualityState({
        data: {
          ...CLEAN,
          status: "fail",
          status_rule: "duplicate_timestamps",
          status_detail: "2 duplicate timestamps",
          duplicates: 2,
        },
      }),
    );

    render(<DataQualityPanel symbol="XAUUSD" />);

    expect(screen.getByTestId("quality-status")).toHaveTextContent("Hata");
    expect(screen.getByTestId("quality-status")).toHaveAttribute("data-status", "fail");
    expect(screen.getByTestId("quality-rule")).toHaveTextContent("duplicate_timestamps");
    expect(screen.getByTestId("quality-count-duplicates")).toHaveTextContent("2");
  });

  it("renders a null outlier count as — with the floor explanation, not zero", () => {
    qualityHook.useDataQuality.mockReturnValue(
      qualityState({
        data: { ...CLEAN, outliers: null, outliers_checked: false, rows: 120 },
      }),
    );

    render(<DataQualityPanel symbol="XAUUSD" />);

    const outliers = screen.getByTestId("quality-count-outliers");
    expect(outliers).toHaveTextContent("—");
    expect(outliers).toHaveAttribute("data-null", "true");
    expect(outliers).not.toHaveTextContent("0");
    // The structural counts did run and stay real zeros.
    expect(screen.getByTestId("quality-count-gaps")).toHaveTextContent("0");
    expect(screen.getByTestId("quality-count-gaps")).toHaveAttribute("data-null", "false");
    expect(screen.getByTestId("quality-null-note")).toHaveTextContent(/eşiğin altında/);
    expect(screen.getByTestId("quality-null-note")).toHaveTextContent(/sıfır değil/);
  });

  it("renders configured:false as not checked instead of a fail verdict", () => {
    qualityHook.useDataQuality.mockReturnValue(
      qualityState({
        data: {
          ...CLEAN,
          configured: false,
          status: "fail",
          status_rule: "store_empty",
          status_detail: "no stored M1 bars for XAUUSD",
          rows: 0,
          first_ts: null,
          last_ts: null,
          duplicates: null,
          gaps: null,
          ohlc_violations: null,
          crossed_quotes: null,
          negative_spread: null,
          negative_volume: null,
          outliers: null,
          outliers_checked: false,
          provenance: null,
        },
      }),
    );

    render(<DataQualityPanel symbol="XAUUSD" />);

    expect(screen.getByTestId("quality-status")).toHaveTextContent("Yapılandırılmadı");
    expect(screen.getByTestId("quality-status")).toHaveAttribute("data-status", "unconfigured");
    expect(screen.queryByText("Hata")).not.toBeInTheDocument();
    expect(screen.getByTestId("quality-unconfigured")).toHaveTextContent(/doğrulama yapılmadı/);
    expect(screen.getByTestId("quality-count-gaps")).toHaveTextContent("—");
    expect(screen.getByTestId("quality-count-duplicates")).toHaveTextContent("—");
    expect(screen.getByTestId("quality-count-outliers")).toHaveTextContent("—");
    expect(screen.getByTestId("quality-null-note")).toHaveTextContent(/kontrol edilmedi/);
    expect(screen.getByTestId("quality-thresholds")).toBeInTheDocument();
    expect(screen.getByTestId("quality-provenance")).toHaveTextContent("Köken kaydı yok");
  });

  it("shows the server's 422 detail verbatim and no invented counts", () => {
    qualityHook.useDataQuality.mockReturnValue(
      qualityState({ data: undefined, error: new ApiError(422, "unknown symbol 'BTCUSD'") }),
    );

    render(<DataQualityPanel symbol="BTCUSD" />);

    expect(screen.getByTestId("quality-error")).toHaveTextContent("unknown symbol 'BTCUSD'");
    expect(screen.getByTestId("quality-error")).toHaveAttribute("role", "alert");
    expect(screen.getByTestId("quality-load")).toHaveTextContent("Hata");
    expect(screen.queryByTestId("quality-count-gaps")).not.toBeInTheDocument();
    expect(screen.queryByTestId("quality-loading")).not.toBeInTheDocument();
  });

  it("marks a real dataset's source and admits a missing sidecar", () => {
    qualityHook.useDataQuality.mockReturnValue(
      qualityState({
        data: {
          ...CLEAN,
          provenance: { ...SYNTHETIC, synthetic: false, source: "provider:dukascopy" },
        },
      }),
    );

    const { unmount } = render(<DataQualityPanel symbol="XAUUSD" />);
    expect(screen.getByTestId("quality-provenance")).toHaveTextContent("Gerçek");
    expect(screen.getByTestId("quality-provenance")).toHaveAttribute("data-kind", "real");
    expect(screen.getByTestId("quality-source")).toHaveTextContent("provider:dukascopy");
    unmount();

    qualityHook.useDataQuality.mockReturnValue(
      qualityState({ data: { ...CLEAN, provenance: null } }),
    );
    render(<DataQualityPanel symbol="XAUUSD" />);
    expect(screen.getByTestId("quality-provenance")).toHaveTextContent("Köken kaydı yok");
    expect(screen.queryByTestId("quality-source")).not.toBeInTheDocument();
  });

  it("offers the API timeframe ladder and refetches on change", () => {
    render(<DataQualityPanel symbol="XAUUSD" />);

    expect(qualityHook.useDataQuality).toHaveBeenLastCalledWith("XAUUSD", "M1");
    const select = screen.getByLabelText("Zaman dilimi");
    expect(select).toHaveValue("M1");
    expect(screen.getByRole("option", { name: "H1" })).toBeInTheDocument();

    fireEvent.change(select, { target: { value: "H1" } });

    expect(qualityHook.useDataQuality).toHaveBeenLastCalledWith("XAUUSD", "H1");
  });

  it("does not fetch or invent a report without a symbol", () => {
    render(<DataQualityPanel symbol="   " />);

    expect(screen.getByText("Sembol girerek kalite raporunu görüntüle.")).toBeInTheDocument();
    expect(qualityHook.useDataQuality).toHaveBeenCalledWith("", "M1");
    expect(screen.queryByTestId("quality-count-gaps")).not.toBeInTheDocument();
  });
});
