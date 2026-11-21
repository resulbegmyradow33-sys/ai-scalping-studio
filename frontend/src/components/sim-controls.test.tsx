import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SimControls } from "./sim-controls";

const dataStatus = vi.hoisted(() => ({
  useDataStatus: vi.fn(),
  providers: ["synthetic"] as string[],
}));

vi.mock("@/hooks/use-data-status", () => ({
  useDataStatus: dataStatus.useDataStatus,
}));

// vitest runs without globals, so testing-library cannot auto-register cleanup.
afterEach(cleanup);

beforeEach(() => {
  dataStatus.providers = ["synthetic"];
  // Read the array lazily so a test can replace `providers` before rendering.
  dataStatus.useDataStatus.mockImplementation(() => ({
    data: {
      symbol: "XAUUSD",
      rows: 1440,
      first_ts: null,
      last_ts: null,
      providers: dataStatus.providers,
    },
    error: undefined,
    isLoading: false,
    mutate: vi.fn(),
  }));
});

function controls(overrides: Partial<{ status: "idle" | "running" }> = {}) {
  return (
    <SimControls
      status={overrides.status ?? "idle"}
      symbol="XAUUSD"
      onStart={vi.fn()}
      onControl={vi.fn()}
    />
  );
}

describe("SimControls", () => {
  it("starts with the selected speed", () => {
    const onStart = vi.fn();
    render(
      <SimControls status="idle" symbol="XAUUSD" onStart={onStart} onControl={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /100×/ }));
    fireEvent.click(screen.getByRole("button", { name: "Başlat" }));
    expect(onStart).toHaveBeenCalledWith(100);
  });
  it("sends 0 for Max", () => {
    const onStart = vi.fn();
    render(
      <SimControls status="idle" symbol="XAUUSD" onStart={onStart} onControl={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Max" }));
    fireEvent.click(screen.getByRole("button", { name: "Başlat" }));
    expect(onStart).toHaveBeenCalledWith(0);
  });
  it("marks the default speed pressed and moves the mark with the selection", () => {
    render(controls());

    expect(screen.getByRole("button", { name: "1×" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "100×" })).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(screen.getByRole("button", { name: "100×" }));

    expect(screen.getByRole("button", { name: "100×" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "1×" })).toHaveAttribute("aria-pressed", "false");
  });
  it("disables reset only while running", () => {
    const { rerender } = render(controls());
    expect(screen.getByRole("button", { name: "Sıfırla" })).toBeEnabled();

    rerender(controls({ status: "running" }));
    expect(screen.getByRole("button", { name: "Sıfırla" })).toBeDisabled();

    rerender(controls({ status: "idle" }));
    expect(screen.getByRole("button", { name: "Sıfırla" })).toBeEnabled();
  });
  it("disables start while running", () => {
    render(controls({ status: "running" }));
    expect(screen.getByRole("button", { name: "Başlat" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Durdur" })).toBeEnabled();
  });
  it("shows the real provider for the simulator symbol and the no-real-money warning", () => {
    dataStatus.providers = ["synthetic", "csv"];

    render(controls());

    expect(
      screen.getByText("Veri kaynağı: synthetic, csv — Gerçek para değil."),
    ).toBeInTheDocument();
    expect(dataStatus.useDataStatus).toHaveBeenCalledWith("XAUUSD");
  });
  it("falls back to bilinmiyor without a provider", () => {
    dataStatus.providers = [];

    render(controls());

    expect(screen.getByText(/Veri kaynağı: bilinmiyor/)).toBeInTheDocument();
    expect(screen.getByText(/Gerçek para değil/)).toBeInTheDocument();
  });
  it("stops a running episode and can continue it afterwards", () => {
    const onControl = vi.fn();
    const { rerender } = render(
      <SimControls
        status="running"
        symbol="XAUUSD"
        onStart={vi.fn()}
        onControl={onControl}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Durdur" }));
    expect(onControl).toHaveBeenCalledWith("stop");

    rerender(
      <SimControls
        status="stopped"
        symbol="XAUUSD"
        onStart={vi.fn()}
        onControl={onControl}
      />,
    );
    const resume = screen.getByRole("button", { name: "Devam" });
    expect(resume).toBeEnabled();
    fireEvent.click(resume);
    expect(onControl).toHaveBeenCalledWith("resume");
  });
});
