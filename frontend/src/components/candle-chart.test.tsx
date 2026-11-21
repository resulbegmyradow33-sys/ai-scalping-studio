import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CandleChart } from "./candle-chart";
import type { BarPayload } from "@/lib/types";

const BAR: BarPayload = {
  ts: "2026-01-05T10:00:00Z",
  open: 1,
  high: 2,
  low: 0.5,
  close: 1.5,
};

beforeEach(() => {
  // jsdom has no 2D canvas: force the "no surface" branch even if a future
  // jsdom version ships a ResizeObserver.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});

afterEach(cleanup);

describe("CandleChart", () => {
  it("renders the waiting state without a canvas surface", () => {
    render(<CandleChart bars={[]} />);

    expect(screen.getByTestId("candle-chart-surface")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Veri bekleniyor…");
  });

  it("does not crash when bars arrive without a canvas surface", () => {
    expect(() =>
      render(<CandleChart bars={[BAR, { ...BAR, ts: "2026-01-05T10:01:00Z" }]} />),
    ).not.toThrow();

    expect(screen.getByRole("status")).toHaveTextContent("Veri bekleniyor…");
  });

  it("mounts and unmounts repeatedly without leaving nodes behind", () => {
    for (let index = 0; index < 3; index += 1) {
      const view = render(<CandleChart bars={[BAR]} trades={[]} />);
      expect(screen.getByTestId("candle-chart-surface")).toBeInTheDocument();
      view.unmount();
      expect(view.container).toBeEmptyDOMElement();
    }
  });
});
