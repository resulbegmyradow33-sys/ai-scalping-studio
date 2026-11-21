import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CandleChart } from "./candle-chart";
import type { BarPayload, TradeOpenPayload } from "@/lib/types";

const mocks = vi.hoisted(() => {
  const series = {
    setData: vi.fn(),
    update: vi.fn(),
    createPriceLine: vi.fn(),
    removePriceLine: vi.fn(),
  };
  const timeScale = { fitContent: vi.fn(), applyOptions: vi.fn() };
  const chart = {
    addSeries: vi.fn(),
    remove: vi.fn(),
    applyOptions: vi.fn(),
    timeScale: vi.fn(),
  };
  const markers = { setMarkers: vi.fn(), detach: vi.fn() };

  return {
    series,
    timeScale,
    chart,
    markers,
    createChart: vi.fn(),
    createSeriesMarkers: vi.fn(),
  };
});

vi.mock("lightweight-charts", () => ({
  CandlestickSeries: { type: "Candlestick" },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createChart: mocks.createChart,
  createSeriesMarkers: mocks.createSeriesMarkers,
}));

const barAt = (index: number): BarPayload => ({
  ts: new Date(Date.UTC(2026, 0, 5, 10, 0, 0) + index * 60_000).toISOString(),
  open: 100 + index,
  high: 101 + index,
  low: 99 + index,
  close: 100.5 + index,
});

const bars = (count: number): BarPayload[] => Array.from({ length: count }, (_, i) => barAt(i));

const position: TradeOpenPayload = {
  side: 1,
  lot: 1,
  entry: 110,
  entry_ts: barAt(0).ts,
  sl: 105,
  tp: 120,
  strategy: "test",
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.chart.addSeries.mockReturnValue(mocks.series);
  mocks.chart.timeScale.mockReturnValue(mocks.timeScale);
  mocks.createChart.mockReturnValue(mocks.chart);
  mocks.createSeriesMarkers.mockReturnValue(mocks.markers);
  mocks.series.createPriceLine.mockImplementation((options: unknown) => ({ options }));
});

afterEach(cleanup);

describe("CandleChart without a canvas surface", () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  });

  it("renders the waiting state", () => {
    render(<CandleChart bars={[]} />);

    expect(screen.getByTestId("candle-chart-surface")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Veri bekleniyor…");
  });

  it("does not crash when bars arrive without a canvas surface", () => {
    expect(() => render(<CandleChart bars={bars(5)} />)).not.toThrow();

    expect(mocks.createChart).not.toHaveBeenCalled();
    expect(screen.getByRole("status")).toHaveTextContent("Veri bekleniyor…");
  });

  it("mounts and unmounts repeatedly without leaving nodes behind", () => {
    for (let index = 0; index < 3; index += 1) {
      const view = render(<CandleChart bars={[barAt(0)]} trades={[]} />);
      expect(screen.getByTestId("candle-chart-surface")).toBeInTheDocument();
      view.unmount();
      expect(view.container).toBeEmptyDOMElement();
    }
  });
});

describe("CandleChart with a canvas surface", () => {
  let disconnectSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    disconnectSpy = vi.fn();
    class FakeResizeObserver {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = disconnectSpy;
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({} as never);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates the chart once, fits after enough bars and appends with update", () => {
    const first = bars(40);
    const { rerender } = render(<CandleChart bars={first} />);

    expect(mocks.createChart).toHaveBeenCalledTimes(1);
    expect(mocks.series.setData).toHaveBeenCalledTimes(1);
    expect(mocks.timeScale.fitContent).toHaveBeenCalledTimes(1);

    rerender(<CandleChart bars={[...first, barAt(40)]} />);

    expect(mocks.createChart).toHaveBeenCalledTimes(1);
    expect(mocks.series.setData).toHaveBeenCalledTimes(1);
    expect(mocks.series.update).toHaveBeenCalledTimes(1);
  });

  it("does not fit until the dataset has enough bars", () => {
    const { rerender } = render(<CandleChart bars={bars(1)} />);

    expect(mocks.series.setData).toHaveBeenCalledTimes(1);
    expect(mocks.timeScale.fitContent).not.toHaveBeenCalled();

    rerender(<CandleChart bars={bars(30)} />);

    expect(mocks.timeScale.fitContent).toHaveBeenCalledTimes(1);
  });

  it("re-arms the fit when the dataset resets", () => {
    const { rerender } = render(<CandleChart bars={bars(40)} />);
    expect(mocks.timeScale.fitContent).toHaveBeenCalledTimes(1);

    rerender(<CandleChart bars={[]} />);
    rerender(<CandleChart bars={bars(40)} />);

    expect(mocks.timeScale.fitContent).toHaveBeenCalledTimes(2);
  });

  it("updates the last candle when only its wick changes", () => {
    const first = bars(40);
    const { rerender } = render(<CandleChart bars={first} />);

    const revised = [...first.slice(0, -1), { ...first[first.length - 1], high: 999 }];
    rerender(<CandleChart bars={revised} />);

    expect(mocks.series.update).toHaveBeenCalledWith(expect.objectContaining({ high: 999 }));
    expect(mocks.series.setData).toHaveBeenCalledTimes(1);
  });

  it("does not redraw markers when only bars change", () => {
    const first = bars(40);
    const { rerender } = render(<CandleChart bars={first} />);

    expect(mocks.markers.setMarkers).toHaveBeenCalledTimes(1);

    rerender(<CandleChart bars={[...first, barAt(40)]} />);

    expect(mocks.markers.setMarkers).toHaveBeenCalledTimes(1);
  });

  it("draws SL/TP for an open position and removes them on close", () => {
    const { rerender } = render(<CandleChart bars={bars(40)} position={position} />);

    expect(mocks.series.createPriceLine).toHaveBeenCalledTimes(2);
    expect(mocks.series.createPriceLine).toHaveBeenCalledWith(
      expect.objectContaining({ price: 105, title: "SL" }),
    );
    expect(mocks.series.createPriceLine).toHaveBeenCalledWith(
      expect.objectContaining({ price: 120, title: "TP" }),
    );

    rerender(<CandleChart bars={bars(40)} position={null} />);

    expect(mocks.series.removePriceLine).toHaveBeenCalledTimes(2);
  });

  it("removes the chart and disconnects the observer on unmount", () => {
    const view = render(<CandleChart bars={bars(40)} />);

    view.unmount();

    expect(mocks.chart.remove).toHaveBeenCalledTimes(1);
    expect(mocks.markers.detach).toHaveBeenCalledTimes(1);
    expect(disconnectSpy).toHaveBeenCalledTimes(1);
  });
});
