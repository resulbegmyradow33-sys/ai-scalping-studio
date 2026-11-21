"use client";

/**
 * Candlestick chart for the simulator stream.
 *
 * lightweight-charts renders into a canvas, so the component is mounted only
 * after a surface check: on the server, in jsdom, or anywhere `getContext`
 * returns null it falls back to the "Veri bekleniyor…" placeholder instead of
 * throwing. The chart itself is created once per mount; data, markers and
 * price lines are applied by separate effects so a trade update never forces a
 * full `setData`.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  LineStyle,
  type CandlestickData,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";

import { toCandles, toMarkers, type Timeframe, type TradeMarker } from "@/lib/chart-data";
import type { BarPayload, TradeOpenPayload, TradeSummary } from "@/lib/types";

export interface CandleChartProps {
  bars: readonly BarPayload[];
  trades?: readonly TradeSummary[];
  position?: TradeOpenPayload | null;
  timeframe?: Timeframe;
  height?: number;
}

const COLORS = {
  background: "#0b0f17",
  text: "#94a3b8",
  grid: "#1e293b",
  border: "#1e293b",
  profit: "#16a34a",
  loss: "#dc2626",
  sl: "#dc2626",
  tp: "#16a34a",
} as const;

/**
 * A canvas surface is required by lightweight-charts; jsdom and SSR have none.
 * ResizeObserver is checked first because the chart relies on it for sizing and
 * because probing `getContext` inside jsdom logs a noisy "not implemented".
 */
export function canUseChartSurface(): boolean {
  if (typeof window === "undefined" || typeof document === "undefined") return false;
  if (typeof ResizeObserver === "undefined") return false;
  if (typeof HTMLCanvasElement === "undefined") return false;

  try {
    return document.createElement("canvas").getContext("2d") !== null;
  } catch {
    return false;
  }
}

function candleKey(candles: CandlestickData<UTCTimestamp>[], timeframe: Timeframe): string {
  const last = candles[candles.length - 1];
  if (!last) return `${timeframe}:0`;
  return `${timeframe}:${candles.length}:${last.time}:${last.close}`;
}

export function CandleChart({
  bars,
  trades = [],
  position = null,
  timeframe = "M1",
  height = 420,
}: CandleChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const fittedRef = useRef(false);
  const dataKeyRef = useRef<string>("");
  const heightRef = useRef(height);
  heightRef.current = height;

  const [supported, setSupported] = useState(false);
  const [chart, setChart] = useState<IChartApi | null>(null);
  const [series, setSeries] = useState<ISeriesApi<"Candlestick"> | null>(null);

  const candles = useMemo(() => toCandles(bars, timeframe), [bars, timeframe]);
  const markers = useMemo(() => toMarkers(trades), [trades]);

  // Probe once on the client: the server and jsdom both render the placeholder.
  useEffect(() => {
    setSupported(canUseChartSurface());
  }, []);

  useEffect(() => {
    if (!supported) return;

    const container = containerRef.current;
    if (!container) return;

    let created: IChartApi;
    try {
      created = createChart(container, {
        width: container.clientWidth || 0,
        height: heightRef.current,
        layout: {
          background: { color: COLORS.background },
          textColor: COLORS.text,
          attributionLogo: false,
        },
        grid: {
          vertLines: { color: COLORS.grid },
          horzLines: { color: COLORS.grid },
        },
        rightPriceScale: { borderColor: COLORS.border },
        timeScale: {
          borderColor: COLORS.border,
          timeVisible: true,
          secondsVisible: false,
        },
        crosshair: {
          vertLine: { color: COLORS.text, labelBackgroundColor: COLORS.grid },
          horzLine: { color: COLORS.text, labelBackgroundColor: COLORS.grid },
        },
      });
    } catch {
      // A partial canvas implementation can still fail inside the library.
      // Keep the placeholder instead of breaking the page.
      setSupported(false);
      return;
    }

    const createdSeries = created.addSeries(CandlestickSeries, {
      upColor: COLORS.profit,
      downColor: COLORS.loss,
      borderVisible: false,
      wickUpColor: COLORS.profit,
      wickDownColor: COLORS.loss,
    });

    markersRef.current = createSeriesMarkers(createdSeries, []);

    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            created.applyOptions({ width: Math.floor(entry.contentRect.width) });
          });

    if (resizeObserver && container) resizeObserver.observe(container);

    fittedRef.current = false;
    dataKeyRef.current = "";

    setChart(created);
    setSeries(createdSeries);

    return () => {
      resizeObserver?.disconnect();
      priceLinesRef.current = [];
      markersRef.current?.detach();
      markersRef.current = null;
      setChart(null);
      setSeries(null);
      created.remove();
    };
  }, [supported]);

  // Height is a layout concern: apply it without rebuilding the chart.
  useEffect(() => {
    if (!chart) return;
    chart.applyOptions({ height });
  }, [chart, height]);

  useEffect(() => {
    if (!series) return;

    const key = candleKey(candles, timeframe);
    if (key === dataKeyRef.current) return;
    dataKeyRef.current = key;

    series.setData(candles);

    if (!fittedRef.current && candles.length > 0) {
      chart?.timeScale().fitContent();
      fittedRef.current = true;
    }
  }, [series, chart, candles, timeframe]);

  useEffect(() => {
    if (!series) return;
    markersRef.current?.setMarkers(markers as SeriesMarker<Time>[]);
  }, [series, markers]);

  // SL/TP lines belong to the open position only; closed positions remove them.
  useEffect(() => {
    if (!series) return;

    for (const line of priceLinesRef.current) series.removePriceLine(line);
    priceLinesRef.current = [];

    if (!position) return;

    if (Number.isFinite(position.sl)) {
      priceLinesRef.current.push(
        series.createPriceLine({
          price: position.sl,
          color: COLORS.sl,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: "SL",
        }),
      );
    }

    if (Number.isFinite(position.tp)) {
      priceLinesRef.current.push(
        series.createPriceLine({
          price: position.tp,
          color: COLORS.tp,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: "TP",
        }),
      );
    }
  }, [series, position]);

  const hasCandles = candles.length > 0;
  const showEmpty = !supported || !hasCandles;

  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-slate-800 bg-surface"
      style={{ height }}
    >
      <div
        ref={containerRef}
        role="img"
        aria-label={`${timeframe} mum grafiği`}
        data-testid="candle-chart-surface"
        className={[
          "absolute inset-0 transition-opacity duration-500 ease-out motion-reduce:transition-none",
          hasCandles ? "opacity-100" : "opacity-0",
        ].join(" ")}
      />
      {showEmpty && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <p role="status" className="text-sm text-slate-400">
            Veri bekleniyor…
          </p>
        </div>
      )}
    </div>
  );
}
