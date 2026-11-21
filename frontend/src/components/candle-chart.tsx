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
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type Time,
} from "lightweight-charts";

import { formatChartPrice, formatChartTime } from "@/lib/chart-format";
import { toCandles, toMarkers, type Timeframe } from "@/lib/chart-data";
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

/** Fit only once a dataset can fill the viewport without absurd bar spacing. */
const MIN_BARS_TO_FIT = 30;

/** Stable default so `useMemo`/marker effects do not rerun on every render. */
const EMPTY_TRADES: readonly TradeSummary[] = [];

const DEFAULT_BAR_SPACING = 6;
const DEFAULT_RIGHT_OFFSET = 5;

type MarkerList = ReturnType<typeof toMarkers>;
type MarkerItem = MarkerList[number];

interface RenderedData {
  length: number;
  lastTime: number | null;
  timeframe: Timeframe;
}

function markerEquals(a: MarkerItem, b: MarkerItem): boolean {
  return a.time === b.time && a.position === b.position && a.shape === b.shape && a.text === b.text;
}

/** Cheap bounds check: the stream only appends trades or drops the oldest. */
function sameMarkerBounds(a: MarkerList, b: MarkerList): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  if (a.length === 0) return true;
  return markerEquals(a[0], b[0]) && markerEquals(a[a.length - 1], b[b.length - 1]);
}

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

export function CandleChart({
  bars,
  trades = EMPTY_TRADES,
  position = null,
  timeframe = "M1",
  height = 420,
}: CandleChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const fittedRef = useRef(false);
  const renderedRef = useRef<RenderedData | null>(null);
  const lastMarkersRef = useRef<MarkerList | null>(null);
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
        localization: {
          locale: "tr-TR",
          // The axis must agree with the cards: Istanbul clock, comma prices.
          timeFormatter: formatChartTime,
          priceFormatter: formatChartPrice,
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
          // `timeVisible` alone still labels ticks from the library's locale;
          // this pins every tick mark to the Istanbul clock.
          tickMarkFormatter: (time: Time) => formatChartTime(time),
          // Sensible spacing until the first full dataset is fitted.
          barSpacing: DEFAULT_BAR_SPACING,
          rightOffset: DEFAULT_RIGHT_OFFSET,
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
    renderedRef.current = null;
    lastMarkersRef.current = null;

    setChart(created);
    setSeries(createdSeries);

    return () => {
      resizeObserver?.disconnect();
      priceLinesRef.current = [];
      markersRef.current?.detach();
      markersRef.current = null;
      renderedRef.current = null;
      lastMarkersRef.current = null;
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

    const previous = renderedRef.current;
    const last = candles[candles.length - 1];

    if (!last) {
      if (previous === null || previous.length !== 0) series.setData([]);
      renderedRef.current = { length: 0, lastTime: null, timeframe };
      fittedRef.current = false;
      return;
    }

    // Same length and same last time is a revision of the last candle
    // (wick-only change). Same length with a newer last time means the 5 000
    // bar window shifted: update() would append a candle the source no longer
    // has, so the whole array must be rewritten.
    const revised =
      previous !== null &&
      previous.timeframe === timeframe &&
      previous.length === candles.length &&
      previous.lastTime === last.time;
    const appended =
      previous !== null &&
      previous.timeframe === timeframe &&
      previous.lastTime !== null &&
      candles.length === previous.length + 1 &&
      last.time > previous.lastTime;

    if (revised || appended) {
      series.update(last);
    } else {
      series.setData(candles);
    }

    renderedRef.current = { length: candles.length, lastTime: last.time, timeframe };

    // A new session restarting at 0–1 bars must be able to fit again once it
    // grows; fitting a single bar would lock spacing for the whole session.
    if (candles.length <= 1) fittedRef.current = false;

    if (!fittedRef.current && candles.length >= MIN_BARS_TO_FIT) {
      chart?.timeScale().fitContent();
      fittedRef.current = true;
    }
  }, [series, chart, candles, timeframe]);

  useEffect(() => {
    if (!series) return;
    if (lastMarkersRef.current && sameMarkerBounds(lastMarkersRef.current, markers)) return;
    markersRef.current?.setMarkers(markers);
    lastMarkersRef.current = markers;
  }, [series, markers]);

  // SL/TP lines belong to the open position only; closed positions remove them.
  useEffect(() => {
    if (!series) return;

    for (const line of priceLinesRef.current) series.removePriceLine(line);
    priceLinesRef.current = [];

    if (!position) return;

    if (Number.isFinite(position.sl) && position.sl > 0) {
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

    if (Number.isFinite(position.tp) && position.tp > 0) {
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
