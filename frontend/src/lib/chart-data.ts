/**
 * Chart view data derived from the simulator stream.
 *
 * The backend streams M1 bars; higher timeframes are aggregated here for
 * display only and never fed back into the simulation. Times are epoch
 * **seconds** (`UTCTimestamp`), because lightweight-charts works in seconds
 * while the wire uses ISO strings.
 */

import type { CandlestickData, SeriesMarker, UTCTimestamp } from "lightweight-charts";

import { formatNumber } from "./format";
import type { BarPayload, TradeSummary } from "./types";

/** A bar already normalized for the chart (`o/h/l/c` short names). */
export interface BarMsg {
  ts: number | string;
  o: number;
  h: number;
  l: number;
  c: number;
}

/** Accepted inputs: wire bars (`open/high/low/close`) or chart bars. */
export type ChartBar = BarMsg | BarPayload;

export type Timeframe = "M1" | "M5" | "M15";

/** Bucket sizes in seconds; M1 only merges bars that share a second. */
export const TIMEFRAME_SECONDS: Record<Timeframe, number> = {
  M1: 60,
  M5: 300,
  M15: 900,
};

export type Candle = CandlestickData<UTCTimestamp>;
export type TradeMarker = SeriesMarker<UTCTimestamp>;

const ENTRY_COLOR = "#16a34a";
const PROFIT_COLOR = "#16a34a";
const LOSS_COLOR = "#dc2626";

interface NormalizedBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

function toSeconds(ts: number | string): number | null {
  const seconds = typeof ts === "number" ? ts : Date.parse(ts) / 1000;
  if (!Number.isFinite(seconds)) return null;
  return Math.trunc(seconds);
}

function normalize(bar: ChartBar): NormalizedBar | null {
  const time = toSeconds(bar.ts);
  if (time === null) return null;

  const isChartBar = "o" in bar;
  const open = isChartBar ? bar.o : bar.open;
  const high = isChartBar ? bar.h : bar.high;
  const low = isChartBar ? bar.l : bar.low;
  const close = isChartBar ? bar.c : bar.close;

  if (![open, high, low, close].every(Number.isFinite)) return null;
  return { time, open, high, low, close };
}

/**
 * Convert stream bars to candles for `timeframe`.
 *
 * Aggregation uses epoch-second buckets, so it is timezone independent. Input
 * order is not trusted: bars are sorted by time before folding, and duplicate
 * times (M1 receives one bar per minute but replays may repeat) merge into a
 * single candle instead of making lightweight-charts throw.
 */
export function toCandles(bars: readonly ChartBar[], timeframe: Timeframe): Candle[] {
  const bucket = TIMEFRAME_SECONDS[timeframe];
  const normalized = bars
    .map(normalize)
    .filter((bar): bar is NormalizedBar => bar !== null)
    .sort((a, b) => a.time - b.time);

  const candles: Candle[] = [];
  let current: Candle | null = null;

  for (const bar of normalized) {
    const time =
      timeframe === "M1" ? bar.time : Math.floor(bar.time / bucket) * bucket;

    if (current && current.time === time) {
      current.high = Math.max(current.high, bar.high);
      current.low = Math.min(current.low, bar.low);
      current.close = bar.close;
      continue;
    }

    current = {
      time: time as UTCTimestamp,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    };
    candles.push(current);
  }

  return candles;
}

/** Signed PnL label for the exit marker (`+12,00` / `-3,50`). */
function pnlLabel(pnl: number): string {
  const sign = pnl > 0 ? "+" : "";
  return `${sign}${formatNumber(pnl)}`;
}

/**
 * Entry/exit markers for closed trades.
 *
 * The result is sorted ascending by time, which lightweight-charts requires.
 * Entry and exit of one trade share their natural order; equal timestamps keep
 * the entry first because `Array.prototype.sort` is stable.
 */
export function toMarkers(trades: readonly TradeSummary[]): TradeMarker[] {
  const markers: TradeMarker[] = [];

  for (const trade of trades) {
    const entryTime = toSeconds(trade.entry_ts);
    const exitTime = toSeconds(trade.exit_ts);

    if (entryTime !== null) {
      markers.push({
        time: entryTime as UTCTimestamp,
        position: "belowBar",
        shape: "arrowUp",
        color: ENTRY_COLOR,
        text: "AL",
      });
    }

    if (exitTime !== null) {
      markers.push({
        time: exitTime as UTCTimestamp,
        position: "aboveBar",
        shape: "arrowDown",
        color: Number.isFinite(trade.pnl) && trade.pnl < 0 ? LOSS_COLOR : PROFIT_COLOR,
        text: pnlLabel(trade.pnl),
      });
    }
  }

  return markers.sort((a, b) => (a.time as number) - (b.time as number));
}
