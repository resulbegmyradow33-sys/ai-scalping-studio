/**
 * Axis formatters for the candlestick chart.
 *
 * lightweight-charts draws on a canvas, so the axis text cannot be inspected in
 * jsdom. Keeping the two formatters pure and separate lets the tests assert the
 * exact Istanbul clock and the tr-TR decimal comma without a canvas.
 *
 * Times are epoch seconds (`UTCTimestamp`) in the simulator stream, while the
 * axis must read Europe/Istanbul like every card on the page. Prices use the
 * same tr-TR grouping as `formatNumber` (`2.004,34`), because lightweight-charts
 * would otherwise render the raw `2004.34`.
 */

import type { Time } from "lightweight-charts";

const LOCALE = "tr-TR";
const TIME_ZONE = "Europe/Istanbul";
const PRICE_DIGITS = 2;

const timeFormatter = new Intl.DateTimeFormat(LOCALE, {
  timeZone: TIME_ZONE,
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const priceFormatter = new Intl.NumberFormat(LOCALE, {
  minimumFractionDigits: PRICE_DIGITS,
  maximumFractionDigits: PRICE_DIGITS,
  useGrouping: true,
});

function toDate(time: Time): Date | null {
  if (typeof time === "number") {
    return Number.isFinite(time) ? new Date(time * 1000) : null;
  }

  if (typeof time === "string") {
    const parsed = Date.parse(time);
    return Number.isNaN(parsed) ? null : new Date(parsed);
  }

  if (typeof time === "object" && time !== null) {
    // A business day carries no clock; UTC midnight keeps the day stable
    // before the Istanbul offset is applied.
    const parsed = Date.UTC(time.year, time.month - 1, time.day);
    return Number.isNaN(parsed) ? null : new Date(parsed);
  }

  return null;
}

/** `HH:mm` in Istanbul, e.g. `03:00` for 00:00 UTC. */
export function formatChartTime(time: Time): string {
  const date = toDate(time);
  return date === null ? "" : timeFormatter.format(date);
}

/** `2.004,34` — tr-TR grouping and comma decimals. */
export function formatChartPrice(price: number): string {
  return Number.isFinite(price) ? priceFormatter.format(price) : "";
}
