import { describe, expect, it } from "vitest";

import { formatChartPrice, formatChartTime } from "./chart-format";

/** Epoch seconds for a UTC instant, the unit lightweight-charts consumes. */
const seconds = (hour: number, minute: number): number =>
  Date.UTC(2026, 0, 5, hour, minute, 0) / 1000;

describe("chart axis formatting", () => {
  it("renders time labels in Europe/Istanbul, not UTC", () => {
    // 00:00 UTC is 03:00 in Istanbul (UTC+3, no DST).
    expect(formatChartTime(seconds(0, 0))).toBe("03:00");
    expect(formatChartTime(seconds(2, 0))).toBe("05:00");
  });

  it("rolls the hour over at the Istanbul midnight", () => {
    expect(formatChartTime(seconds(21, 34))).toBe("00:34");
  });

  it("renders prices with tr-TR grouping and comma decimals", () => {
    expect(formatChartPrice(2004.34)).toBe("2.004,34");
    expect(formatChartPrice(2004.3)).toBe("2.004,30");
    expect(formatChartPrice(-12.5)).toBe("-12,50");
  });
});
