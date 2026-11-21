import { describe, expect, it } from "vitest";
import { toCandles, toMarkers } from "./chart-data";

const bar = (ts: number, o: number, h: number, l: number, c: number) => ({ ts, o, h, l, c });

describe("chart data", () => {
  it("maps M1 bars 1:1", () => {
    const out = toCandles([bar(1, 1, 2, 0.5, 1.5)], "M1");
    expect(out).toEqual([{ time: 1, open: 1, high: 2, low: 0.5, close: 1.5 }]);
  });
  it("aggregates to 5m buckets", () => {
    const bars = Array.from({ length: 5 }, (_, i) => bar(i * 60, 10 + i, 12 + i, 9 + i, 11 + i));
    const out = toCandles(bars, "M5");
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ open: 10, high: 16, low: 9, close: 15 });
  });
  it("drops bars with non-finite prices", () => {
    expect(toCandles([bar(1, NaN, 2, 1, 1)], "M1")).toHaveLength(0);
  });
  it("places buy/sell markers at trade times", () => {
    const m = toMarkers([{ entry_ts: "2026-01-05T10:00:00Z", exit_ts: "2026-01-05T10:04:00Z", side: 1, pnl: 12 } as never]);
    expect(m.map((x) => x.position)).toEqual(["belowBar", "aboveBar"]);
    expect(m[0].text).toBe("AL");
    expect(m[1].text).toContain("12");
  });
});
