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

describe("chart data edges", () => {
  it("sorts unsorted input", () => {
    const out = toCandles([bar(120, 2, 3, 1, 2.5), bar(60, 1, 2, 0.5, 1.5)], "M1");
    expect(out.map((candle) => candle.time)).toEqual([60, 120]);
  });

  it("merges duplicate timestamps keeping the last close", () => {
    const out = toCandles([bar(60, 10, 12, 9, 11), bar(60, 11, 15, 8, 14)], "M1");
    expect(out).toEqual([{ time: 60, open: 10, high: 15, low: 8, close: 14 }]);
  });

  it("drops bars with unparseable timestamps", () => {
    const out = toCandles(
      [bar(NaN, 1, 2, 1, 1.5), { ts: "not-a-date", o: 1, h: 2, l: 1, c: 1.5 }],
      "M1",
    );
    expect(out).toHaveLength(0);
  });

  it("buckets M15 by epoch seconds", () => {
    const out = toCandles([bar(0, 1, 2, 0.5, 1.5), bar(899, 2, 3, 1, 2.5), bar(900, 3, 4, 2, 3.5)], "M15");
    expect(out.map((candle) => candle.time)).toEqual([0, 900]);
    expect(out[0]).toMatchObject({ open: 1, high: 3, low: 0.5, close: 2.5 });
  });

  it("accepts ISO-string timestamps", () => {
    const out = toCandles(
      [{ ts: "2026-01-05T10:00:00Z", o: 1, h: 2, l: 0.5, c: 1.5 }],
      "M1",
    );
    expect(out[0].time).toBe(Date.parse("2026-01-05T10:00:00Z") / 1000);
  });

  it("labels a non-finite pnl with a dash", () => {
    const m = toMarkers([
      { entry_ts: "2026-01-05T10:00:00Z", exit_ts: "2026-01-05T10:04:00Z", pnl: NaN } as never,
    ]);
    expect(m[1].text).toBe("—");
  });
});
