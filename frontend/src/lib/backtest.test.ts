import { describe, expect, it } from "vitest";

import { completedRuns, equitySeries, hasMetrics } from "./backtest";
import type { RunSummary } from "./types";

function run(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: "run-1",
    status: "completed",
    created_at: "2026-11-20T10:00:00Z",
    steps: 10,
    reward_mean: 0.1,
    ...overrides,
  };
}

describe("completedRuns", () => {
  it("keeps only completed runs, case-insensitively", () => {
    const out = completedRuns([
      run({ run_id: "a", status: "completed" }),
      run({ run_id: "b", status: "COMPLETED" }),
      run({ run_id: "c", status: "running" }),
      run({ run_id: "d", status: "failed" }),
      run({ run_id: "e", status: "paused" }),
    ]);

    expect(out.map((item) => item.run_id)).toEqual(["a", "b"]);
  });

  it("returns an empty list when nothing finished", () => {
    expect(completedRuns([run({ status: "running" })])).toEqual([]);
  });
});

describe("equitySeries", () => {
  it("keeps finite marks at their original index", () => {
    expect(equitySeries([100, 101.5, null, "x", 99])).toEqual([
      { time: 0, value: 100 },
      { time: 1, value: 101.5 },
      { time: 4, value: 99 },
    ]);
  });

  it("returns empty for a missing, non-array or non-numeric value", () => {
    expect(equitySeries(undefined)).toEqual([]);
    expect(equitySeries(null)).toEqual([]);
    expect(equitySeries({ equity: [1, 2] })).toEqual([]);
    expect(equitySeries([Number.NaN, Number.POSITIVE_INFINITY])).toEqual([]);
  });
});

describe("hasMetrics", () => {
  it("accepts a payload with trades and net profit", () => {
    expect(hasMetrics({ trades: 0, net_profit: 0 })).toBe(true);
  });

  it("rejects a missing, empty or headline-less object", () => {
    expect(hasMetrics(undefined)).toBe(false);
    expect(hasMetrics(null)).toBe(false);
    expect(hasMetrics([])).toBe(false);
    expect(hasMetrics({})).toBe(false);
    expect(hasMetrics({ trades: 3 })).toBe(false);
    expect(hasMetrics({ trades: 3, net_profit: Number.NaN })).toBe(false);
  });
});
