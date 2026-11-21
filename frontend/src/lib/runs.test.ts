import { describe, expect, it } from "vitest";

import { pickLatestRun, toSeries } from "./runs";

describe("runs helpers", () => {
  it("picks the newest by created_at", () => {
    const runs = [
      { id: "a", created_at: "2026-01-01T00:00:00Z" },
      { id: "b", created_at: "2026-01-03T00:00:00Z" },
    ] as never;
    expect(pickLatestRun(runs)?.id).toBe("b");
  });

  it("returns null for empty", () => {
    expect(pickLatestRun([])).toBeNull();
  });

  it("sorts training points by step and drops NaN", () => {
    const out = toSeries([
      { step: 2, value: 1 },
      { step: 1, value: NaN },
      { step: 0, value: 3 },
    ]);
    expect(out).toEqual([
      { time: 0, value: 3 },
      { time: 2, value: 1 },
    ]);
  });

  it("keeps the last value for a repeated step and drops non-finite steps", () => {
    const out = toSeries([
      { step: 5, value: 1 },
      { step: 5, value: 2 },
      { step: Infinity, value: 9 },
      { step: 3, value: Infinity },
    ]);
    expect(out).toEqual([{ time: 5, value: 2 }]);
  });
});
