import { describe, expect, it } from "vitest";

import { classifyPF, costShare, drawdownSeries } from "./metrics";

describe("metrics helpers", () => {
  it("classifies profit factor", () => {
    expect(classifyPF(1.6)).toBe("good");
    expect(classifyPF(1.1)).toBe("ok");
    expect(classifyPF(0.8)).toBe("bad");
    expect(classifyPF(0)).toBe("bad");
    // The band edges: 1.0 and 1.3 are still "ok", 1.31 is good.
    expect(classifyPF(1.0)).toBe("ok");
    expect(classifyPF(1.3)).toBe("ok");
    expect(classifyPF(1.31)).toBe("good");
    // An infinite factor means wins and no losses: the best case.
    expect(classifyPF(Number.POSITIVE_INFINITY)).toBe("good");
    // Unknown or nonsensical values are not a good factor.
    expect(classifyPF(null)).toBe("bad");
    expect(classifyPF(Number.NaN)).toBe("bad");
    expect(classifyPF(Number.NEGATIVE_INFINITY)).toBe("bad");
  });

  it("computes drawdown from running peak", () => {
    expect(drawdownSeries([100, 120, 90, 110])).toEqual([0, 0, -25, -8.333333333333332]);
  });

  it("reports -100 at zero equity and follows a loss-only curve", () => {
    expect(drawdownSeries([100, 0])).toEqual([0, -100]);
    expect(drawdownSeries([100, 80, 60])).toEqual([0, -20, -40]);
  });

  it("ignores a non-finite mark instead of producing NaN", () => {
    expect(drawdownSeries([100, Number.NaN, 90])).toEqual([0, 0, -10]);
    expect(drawdownSeries([100, Number.POSITIVE_INFINITY, 90])).toEqual([0, 0, -10]);
  });

  it("computes cost share of gross profit", () => {
    expect(costShare({ total_costs: 25, net_profit: 75 } as never)).toBe(0.25);
    expect(costShare({ total_costs: 25, net_profit: 0 } as never)).toBe(1);
  });

  it("handles a zero or negative gross denominator", () => {
    // Gross is exactly zero with costs paid: they are all of it.
    expect(costShare({ total_costs: 25, net_profit: -25 } as never)).toBe(1);
    // Zero gross and no costs: there is no share to report.
    expect(costShare({ total_costs: 0, net_profit: 0 } as never)).toBe(0);
    // Negative gross: still capped at 1 rather than a negative share.
    expect(costShare({ total_costs: 10, net_profit: -30 } as never)).toBe(1);
  });
});
