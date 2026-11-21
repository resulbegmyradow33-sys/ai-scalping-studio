import { describe, expect, it } from "vitest";

import { classifyPF, costShare, drawdownSeries } from "./metrics";

describe("metrics helpers", () => {
  it("classifies profit factor", () => {
    expect(classifyPF(1.6)).toBe("good");
    expect(classifyPF(1.1)).toBe("ok");
    expect(classifyPF(0.8)).toBe("bad");
    expect(classifyPF(0)).toBe("bad");
    // The backend serialises a non-finite profit factor as `null`.
    expect(classifyPF(null)).toBe("bad");
    expect(classifyPF(Number.POSITIVE_INFINITY)).toBe("bad");
  });

  it("computes drawdown from running peak", () => {
    expect(drawdownSeries([100, 120, 90, 110])).toEqual([0, 0, -25, -8.333333333333332]);
  });

  it("computes cost share of gross profit", () => {
    expect(costShare({ total_costs: 25, net_profit: 75 } as never)).toBe(0.25);
    expect(costShare({ total_costs: 25, net_profit: 0 } as never)).toBe(1);
  });
});
