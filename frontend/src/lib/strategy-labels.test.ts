import { describe, expect, it } from "vitest";

import { isLegacyStrategy } from "./strategy-labels";

describe("isLegacyStrategy", () => {
  it("recognises the legacy panel labels", () => {
    expect(isLegacyStrategy("momentum")).toBe(true);
    expect(isLegacyStrategy("flow_momentum")).toBe(true);
    expect(isLegacyStrategy("legacy_baseline:flow_momentum")).toBe(true);
  });

  it("leaves current and absent labels alone", () => {
    expect(isLegacyStrategy("dsl")).toBe(false);
    expect(isLegacyStrategy("no_trade")).toBe(false);
    expect(isLegacyStrategy("")).toBe(false);
    expect(isLegacyStrategy(null)).toBe(false);
    expect(isLegacyStrategy(undefined)).toBe(false);
  });
});
