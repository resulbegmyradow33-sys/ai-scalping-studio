import { describe, expect, it } from "vitest";

import { metadata } from "./layout";

describe("BacktestingLayout metadata", () => {
  it("uses a unique document title", () => {
    expect(metadata.title).toBe("Backtesting — AI Scalping Studio");
  });
});
