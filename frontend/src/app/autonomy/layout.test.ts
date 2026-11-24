import { describe, expect, it } from "vitest";

import { metadata } from "./layout";

describe("autonomy layout", () => {
  it("gives the route its own document title", () => {
    expect(metadata.title).toBe("Otonomi — AI Scalping Studio");
  });
});
