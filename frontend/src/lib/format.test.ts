import { describe, expect, it } from "vitest";
import { formatMoney, formatNumber, formatPct, formatTs } from "./format";

describe("tr-TR formatting", () => {
  it("uses dot for thousands and comma for decimals", () => {
    expect(formatNumber(2350.5, 2)).toBe("2.350,50");
  });
  it("formats money with currency", () => {
    expect(formatMoney(10000)).toBe("$10.000,00");
    expect(formatMoney(-10000)).toBe("-$10.000,00");
  });
  it("formats percent with sign", () => {
    expect(formatPct(1.234)).toBe("+%1,23");
    expect(formatPct(-2)).toBe("-%2,00");
  });
  it("renders em dash for null", () => {
    expect(formatNumber(null)).toBe("—");
    expect(formatTs(null)).toBe("—");
  });
  it("formats timestamps as DD.MM.YYYY HH:mm in Istanbul time", () => {
    expect(formatTs("2026-01-05T10:04:00Z")).toBe("05.01.2026 13:04");
    expect(formatTs("2026-01-05T22:30:00Z")).toBe("06.01.2026 01:30");
  });
});
