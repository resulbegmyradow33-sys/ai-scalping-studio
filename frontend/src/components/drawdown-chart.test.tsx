import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { DrawdownChart } from "./drawdown-chart";

afterEach(cleanup);

function linePath(container: HTMLElement): string {
  const line = container.querySelector("[data-testid=drawdown-line]");
  if (!line) throw new Error("no drawdown line rendered");
  return line.getAttribute("d") ?? "";
}

describe("DrawdownChart", () => {
  it("renders an empty state instead of an empty svg", () => {
    const { container } = render(<DrawdownChart values={[]} />);

    expect(screen.getByText("Veri yok")).toBeInTheDocument();
    expect(container.querySelector("svg")).toBeNull();
  });

  it("draws a flat series on the zero line without NaN", () => {
    const { container } = render(<DrawdownChart values={[100, 100, 100]} />);

    expect(linePath(container)).toBe("M 0.00,0.00 L 50.00,0.00 L 100.00,0.00");
    expect(linePath(container)).not.toMatch(/NaN/);
    expect(container.querySelector("svg")).toHaveAttribute("viewBox", "0 0 100 40");
    expect(container.querySelector("svg")).toHaveAttribute("preserveAspectRatio", "none");
  });

  it("puts a full drawdown at the bottom of the chart", () => {
    const { container } = render(<DrawdownChart values={[100, 0]} />);

    expect(linePath(container)).toBe("M 0.00,0.00 L 100.00,40.00");
    expect(
      container.querySelector("[data-testid=drawdown-line]"),
    ).toHaveAttribute("vector-effect", "non-scaling-stroke");
  });
});
