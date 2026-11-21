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

  it("draws a flat series on the inset zero line without NaN", () => {
    const { container } = render(<DrawdownChart values={[100, 100, 100]} />);

    // 0.75 = half the 1.5 stroke width, so the top edge does not clip it.
    expect(linePath(container)).toBe(
      "M 0.00,0.75 L 50.00,0.75 L 100.00,0.75",
    );
    expect(linePath(container)).not.toMatch(/NaN/);
    expect(container.querySelector("svg")).toHaveAttribute("viewBox", "0 0 100 40");
    expect(container.querySelector("svg")).toHaveAttribute("preserveAspectRatio", "none");
    expect(container.querySelector("[data-testid=drawdown-zero]")).toHaveAttribute(
      "y1",
      "0.75",
    );
  });

  it("puts a full drawdown at the bottom of the chart", () => {
    const { container } = render(<DrawdownChart values={[100, 0]} />);

    expect(linePath(container)).toBe("M 0.00,0.75 L 100.00,40.00");
    expect(
      container.querySelector("[data-testid=drawdown-line]"),
    ).toHaveAttribute("vector-effect", "non-scaling-stroke");
  });

  it("paints a single mark as a circle instead of an invisible path", () => {
    const { container } = render(<DrawdownChart values={[100]} />);

    const circle = container.querySelector("circle");
    expect(circle).not.toBeNull();
    expect(circle).toHaveAttribute("cx", "50.00");
    expect(circle).toHaveAttribute("cy", "0.75");
    expect(circle?.getAttribute("cx")).not.toMatch(/NaN/);
    expect(circle?.getAttribute("cy")).not.toMatch(/NaN/);
    expect(container.querySelector("[data-testid=drawdown-line]")).toBeNull();
  });

  it("downsamples a long series to 64 points, keeping the ends", () => {
    const values = Array.from({ length: 200 }, (_, index) => 100 - index / 2);
    const { container } = render(<DrawdownChart values={values} />);

    const coords = linePath(container).replace(/^M /, "").split(" L ");
    expect(coords).toHaveLength(64);
    expect(coords[0]).toBe("0.00,0.75");
    expect(coords.at(-1)).toMatch(/^100\.00,39\./);
    expect(linePath(container)).not.toMatch(/NaN/);
  });
});
