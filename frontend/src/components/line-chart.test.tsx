import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { LineChart } from "./line-chart";

afterEach(cleanup);

function coords(container: HTMLElement): string[] {
  const polyline = container.querySelector("polyline");
  if (!polyline) throw new Error("no polyline rendered");
  return (polyline.getAttribute("points") ?? "").trim().split(/\s+/);
}

describe("LineChart", () => {
  it("renders an empty state instead of an empty svg", () => {
    const { container } = render(<LineChart title="Ödül" points={[]} />);

    expect(screen.getByText("Veri yok")).toBeInTheDocument();
    expect(container.querySelector("svg")).toBeNull();
  });

  it("drops non-finite values before drawing", () => {
    const { container } = render(
      <LineChart
        title="Kayıp"
        points={[
          { time: 0, value: Number.NaN },
          { time: 1, value: 2 },
          { time: 2, value: Number.POSITIVE_INFINITY },
          { time: 3, value: 4 },
        ]}
      />,
    );

    expect(coords(container)).toHaveLength(2);
  });

  it("paints a single point as a circle instead of an invisible polyline", () => {
    const { container } = render(
      <LineChart title="Entropi" points={[{ time: 10, value: 1 }]} />,
    );

    const circle = container.querySelector("circle");
    expect(circle).not.toBeNull();
    expect(circle?.getAttribute("cx")).not.toMatch(/NaN/);
    expect(circle?.getAttribute("cy")).not.toMatch(/NaN/);
    expect(container.querySelector("polyline")).toBeNull();
  });

  it("positions points by time, not by array index", () => {
    const { container } = render(
      <LineChart
        title="Ödül"
        points={[
          { time: 0, value: 0 },
          { time: 1, value: 1 },
          { time: 100, value: 0 },
        ]}
      />,
    );

    const rendered = coords(container);
    expect(rendered[0]).toBe("0.00,40.00");
    expect(rendered[1]).toBe("1.00,0.00");
    expect(rendered[2]).toBe("100.00,40.00");
  });

  it("centres a flat series on the middle line", () => {
    const { container } = render(
      <LineChart
        title="Entropi"
        points={[
          { time: 0, value: 5 },
          { time: 10, value: 5 },
        ]}
      />,
    );

    expect(coords(container)).toEqual(["0.00,20.00", "100.00,20.00"]);
  });

  it("exposes the title through the svg's accessible name", () => {
    const { container } = render(
      <LineChart title="Bölüm uzunluğu" points={[{ time: 0, value: 1 }, { time: 1, value: 2 }]} />,
    );

    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("viewBox", "0 0 100 40");
    expect(svg).toHaveAttribute("preserveAspectRatio", "none");
    expect(container.querySelector("polyline")).toHaveAttribute(
      "vector-effect",
      "non-scaling-stroke",
    );
    expect(screen.getByTitle("Bölüm uzunluğu")).toBeInTheDocument();
  });
});
