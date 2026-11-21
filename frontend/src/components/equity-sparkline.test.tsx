import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { EquitySparkline } from "./equity-sparkline";

afterEach(cleanup);

function points(container: HTMLElement): string[] {
  const polyline = container.querySelector("polyline");
  if (!polyline) throw new Error("no polyline rendered");
  return (polyline.getAttribute("points") ?? "").trim().split(/\s+/);
}

describe("EquitySparkline", () => {
  it("renders nothing for fewer than two points", () => {
    const { container } = render(<EquitySparkline values={[10]} />);

    expect(container.querySelector("polyline")).toBeNull();
  });

  it("downsamples a long series to 64 evenly spaced points, keeping the ends", () => {
    const values = Array.from({ length: 5_000 }, (_, index) => index);
    const { container } = render(<EquitySparkline values={values} />);

    const rendered = points(container);
    expect(rendered).toHaveLength(64);
    expect(rendered[0]).toBe("0.00,28.00");
    expect(rendered.at(-1)).toBe("100.00,0.00");
  });

  it("uses the height prop instead of ignoring it", () => {
    const { container } = render(<EquitySparkline values={[1, 2]} height={40} />);

    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("viewBox", "0 0 100 40");
    expect(svg).toHaveStyle({ height: "40px" });
  });
});
