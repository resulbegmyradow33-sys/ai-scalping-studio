import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import type { Metrics } from "@/lib/types";

import { MetricGrid } from "./metric-grid";

afterEach(cleanup);

const METRICS: Metrics = {
  net_profit: 75,
  profit_factor: 1.6,
  sharpe: 1.2,
  max_drawdown: 0.25,
  win_rate: 0.55,
  expectancy_r: 0.4,
  trades: 12,
  total_costs: 25,
  cost_ratio: 0.25,
  avg_bars: 3,
};

describe("MetricGrid", () => {
  it("renders every metric, multiplying fraction fields by 100 once", () => {
    render(<MetricGrid metrics={METRICS} />);

    expect(screen.getByTestId("metric-net-profit")).toHaveTextContent("$75,00");
    expect(screen.getByTestId("metric-profit-factor")).toHaveTextContent("1,60");
    expect(screen.getByTestId("metric-profit-factor")).toHaveClass("text-profit");
    expect(screen.getByTestId("metric-sharpe")).toHaveTextContent("1,20");
    expect(screen.getByTestId("metric-max-drawdown")).toHaveTextContent("-%25,00");
    expect(screen.getByTestId("metric-win-rate")).toHaveTextContent("%55,0");
    expect(screen.getByTestId("metric-expectancy")).toHaveTextContent("0,40");
    expect(screen.getByTestId("metric-trades")).toHaveTextContent("12");
    expect(screen.getByTestId("metric-total-costs")).toHaveTextContent("$25,00");
    expect(screen.getByTestId("metric-cost-share")).toHaveTextContent("%25,0");
  });

  it("shows dashes without a run instead of NaN or a stale value", () => {
    render(<MetricGrid metrics={null} />);

    expect(screen.getByTestId("metric-net-profit")).toHaveTextContent("—");
    expect(screen.getByTestId("metric-profit-factor")).toHaveTextContent("—");
    expect(screen.getByTestId("metric-max-drawdown")).toHaveTextContent("—");
    expect(screen.getByTestId("metric-cost-share")).toHaveTextContent("—");
    expect(screen.getByTestId("metric-profit-factor")).not.toHaveClass("text-loss");
  });

  it("renders an infinite profit factor as a dash, not as a good value", () => {
    render(
      <MetricGrid
        metrics={{ ...METRICS, profit_factor: Number.POSITIVE_INFINITY }}
      />,
    );

    expect(screen.getByTestId("metric-profit-factor")).toHaveTextContent("—");
  });
});
