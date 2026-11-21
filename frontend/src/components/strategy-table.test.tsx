import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { StrategyTable, WalkForwardTable } from "./strategy-table";

afterEach(cleanup);

describe("StrategyTable", () => {
  it("always exposes the five backend strategy rows", () => {
    render(<StrategyTable />);

    // One header row plus five strategy rows.
    expect(screen.getAllByRole("row")).toHaveLength(6);
    for (const strategy of ["momentum", "reversal", "sweep", "fvg", "trend"]) {
      expect(screen.getByText(strategy)).toBeInTheDocument();
    }
    expect(screen.getAllByText("—")).toHaveLength(20);
  });

  it("merges a supplied row into the frozen order", () => {
    render(
      <StrategyTable
        rows={[
          {
            strategy: "sweep",
            trades: 8,
            winRate: 0.625,
            avgR: 0.3,
            netR: 2.4,
          },
        ]}
      />,
    );

    const row = screen.getByText("sweep").closest("tr");
    expect(row).toHaveTextContent("8");
    expect(row).toHaveTextContent("%62,5");
    expect(row).toHaveTextContent("0,30");
    expect(row).toHaveTextContent("2,40");
  });
});

describe("WalkForwardTable", () => {
  it("renders one explanatory row when the run has no folds", () => {
    render(<WalkForwardTable />);

    expect(screen.getByText("Walk-forward henüz koşulmadı")).toBeInTheDocument();
  });

  it("renders a fold with a signed drawdown", () => {
    render(
      <WalkForwardTable
        folds={[
          {
            fold: 1,
            testStart: "2026-01-05T10:04:00Z",
            testEnd: "2026-02-05T10:04:00Z",
            profitFactor: 1.4,
            maxDrawdown: 0.2,
          },
        ]}
      />,
    );

    const row = screen.getByText("1").closest("tr");
    expect(row).toHaveTextContent("05.01.2026 13:04 – 05.02.2026 13:04");
    expect(row).toHaveTextContent("1,40");
    expect(row).toHaveTextContent("-%20,00");
  });
});
