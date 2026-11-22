import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CalibrateResponse, StrategyCard } from "@/lib/types";

import { StrategyCardPanel } from "./strategy-card";

afterEach(cleanup);

const card: StrategyCard = {
  active: true,
  strategy: "flow_momentum",
  timeframe: "H1",
  params: {
    timeframe: "H1",
    flow_quantile: 0.85,
    flow_window: 480,
    sl_atr_mult: 3,
    tp_atr_mult: 6,
    max_bars_in_trade: 48,
  },
  train: {
    trades: 1718,
    expectancy_r: 0.042,
    profit_factor: 1.08,
    win_rate: 0.4,
    max_drawdown: 0.4,
    total_r: 72.6,
  },
  holdout: {
    trades: 215,
    expectancy_r: 0.159,
    profit_factor: 1.32,
    win_rate: 0.46,
    max_drawdown: 0.081,
    total_r: 34.2,
  },
  digest: "abc",
  created_at: "2026-11-22T05:36:40+00:00",
  note: "holdout reused across nightly runs",
};

describe("StrategyCardPanel", () => {
  it("shows the inactive fallback without a card", () => {
    render(<StrategyCardPanel card={{ active: false, strategy: "momentum" }} />);

    expect(screen.getByTestId("strategy-card-state")).toHaveTextContent(
      "kart yok · M1 momentum",
    );
    expect(screen.getByText(/Henüz terfi etmiş bir kural kartı yok/)).toBeInTheDocument();
  });

  it("renders the promoted rule and its holdout evidence", () => {
    render(<StrategyCardPanel card={card} />);

    expect(screen.getByTestId("strategy-card-state")).toHaveTextContent("flow + momentum");
    expect(screen.getByText("üst %15 · 480 bar")).toBeInTheDocument();
    expect(screen.getByText("3 / 6 ATR · 48 bar")).toBeInTheDocument();
    expect(screen.getByTestId("strategy-card-holdout")).toHaveTextContent(
      "+0.159 R · PF 1.32 · DD -%8,10 · 215 işlem",
    );
    expect(screen.getByText("holdout reused across nightly runs")).toBeInTheDocument();
  });

  it("reports a rejected calibration honestly", () => {
    const result = {
      promoted: false,
      reason: "holdout expectancy -0.0100R is not positive",
      digest: "abc",
      evaluated: 27,
      candidate: {
        params: card.params!,
        train: card.train!,
        holdout: card.holdout!,
      },
      incumbent: null,
      card,
    } satisfies CalibrateResponse;
    render(<StrategyCardPanel card={card} result={result} />);

    expect(screen.getByTestId("calibrate-result")).toHaveTextContent(
      "Aday reddedildi — holdout expectancy -0.0100R is not positive",
    );
  });

  it("runs the recalibration action", () => {
    const onCalibrate = vi.fn();
    render(<StrategyCardPanel card={card} onCalibrate={onCalibrate} />);

    fireEvent.click(screen.getByTestId("calibrate-button"));
    expect(onCalibrate).toHaveBeenCalledTimes(1);
  });

  it("disables the button while calibration runs", () => {
    render(<StrategyCardPanel card={card} running onCalibrate={vi.fn()} />);

    expect(screen.getByTestId("calibrate-button")).toBeDisabled();
    expect(screen.getByTestId("calibrate-button")).toHaveTextContent("Kalibre ediliyor…");
  });
});
