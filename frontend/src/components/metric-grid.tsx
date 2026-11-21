/**
 * The nine headline metrics of one backtest.
 *
 * Percentages arrive from the backend as fractions (`0.0098`), so every value
 * is multiplied by 100 exactly once, here. Drawdown is stored as a positive
 * fraction and is rendered with a minus sign: a 25 % drawdown is a loss, not a
 * `+%25` gain.
 */

import { formatMoney, formatNumber, formatPct } from "@/lib/format";
import { classifyPF, costShare, formatFractionPct } from "@/lib/metrics";
import type { Metrics } from "@/lib/types";

import { StatCard } from "./stat-card";
import type { StatTone } from "./stat-card";

const PF_TONE: Record<ReturnType<typeof classifyPF>, StatTone> = {
  good: "profit",
  ok: "neutral",
  bad: "loss",
};

/** One metric by key; `null` covers both "no run" and a non-finite wire value. */
function metric(metrics: Metrics | null, key: keyof Metrics): number | null {
  if (metrics === null) return null;
  const value = metrics[key];
  return Number.isFinite(value) ? value : null;
}

export interface MetricGridProps {
  /** `null` while the run detail is loading or absent: every card shows a dash. */
  metrics: Metrics | null;
}

export function MetricGrid({ metrics }: MetricGridProps) {
  const netProfit = metric(metrics, "net_profit");
  const profitFactor = metric(metrics, "profit_factor");
  const sharpe = metric(metrics, "sharpe");
  const maxDrawdown = metric(metrics, "max_drawdown");
  const winRate = metric(metrics, "win_rate");
  const expectancy = metric(metrics, "expectancy_r");
  const trades = metric(metrics, "trades");
  const totalCosts = metric(metrics, "total_costs");
  const share = metrics === null ? null : costShare(metrics);

  const netTone: StatTone =
    netProfit === null || netProfit === 0 ? "neutral" : netProfit > 0 ? "profit" : "loss";
  // A missing factor is shown grey, not red: there is nothing to classify yet.
  const pfTone: StatTone =
    profitFactor === null ? "neutral" : PF_TONE[classifyPF(profitFactor)];
  const drawdownPct = maxDrawdown === null ? null : -Math.abs(maxDrawdown) * 100;

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-5">
      <StatCard
        label="Net kâr"
        value={formatMoney(netProfit)}
        tone={netTone}
        testId="metric-net-profit"
      />
      <StatCard
        label="Profit Factor"
        value={formatNumber(profitFactor, 2)}
        tone={pfTone}
        testId="metric-profit-factor"
      />
      <StatCard
        label="Sharpe"
        value={formatNumber(sharpe, 2)}
        testId="metric-sharpe"
      />
      <StatCard
        label="Max DD"
        value={formatPct(drawdownPct)}
        tone={maxDrawdown !== null && maxDrawdown > 0 ? "loss" : "neutral"}
        testId="metric-max-drawdown"
      />
      <StatCard
        label="Win rate"
        value={formatFractionPct(winRate, 1)}
        testId="metric-win-rate"
      />
      <StatCard
        label="Expectancy (R)"
        value={formatNumber(expectancy, 2)}
        testId="metric-expectancy"
      />
      <StatCard
        label="İşlem sayısı"
        value={formatNumber(trades, 0)}
        testId="metric-trades"
      />
      <StatCard
        label="Toplam maliyet"
        value={formatMoney(totalCosts)}
        testId="metric-total-costs"
      />
      <StatCard
        label="Maliyet/kâr"
        value={formatFractionPct(share, 1)}
        hint="brüt kârın payı"
        testId="metric-cost-share"
      />
    </div>
  );
}
