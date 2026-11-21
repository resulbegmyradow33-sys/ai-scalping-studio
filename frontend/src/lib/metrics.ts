/**
 * Performance-screen maths.
 *
 * Every helper accepts the values the backend actually sends: `profit_factor`
 * arrives as `null` when it is infinite, and a run can have no trades at all.
 * A missing measurement is a dash on screen, never a `NaN`.
 */

import { formatNumber } from "./format";
import type { Metrics } from "./types";

export type PFClass = "good" | "ok" | "bad";

/**
 * Profit-factor banding. `> 1.3` is good, `1.0–1.3` is acceptable and anything
 * below 1.0 is bad. The boundaries are inclusive on the "ok" side: 1.3 is not
 * yet good.
 *
 * A positive infinity means wins and no losing trades — the best possible
 * factor — so it is good. `null` (what the backend sends when the factor is
 * not representable) and `NaN` are unknown, not good: they stay bad.
 */
export function classifyPF(value: number | null): PFClass {
  if (value === null || Number.isNaN(value)) return "bad";
  if (value === Number.POSITIVE_INFINITY) return "good";
  if (!Number.isFinite(value) || value < 1.0) return "bad";
  return value > 1.3 ? "good" : "ok";
}

/**
 * Drawdown as a percentage below the running peak, one value per equity mark.
 *
 * The formula is `(value - peak) / peak * 100` — not `(value / peak - 1) * 100`
 * — because the latter drifts by a few ULPs and would make an exact expected
 * value like `-8.333333333333332` unreproducible. The first point is always 0
 * (it is the peak by definition) and zero equity is -100 %.
 *
 * A non-finite mark (a failed read) contributes 0 instead of poisoning the
 * path with `NaN`; the chart still renders every other point.
 */
export function drawdownSeries(equity: readonly number[]): number[] {
  const series: number[] = [];
  let peak = 0;

  for (const value of equity) {
    if (!Number.isFinite(value)) {
      series.push(0);
      continue;
    }
    if (value > peak) peak = value;

    const drawdown = peak > 0 ? ((value - peak) / peak) * 100 : 0;
    series.push(Math.min(drawdown, 0));
  }

  // Declared explicitly: a series that opens at zero equity is at its peak,
  // even though the arithmetic above would also report -100 for a later zero.
  if (series.length > 0) series[0] = 0;

  return series;
}

/**
 * Cost as a share of gross profit.
 *
 * The wire sends `net_profit` and `total_costs`, not gross profit, so gross is
 * reconstructed as `net_profit + total_costs`. When that denominator is not
 * positive there is no profit to compare against: report 1 (all of it) if
 * costs were paid, otherwise 0. This keeps the card meaningful for a losing
 * run instead of dividing by zero or showing a negative share.
 */
export function costShare(metrics: Metrics): number {
  const costs = metrics.total_costs;
  const gross = metrics.net_profit + costs;

  if (Number.isFinite(gross) && gross > 0) {
    const share = costs / gross;
    return Number.isFinite(share) ? share : 0;
  }

  return Number.isFinite(costs) && costs > 0 ? 1 : 0;
}

/**
 * Unsigned percent text for a fraction: `0.55` → `%55,0`. Win rate and cost
 * share are magnitudes, so they must not borrow `formatPct`'s `+` sign.
 */
export function formatFractionPct(fraction: number | null, digits = 1): string {
  if (fraction === null || !Number.isFinite(fraction)) return formatNumber(null);
  // Same `%` prefix as `formatPct`: tr-TR writes %55,0, not 55,0 %.
  return `%${formatNumber(fraction * 100, digits)}`;
}
