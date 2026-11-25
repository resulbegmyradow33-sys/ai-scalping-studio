/**
 * Backtesting-screen helpers.
 *
 * The backend's `POST /api/backtest` accepts only `run_id` and `symbol`, and
 * its response carries a report path plus the same `Metrics` object as a
 * simulator run. Everything here treats a missing or malformed value as
 * absent: a run that is not exactly `completed` cannot be replayed, and an
 * equity series that the response does not contain stays empty instead of
 * being reconstructed in the browser.
 */

import type { Metrics, RunSummary } from "./types";

/** A chart-ready mark: `time` is the array index, which is strictly increasing. */
export interface EquityPoint {
  time: number;
  value: number;
}

/**
 * The runs `/api/backtest` can replay.
 *
 * The route rejects live and failed statuses with a 409, so the selector offers
 * only `completed`. The comparison is case-insensitive because the manifest's
 * status string is not normalised by the API.
 */
export function completedRuns(runs: readonly RunSummary[]): RunSummary[] {
  return runs.filter((run) => run.status.toLowerCase() === "completed");
}

/**
 * Finite marks from an optional `equity` field.
 *
 * The current `BacktestResponse` has no equity array, so a response without it
 * returns `[]` and the page says so. Non-finite or non-numeric entries are
 * dropped rather than charted as zero; `time` keeps the original array index,
 * so a hole stays a gap instead of shifting later marks left.
 */
export function equitySeries(value: unknown): EquityPoint[] {
  if (!Array.isArray(value)) return [];

  const points: EquityPoint[] = [];
  value.forEach((item, index) => {
    if (typeof item === "number" && Number.isFinite(item)) {
      points.push({ time: index, value: item });
    }
  });
  return points;
}

/**
 * Whether the response carries enough of a metrics object to render.
 *
 * A missing or empty body is rejected here, before `MetricGrid` runs: the grid
 * can show a dash for one absent field, but a wholly absent object would make
 * derived values such as the cost share collapse to a zero the API never sent.
 * `trades` and `net_profit` are the two fields every real `Metrics` payload
 * has, so they gate the rest.
 */
export function hasMetrics(value: unknown): value is Metrics {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.trades === "number" &&
    Number.isFinite(record.trades) &&
    typeof record.net_profit === "number" &&
    Number.isFinite(record.net_profit)
  );
}
