/**
 * Training-run helpers.
 *
 * The backend has no `/api/runs` yet (the RL plan adds it), so every helper
 * here accepts missing or malformed data: an absent run is an empty state, not
 * an error, and a broken checkpoint is dropped instead of reaching an SVG.
 */

import type { RunSummary, TrainingPoint } from "./types";

/** A chart-ready point: `step` renamed to `time` for a monotonic x axis. */
export interface SeriesPoint {
  time: number;
  value: number;
}

/** Arm order is frozen on the backend; the table keeps the same order. */
export const BANDIT_STRATEGIES = ["momentum", "reversal", "sweep", "fvg", "trend"] as const;

/**
 * Newest run by `created_at`. The list is not sorted by the contract, and a
 * malformed timestamp sorts last instead of poisoning the comparison.
 */
export function pickLatestRun(runs: readonly RunSummary[]): RunSummary | null {
  let latest: RunSummary | null = null;
  let latestTime = Number.NEGATIVE_INFINITY;

  for (const run of runs) {
    const parsed = Date.parse(run.created_at);
    const time = Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
    if (latest === null || time > latestTime) {
      latest = run;
      latestTime = time;
    }
  }

  return latest;
}

/**
 * Sort by step, drop non-finite values and collapse repeated steps to the last
 * value. The chart needs strictly increasing x values; a NaN checkpoint is a
 * failed measurement, not a zero.
 */
export function toSeries(points: readonly TrainingPoint[]): SeriesPoint[] {
  const byStep = new Map<number, number>();

  for (const point of points) {
    if (!Number.isFinite(point.step) || !Number.isFinite(point.value)) continue;
    // Later checkpoints win: a retried step supersedes its earlier value.
    byStep.set(point.step, point.value);
  }

  return [...byStep.entries()]
    .sort(([left], [right]) => left - right)
    .map(([time, value]) => ({ time, value }));
}
