/**
 * A sparkline for the equity curve. Inline SVG instead of a chart library: the
 * card is a glanceable shape, not an interactive chart, so it stays out of the
 * accessibility tree (`aria-hidden`) and the numeric value carries the meaning.
 */

import { useMemo } from "react";

export interface EquitySparklineProps {
  values: readonly number[];
  height?: number;
}

const WIDTH = 100;
/** A sparkline is a shape, not a data table: 64 points per frame is plenty. */
const MAX_POINTS = 64;

/**
 * Evenly spaced indices over the whole series, always keeping the first and
 * last. The full 5 000-frame buffer would otherwise rebuild an unusable path
 * on every equity tick.
 */
function downsample(values: readonly number[]): readonly number[] {
  if (values.length <= MAX_POINTS) return values;
  const stride = (values.length - 1) / (MAX_POINTS - 1);
  const sampled: number[] = [];
  for (let index = 0; index < MAX_POINTS; index += 1) {
    sampled.push(values[Math.round(index * stride)]);
  }
  return sampled;
}

export function EquitySparkline({ values, height = 28 }: EquitySparklineProps) {
  const points = useMemo(() => {
    // One point is not a curve; an empty or single-value series renders nothing
    // rather than a misleading flat line.
    if (values.length < 2) return null;

    const sampled = downsample(values);

    // A loop, not `Math.min(...values)`: a 5 000-argument spread can overflow
    // the call stack on some engines.
    let low = sampled[0];
    let high = sampled[0];
    for (const value of sampled) {
      if (value < low) low = value;
      if (value > high) high = value;
    }

    const span = high - low || 1;
    const step = WIDTH / (sampled.length - 1);

    return sampled
      .map((value, index) => {
        const x = index * step;
        const y = height - ((value - low) / span) * height;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [values, height]);

  if (points === null) return null;

  const rising = values[values.length - 1] >= values[0];

  return (
    <svg
      aria-hidden="true"
      viewBox={`0 0 ${WIDTH} ${height}`}
      preserveAspectRatio="none"
      style={{ height }}
      className={`mt-1 block w-full ${rising ? "text-profit" : "text-loss"}`}
    >
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
