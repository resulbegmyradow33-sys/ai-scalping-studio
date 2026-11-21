/**
 * A sparkline for the equity curve. Inline SVG instead of a chart library: the
 * card is a glanceable shape, not an interactive chart, so it stays out of the
 * accessibility tree (`aria-hidden`) and the numeric value carries the meaning.
 */

export interface EquitySparklineProps {
  values: readonly number[];
  height?: number;
}

const WIDTH = 100;

export function EquitySparkline({ values, height = 28 }: EquitySparklineProps) {
  // One point is not a curve; an empty or single-value series renders nothing
  // rather than a misleading flat line.
  if (values.length < 2) return null;

  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const step = WIDTH / (values.length - 1);

  const points = values
    .map((value, index) => {
      const x = index * step;
      const y = height - ((value - low) / span) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  const rising = values[values.length - 1] >= values[0];

  return (
    <svg
      aria-hidden="true"
      viewBox={`0 0 ${WIDTH} ${height}`}
      preserveAspectRatio="none"
      className={`mt-1 block h-7 w-full ${rising ? "text-profit" : "text-loss"}`}
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
