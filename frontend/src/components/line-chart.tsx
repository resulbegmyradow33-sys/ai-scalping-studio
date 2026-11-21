/**
 * A pure SVG line chart for training curves.
 *
 * No chart library: the training page plots a handful of checkpoints, so one
 * `<polyline>` in a fixed 100×40 viewBox is enough. `preserveAspectRatio="none"`
 * stretches it to the card and `vector-effect="non-scaling-stroke"` keeps the
 * stroke from stretching with it.
 */

import type { SeriesPoint } from "@/lib/runs";

const WIDTH = 100;
const HEIGHT = 40;

export interface LineChartProps {
  title: string;
  points: readonly SeriesPoint[];
  color?: string;
}

export function LineChart({ title, points, color = "currentColor" }: LineChartProps) {
  // Drop bad points here too: callers are expected to use `toSeries`, but a
  // NaN coordinate would make the whole polyline disappear.
  const valid = points.filter(
    (point) => Number.isFinite(point.time) && Number.isFinite(point.value),
  );

  if (valid.length === 0) {
    return <p className="py-6 text-center text-xs text-slate-500">Veri yok</p>;
  }

  let low = valid[0].value;
  let high = valid[0].value;
  for (const point of valid) {
    if (point.value < low) low = point.value;
    if (point.value > high) high = point.value;
  }

  const span = high - low;
  const coords = valid.map((point, index) => {
    // One point is not a curve: centre it instead of dividing by zero. A flat
    // series sits on the middle line for the same reason.
    const x = valid.length === 1 ? WIDTH / 2 : (index / (valid.length - 1)) * WIDTH;
    const y = span === 0 ? HEIGHT / 2 : HEIGHT - ((point.value - low) / span) * HEIGHT;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  return (
    <svg
      role="img"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      className="block h-28 w-full"
    >
      <title>{title}</title>
      <polyline
        points={coords.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
