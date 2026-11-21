/**
 * A pure SVG line chart for training curves.
 *
 * No chart library: the training page plots a handful of checkpoints, so one
 * `<polyline>` in a fixed 100×40 viewBox is enough. `preserveAspectRatio="none"`
 * stretches it to the card and `vector-effect="non-scaling-stroke"` keeps the
 * stroke from stretching with it.
 *
 * Input must be sorted by `time`; `toSeries` guarantees that. The x axis is
 * positioned by `time`, not by array index, so two series with the same number
 * of checkpoints but different step spacing render differently. A single point
 * is a `<circle>`: a one-vertex polyline paints no stroke.
 */

import type { SeriesPoint } from "@/lib/runs";

const WIDTH = 100;
const HEIGHT = 40;
const POINT_RADIUS = 1.5;

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

  const valueSpan = high - low;
  const flat = !Number.isFinite(valueSpan) || valueSpan === 0;

  const firstTime = valid[0].time;
  const lastTime = valid[valid.length - 1].time;
  const timeSpan = lastTime - firstTime;
  const singleTime = !Number.isFinite(timeSpan) || timeSpan <= 0;

  const coords = valid.map((point) => {
    const x = singleTime ? WIDTH / 2 : ((point.time - firstTime) / timeSpan) * WIDTH;
    const y = flat ? HEIGHT / 2 : HEIGHT - ((point.value - low) / valueSpan) * HEIGHT;
    return { x, y };
  });

  return (
    <svg
      role="img"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      style={{ color }}
      className="block h-28 w-full"
    >
      <title>{title}</title>
      {coords.length === 1 ? (
        <circle
          cx={coords[0].x}
          cy={coords[0].y}
          r={POINT_RADIUS}
          fill="currentColor"
        />
      ) : (
        <polyline
          points={coords.map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ")}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}
