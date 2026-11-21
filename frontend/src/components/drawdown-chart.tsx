/**
 * Drawdown curve: the percentage below the running equity peak, drawn as an
 * area under the zero line.
 *
 * Fixed 0…-100 % scale rather than a per-run zoom: the height of the shape is
 * comparable between runs, and a -20 % drawdown looks like -20 % instead of
 * filling the card. Pure SVG for the same reason as the other small charts:
 * one path, no library, and `preserveAspectRatio="none"` with a non-scaling
 * stroke keeps it crisp at any width.
 *
 * The walk over the equity marks and both path strings are memoised on
 * `values`, and a long curve is downsampled to 64 evenly spaced marks (ends
 * included). A revalidation must not rebuild a multi-thousand-point path.
 */

import { useMemo } from "react";

import { drawdownSeries } from "@/lib/metrics";

const WIDTH = 100;
const HEIGHT = 40;
const STROKE_WIDTH = 1.5;
/**
 * 0 % is drawn half a stroke below the top edge; centred on y=0 the upper half
 * of the stroke would be clipped by the viewBox.
 */
const ZERO_Y = STROKE_WIDTH / 2;
const POINT_RADIUS = 1.5;
/** A chart is a glanceable shape; 64 marks per frame is plenty. */
const MAX_POINTS = 64;

export interface DrawdownChartProps {
  /** Equity marks, oldest first. The drawdown is derived here. */
  values: readonly number[];
}

/**
 * Evenly spaced indices over the whole curve, always keeping the first and the
 * last. Sampling happens *after* the drawdown walk so an omitted mark can
 * never hide a peak.
 */
function downsample(series: readonly number[]): readonly number[] {
  if (series.length <= MAX_POINTS) return series;
  const stride = (series.length - 1) / (MAX_POINTS - 1);
  const sampled: number[] = [];
  for (let index = 0; index < MAX_POINTS; index += 1) {
    sampled.push(series[Math.round(index * stride)]);
  }
  return sampled;
}

interface ChartGeometry {
  line: string;
  area: string;
  /** Exactly one mark: a circle, because a one-vertex path paints nothing. */
  point: { x: number; y: number } | null;
}

function geometry(series: readonly number[]): ChartGeometry | null {
  if (series.length === 0) return null;

  const step = series.length > 1 ? WIDTH / (series.length - 1) : 0;
  const coords = series.map((value, index) => {
    const x = series.length > 1 ? index * step : WIDTH / 2;
    // Fixed scale from the inset zero line to the bottom edge. Clamp so a
    // malformed mark cannot push the path outside the viewBox.
    const y = Math.min(
      HEIGHT,
      Math.max(ZERO_Y, ZERO_Y + (-value / 100) * (HEIGHT - ZERO_Y)),
    );
    return { x, y };
  });

  if (coords.length === 1) {
    return { line: "", area: "", point: coords[0] };
  }

  const points = coords
    .map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" L ");
  const zero = ZERO_Y.toFixed(2);

  return {
    line: `M ${points}`,
    // Close the shape along the zero line to fill the area between curve and 0.
    area: `M 0.00,${zero} L ${points} L ${WIDTH.toFixed(2)},${zero} Z`,
    point: null,
  };
}

export function DrawdownChart({ values }: DrawdownChartProps) {
  const chart = useMemo(
    () => geometry(downsample(drawdownSeries(values))),
    [values],
  );

  if (chart === null) {
    return <p className="py-6 text-center text-xs text-slate-500">Veri yok</p>;
  }

  return (
    <svg
      role="img"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      className="block h-24 w-full text-loss"
    >
      <title>Drawdown</title>
      <line
        data-testid="drawdown-zero"
        x1={0}
        y1={ZERO_Y}
        x2={WIDTH}
        y2={ZERO_Y}
        stroke="currentColor"
        strokeOpacity={0.3}
        vectorEffect="non-scaling-stroke"
      />
      {chart.point === null ? (
        <>
          <path d={chart.area} fill="currentColor" opacity={0.18} />
          <path
            data-testid="drawdown-line"
            d={chart.line}
            fill="none"
            stroke="currentColor"
            strokeWidth={STROKE_WIDTH}
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        </>
      ) : (
        <circle
          cx={chart.point.x.toFixed(2)}
          cy={chart.point.y.toFixed(2)}
          r={POINT_RADIUS}
          fill="currentColor"
        />
      )}
    </svg>
  );
}
