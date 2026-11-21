/**
 * Drawdown curve: the percentage below the running equity peak, drawn as an
 * area under the zero line.
 *
 * Fixed 0…-100 % scale rather than a per-run zoom: the height of the shape is
 * comparable between runs, and a -20 % drawdown looks like -20 % instead of
 * filling the card. Pure SVG for the same reason as the other small charts:
 * one path, no library, and `preserveAspectRatio="none"` with a non-scaling
 * stroke keeps it crisp at any width.
 */

import { drawdownSeries } from "@/lib/metrics";

const WIDTH = 100;
const HEIGHT = 40;

export interface DrawdownChartProps {
  /** Equity marks, oldest first. The drawdown is derived here. */
  values: readonly number[];
}

export function DrawdownChart({ values }: DrawdownChartProps) {
  const series = drawdownSeries(values);

  if (series.length === 0) {
    return <p className="py-6 text-center text-xs text-slate-500">Veri yok</p>;
  }

  const step = series.length > 1 ? WIDTH / (series.length - 1) : 0;

  const points = series.map((value, index) => {
    const x = series.length > 1 ? index * step : WIDTH / 2;
    // Fixed scale: 0 % sits on the top edge, -100 % on the bottom. Clamp so a
    // malformed mark below -100 % cannot push the path outside the viewBox.
    const y = Math.min(HEIGHT, Math.max(0, (-value / 100) * HEIGHT));
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  const line = `M ${points.join(" L ")}`;
  // Close the shape along the zero line to fill the area between curve and 0.
  const area = `M 0.00,0.00 L ${points.join(" L ")} L ${WIDTH.toFixed(2)},0.00 Z`;

  return (
    <svg
      role="img"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      className="block h-24 w-full text-loss"
    >
      <title>Drawdown</title>
      <path d={area} fill="currentColor" opacity={0.18} />
      <path
        data-testid="drawdown-line"
        d={line}
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
