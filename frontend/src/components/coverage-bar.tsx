/**
 * A one-line visual of stored coverage.
 *
 * The API returns first/last timestamps only, so the bar cannot show gaps or
 * density: it says "covered" or "empty" and leaves the exact range to the
 * provider cards. The accessible name carries the same range as text.
 */

import { formatTs } from "@/lib/format";

export interface CoverageBarProps {
  firstTs: string | null;
  lastTs: string | null;
  rows: number;
}

export function CoverageBar({ firstTs, lastTs, rows }: CoverageBarProps) {
  const empty = rows === 0 || firstTs === null || lastTs === null;
  const label = empty
    ? "Kapsam: veri yok"
    : `Kapsam: ${formatTs(firstTs)} → ${formatTs(lastTs)}, ${rows} bar`;

  return (
    <div
      role="img"
      aria-label={label}
      className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800"
    >
      <div
        className={
          empty ? "h-full w-0" : "h-full w-full rounded-full bg-accent/70"
        }
      />
    </div>
  );
}
