import type { ReactNode } from "react";

/**
 * One read-only metric. The value uses tabular figures so a stream of prices
 * does not make the card width jump, and `live` marks the single element that
 * should be announced by assistive tech as equity ticks.
 */

export type StatTone = "neutral" | "profit" | "loss";

export interface StatCardProps {
  label: string;
  value: ReactNode;
  tone?: StatTone;
  hint?: ReactNode;
  live?: boolean;
  testId?: string;
  /** Optional visual under the value, e.g. an equity sparkline. */
  children?: ReactNode;
}

const TONE_CLASS: Record<StatTone, string> = {
  neutral: "text-slate-100",
  profit: "text-profit",
  loss: "text-loss",
};

export function StatCard({
  label,
  value,
  tone = "neutral",
  hint,
  live = false,
  testId,
  children,
}: StatCardProps) {
  return (
    <div className="min-w-0 rounded-md border border-slate-800 bg-surface px-3 py-2">
      <p className="truncate text-xs text-slate-400" title={label}>
        {label}
      </p>
      <p
        data-testid={testId}
        aria-live={live ? "polite" : undefined}
        aria-atomic={live ? true : undefined}
        className={`mt-1 text-base tabular-nums ${TONE_CLASS[tone]}`}
      >
        {value}
      </p>
      {hint !== undefined && <p className="mt-0.5 text-xs text-slate-500">{hint}</p>}
      {children}
    </div>
  );
}
