/**
 * A visible marker for a legacy-path value. The text is real DOM content, not
 * a background colour or a `title`, so it survives zoom, high contrast and a
 * screen reader without an `aria-label` on a plain span.
 */

export interface LegacyBadgeProps {
  testId?: string;
}

export function LegacyBadge({ testId }: LegacyBadgeProps) {
  return (
    <span
      data-testid={testId}
      className="rounded-sm border border-amber-700/50 bg-amber-900/20 px-1.5 py-0.5 text-[0.65rem] uppercase tracking-wide text-amber-300"
    >
      legacy baseline
    </span>
  );
}
