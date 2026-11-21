/**
 * One data provider in the store status grid.
 *
 * `GET /api/data/status` reports coverage for a symbol, not per provider. An
 * active card therefore shows the stored range but `—` for rows: attributing
 * the symbol total to each contributing provider would be a lie. Inactive
 * cards read `0 bar`.
 */

import { formatNumber, formatTs } from "@/lib/format";
import type { ProviderName } from "@/lib/types";

export interface ProviderCardProps {
  provider: ProviderName;
  label: string;
  active: boolean;
  /** `null` when the provider contributed but the API has no per-provider split. */
  rows: number | null;
  firstTs: string | null;
  lastTs: string | null;
}

export function ProviderCard({
  provider,
  label,
  active,
  rows,
  firstTs,
  lastTs,
}: ProviderCardProps) {
  const range =
    active && firstTs !== null && lastTs !== null
      ? `${formatTs(firstTs)} → ${formatTs(lastTs)}`
      : "—";

  return (
    <article
      data-testid={`provider-card-${provider}`}
      className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-3 py-2.5"
    >
      <header className="flex items-center gap-2">
        <span
          data-testid={`provider-dot-${provider}`}
          data-state={active ? "ready" : "empty"}
          aria-hidden="true"
          className={`size-2 shrink-0 rounded-full ${active ? "bg-emerald-400" : "bg-slate-600"}`}
        />
        <h3 className="text-sm font-medium text-slate-100">{label}</h3>
        <span className="ml-auto text-xs text-slate-400">
          {active ? "Veri var" : "Veri yok"}
        </span>
      </header>
      <p
        data-testid={`provider-range-${provider}`}
        className="truncate text-xs tabular-nums text-slate-400"
      >
        {range}
      </p>
      <p
        data-testid={`provider-rows-${provider}`}
        className="text-sm font-medium tabular-nums text-slate-200"
      >
        {rows === null ? "—" : `${formatNumber(rows, 0)} bar`}
      </p>
    </article>
  );
}
