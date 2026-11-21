import { formatMoney, formatNumber, formatPct } from "@/lib/format";
import type { TradeOpenPayload } from "@/lib/types";

/**
 * The open position, or an explicit flat state. Unrealised PnL is not on the
 * wire yet: the backend sends account equity, so the card reports equity and
 * the session's day PnL instead of inventing a position PnL.
 */

export interface PositionCardProps {
  position: TradeOpenPayload | null;
  equity: number | null;
  /** Session change of equity in percent, already computed by the page. */
  dayPnl: number | null;
}

function pnlTone(value: number | null): string {
  if (value === null || value === 0) return "text-slate-300";
  return value > 0 ? "text-profit" : "text-loss";
}

export function PositionCard({ position, equity, dayPnl }: PositionCardProps) {
  return (
    <section
      aria-labelledby="position-heading"
      className="rounded-md border border-slate-800 bg-surface px-3 py-2"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 id="position-heading" className="text-sm font-semibold text-slate-100">
          Pozisyon
        </h2>
        {position !== null && (
          <span
            className={`rounded-sm border px-2 py-0.5 text-xs font-medium ${
              position.side === 1
                ? "border-profit/40 bg-profit/10 text-profit"
                : "border-loss/40 bg-loss/10 text-loss"
            }`}
          >
            {position.side === 1 ? "AL" : "SAT"}
          </span>
        )}
      </div>

      {position === null ? (
        <p className="mt-2 text-sm text-slate-400">Pozisyon yok</p>
      ) : (
        <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
          <dt className="text-slate-400">Strateji</dt>
          <dd className="truncate text-right text-slate-200" title={position.strategy}>
            {position.strategy}
          </dd>
          <dt className="text-slate-400">Giriş</dt>
          <dd className="text-right tabular-nums text-slate-200">
            {formatNumber(position.entry, 2)}
          </dd>
          <dt className="text-slate-400">SL</dt>
          <dd className="text-right tabular-nums text-loss">{formatNumber(position.sl, 2)}</dd>
          <dt className="text-slate-400">TP</dt>
          <dd className="text-right tabular-nums text-profit">{formatNumber(position.tp, 2)}</dd>
          <dt className="text-slate-400">Lot</dt>
          <dd className="text-right tabular-nums text-slate-200">{formatNumber(position.lot, 2)}</dd>
        </dl>
      )}

      <dl className="mt-3 flex items-baseline justify-between gap-2 border-t border-slate-800 pt-2 text-xs">
        <dt className="text-slate-400">Equity</dt>
        <dd className="tabular-nums text-slate-200">{formatMoney(equity)}</dd>
      </dl>
      <dl className="mt-1 flex items-baseline justify-between gap-2 text-xs">
        <dt className="text-slate-400">Günlük PnL</dt>
        <dd className={`tabular-nums ${pnlTone(dayPnl)}`}>{formatPct(dayPnl)}</dd>
      </dl>
    </section>
  );
}
