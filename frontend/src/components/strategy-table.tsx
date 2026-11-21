/**
 * Strategy breakdown and walk-forward result tables.
 *
 * Both render fixed outer wrappers that can scroll horizontally and are
 * keyboard-focusable, following the run table. The strategy order is the
 * backend's frozen arm order: five rows always exist, with dashes where the
 * current API has no breakdown yet. Walk-forward falls back to a single
 * explanatory row rather than an empty `<tbody>`.
 */

import { formatNumber, formatPct, formatTs } from "@/lib/format";
import { formatFractionPct } from "@/lib/metrics";
import { BANDIT_STRATEGIES } from "@/lib/runs";

export interface StrategyRow {
  strategy: string;
  trades: number | null;
  /** Fraction (`0.55`), not percent. */
  winRate: number | null;
  avgR: number | null;
  netR: number | null;
}

export interface WalkForwardFold {
  fold: number;
  testStart: string;
  testEnd: string;
  profitFactor: number | null;
  /** Positive fraction (`0.2`), not percent. */
  maxDrawdown: number | null;
}

const SCROLL_REGION =
  "overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const HEAD_CELL = "px-2 py-1.5 font-medium";
const BODY_CELL = "px-2 py-1.5 tabular-nums text-slate-300";

function blankStrategy(strategy: string): StrategyRow {
  return { strategy, trades: null, winRate: null, avgR: null, netR: null };
}

/** Guard a wire value before it reaches `Intl`: `NaN` would throw. */
function finite(value: number | null): number | null {
  return value !== null && Number.isFinite(value) ? value : null;
}

export interface StrategyTableProps {
  rows?: readonly StrategyRow[];
}

export function StrategyTable({ rows = [] }: StrategyTableProps) {
  const byStrategy = new Map(rows.map((row) => [row.strategy, row]));
  const ordered = BANDIT_STRATEGIES.map(
    (strategy) => byStrategy.get(strategy) ?? blankStrategy(strategy),
  );

  return (
    <div role="region" aria-label="Strateji kırılımı" tabIndex={0} className={SCROLL_REGION}>
      <table className="w-full min-w-[26rem] border-collapse text-left text-xs">
        <caption className="sr-only">Strateji bazlı kırılım</caption>
        <thead className="bg-slate-900/60 text-slate-400">
          <tr>
            <th scope="col" className={HEAD_CELL}>Strateji</th>
            <th scope="col" className={`${HEAD_CELL} text-right`}>İşlem</th>
            <th scope="col" className={`${HEAD_CELL} text-right`}>Win rate</th>
            <th scope="col" className={`${HEAD_CELL} text-right`}>Ort. R</th>
            <th scope="col" className={`${HEAD_CELL} text-right`}>Net R</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map((row) => (
            <tr key={row.strategy} className="border-t border-slate-800/60">
              <th scope="row" className="px-2 py-1.5 font-normal text-slate-300">
                {row.strategy}
              </th>
              <td className={`${BODY_CELL} text-right`}>{formatNumber(finite(row.trades), 0)}</td>
              <td className={`${BODY_CELL} text-right`}>
                {formatFractionPct(finite(row.winRate), 1)}
              </td>
              <td className={`${BODY_CELL} text-right`}>{formatNumber(finite(row.avgR), 2)}</td>
              <td className={`${BODY_CELL} text-right`}>{formatNumber(finite(row.netR), 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One fold's test window; `—` when either boundary is missing. */
function testRange(fold: WalkForwardFold): string {
  if (!fold.testStart || !fold.testEnd) return formatTs(null);
  return `${formatTs(fold.testStart)} – ${formatTs(fold.testEnd)}`;
}

export interface WalkForwardTableProps {
  folds?: readonly WalkForwardFold[];
}

export function WalkForwardTable({ folds = [] }: WalkForwardTableProps) {
  return (
    <div role="region" aria-label="Walk-forward sonuçları" tabIndex={0} className={SCROLL_REGION}>
      <table className="w-full min-w-[26rem] border-collapse text-left text-xs">
        <caption className="sr-only">Walk-forward fold sonuçları</caption>
        <thead className="bg-slate-900/60 text-slate-400">
          <tr>
            <th scope="col" className={HEAD_CELL}>Fold</th>
            <th scope="col" className={HEAD_CELL}>Test aralığı</th>
            <th scope="col" className={`${HEAD_CELL} text-right`}>PF</th>
            <th scope="col" className={`${HEAD_CELL} text-right`}>DD</th>
          </tr>
        </thead>
        <tbody>
          {folds.length === 0 ? (
            <tr className="border-t border-slate-800/60">
              <td colSpan={4} className="px-2 py-3 text-center text-slate-400">
                Walk-forward henüz koşulmadı
              </td>
            </tr>
          ) : (
            folds.map((fold) => {
              const drawdown = finite(fold.maxDrawdown);
              return (
                <tr key={fold.fold} className="border-t border-slate-800/60">
                  <th scope="row" className="px-2 py-1.5 font-normal text-slate-300">
                    {formatNumber(finite(fold.fold), 0)}
                  </th>
                  <td className={`${BODY_CELL}`}>{testRange(fold)}</td>
                  <td className={`${BODY_CELL} text-right`}>
                    {formatNumber(finite(fold.profitFactor), 2)}
                  </td>
                  <td className={`${BODY_CELL} text-right`}>
                    {formatPct(drawdown === null ? null : -Math.abs(drawdown) * 100)}
                  </td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
