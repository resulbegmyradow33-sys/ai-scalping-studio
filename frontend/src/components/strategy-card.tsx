"use client";

/**
 * The live strategy card: which rule the panel replays and how it earned the
 * seat. Recalibration is the "keep learning" action: it re-selects parameters
 * on the training slice and only promotes them when the holdout gate passes.
 */

import { formatPct, formatTs } from "@/lib/format";
import type {
  CalibrateResponse,
  ForwardResult,
  RuleMetrics,
  StrategyCard,
} from "@/lib/types";

export interface StrategyCardPanelProps {
  card: StrategyCard | undefined;
  loading?: boolean;
  error?: Error | null;
  running?: boolean;
  result?: CalibrateResponse | null;
  calibrateError?: string | null;
  onCalibrate?: () => void;
}

function forwardLine(result: ForwardResult | undefined): string | null {
  if (result === undefined || !Number.isFinite(result.expectancy_r)) return null;
  const r = result.expectancy_r ?? 0;
  const pf = Number.isFinite(result.profit_factor)
    ? (result.profit_factor ?? 0).toFixed(2)
    : "—";
  const months =
    result.start && result.end
      ? `${result.start.slice(0, 7)}→${result.end.slice(0, 7)}`
      : "ileri";
  return `${r >= 0 ? "+" : ""}${r.toFixed(3)} R · PF ${pf} · ${result.trades ?? 0} işlem (${months})`;
}

function holdoutLine(metrics: RuleMetrics | undefined): string {
  if (metrics === undefined) return "—";
  const r = Number.isFinite(metrics.expectancy_r)
    ? `${metrics.expectancy_r >= 0 ? "+" : ""}${metrics.expectancy_r.toFixed(3)} R`
    : "—";
  const pf = Number.isFinite(metrics.profit_factor) ? metrics.profit_factor.toFixed(2) : "—";
  const dd = Number.isFinite(metrics.max_drawdown)
    ? formatPct(-Math.abs(metrics.max_drawdown) * 100)
    : "—";
  return `${r} · PF ${pf} · DD ${dd} · ${metrics.trades} işlem`;
}

export function StrategyCardPanel({
  card,
  loading = false,
  error = null,
  running = false,
  result = null,
  calibrateError = null,
  onCalibrate,
}: StrategyCardPanelProps) {
  const active = card?.active === true && card.params !== undefined;
  return (
    <section
      aria-labelledby="strategy-card-heading"
      className="rounded-md border border-slate-800 bg-surface px-3 py-2"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-2">
          <h2 id="strategy-card-heading" className="text-sm font-semibold text-slate-100">
            Canlı strateji
          </h2>
          <span
            data-testid="strategy-card-state"
            className={
              active
                ? "rounded-sm border border-profit/40 bg-profit/10 px-1.5 py-0.5 text-xs text-profit"
                : "rounded-sm border border-slate-700 px-1.5 py-0.5 text-xs text-slate-400"
            }
          >
            {active ? "flow + momentum" : "kart yok · M1 momentum"}
          </span>
        </div>
        <button
          type="button"
          data-testid="calibrate-button"
          onClick={onCalibrate}
          disabled={running || onCalibrate === undefined}
          className="rounded-sm border border-accent bg-accent/20 px-2 py-0.5 text-xs font-medium text-slate-100 transition-colors hover:bg-accent/30 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          {running ? "Kalibre ediliyor…" : "Yeniden kalibre et"}
        </button>
      </div>

      {loading && <p className="mt-1 text-xs text-slate-500">Kart okunuyor…</p>}
      {error !== null && !loading && (
        <p className="mt-1 text-xs text-loss">Kart alınamadı: {error.message}</p>
      )}

      {!loading && error === null && active && card.params !== undefined && (
        <dl className="mt-1 flex flex-col gap-0.5 text-xs text-slate-300">
          <div className="flex flex-wrap gap-x-3">
            <div className="flex gap-1">
              <dt className="text-slate-500">Zaman dilimi</dt>
              <dd>{card.params.timeframe}</dd>
            </div>
            <div className="flex gap-1">
              <dt className="text-slate-500">Akış eşiği</dt>
              <dd>
                üst %{Math.round((1 - card.params.flow_quantile) * 100)} ·{" "}
                {card.params.flow_window} bar
              </dd>
            </div>
            <div className="flex gap-1">
              <dt className="text-slate-500">SL / TP</dt>
              <dd>
                {card.params.sl_atr_mult} / {card.params.tp_atr_mult} ATR ·{" "}
                {card.params.max_bars_in_trade} bar
              </dd>
            </div>
          </div>
          <div className="flex flex-wrap gap-1">
            <dt className="text-slate-500">Holdout</dt>
            <dd data-testid="strategy-card-holdout">{holdoutLine(card.holdout)}</dd>
          </div>
          {forwardLine(card.forward) !== null && (
            <div className="flex flex-wrap gap-1">
              <dt className="text-slate-500">İleri test</dt>
              <dd
                data-testid="strategy-card-forward"
                className={(card.forward?.expectancy_r ?? 0) >= 0 ? "text-profit" : "text-loss"}
              >
                {forwardLine(card.forward)}
              </dd>
            </div>
          )}
          <div className="flex flex-wrap gap-1 text-slate-500">
            <dt>Kart</dt>
            <dd>{formatTs(card.created_at ?? null)}</dd>
          </div>
          {card.note ? <p className="text-slate-500">{card.note}</p> : null}
        </dl>
      )}

      {!loading && error === null && !active && (
        <p className="mt-1 text-xs text-slate-400">
          Henüz terfi etmiş bir kural kartı yok; panel doğrulanmamış M1 momentumunu
          oynatıyor. Kalibrasyon en az 5 yıllık H1 verisi ister.
        </p>
      )}

      {calibrateError !== null && (
        <p role="alert" className="mt-1 text-xs text-loss">
          Kalibrasyon başarısız: {calibrateError}
        </p>
      )}

      {result !== null && (
        <p
          data-testid="calibrate-result"
          className={`mt-1 text-xs ${result.promoted ? "text-profit" : "text-slate-400"}`}
        >
          {result.promoted
            ? `Yeni kart devreye alındı — holdout ${holdoutLine(result.candidate.holdout)}`
            : `Aday reddedildi — ${result.reason}`}
        </p>
      )}
    </section>
  );
}
