"use client";

/**
 * The simulator screen: controls, the live candle chart, and the position and
 * account panel. All state comes from `useSimulation` (the socket) and
 * `useHealth` (SWR); the page only derives display values, so an empty backend
 * renders the waiting states instead of an error.
 */

import { useMemo } from "react";

import { CandleChart } from "@/components/candle-chart";
import { EquitySparkline } from "@/components/equity-sparkline";
import { PositionCard } from "@/components/position-card";
import { SimControls } from "@/components/sim-controls";
import { StatCard } from "@/components/stat-card";
import { useHealth } from "@/hooks/use-health";
import { useSimulation } from "@/hooks/use-simulation";
import { API_BASE } from "@/lib/api";
import { formatMoney, formatNumber, formatPct } from "@/lib/format";
import type { ControlAction, SimState } from "@/lib/types";

const STATUS_LABEL: Record<SimState, string> = {
  idle: "Hazır",
  running: "Çalışıyor",
  paused: "Duraklatıldı",
  stopped: "Durduruldu",
  finished: "Bitti",
  error: "Hata",
};

/** Both start-time 409s come from the backend's RunnerError. */
const MISSING_DATA = /no data|not enough bars/i;
const MISSING_DATA_HINT = "Veri yok — Veri ekranından sentetik veri üret.";
const REQUEST_REJECTED = "İstek reddedildi — sunucu bu işlemi kabul etmedi.";

/** The host shown in the offline banner, e.g. `127.0.0.1:8000`. */
const API_HOST = (() => {
  try {
    return new URL(API_BASE).host;
  } catch {
    return API_BASE;
  }
})();

function describeStartError(error: string | null, status: number | null): string | null {
  if (error === null) return null;
  // Prefer the status: a 4xx detail is backend English, not UI copy.
  if (status === 409) return MISSING_DATA_HINT;
  if (status !== null && status >= 400 && status < 500) return REQUEST_REJECTED;
  // Older call paths without a status still carry the RunnerError prose.
  return MISSING_DATA.test(error) ? MISSING_DATA_HINT : error;
}

export default function SimulatorPage() {
  const sim = useSimulation();
  const health = useHealth();

  const lastEquity = sim.equity.at(-1) ?? null;
  const anchor = sim.equityAnchor;
  const sessionPnl =
    lastEquity !== null && anchor !== null && anchor !== 0
      ? ((lastEquity.equity - anchor) / anchor) * 100
      : null;
  const pnlTone = sessionPnl === null || sessionPnl === 0 ? "neutral" : sessionPnl > 0 ? "profit" : "loss";

  const lastBar = sim.bars.at(-1) ?? null;
  // A new array per frame would rebuild the sparkline path even when only the
  // last point changed; memoise on the stream's own reference.
  const equityValues = useMemo(() => sim.equity.map((point) => point.equity), [sim.equity]);

  const startError = describeStartError(sim.error, sim.errorStatus);
  const healthDown = Boolean(health.error) || health.data?.status !== "ok";
  // A down backend already gets its own banner; repeating "network request
  // failed" underneath it is noise. A real 409 still matters and stays visible.
  const showStartError = startError !== null && !(healthDown && startError !== MISSING_DATA_HINT);

  const handleStart = (speed: number): void => {
    void sim.start(speed);
  };

  const handleControl = (action: ControlAction): void => {
    void sim.control(action);
  };

  return (
    <section aria-labelledby="simulator-heading" className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 id="simulator-heading" className="text-xl font-semibold text-slate-100">
          Simülatör
        </h1>
        <span
          data-testid="status-chip"
          aria-live="polite"
          aria-atomic="true"
          className="rounded-sm border border-slate-700 px-2 py-0.5 text-xs text-slate-300"
        >
          {STATUS_LABEL[sim.status]}
        </span>
      </div>

      {healthDown && (
        <div
          role="status"
          className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200"
        >
          Backend&apos;e ulaşılamıyor ({API_HOST})
        </div>
      )}

      {showStartError && (
        <div
          role="alert"
          className="rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-slate-100"
        >
          {startError}
        </div>
      )}

      <SimControls status={sim.status} onStart={handleStart} onControl={handleControl} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="flex min-w-0 flex-col gap-2 lg:col-span-9">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold text-slate-100">Fiyat (M1)</h2>
            <dl className="flex items-baseline gap-4 text-xs">
              <div className="flex items-baseline gap-1.5">
                <dt className="text-slate-400">Bar</dt>
                <dd data-testid="bar-count" className="tabular-nums text-slate-200">
                  {sim.bars.length}
                </dd>
              </div>
              <div className="flex items-baseline gap-1.5">
                <dt className="text-slate-400">Son fiyat</dt>
                <dd data-testid="last-price" className="tabular-nums text-slate-200">
                  {formatNumber(lastBar?.close ?? null, 2)}
                </dd>
              </div>
            </dl>
          </div>
          <CandleChart
            bars={sim.bars}
            trades={sim.trades}
            position={sim.position}
            height={380}
          />
        </div>

        {/* The stat grid follows the sidebar's own width, not the viewport. */}
        <aside
          aria-label="Pozisyon ve hesap"
          className="@container flex min-w-0 flex-col gap-2 lg:col-span-3"
        >
          <PositionCard
            position={sim.position}
            equity={lastEquity?.equity ?? null}
            sessionPnl={sessionPnl}
          />

          <div className="grid grid-cols-1 gap-2 @min-[18rem]:grid-cols-2">
            <StatCard
              label="Bakiye"
              testId="balance-value"
              value={formatMoney(lastEquity?.balance ?? null)}
            />
            <StatCard
              label="Equity"
              testId="equity-value"
              value={formatMoney(lastEquity?.equity ?? null)}
              tone={pnlTone}
            >
              <EquitySparkline values={equityValues} />
            </StatCard>
            <StatCard
              label="Seans PnL"
              testId="day-pnl-value"
              value={formatPct(sessionPnl)}
              tone={pnlTone}
            />
            <StatCard
              label="Seçili strateji"
              testId="strategy-value"
              value={sim.lastDecision?.strategy ?? "—"}
            />
            <StatCard
              label="Bloklanan aksiyon"
              testId="risk-blocked-value"
              value={sim.riskBlocked ?? "—"}
            />
          </div>
        </aside>
      </div>
    </section>
  );
}
