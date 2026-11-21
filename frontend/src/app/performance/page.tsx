"use client";

/**
 * The performance screen: metrics, drawdown, strategy split and walk-forward.
 *
 * `/api/runs` belongs to the RL plan and does not exist on the current backend.
 * A 404 (or a backend that is down) therefore means "no backtest yet", never a
 * crash: SWR retries are off, so an absent endpoint cannot become a polling
 * loop. `RunDetail` carries metrics and config only today; the equity curve,
 * strategy split and folds are not on the wire yet, so those components get
 * empty inputs and render their explicit no-data states.
 */

import Link from "next/link";
import useSWR from "swr";

import { DrawdownChart } from "@/components/drawdown-chart";
import { MetricGrid } from "@/components/metric-grid";
import { RunStatusChip } from "@/components/run-table";
import { StrategyTable, WalkForwardTable } from "@/components/strategy-table";
import type { StrategyRow, WalkForwardFold } from "@/components/strategy-table";
import { api } from "@/lib/api";
import { pickLatestRun } from "@/lib/runs";
import type { RunDetail, RunSummary } from "@/lib/types";

const RUNS_KEY = "api/runs";
const NO_RETRY = { shouldRetryOnError: false } as const;

/** Not in the current `RunDetail`; empty arrays keep the components honest. */
const NO_EQUITY: readonly number[] = [];
const NO_STRATEGIES: readonly StrategyRow[] = [];
const NO_FOLDS: readonly WalkForwardFold[] = [];

export default function PerformancePage() {
  const runsState = useSWR<RunSummary[]>(RUNS_KEY, api.getRuns, NO_RETRY);
  const runs = runsState.data ?? [];
  const selected = pickLatestRun(runs);

  const runId = selected?.run_id ?? null;
  const detailKey = runId === null ? null : `${RUNS_KEY}/${encodeURIComponent(runId)}`;
  const detailState = useSWR<RunDetail>(
    detailKey,
    runId === null ? null : () => api.getRun(runId),
    NO_RETRY,
  );
  const detail = detailState.data;
  const status = detail?.status ?? selected?.status ?? "";

  return (
    <section aria-labelledby="performance-heading" className="flex flex-col gap-4">
      <div>
        <h1 id="performance-heading" className="text-xl font-semibold text-slate-100">
          Performans
        </h1>
        <p className="text-xs text-slate-400">
          Backtest metrikleri — gerçek para değil.
        </p>
      </div>

      {runsState.isLoading ? (
        <p className="text-sm text-slate-400">Yükleniyor…</p>
      ) : selected === null ? (
        <div className="rounded-md border border-slate-800 bg-slate-900/40 px-4 py-8 text-center">
          <h2 className="text-base font-semibold text-slate-100">Henüz backtest yok</h2>
          <p className="mt-1 text-sm text-slate-400">
            Bir backtest koşusu tamamlandığında metrikler, drawdown ve strateji
            kırılımı burada görünecek.
          </p>
          <Link
            href="/"
            className="mt-4 inline-block rounded-sm border border-accent bg-accent/15 px-3 py-1.5 text-sm font-medium text-slate-100 hover:bg-accent/25 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Simülatöre git
          </Link>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-3 py-2">
            <span className="text-xs text-slate-400">Koşu</span>
            <span className="truncate font-mono text-sm text-slate-100">{runId}</span>
            <RunStatusChip status={status} />
          </div>

          {detailState.error && (
            <p role="status" className="text-xs text-slate-400">
              Bu koşunun metrikleri yüklenemedi.
            </p>
          )}

          <section aria-labelledby="metrics-heading">
            <h2 id="metrics-heading" className="mb-2 text-sm font-semibold text-slate-100">
              Metrikler
            </h2>
            <MetricGrid metrics={detail?.metrics ?? null} />
          </section>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <section aria-labelledby="drawdown-heading" className="min-w-0">
              <h2 id="drawdown-heading" className="mb-2 text-sm font-semibold text-slate-100">
                Drawdown
              </h2>
              <figure className="rounded-md border border-slate-800 bg-slate-900/40 p-3">
                <figcaption className="mb-1 text-xs text-slate-400">
                  Tepeye göre yüzde
                </figcaption>
                <DrawdownChart values={NO_EQUITY} />
              </figure>
            </section>

            <section aria-labelledby="strategies-heading" className="min-w-0">
              <h2 id="strategies-heading" className="mb-2 text-sm font-semibold text-slate-100">
                Strateji kırılımı
              </h2>
              <StrategyTable rows={NO_STRATEGIES} />
            </section>
          </div>

          <section aria-labelledby="walkforward-heading">
            <h2 id="walkforward-heading" className="mb-2 text-sm font-semibold text-slate-100">
              Walk-forward
            </h2>
            <WalkForwardTable folds={NO_FOLDS} />
          </section>
        </div>
      )}
    </section>
  );
}
