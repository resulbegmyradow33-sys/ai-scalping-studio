"use client";

/**
 * The training screen: run selection, four PPO curves, bandit weights and the
 * run's progress. `/api/runs` belongs to the RL plan and does not exist yet, so
 * a 404 (or a backend that is down) is the empty state, never a crash. SWR
 * retries are off for these keys: an absent endpoint must not become a polling
 * loop.
 */

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { LegacyBadge } from "@/components/legacy-badge";
import { LineChart } from "@/components/line-chart";
import { RunStatusChip, RunTable } from "@/components/run-table";
import { api } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import { BANDIT_STRATEGIES, pickLatestRun, toSeries } from "@/lib/runs";
import type { SeriesPoint } from "@/lib/runs";
import type { RunControlAction, RunDetail, RunSummary } from "@/lib/types";

const RUNS_KEY = "api/runs";
const CONTROL_TITLE = "Eğitim altyapısı henüz yok";
/** `POST /api/runs/{id}/control` ships with the RL API; flip when it lands. */
const CONTROL_READY = false;
const NO_RETRY = { shouldRetryOnError: false } as const;

const BUTTON_BASE =
  "rounded-sm border px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

function controlClass(enabled: boolean): string {
  return enabled
    ? `${BUTTON_BASE} border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100`
    : `${BUTTON_BASE} cursor-not-allowed border-slate-800 text-slate-600`;
}

function finiteNumber(value: number | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** `total_steps` lives in the run config until the detail schema promotes it. */
function configTotal(config: Record<string, unknown> | undefined): number | null {
  const value = config?.total_steps;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

interface CurveCardProps {
  label: string;
  points: readonly SeriesPoint[];
  color: string;
}

function CurveCard({ label, points, color }: CurveCardProps) {
  const latest = points.at(-1)?.value ?? null;

  return (
    <figure className="min-w-0 rounded-md border border-slate-800 bg-slate-900/40 p-3">
      <figcaption className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-xs font-medium text-slate-300">{label}</span>
        <span className="text-xs tabular-nums text-slate-400">{formatNumber(latest, 3)}</span>
      </figcaption>
      <LineChart title={label} points={points} color={color} />
    </figure>
  );
}

export default function TrainingPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);

  const runsState = useSWR<RunSummary[]>(RUNS_KEY, api.getRuns, NO_RETRY);
  const runs = runsState.data ?? [];
  const latest = pickLatestRun(runs);
  const selected =
    runs.find((run) => run.run_id === selectedId) ?? latest;

  const runId = selected?.run_id ?? null;
  const detailKey = runId === null ? null : `${RUNS_KEY}/${encodeURIComponent(runId)}`;
  const detailState = useSWR<RunDetail>(
    detailKey,
    runId === null ? null : () => api.getRun(runId),
    NO_RETRY,
  );
  const detail = detailState.data;

  const series = detail?.series;
  const reward = toSeries(series?.reward ?? []);
  const episodeLength = toSeries(series?.episode_length ?? []);
  const entropy = toSeries(series?.entropy ?? []);
  const loss = toSeries(series?.loss ?? []);

  const status = detail?.status ?? selected?.status ?? "";

  const step = finiteNumber(detail?.step ?? selected?.steps);
  // A zero from the detail API is a placeholder, not a real denominator; fall
  // back to the run config instead of hiding the whole progress bar.
  const detailTotal = finiteNumber(detail?.total_steps);
  const totalSteps =
    detailTotal !== null && detailTotal > 0 ? detailTotal : configTotal(detail?.config);
  const totalValue = totalSteps ?? 0;
  const hasTotal = totalValue > 0;
  // Without a known total the bar is indeterminate: reporting `step / 1`
  // would announce 100% for a run that may just have started.
  const progressNow = hasTotal ? Math.min(Math.max(step ?? 0, 0), totalValue) : 0;
  const progressPct = hasTotal ? (progressNow / totalValue) * 100 : 0;

  const stopEnabled =
    CONTROL_READY && (status === "running" || status === "paused");
  const resumeEnabled =
    CONTROL_READY && (status === "paused" || status === "stopped");
  const stopTitle = CONTROL_READY ? "Bu koşu durdurulamaz" : CONTROL_TITLE;
  const resumeTitle = CONTROL_READY ? "Bu koşu devam ettirilemez" : CONTROL_TITLE;

  const weights = detail?.bandit ?? [];
  const banditRows = BANDIT_STRATEGIES.map((strategy) => {
    const match = weights.find((entry) => entry.strategy === strategy);
    return {
      strategy,
      weight: match && Number.isFinite(match.weight) ? match.weight : null,
    };
  });

  const handleControl = (action: RunControlAction): void => {
    if (runId === null) return;
    if (action === "stop" ? !stopEnabled : !resumeEnabled) return;

    setControlError(null);
    void api.controlRun(runId, action).catch(() => {
      setControlError("Kontrol isteği gönderilemedi.");
    });
  };

  return (
    <section aria-labelledby="training-heading" className="flex flex-col gap-4">
      <div>
        <h1 id="training-heading" className="text-xl font-semibold text-slate-100">
          Eğitim
        </h1>
        <p className="text-xs text-slate-400">
          PPO eğitimi — sentetik veri, gerçek para değil.
        </p>
      </div>

      {runsState.isLoading ? (
        <p className="text-sm text-slate-400">Yükleniyor…</p>
      ) : runs.length === 0 ? (
        <div className="rounded-md border border-slate-800 bg-slate-900/40 px-4 py-8 text-center">
          <h2 className="text-base font-semibold text-slate-100">Henüz eğitim yok</h2>
          <p className="mt-1 text-sm text-slate-400">
            Eğitim başlatıldığında ölçümler burada görünecek.
          </p>
          <Link
            href="/data"
            className="mt-4 inline-block rounded-sm border border-accent bg-accent/15 px-3 py-1.5 text-sm font-medium text-slate-100 hover:bg-accent/25 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Veri ekranına git
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          <div className="flex min-w-0 flex-col gap-4 lg:col-span-8">
            <div className="rounded-md border border-slate-800 bg-slate-900/40 px-3 py-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <span className="truncate font-mono text-sm text-slate-100">
                    {runId}
                  </span>
                  {selected && <RunStatusChip status={status} />}
                </div>
                <p
                  data-testid="training-progress"
                  className="text-xs tabular-nums text-slate-300"
                >
                  {formatNumber(step, 0)} / {formatNumber(hasTotal ? totalValue : null, 0)}
                </p>
              </div>

              <div
                role="progressbar"
                aria-label="Eğitim ilerlemesi"
                aria-valuemin={0}
                aria-valuemax={hasTotal ? totalValue : undefined}
                aria-valuenow={hasTotal ? progressNow : undefined}
                className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-800"
              >
                <div className="h-full bg-accent" style={{ width: `${progressPct}%` }} />
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleControl("stop")}
                  disabled={!stopEnabled}
                  title={stopTitle}
                  className={controlClass(stopEnabled)}
                >
                  Durdur
                </button>
                <button
                  type="button"
                  onClick={() => handleControl("resume")}
                  disabled={!resumeEnabled}
                  title={resumeTitle}
                  className={controlClass(resumeEnabled)}
                >
                  Devam
                </button>
                {controlError && (
                  <p role="alert" className="text-xs text-loss">
                    {controlError}
                  </p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <CurveCard label="Ödül" points={reward} color="var(--color-profit)" />
              <CurveCard
                label="Bölüm uzunluğu"
                points={episodeLength}
                color="var(--color-accent)"
              />
              <CurveCard label="Entropi" points={entropy} color="var(--color-neutral)" />
              <CurveCard label="Kayıp" points={loss} color="var(--color-loss)" />
            </div>
          </div>

          <aside className="flex min-w-0 flex-col gap-4 lg:col-span-4">
            <section aria-labelledby="runs-heading" className="min-w-0">
              <h2 id="runs-heading" className="mb-2 text-sm font-semibold text-slate-100">
                Koşular
              </h2>
              <RunTable
                runs={runs}
                selectedId={runId}
                onSelect={setSelectedId}
              />
            </section>

            <section aria-labelledby="bandit-heading" className="min-w-0">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <h2 id="bandit-heading" className="text-sm font-semibold text-slate-100">
                  Bandit ağırlıkları
                </h2>
                <LegacyBadge testId="bandit-legacy" />
              </div>
              <div
                role="region"
                aria-label="Bandit ağırlıkları"
                tabIndex={0}
                className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                <table
                  data-testid="bandit-weights"
                  className="w-full min-w-[18rem] border-collapse text-left text-xs"
                >
                  <caption className="sr-only">
                    Bandit strateji ağırlıkları (legacy baseline)
                  </caption>
                  <thead className="bg-slate-900/60 text-slate-400">
                    <tr>
                      <th scope="col" className="px-2 py-1.5 font-medium">Strateji</th>
                      <th scope="col" className="px-2 py-1.5 text-right font-medium">Ağırlık</th>
                    </tr>
                  </thead>
                  <tbody>
                    {banditRows.map((row) => (
                      <tr key={row.strategy} className="border-t border-slate-800/60">
                        <th scope="row" className="px-2 py-1.5 font-normal text-slate-300">
                          {row.strategy}
                        </th>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.weight, 3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </aside>
        </div>
      )}
    </section>
  );
}
