/**
 * Training-run list.
 *
 * The horizontal wrapper is focusable on purpose: on a narrow screen the table
 * scrolls sideways, and a keyboard user needs a focus target that can scroll
 * it. The run id is a button so selection follows the tab order.
 */

import { formatNumber, formatTs } from "@/lib/format";
import type { RunSummary } from "@/lib/types";

const STATUS_LABEL: Record<string, string> = {
  queued: "Sırada",
  running: "Çalışıyor",
  paused: "Duraklatıldı",
  stopping: "Durduruluyor",
  stopped: "Durduruldu",
  completed: "Bitti",
  finished: "Bitti",
  failed: "Hata",
  error: "Hata",
};

const STATUS_CLASS: Record<string, string> = {
  queued: "border-slate-700 text-slate-300",
  running: "border-accent/50 bg-accent/10 text-accent",
  paused: "border-amber-500/50 bg-amber-500/10 text-amber-300",
  stopping: "border-amber-500/50 bg-amber-500/10 text-amber-300",
  stopped: "border-slate-700 text-slate-300",
  completed: "border-profit/50 bg-profit/10 text-profit",
  finished: "border-profit/50 bg-profit/10 text-profit",
  failed: "border-loss/50 bg-loss/10 text-loss",
  error: "border-loss/50 bg-loss/10 text-loss",
};

export function RunStatusChip({ status }: { status: string }) {
  const key = status.toLowerCase();
  const label = STATUS_LABEL[key] ?? status;
  const tone = STATUS_CLASS[key] ?? "border-slate-700 text-slate-300";

  return (
    <span className={`inline-block rounded-sm border px-2 py-0.5 text-xs ${tone}`}>
      {label}
    </span>
  );
}

export interface RunTableProps {
  runs: readonly RunSummary[];
  selectedId: string | null;
  onSelect: (runId: string) => void;
}

export function RunTable({ runs, selectedId, onSelect }: RunTableProps) {
  return (
    <div
      role="region"
      aria-label="Eğitim koşuları"
      tabIndex={0}
      className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <table className="w-full min-w-[34rem] border-collapse text-left text-xs">
        <caption className="sr-only">Eğitim koşuları</caption>
        <thead className="bg-slate-900/60 text-slate-400">
          <tr>
            <th scope="col" className="px-2 py-1.5 font-medium">Koşu</th>
            <th scope="col" className="px-2 py-1.5 font-medium">Durum</th>
            <th scope="col" className="px-2 py-1.5 text-right font-medium">Adım</th>
            <th scope="col" className="px-2 py-1.5 text-right font-medium">Ort. ödül</th>
            <th scope="col" className="px-2 py-1.5 font-medium">Tarih</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const selected = run.run_id === selectedId;
            return (
              <tr
                key={run.run_id}
                className={selected ? "bg-slate-800/70" : "hover:bg-slate-900/60"}
              >
                <th scope="row" className="px-2 py-1.5 font-normal">
                  <button
                    type="button"
                    aria-pressed={selected}
                    onClick={() => onSelect(run.run_id)}
                    className="rounded-sm font-mono text-slate-100 underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  >
                    {run.run_id}
                  </button>
                </th>
                <td className="px-2 py-1.5">
                  <RunStatusChip status={run.status} />
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">
                  {formatNumber(run.steps, 0)}
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">
                  {formatNumber(run.reward_mean, 2)}
                </td>
                <td className="px-2 py-1.5 tabular-nums text-slate-400">
                  {formatTs(run.created_at)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
