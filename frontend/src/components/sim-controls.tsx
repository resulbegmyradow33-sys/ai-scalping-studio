"use client";

/**
 * Playback controls for the simulator.
 *
 * The speed is local UI state: the parent only learns it when the user starts,
 * so changing the speed never restarts a running episode. Buttons are plain
 * `<button>` elements with `aria-pressed` on the speed group, which keeps the
 * tab order equal to the visual order.
 */

import { useState } from "react";

import { useDataStatus } from "@/hooks/use-data-status";
import type { ControlAction, SimState } from "@/lib/types";

const SPEEDS = [
  { label: "1×", value: 1 },
  { label: "10×", value: 10 },
  { label: "100×", value: 100 },
  { label: "Max", value: 0 },
] as const;

export interface SimControlsProps {
  status: SimState;
  /** Symbol the simulator replays; the source label follows its stored data. */
  symbol: string;
  onStart: (speed: number) => void;
  onControl: (action: ControlAction) => void;
}

/** Wire provider ids → display labels; unknown ids stay readable as-is. */
const PROVIDER_LABELS: Readonly<Record<string, string>> = {
  synthetic: "sentetik",
  csv: "CSV",
  csv_no_spread: "CSV",
  dukascopy: "Dukascopy",
};

/** `["synthetic", "csv"]` → `sentetik, CSV`; no providers → `bilinmiyor`. */
export function formatDataSources(providers: readonly string[]): string {
  if (providers.length === 0) return "bilinmiyor";
  return providers.map((id) => PROVIDER_LABELS[id] ?? id).join(", ");
}

const BUTTON_BASE =
  "rounded-sm border px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

function actionClass(enabled: boolean, primary = false): string {
  if (!enabled) {
    return `${BUTTON_BASE} cursor-not-allowed border-slate-800 text-slate-600`;
  }
  if (primary) {
    return `${BUTTON_BASE} border-accent bg-accent/15 font-medium text-slate-100 hover:bg-accent/25`;
  }
  return `${BUTTON_BASE} border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100`;
}

function speedClass(selected: boolean): string {
  if (selected) {
    return `${BUTTON_BASE} border-accent bg-accent/20 font-medium text-slate-100`;
  }
  return `${BUTTON_BASE} border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-100`;
}

export function SimControls({ status, symbol, onStart, onControl }: SimControlsProps) {
  const [speed, setSpeed] = useState(1);

  // The data page owns the same SWR key, so this is not a second poll loop;
  // SWR dedupes and shares the 30 s refresh.
  const providers = useDataStatus(symbol).data?.providers ?? [];
  const source = formatDataSources(providers);

  const running = status === "running";
  const paused = status === "paused";
  const startEnabled = !running && !paused;
  const stopEnabled = running || paused;
  // A stopped episode can be continued: the backend's resume clears its stop
  // event. Without this, `Durdur` would leave `Devam` permanently disabled.
  const resumeEnabled = paused || status === "stopped";

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => onStart(speed)}
          disabled={!startEnabled}
          className={actionClass(startEnabled, true)}
        >
          Başlat
        </button>
        <button
          type="button"
          onClick={() => onControl("stop")}
          disabled={!stopEnabled}
          className={actionClass(stopEnabled)}
        >
          Durdur
        </button>
        <button
          type="button"
          onClick={() => onControl("resume")}
          disabled={!resumeEnabled}
          className={actionClass(resumeEnabled)}
        >
          Devam
        </button>
        <button
          type="button"
          onClick={() => onControl("reset")}
          disabled={status === "running"}
          className={actionClass(status !== "running")}
        >
          Sıfırla
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <p className="text-xs text-slate-400">
          Veri kaynağı: {source} · Gerçek para değil
        </p>
        <div role="group" aria-label="Oynatma hızı" className="flex items-center gap-1">
          {SPEEDS.map((option) => (
            <button
              key={option.label}
              type="button"
              aria-pressed={speed === option.value}
              onClick={() => setSpeed(option.value)}
              className={speedClass(speed === option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
