"use client";

/**
 * Market Intelligence: the descriptive regime state of one stored slice.
 *
 * Every value on this page comes from `GET /api/regime`; nothing is derived or
 * invented in the browser. Labels describe the recent path — they are weather,
 * not a forecast — and the footnote says so.
 */

import { useState } from "react";

import { useRegime, useTimeframes } from "@/hooks/use-regime";
import { ApiError } from "@/lib/api";
import { formatNumber, formatTs } from "@/lib/format";
import type { RegimeCurrent, RegimePoint } from "@/lib/types";

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const LABEL_CLASS = "text-xs text-slate-400";

/** Cells rendered for the timeline; the API may return up to 500 points. */
const TIMELINE_POINTS = 120;

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "Rejim verisi alınamadı.";
}

/** Colour by the summary label the backend returned; no new categories. */
function regimeTone(regime: string): string {
  if (regime === "unusual") return "bg-fuchsia-400";
  if (regime === "breakout_up") return "bg-emerald-300";
  if (regime === "breakout_down") return "bg-rose-300";
  if (regime.startsWith("up")) return "bg-emerald-500";
  if (regime.startsWith("down")) return "bg-rose-500";
  return "bg-slate-500";
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-baseline gap-1 rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-sm font-medium text-slate-100">{value}</span>
    </span>
  );
}

function CurrentCard({ current }: { current: RegimeCurrent }) {
  return (
    <article
      data-testid="regime-current"
      className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
    >
      <header className="flex flex-wrap items-baseline gap-2">
        <span className="text-xs text-slate-400">Şu anki rejim</span>
        <strong
          data-testid="current-regime"
          className="font-mono text-lg font-semibold text-slate-100"
        >
          {current.regime}
        </strong>
        <span data-testid="current-ts" className="ml-auto text-xs tabular-nums text-slate-400">
          {formatTs(current.ts)}
        </span>
      </header>
      <div className="flex flex-wrap gap-2">
        <span data-testid="current-trend">
          <Chip label="trend" value={`${current.trend} (${current.trend_strength})`} />
        </span>
        <span data-testid="current-volatility">
          <Chip label="volatilite" value={current.volatility} />
        </span>
        <span data-testid="current-expansion">
          <Chip label="genişleme" value={current.expansion} />
        </span>
        <span data-testid="current-breakout">
          <Chip label="breakout" value={current.breakout} />
        </span>
        <span data-testid="current-unusual">
          <Chip label="sıra dışı" value={current.unusual ? "evet" : "hayır"} />
        </span>
      </div>
    </article>
  );
}

function Timeline({ points }: { points: readonly RegimePoint[] }) {
  const visible = points.slice(-TIMELINE_POINTS);
  return (
    <div className="flex flex-col gap-1.5">
      <h2 className="text-sm font-semibold text-slate-100">Rejim zaman çizgisi</h2>
      <div
        data-testid="regime-timeline"
        role="img"
        aria-label={`Son ${visible.length} barın rejim etiketleri`}
        className="flex h-10 items-stretch gap-px overflow-hidden rounded-sm border border-slate-800 bg-slate-950/60 p-1"
      >
        {visible.map((point) => (
          <span
            key={point.ts}
            data-testid="regime-point"
            data-regime={point.regime}
            title={`${formatTs(point.ts)} · ${point.regime}`}
            className={`min-w-0 flex-1 rounded-[1px] ${regimeTone(point.regime)}`}
          />
        ))}
      </div>
      <p className="text-xs text-slate-500">
        {visible.length} bar · renkler özet rejim etiketidir; her hücrenin
        üzerine gelince saat ve etiket görünür.
      </p>
    </div>
  );
}

export default function IntelligencePage() {
  const [symbol, setSymbol] = useState("XAUUSD");
  const [timeframe, setTimeframe] = useState("M1");

  const cleanSymbol = symbol.trim();
  const timeframesState = useTimeframes();
  const timeframes =
    timeframesState.data !== undefined && timeframesState.data.length > 0
      ? timeframesState.data.map((item) => item.timeframe)
      : ["M1"];

  const regimeState = useRegime(cleanSymbol, timeframe);
  const data = regimeState.data;
  const rows = data?.rows ?? 0;
  const current = data?.current ?? null;
  const counts = data?.counts ?? [];
  const series = data?.series ?? [];

  const statusDetail =
    regimeState.error === undefined ? null : errorDetail(regimeState.error);

  return (
    <section aria-labelledby="intelligence-heading" className="flex flex-col gap-5">
      <div>
        <h1 id="intelligence-heading" className="text-xl font-semibold text-slate-100">
          Piyasa Zekâsı
        </h1>
        <p className="text-xs text-slate-400">
          Kural tabanlı rejim etiketleri — piyasanın son hâlini tanımlar, fiyat
          tahmini değildir. Gerçek para yok.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <label htmlFor="regime-symbol" className={LABEL_CLASS}>
            Sembol
          </label>
          <input
            id="regime-symbol"
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
            autoComplete="off"
            spellCheck={false}
            className={FIELD_CLASS}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="regime-timeframe" className={LABEL_CLASS}>
            Zaman dilimi
          </label>
          <select
            id="regime-timeframe"
            value={timeframe}
            onChange={(event) => setTimeframe(event.target.value)}
            className={FIELD_CLASS}
          >
            {timeframes.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
        <span className="pb-2 text-xs tabular-nums text-slate-500">
          {data === undefined ? "" : `${formatNumber(rows, 0)} bar`}
        </span>
      </div>

      {statusDetail !== null && (
        <p data-testid="regime-error" role="status" className="text-sm text-amber-200">
          Rejim verisi alınamadı ({statusDetail}).
        </p>
      )}

      {cleanSymbol === "" ? (
        <p className="text-sm text-slate-400">Sembol girerek rejimi görüntüle.</p>
      ) : data === undefined ? (
        statusDetail === null ? (
          <p data-testid="regime-loading" className="text-sm text-slate-400">
            Yükleniyor…
          </p>
        ) : null
      ) : rows === 0 ? (
        <p data-testid="regime-empty" className="text-sm text-slate-400">
          Bu sembol ve zaman diliminde veri yok — Veri sayfasından içe aktar.
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          {current !== null && <CurrentCard current={current} />}

          <section aria-labelledby="regime-counts-heading" className="flex flex-col gap-2">
            <h2 id="regime-counts-heading" className="text-sm font-semibold text-slate-100">
              Rejim dağılımı
            </h2>
            <ul className="flex flex-col gap-1.5">
              {counts.map((row) => (
                <li key={row.regime} data-testid={`regime-count-${row.regime}`}>
                  <div className="flex items-baseline gap-2 text-xs">
                    <span className="w-40 shrink-0 truncate font-mono text-slate-200">
                      {row.regime}
                    </span>
                    <span className="w-20 shrink-0 text-right tabular-nums text-slate-400">
                      {formatNumber(row.bars, 0)} bar
                    </span>
                    <span className="w-14 shrink-0 text-right tabular-nums text-slate-500">
                      {(row.share * 100).toFixed(1)}%
                    </span>
                    <span className="h-2 flex-1 overflow-hidden rounded-sm bg-slate-800">
                      <span
                        className={`block h-full ${regimeTone(row.regime)}`}
                        style={{ width: `${Math.max(0, Math.min(100, row.share * 100))}%` }}
                      />
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          </section>

          <Timeline points={series} />

          <p className="text-xs text-slate-500">
            Dağılım tüm store dilimini kapsar; zaman çizgisi yalnızca son
            barları gösterir. Rejim bir tahmin değil, geçmiş yolun tanımıdır.
          </p>
        </div>
      )}
    </section>
  );
}
