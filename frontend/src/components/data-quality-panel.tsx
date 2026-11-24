"use client";

/**
 * Read-only quality report for one stored slice.
 *
 * The panel reuses the page's symbol field and owns only the timeframe select.
 * Every number comes from `GET /api/data/quality`: a null count renders `—`
 * with a note saying why, because a zero would claim a check that never ran.
 * `configured: false` is shown as "not checked", not as a failure verdict.
 */

import { useState } from "react";

import { useDataQuality } from "@/hooks/use-data-quality";
import { useTimeframes } from "@/hooks/use-regime";
import { ApiError } from "@/lib/api";
import { formatNumber, formatTs } from "@/lib/format";
import type { DataQuality, DataQualityStatus } from "@/lib/types";

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const LABEL_CLASS = "text-xs text-slate-400";

const STATUS_LABEL: Record<DataQualityStatus, string> = {
  ok: "Sağlıklı",
  warn: "Uyarı",
  fail: "Hata",
};

const STATUS_TONE: Record<DataQualityStatus, string> = {
  ok: "border-emerald-600/60 bg-emerald-500/10 text-emerald-200",
  warn: "border-amber-600/60 bg-amber-500/10 text-amber-200",
  fail: "border-rose-600/60 bg-rose-500/10 text-rose-200",
};

const UNCONFIGURED_TONE = "border-slate-700 bg-slate-800/60 text-slate-300";

/**
 * One row per validator count. `outliers` can be null on a short slice while
 * the structural counts are real zeros, so each cell is rendered on its own.
 */
const COUNT_ROWS = [
  { key: "duplicates", label: "Yinelenen zaman damgası" },
  { key: "gaps", label: "Zaman boşlukları" },
  { key: "ohlc_violations", label: "OHLC tutarsızlığı" },
  { key: "crossed_quotes", label: "Kesişen kotasyon" },
  { key: "negative_spread", label: "Negatif spread" },
  { key: "negative_volume", label: "Negatif hacim" },
  { key: "outliers", label: "Aykırı değer (outlier)" },
] as const;

export interface DataQualityPanelProps {
  /** The page's symbol field; the panel never adds a second symbol input. */
  symbol: string;
}

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "bilinmeyen hata";
}

/** Why a `—` appears; null when every count was actually checked. */
function nullNote(quality: DataQuality): string | null {
  if (!COUNT_ROWS.some((row) => quality[row.key] === null)) return null;
  if (!quality.configured) {
    return "Bu dilimde doğrulanacak bar yok; sayaçlar kontrol edilmedi. Bu yüzden 0 değil — gösterilir.";
  }
  return `Outlier taraması çalışmadı: ${formatNumber(quality.rows, 0)} satır, ${formatNumber(
    quality.thresholds.outlier_min_rows,
    0,
  )} satırlık eşiğin altında (veya tarama kapalı). — "kontrol edilmedi" demektir, sıfır değil.`;
}

function CountRow({ quality, row }: { quality: DataQuality; row: (typeof COUNT_ROWS)[number] }) {
  const value = quality[row.key];
  return (
    <tr className="border-t border-slate-800/70">
      <th scope="row" className="py-1 pr-4 text-left font-normal text-slate-300">
        {row.label}
      </th>
      <td
        data-testid={`quality-count-${row.key}`}
        data-null={value === null ? "true" : "false"}
        className="py-1 text-right tabular-nums text-slate-100"
      >
        {value === null ? "—" : formatNumber(value, 0)}
      </td>
    </tr>
  );
}

function QualityBody({ quality }: { quality: DataQuality }) {
  const provenance = quality.provenance;
  const note = nullNote(quality);

  return (
    <div className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3">
      <header className="flex flex-wrap items-center gap-2">
        <span
          data-testid="quality-status"
          data-status={quality.configured ? quality.status : "unconfigured"}
          className={`rounded-sm border px-2 py-0.5 text-xs font-medium ${
            quality.configured ? STATUS_TONE[quality.status] : UNCONFIGURED_TONE
          }`}
        >
          {quality.configured ? STATUS_LABEL[quality.status] : "Yapılandırılmadı"}
        </span>
        <span className="text-xs tabular-nums text-slate-400">
          {quality.symbol} · {quality.timeframe} · {formatNumber(quality.rows, 0)} bar
        </span>
        <code data-testid="quality-rule" className="text-xs text-slate-500">
          {quality.status_rule}
        </code>
      </header>

      <p data-testid="quality-detail" className="text-sm text-slate-300">
        {quality.status_detail}
      </p>

      {!quality.configured && (
        <p data-testid="quality-unconfigured" className="text-xs text-amber-200">
          Bu dilim için doğrulama yapılmadı: depoda {quality.timeframe} barı yok.
          Sayımlar tahmin edilmez; aşağıdaki sayaçlar bu yüzden — gösterir.
        </p>
      )}

      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="flex flex-col gap-0.5">
          <dt className={LABEL_CLASS}>İlk bar</dt>
          <dd data-testid="quality-first-ts" className="text-sm tabular-nums text-slate-200">
            {formatTs(quality.first_ts)}
          </dd>
        </div>
        <div className="flex flex-col gap-0.5">
          <dt className={LABEL_CLASS}>Son bar</dt>
          <dd data-testid="quality-last-ts" className="text-sm tabular-nums text-slate-200">
            {formatTs(quality.last_ts)}
          </dd>
        </div>
        <div className="flex flex-col gap-0.5">
          <dt className={LABEL_CLASS}>Köken</dt>
          <dd
            data-testid="quality-provenance"
            data-kind={
              provenance === null ? "missing" : provenance.synthetic ? "synthetic" : "real"
            }
            className="text-sm text-slate-200"
          >
            {provenance === null
              ? "Köken kaydı yok"
              : provenance.synthetic
                ? "Sentetik"
                : "Gerçek"}
          </dd>
        </div>
        {provenance !== null && (
          <div className="flex flex-col gap-0.5 sm:col-span-2">
            <dt className={LABEL_CLASS}>Kaynak</dt>
            <dd
              data-testid="quality-source"
              className="break-all font-mono text-xs text-slate-200"
            >
              {provenance.source}
            </dd>
          </div>
        )}
      </dl>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">
            Doğrulama sayaçları — {quality.symbol} {quality.timeframe}
          </caption>
          <thead>
            <tr>
              <th scope="col" className="pb-1 pr-4 text-left text-xs font-medium text-slate-400">
                Kontrol
              </th>
              <th scope="col" className="pb-1 text-right text-xs font-medium text-slate-400">
                Adet
              </th>
            </tr>
          </thead>
          <tbody>
            {COUNT_ROWS.map((row) => (
              <CountRow key={row.key} quality={quality} row={row} />
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-500">
        0 = kontrol edildi ve temiz; — = kontrol edilmedi.
      </p>
      {note !== null && (
        <p data-testid="quality-null-note" className="text-xs text-slate-400">
          {note}
        </p>
      )}

      <p data-testid="quality-thresholds" className="text-xs text-slate-500">
        Eşikler: boşluk ≥ {formatNumber(quality.thresholds.gap_threshold_minutes, 0)} dk ·
        getiri z {formatNumber(quality.thresholds.outlier_return_z, 1)} · hacim z{" "}
        {formatNumber(quality.thresholds.outlier_volume_z, 1)} · outlier min{" "}
        {formatNumber(quality.thresholds.outlier_min_rows, 0)} satır
      </p>
    </div>
  );
}

export function DataQualityPanel({ symbol }: DataQualityPanelProps) {
  const [timeframe, setTimeframe] = useState("M1");

  const timeframesState = useTimeframes();
  const timeframes =
    timeframesState.data !== undefined && timeframesState.data.length > 0
      ? timeframesState.data.map((item) => item.timeframe)
      : ["M1"];

  const cleanSymbol = symbol.trim();
  const qualityState = useDataQuality(cleanSymbol, timeframe);
  const quality = qualityState.data;

  // An error with a previous report still shows that report, plus the note.
  const statusDetail =
    qualityState.error === undefined ? null : errorDetail(qualityState.error);
  const loading = qualityState.isLoading && quality === undefined;
  const loadLabel =
    statusDetail !== null
      ? "Hata"
      : quality !== undefined
        ? "Güncel"
        : loading
          ? "Yükleniyor…"
          : "Bekliyor";

  return (
    <section aria-labelledby="quality-heading" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 id="quality-heading" className="text-sm font-semibold text-slate-100">
            Veri kalitesi
          </h2>
          <p className="text-xs text-slate-400">
            Salt okunur doğrulama — hiçbir bar onarılmaz, doldurulmaz veya silinmez.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1">
            <label htmlFor="quality-timeframe" className={LABEL_CLASS}>
              Zaman dilimi
            </label>
            <select
              id="quality-timeframe"
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
          {cleanSymbol !== "" && (
            <span className="pb-2 text-xs tabular-nums text-slate-400">
              Sembol: {cleanSymbol}
            </span>
          )}
          <span
            data-testid="quality-load"
            role="status"
            aria-live="polite"
            className="pb-2 text-xs text-slate-400"
          >
            {loadLabel}
          </span>
        </div>
      </div>

      {statusDetail !== null && (
        <p data-testid="quality-error" role="alert" className="text-sm text-rose-300">
          Kalite raporu alınamadı ({statusDetail}).
        </p>
      )}

      {cleanSymbol === "" ? (
        <p className="text-sm text-slate-400">
          Sembol girerek kalite raporunu görüntüle.
        </p>
      ) : quality === undefined ? (
        statusDetail === null && (
          <p data-testid="quality-loading" className="text-sm text-slate-400">
            Yükleniyor…
          </p>
        )
      ) : (
        <QualityBody quality={quality} />
      )}
    </section>
  );
}
