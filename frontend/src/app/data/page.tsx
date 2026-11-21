"use client";

/**
 * The data screen: what the local store holds and how to add more.
 *
 * The status hook is keyed by the symbol field, so the page never polls a
 * second time on its own (the hook owns the 30 s refresh). The download form
 * is one request at a time; a successful call refetches the status key and
 * reports the added rows, a failure shows the backend's `detail` verbatim.
 */

import { useState } from "react";
import type { FormEvent } from "react";

import { CoverageBar } from "@/components/coverage-bar";
import { ProviderCard } from "@/components/provider-card";
import { useDataStatus } from "@/hooks/use-data-status";
import { ApiError, api } from "@/lib/api";
import { formatNumber, formatTs } from "@/lib/format";
import type { ProviderName } from "@/lib/types";

const PROVIDERS = [
  { id: "synthetic", label: "Sentetik" },
  { id: "csv", label: "CSV" },
  { id: "dukascopy", label: "Dukascopy" },
] as const satisfies readonly { id: ProviderName; label: string }[];

type Phase = "idle" | "running" | "done" | "error";

const PHASE_LABEL: Record<Phase, string> = {
  idle: "Hazır",
  running: "İndiriliyor…",
  done: "Tamamlandı",
  error: "Hata",
};

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const LABEL_CLASS = "text-xs text-slate-400";

/** `YYYY-MM-DD` for a date input; `offset` is in days from today. */
function isoDay(offset: number): string {
  const date = new Date();
  date.setDate(date.getDate() + offset);
  return date.toISOString().slice(0, 10);
}

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "İndirme başarısız oldu.";
}

export default function DataPage() {
  const [symbol, setSymbol] = useState("XAUUSD");
  const [provider, setProvider] = useState<ProviderName>("synthetic");
  const [start, setStart] = useState(() => isoDay(-9));
  const [end, setEnd] = useState(() => isoDay(0));
  const [seed, setSeed] = useState("7");
  const [path, setPath] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [lastRows, setLastRows] = useState<number | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const cleanSymbol = symbol.trim();
  const statusState = useDataStatus(cleanSymbol);
  const status = statusState.data;
  const rows = status?.rows ?? 0;
  const providers = status?.providers ?? [];

  const statusDetail =
    statusState.error === undefined
      ? null
      : statusState.error instanceof ApiError
        ? statusState.error.detail
        : statusState.error instanceof Error
          ? statusState.error.message
          : "bilinmeyen hata";

  const busy = phase === "running";
  const missingPath = provider === "csv" && path.trim() === "";
  const canSubmit =
    cleanSymbol !== "" && start !== "" && end !== "" && !missingPath && !busy;

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (!canSubmit) return;

    setPhase("running");
    setLastRows(null);
    setDownloadError(null);

    void (async () => {
      try {
        const trimmedSeed = seed.trim();
        const result = await api.downloadData({
          provider,
          symbol: cleanSymbol,
          start: `${start}T00:00:00Z`,
          end: `${end}T23:59:00Z`,
          seed: provider === "synthetic" && trimmedSeed !== "" ? Number(trimmedSeed) : null,
          path: provider === "csv" ? path.trim() : null,
        });
        setLastRows(result.rows);
        setPhase("done");
        // Refetch only after the write landed, so the cards show the new rows.
        void statusState.mutate();
      } catch (error) {
        setDownloadError(errorDetail(error));
        setPhase("error");
      }
    })();
  };

  return (
    <section aria-labelledby="data-heading" className="flex flex-col gap-5">
      <div>
        <h1 id="data-heading" className="text-xl font-semibold text-slate-100">
          Veri
        </h1>
        <p className="text-xs text-slate-400">
          Yerel Parquet deposu — M1 barlar. Gerçek para değil.
        </p>
      </div>

      <section aria-labelledby="coverage-heading" className="flex flex-col gap-2">
        <div className="flex flex-wrap items-baseline gap-3">
          <h2 id="coverage-heading" className="text-sm font-semibold text-slate-100">
            Kapsam
          </h2>
          <span className="text-xs tabular-nums text-slate-400">{cleanSymbol}</span>
        </div>

        {statusDetail !== null && (
          <p role="status" className="text-sm text-amber-200">
            Veri durumu alınamadı ({statusDetail}).
          </p>
        )}

        {cleanSymbol === "" ? (
          <p className="text-sm text-slate-400">Sembol girerek kapsamı görüntüle.</p>
        ) : statusState.isLoading && status === undefined ? (
          <p className="text-sm text-slate-400">Yükleniyor…</p>
        ) : rows === 0 ? (
          <p data-testid="data-empty" className="text-sm text-slate-400">
            Henüz veri yok — aşağıdaki formdan sentetik veri üret veya CSV içe aktar.
          </p>
        ) : (
          <div className="flex flex-col gap-1.5">
            <CoverageBar firstTs={status?.first_ts ?? null} lastTs={status?.last_ts ?? null} rows={rows} />
            <p className="text-xs tabular-nums text-slate-400">
              {formatNumber(rows, 0)} bar · {formatTs(status?.first_ts ?? null)} →{" "}
              {formatTs(status?.last_ts ?? null)}
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
          {PROVIDERS.map((item) => {
            const active = providers.includes(item.id);
            return (
              <ProviderCard
                key={item.id}
                provider={item.id}
                label={item.label}
                active={active}
                rows={active ? rows : 0}
                firstTs={active ? (status?.first_ts ?? null) : null}
                lastTs={active ? (status?.last_ts ?? null) : null}
              />
            );
          })}
        </div>
        <p className="text-xs text-slate-500">
          Kartlardaki aralık ve satır sayısı sembolün depo toplamıdır; nokta
          sağlayıcının veriye katkısını gösterir.
        </p>
      </section>

      <form
        data-testid="download-form"
        onSubmit={handleSubmit}
        aria-busy={busy}
        className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold text-slate-100">Veri indir</h2>
          <span data-testid="download-state" className="text-xs text-slate-400" aria-live="polite">
            {PHASE_LABEL[phase]}
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <div className="flex flex-col gap-1">
            <label htmlFor="data-provider" className={LABEL_CLASS}>
              Sağlayıcı
            </label>
            <select
              id="data-provider"
              value={provider}
              onChange={(event) => {
                const next = event.target.value as ProviderName;
                setProvider(next);
                setLastRows(null);
                setDownloadError(null);
                setPhase("idle");
              }}
              className={FIELD_CLASS}
            >
              {PROVIDERS.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor="data-symbol" className={LABEL_CLASS}>
              Sembol
            </label>
            <input
              id="data-symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
              required
              autoComplete="off"
              spellCheck={false}
              className={FIELD_CLASS}
            />
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor="data-start" className={LABEL_CLASS}>
              Başlangıç
            </label>
            <input
              id="data-start"
              type="date"
              value={start}
              onChange={(event) => setStart(event.target.value)}
              required
              className={FIELD_CLASS}
            />
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor="data-end" className={LABEL_CLASS}>
              Bitiş
            </label>
            <input
              id="data-end"
              type="date"
              value={end}
              onChange={(event) => setEnd(event.target.value)}
              required
              className={FIELD_CLASS}
            />
          </div>

          {provider === "synthetic" ? (
            <div className="flex flex-col gap-1">
              <label htmlFor="data-seed" className={LABEL_CLASS}>
                Seed
              </label>
              <input
                id="data-seed"
                type="number"
                inputMode="numeric"
                value={seed}
                onChange={(event) => setSeed(event.target.value)}
                className={FIELD_CLASS}
              />
            </div>
          ) : provider === "csv" ? (
            <div className="flex flex-col gap-1 sm:col-span-2">
              <label htmlFor="data-path" className={LABEL_CLASS}>
                CSV dosyası
              </label>
              <input
                id="data-path"
                value={path}
                onChange={(event) => setPath(event.target.value)}
                required
                autoComplete="off"
                spellCheck={false}
                placeholder="/yol/xauusd.csv"
                className={FIELD_CLASS}
              />
            </div>
          ) : (
            <p className="self-end pb-2 text-xs text-slate-400 sm:col-span-2">
              Dukascopy aralığı sunucudan indirir; ağ yoksa 503 döner.
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-sm border border-accent bg-accent/15 px-3 py-1.5 text-sm font-medium text-slate-100 hover:bg-accent/25 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600"
          >
            İndir
          </button>
          <span className="text-xs text-slate-500">
            İsteğe bağlı; indirme senkron çalışır.
          </span>
        </div>

        {phase === "done" && lastRows !== null && (
          <p data-testid="download-rows" role="status" className="text-sm text-emerald-300">
            Tamamlandı — {formatNumber(lastRows, 0)} satır eklendi.
          </p>
        )}

        {phase === "error" && downloadError !== null && (
          <p data-testid="download-error" role="alert" className="text-sm text-rose-300">
            {downloadError}
          </p>
        )}
      </form>
    </section>
  );
}
