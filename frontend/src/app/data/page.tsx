"use client";

/**
 * The data screen: what the local store holds and how to add more.
 *
 * The status hook is keyed by the symbol field, so the page never polls a
 * second time on its own (the hook owns the 30 s refresh). The download form
 * is one request at a time; a successful call refetches the status key and
 * reports the added rows, a failure shows the backend's `detail` verbatim.
 */

import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { CoverageBar } from "@/components/coverage-bar";
import { ProvenancePanel } from "@/components/provenance-panel";
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

/**
 * Mirrors `ImportFileName` in the API: a bare file name in `<data_root>/imports`,
 * never a path. The input sends the trimmed value as the request's `path`.
 */
const CSV_NAME_PATTERN = "^[A-Za-z0-9][A-Za-z0-9._-]*$";
const CSV_NAME_MAX_LENGTH = 255;
const CSV_NAME_RE = new RegExp(CSV_NAME_PATTERN);

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
  // Dates start empty so the prerendered HTML and the hydrated client agree;
  // the first paint would otherwise bake the build day into the static page.
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [seed, setSeed] = useState("7");
  const [fileName, setFileName] = useState("");
  const [fileNameTouched, setFileNameTouched] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [pending, setPending] = useState(false);
  const [lastRows, setLastRows] = useState<number | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    setStart(isoDay(-9));
    setEnd(isoDay(0));
  }, []);

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

  const busy = pending;
  const cleanFileName = fileName.trim();
  const missingPath = provider === "csv" && cleanFileName === "";
  const invalidPathName =
    provider === "csv" &&
    cleanFileName !== "" &&
    (cleanFileName.length > CSV_NAME_MAX_LENGTH || !CSV_NAME_RE.test(cleanFileName));
  // Errors appear after the field is left (or a bad value is submitted), never
  // while the user is still typing; editing clears them until the next blur.
  const pathError =
    provider === "csv" && fileNameTouched && (missingPath || invalidPathName)
      ? missingPath
        ? "Dosya adı gerekli."
        : "Yalnızca dosya adı girin (ör. xauusd_h1.csv); /, \\ ve .. kullanılamaz."
      : null;
  const canSubmit =
    cleanSymbol !== "" &&
    start !== "" &&
    end !== "" &&
    !missingPath &&
    !invalidPathName &&
    !busy;

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (!canSubmit) {
      // A blocked submit still explains the field, matching the blur path.
      if (provider === "csv" && (missingPath || invalidPathName)) setFileNameTouched(true);
      return;
    }

    setPending(true);
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
          path: provider === "csv" ? cleanFileName : null,
        });
        setLastRows(result.rows);
        setPhase("done");
        // Refetch only after the write landed, so the cards show the new rows.
        void statusState.mutate();
      } catch (error) {
        setDownloadError(errorDetail(error));
        setPhase("error");
      } finally {
        setPending(false);
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
          <p data-testid="status-error" role="status" className="text-sm text-amber-200">
            Veri durumu alınamadı ({statusDetail}).
          </p>
        )}

        {cleanSymbol === "" ? (
          <p className="text-sm text-slate-400">Sembol girerek kapsamı görüntüle.</p>
        ) : statusState.isLoading && status === undefined ? (
          <p className="text-sm text-slate-400">Yükleniyor…</p>
        ) : rows === 0 ? (
          // With a poll error the store state is unknown; the note above
          // already explains it, and "no data" would be a false claim.
          statusDetail === null ? (
            <p data-testid="data-empty" className="text-sm text-slate-400">
              Henüz veri yok — aşağıdaki formdan sentetik veri üret veya CSV içe aktar.
            </p>
          ) : null
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
                // The API has no per-provider row split, so an active
                // card cannot claim the symbol total as its own.
                rows={active ? null : 0}
                firstTs={active ? (status?.first_ts ?? null) : null}
                lastTs={active ? (status?.last_ts ?? null) : null}
              />
            );
          })}
        </div>
        <p className="text-xs text-slate-500">
          Sağlayıcı kartlarındaki satır sayısı API sağlayıcı bazlı kırılım
          vermediği için —; sembol toplamı üstteki kapsam satırındadır.
        </p>
      </section>

      {/* An empty store has nothing to describe; an old store with rows and no
          sidecar gets the explicit missing state instead of a guess. */}
      {rows > 0 && <ProvenancePanel provenance={status?.provenance} />}

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
              disabled={busy}
              onChange={(event) => {
                const next = event.target.value as ProviderName;
                setProvider(next);
                setLastRows(null);
                setDownloadError(null);
                setFileNameTouched(false);
                // A pending request still owns the form; only an idle form
                // resets its job state on a provider switch.
                if (!pending) setPhase("idle");
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
                CSV dosya adı
              </label>
              <p id="data-path-hint" className="text-xs text-slate-500">
                Dosyayı backend&apos;in veri kökündeki imports/ klasörüne koyun; alan
                yalnızca dosya adını alır (ör. xauusd_h1.csv).
              </p>
              <input
                id="data-path"
                value={fileName}
                onChange={(event) => {
                  setFileName(event.target.value);
                  // Editing clears the message; blur or submit re-checks it.
                  if (fileNameTouched) setFileNameTouched(false);
                }}
                onBlur={() => setFileNameTouched(true)}
                required
                maxLength={CSV_NAME_MAX_LENGTH}
                pattern={CSV_NAME_PATTERN}
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                placeholder="xauusd_h1.csv"
                aria-invalid={pathError !== null}
                aria-describedby={
                  pathError !== null ? "data-path-hint data-path-error" : "data-path-hint"
                }
                className={FIELD_CLASS}
              />
              {pathError !== null && (
                <p id="data-path-error" data-testid="path-error" className="text-xs text-rose-300">
                  {pathError}
                </p>
              )}
              {/* One polite region for the field: a shown live region is not
                  announced reliably, a changed one is. */}
              <p className="sr-only" aria-live="polite">
                {pathError ?? ""}
              </p>
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
