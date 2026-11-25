"use client";

/**
 * Paper Trading: the `/api/paper` ledger as the backend stores it.
 *
 * Every number here comes from a response. A missing balance, an unread table
 * or an unknown hash is a dash or an explicit sentence, never a zero. The page
 * opens a session record, closes a running one and reads everything else; it
 * never steps a runner and never talks to a broker.
 *
 * Opening is a record write: the server stores the submitted symbol,
 * timeframe, strategy, model version and config verbatim, generates the id and
 * starts nothing. No champion is selected here, and the form says so.
 *
 * Polling is tied to the session status: a `running` session is re-read every
 * five seconds, and the timer is dropped on unmount, on selection, on
 * pagination and as soon as the session is no longer running.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { LegacyBadge } from "@/components/legacy-badge";
import { LineChart } from "@/components/line-chart";
import { ApiError, api } from "@/lib/api";
import { formatMoney, formatNumber, formatPct, formatTs } from "@/lib/format";
import type { SeriesPoint } from "@/lib/runs";
import { isLegacyStrategy } from "@/lib/strategy-labels";
import type {
  PaperEquityPoint,
  PaperOrder,
  PaperPage,
  PaperPosition,
  PaperSessionCreated,
  PaperSessionCreateRequest,
  PaperSessionItem,
  PaperSessionSummary,
  PaperTimeframe,
  PaperTrade,
} from "@/lib/types";

const PAGE_SIZE = 20;
const POLL_MS = 5_000;

const STATUS_LABEL: Record<string, string> = {
  running: "Çalışıyor",
  closed: "Kapalı",
  stopped: "Durduruldu",
  failed: "Hata",
};

const STATUS_TONE: Record<string, string> = {
  running: "border-accent/50 bg-accent/10 text-accent",
  closed: "border-slate-700 bg-slate-800/40 text-slate-300",
  stopped: "border-slate-700 bg-slate-800/40 text-slate-300",
  failed: "border-loss/50 bg-loss/10 text-rose-300",
};

const BUTTON_BASE =
  "rounded-sm border px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const PRIMARY_BUTTON = `${BUTTON_BASE} border-accent bg-accent/15 font-medium text-slate-100 hover:bg-accent/25 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;
const SECONDARY_BUTTON = `${BUTTON_BASE} border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;
const PAGER_BUTTON = `${BUTTON_BASE} px-2 py-1 text-xs border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent aria-[invalid=true]:border-rose-500/70";
const LABEL_CLASS = "text-xs text-slate-400";
const HINT_CLASS = "text-xs text-slate-400";
const FIELD_ERROR_CLASS = "text-xs text-rose-300";

// --- helpers -----------------------------------------------------------------

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "İstek başarısız oldu.";
}

function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

/** A wire number or `null`; NaN and infinities never reach the screen. */
function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function recordField(source: Record<string, unknown> | null, key: string) {
  if (source === null) return null;
  const value = source[key];
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function textField(source: Record<string, unknown> | null, key: string): string | null {
  if (source === null) return null;
  const value = source[key];
  return typeof value === "string" && value !== "" ? value : null;
}

function pageRange(page: PaperPage<unknown>): string {
  if (page.total === 0) return "0 kayıt";
  const first = page.offset + 1;
  const last = page.offset + page.items.length;
  return `${formatNumber(first, 0)}–${formatNumber(last, 0)} / ${formatNumber(page.total, 0)}`;
}

function PaperStatusChip({ status, testId }: { status: string; testId: string }) {
  const key = status.toLowerCase();
  return (
    <span
      data-testid={testId}
      data-status={key}
      className={`inline-block rounded-sm border px-2 py-0.5 text-xs ${
        STATUS_TONE[key] ?? "border-slate-700 text-slate-300"
      }`}
    >
      {STATUS_LABEL[key] ?? status}
    </span>
  );
}

function Fact({
  label,
  value,
  testId,
  mono = false,
}: {
  label: string;
  value: ReactNode;
  testId?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-x-2">
      <dt className="text-slate-400">{label}</dt>
      <dd
        data-testid={testId}
        className={`tabular-nums text-slate-300 ${mono ? "break-all font-mono" : ""}`}
      >
        {value}
      </dd>
    </div>
  );
}

// --- bounded table -----------------------------------------------------------

interface PageLike {
  items: unknown[];
  total: number;
  limit: number;
  offset: number;
}

interface LedgerTableProps {
  captionId: string;
  caption: string;
  page: PageLike | null;
  error: string | null;
  loading: boolean;
  empty: string;
  prevLabel?: string;
  nextLabel?: string;
  onPage?: (offset: number) => void;
  children: ReactNode;
}

/**
 * One bounded table: a focusable scroll region named by its caption, sticky
 * header cells, and server-side paging. The pager is hidden while the page is
 * unknown, so a failed read never offers a next page that does not exist.
 */
function LedgerTable({
  captionId,
  caption,
  page,
  error,
  loading,
  empty,
  prevLabel = "Önceki",
  nextLabel = "Sonraki",
  onPage,
  children,
}: LedgerTableProps) {
  const atStart = page === null || page.offset === 0;
  const atEnd = page === null || page.offset + page.items.length >= page.total;
  const prevOffset = page === null ? 0 : Math.max(0, page.offset - page.limit);
  const nextOffset = page === null ? 0 : page.offset + page.limit;

  return (
    <div className="flex flex-col gap-1">
      <div
        role="region"
        aria-labelledby={captionId}
        tabIndex={0}
        className="max-h-80 overflow-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent [&_tbody_th]:sticky [&_tbody_th]:left-0 [&_tbody_th]:bg-slate-900/95 [&_thead_th]:sticky [&_thead_th]:top-0 [&_thead_th]:z-10 [&_thead_th]:bg-slate-900"
      >
        <table className="w-full min-w-[40rem] border-collapse text-left text-xs">
          <caption id={captionId} className="px-3 py-2 text-left text-slate-400">
            {caption}
          </caption>
          {children}
        </table>
      </div>

      {error !== null && (
        <p role="alert" className="text-xs text-rose-300">
          {error}
        </p>
      )}

      {page === null && error === null && (
        <p className="text-xs text-slate-400">{loading ? "Yükleniyor…" : "Tablo okunamadı."}</p>
      )}

      {page !== null && page.items.length === 0 && (
        <p className="text-xs text-slate-400">{empty}</p>
      )}

      {page !== null && onPage !== undefined && (
        <div
          role="group"
          aria-label={`${caption} sayfalama`}
          className="flex flex-wrap items-center justify-end gap-2"
        >
          <span aria-live="polite" className="text-xs tabular-nums text-slate-400">
            {pageRange(page)}
          </span>
          <button
            type="button"
            disabled={atStart}
            onClick={() => onPage(prevOffset)}
            className={PAGER_BUTTON}
          >
            {prevLabel}
          </button>
          <button
            type="button"
            disabled={atEnd}
            onClick={() => onPage(nextOffset)}
            className={PAGER_BUTTON}
          >
            {nextLabel}
          </button>
        </div>
      )}
    </div>
  );
}

// --- equity reading ----------------------------------------------------------

/**
 * Chart-ready equity points from one bounded page. The wire page is already
 * chronological, but the chart needs strictly increasing x values, so the
 * points are sorted by time and a repeated timestamp keeps its last mark.
 */
function equitySeries(items: readonly PaperEquityPoint[]): SeriesPoint[] {
  const byTime = new Map<number, number>();
  for (const item of items) {
    const time = Date.parse(item.ts);
    if (!Number.isFinite(time) || !Number.isFinite(item.equity)) continue;
    byTime.set(time, item.equity);
  }
  return [...byTime.entries()]
    .sort(([left], [right]) => left - right)
    .map(([time, value]) => ({ time, value }));
}

interface EquityWindow {
  points: SeriesPoint[];
  first: number;
  last: number;
  change: number;
  changePct: number | null;
}

/**
 * The window's first/last equity and their difference, or `null` when no mark
 * is usable. This is a window delta, not the account's total P&L: the ledger
 * exposes no starting balance, so the page never pretends to know one.
 */
function equityWindow(items: readonly PaperEquityPoint[]): EquityWindow | null {
  const points = equitySeries(items);
  if (points.length === 0) return null;

  const first = points[0].value;
  const last = points[points.length - 1].value;
  const change = last - first;
  return {
    points,
    first,
    last,
    change,
    changePct: first === 0 ? null : (change / first) * 100,
  };
}

function signedMoney(value: number): string {
  return value > 0 ? `+${formatMoney(value)}` : formatMoney(value);
}

// --- start-session form ------------------------------------------------------

const PAPER_TIMEFRAMES: readonly PaperTimeframe[] = [
  "M1",
  "M5",
  "M15",
  "M30",
  "H1",
  "H4",
  "D1",
];

/** Mirrors `SymbolId` in the schema: one safe path segment, at most 32 chars. */
const SYMBOL_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

const START_FIELD_ID = {
  symbol: "paper-start-symbol",
  timeframe: "paper-start-timeframe",
  strategy: "paper-start-strategy",
  modelVersion: "paper-start-model-version",
  config: "paper-start-config",
} as const;

type StartField = keyof typeof START_FIELD_ID;

const START_FIELD_ORDER: readonly StartField[] = [
  "symbol",
  "timeframe",
  "strategy",
  "modelVersion",
  "config",
];

interface StartValues {
  symbol: string;
  timeframe: string;
  strategy: string;
  modelVersion: string;
  config: string;
}

type StartErrors = Partial<Record<StartField, string>>;

const START_DEFAULTS: StartValues = {
  symbol: "XAUUSD",
  timeframe: "M1",
  strategy: "",
  modelVersion: "",
  config: "",
};

function parseConfig(raw: string): { value: Record<string, unknown> | null; error: string | null } {
  if (raw.trim() === "") return { value: null, error: null };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { value: null, error: "Yapılandırma geçerli JSON değil." };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return {
      value: null,
      error: "Yapılandırma bir JSON nesnesi olmalı (örn. {\"key\": \"value\"}).",
    };
  }
  return { value: parsed as Record<string, unknown>, error: null };
}

/** The same bounds the API schema enforces, checked before a request is sent. */
function validateStart(values: StartValues): StartErrors {
  const errors: StartErrors = {};

  const symbol = values.symbol.trim();
  if (symbol === "") {
    errors.symbol = "Sembol gerekli (örn. XAUUSD).";
  } else if (symbol.length > 32) {
    errors.symbol = "Sembol en fazla 32 karakter olabilir.";
  } else if (!SYMBOL_PATTERN.test(symbol)) {
    errors.symbol =
      "Sembol harf veya rakamla başlamalı; yalnızca harf, rakam, nokta, tire ve alt çizgi içerebilir.";
  }

  if (!PAPER_TIMEFRAMES.includes(values.timeframe as PaperTimeframe)) {
    errors.timeframe = "Zaman dilimi listeden seçilmeli.";
  }

  const strategy = values.strategy.trim();
  if (strategy === "") {
    errors.strategy = "Strateji gerekli (örn. dsl veya manual).";
  } else if (strategy.length > 64) {
    errors.strategy = "Strateji en fazla 64 karakter olabilir.";
  }

  if (values.modelVersion.trim().length > 128) {
    errors.modelVersion = "Model sürümü en fazla 128 karakter olabilir.";
  }

  const config = parseConfig(values.config);
  if (config.error !== null) errors.config = config.error;

  return errors;
}

interface StartSessionFormProps {
  /** Called after the server recorded the session; the page selects it. */
  onCreated: (created: PaperSessionCreated) => void;
}

/**
 * Record one paper session. This is a write of what the operator submits, not
 * a strategy picker: the server stores the fields verbatim, generates the id
 * and starts no runner, so no champion can be auto-selected here.
 */
function StartSessionForm({ onCreated }: StartSessionFormProps) {
  const [values, setValues] = useState<StartValues>(START_DEFAULTS);
  const [touched, setTouched] = useState<Partial<Record<StartField, boolean>>>({});
  const [submitted, setSubmitted] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [conflict, setConflict] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const formRef = useRef<HTMLFormElement>(null);
  const noticeRef = useRef<HTMLParagraphElement>(null);

  const errors = validateStart(values);
  const visibleErrors: StartErrors = submitted
    ? errors
    : Object.fromEntries(
        Object.entries(errors).filter(([field]) => touched[field as StartField]),
      );

  const errorEntries = START_FIELD_ORDER.filter(
    (field) => visibleErrors[field] !== undefined,
  ).map((field) => ({ field, message: visibleErrors[field] as string }));

  // The success notice is a focus target: the form is longer than the fold on
  // a phone, and a pointer user should not have to hunt for the outcome.
  useEffect(() => {
    if (notice !== null) noticeRef.current?.focus();
  }, [notice]);

  const handleChange = (field: StartField, value: string): void => {
    setValues((current) => ({ ...current, [field]: value }));
    setTouched((current) => ({ ...current, [field]: false }));
  };

  const focusFirstError = (found: StartErrors): void => {
    const first = START_FIELD_ORDER.find((field) => found[field] !== undefined);
    if (first === undefined) return;
    formRef.current?.querySelector<HTMLElement>(`#${START_FIELD_ID[first]}`)?.focus();
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (starting) return;

    setSubmitted(true);
    setStartError(null);
    setConflict(null);
    setNotice(null);

    const found = validateStart(values);
    if (Object.keys(found).length > 0) {
      focusFirstError(found);
      return;
    }

    const config = parseConfig(values.config);
    const payload: PaperSessionCreateRequest = {
      symbol: values.symbol.trim(),
      timeframe: values.timeframe as PaperTimeframe,
      strategy: values.strategy.trim(),
      model_version: values.modelVersion.trim() === "" ? null : values.modelVersion.trim(),
      ...(config.value === null ? {} : { config: config.value }),
    };

    setStarting(true);
    void api
      .openPaperSession(payload)
      .then((created) => {
        setNotice(
          `Oturum kaydedildi: ${created.session_id}. Durum: ${created.status}; başlangıç: ${formatTs(
            created.started_at,
          )}. Liste yenilendi; sunucu yalnızca gönderilen alanları kaydetti.`,
        );
        onCreated(created);
      })
      .catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 409) {
          setConflict(error.detail);
        } else if (error instanceof ApiError && error.status === 422) {
          // Pydantic's message list arrives joined by the API client.
          setStartError(`Sunucu doğrulaması reddetti (422): ${error.detail}`);
        } else {
          setStartError(errorDetail(error));
        }
      })
      .finally(() => {
        setStarting(false);
      });
  };

  const describedBy = (field: StartField, always?: string): string | undefined => {
    const ids = always === undefined ? [] : [always];
    if (visibleErrors[field] !== undefined) ids.push(`${START_FIELD_ID[field]}-error`);
    return ids.length === 0 ? undefined : ids.join(" ");
  };

  return (
    <form
      ref={formRef}
      data-testid="paper-start-form"
      aria-labelledby="paper-start-heading"
      aria-describedby="paper-start-note"
      aria-busy={starting}
      noValidate
      onSubmit={handleSubmit}
      className="flex flex-col gap-4 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="paper-start-heading" className="text-sm font-semibold text-slate-100">
          Yeni oturum kaydı
        </h2>
        <span className="text-xs text-slate-400">Yalnızca defter kaydı — runner başlatmaz</span>
      </div>

      <p id="paper-start-note" className="text-xs text-slate-400">
        Bu form <code>/api/paper/sessions</code> üzerine bir kayıt yazar; emir gönderilmez,
        gerçek para yoktur. Sunucu gönderdiğiniz sembol, zaman dilimi, strateji, model sürümü ve
        yapılandırmayı olduğu gibi saklar; şampiyon seçilmez ve model sizin yerinize atanmaz.
      </p>

      <fieldset className="flex flex-col gap-3">
        <legend className="mb-1 text-xs font-medium text-slate-300">Kimlik ve atıf</legend>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="flex flex-col gap-1">
            <label htmlFor={START_FIELD_ID.symbol} className={LABEL_CLASS}>
              Sembol <span aria-hidden="true">*</span>
            </label>
            <input
              id={START_FIELD_ID.symbol}
              name="symbol"
              type="text"
              required
              maxLength={32}
              pattern="[A-Za-z0-9][A-Za-z0-9._-]*"
              autoComplete="off"
              autoCapitalize="none"
              spellCheck={false}
              value={values.symbol}
              disabled={starting}
              aria-invalid={visibleErrors.symbol !== undefined}
              aria-describedby={describedBy(
                "symbol",
                `${START_FIELD_ID.symbol}-hint`,
              )}
              onChange={(event) => handleChange("symbol", event.target.value)}
              onBlur={() => setTouched((current) => ({ ...current, symbol: true }))}
              className={FIELD_CLASS}
            />
            <p id={`${START_FIELD_ID.symbol}-hint`} className={HINT_CLASS}>
              Harf veya rakamla başlar; 32 karaktere kadar.
            </p>
            {visibleErrors.symbol !== undefined && (
              <p id={`${START_FIELD_ID.symbol}-error`} className={FIELD_ERROR_CLASS}>
                {visibleErrors.symbol}
              </p>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor={START_FIELD_ID.timeframe} className={LABEL_CLASS}>
              Zaman dilimi <span aria-hidden="true">*</span>
            </label>
            <select
              id={START_FIELD_ID.timeframe}
              name="timeframe"
              required
              value={values.timeframe}
              disabled={starting}
              aria-invalid={visibleErrors.timeframe !== undefined}
              aria-describedby={describedBy("timeframe")}
              onChange={(event) => handleChange("timeframe", event.target.value)}
              onBlur={() => setTouched((current) => ({ ...current, timeframe: true }))}
              className={FIELD_CLASS}
            >
              {PAPER_TIMEFRAMES.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            {visibleErrors.timeframe !== undefined && (
              <p id={`${START_FIELD_ID.timeframe}-error`} className={FIELD_ERROR_CLASS}>
                {visibleErrors.timeframe}
              </p>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor={START_FIELD_ID.strategy} className={LABEL_CLASS}>
              Strateji <span aria-hidden="true">*</span>
            </label>
            <input
              id={START_FIELD_ID.strategy}
              name="strategy"
              type="text"
              required
              maxLength={64}
              autoComplete="off"
              autoCapitalize="none"
              spellCheck={false}
              placeholder="manual"
              value={values.strategy}
              disabled={starting}
              aria-invalid={visibleErrors.strategy !== undefined}
              aria-describedby={describedBy("strategy", `${START_FIELD_ID.strategy}-hint`)}
              onChange={(event) => handleChange("strategy", event.target.value)}
              onBlur={() => setTouched((current) => ({ ...current, strategy: true }))}
              className={FIELD_CLASS}
            />
            <p id={`${START_FIELD_ID.strategy}-hint`} className={HINT_CLASS}>
              Serbest metin; API kaydettiği değeri saklar. En fazla 64 karakter.
            </p>
            {visibleErrors.strategy !== undefined && (
              <p id={`${START_FIELD_ID.strategy}-error`} className={FIELD_ERROR_CLASS}>
                {visibleErrors.strategy}
              </p>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor={START_FIELD_ID.modelVersion} className={LABEL_CLASS}>
              Model sürümü <span className="text-slate-500">(isteğe bağlı)</span>
            </label>
            <input
              id={START_FIELD_ID.modelVersion}
              name="model_version"
              type="text"
              maxLength={128}
              autoComplete="off"
              autoCapitalize="none"
              spellCheck={false}
              placeholder="hash-…"
              value={values.modelVersion}
              disabled={starting}
              aria-invalid={visibleErrors.modelVersion !== undefined}
              aria-describedby={describedBy(
                "modelVersion",
                `${START_FIELD_ID.modelVersion}-hint`,
              )}
              onChange={(event) => handleChange("modelVersion", event.target.value)}
              onBlur={() => setTouched((current) => ({ ...current, modelVersion: true }))}
              className={`${FIELD_CLASS} font-mono`}
            />
            <p id={`${START_FIELD_ID.modelVersion}-hint`} className={HINT_CLASS}>
              Boş bırakılırsa <code>null</code> kaydedilir; sayfa sürüm uydurmaz.
            </p>
            {visibleErrors.modelVersion !== undefined && (
              <p id={`${START_FIELD_ID.modelVersion}-error`} className={FIELD_ERROR_CLASS}>
                {visibleErrors.modelVersion}
              </p>
            )}
          </div>
        </div>
      </fieldset>

      <div className="flex flex-col gap-1">
        <label htmlFor={START_FIELD_ID.config} className={LABEL_CLASS}>
          Yapılandırma <span className="text-slate-500">(JSON nesnesi, isteğe bağlı)</span>
        </label>
        <textarea
          id={START_FIELD_ID.config}
          name="config"
          rows={3}
          spellCheck={false}
          placeholder={'{"note": "manual session"}'}
          value={values.config}
          disabled={starting}
          aria-invalid={visibleErrors.config !== undefined}
          aria-describedby={describedBy("config", `${START_FIELD_ID.config}-hint`)}
          onChange={(event) => handleChange("config", event.target.value)}
          onBlur={() => setTouched((current) => ({ ...current, config: true }))}
          className={`${FIELD_CLASS} font-mono text-xs`}
        />
        <p id={`${START_FIELD_ID.config}-hint`} className={HINT_CLASS}>
          Sunucu bunu oturumun <code>config.extra</code> alanına yazar; runner ayarı değildir.
        </p>
        {visibleErrors.config !== undefined && (
          <p id={`${START_FIELD_ID.config}-error`} className={FIELD_ERROR_CLASS}>
            {visibleErrors.config}
          </p>
        )}
      </div>

      {submitted && errorEntries.length > 0 && (
        <div
          data-testid="paper-start-errors"
          role="alert"
          className="rounded-sm border border-rose-500/40 bg-rose-500/10 px-3 py-2"
        >
          <p className="text-sm font-medium text-rose-200">
            Form gönderilemedi — {errorEntries.length} alan düzeltilmeli:
          </p>
          <ul className="mt-1 list-inside list-disc text-xs text-rose-200">
            {errorEntries.map((entry) => (
              <li key={entry.field}>
                <a
                  href={`#${START_FIELD_ID[entry.field]}`}
                  className="underline underline-offset-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  {entry.message}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <button type="submit" disabled={starting} className={PRIMARY_BUTTON}>
          {starting ? "Kaydediliyor…" : "Oturumu kaydet"}
        </button>
        <p className="text-xs text-slate-400">
          Kayıt sonrası liste yenilenir; oturum <code>running</code> durumunda açılır ve hiçbir
          bar işlenmez.
        </p>
      </div>

      {conflict !== null && (
        <p data-testid="paper-start-conflict" role="alert" className="text-sm text-amber-200">
          Çakışma (409): {conflict} Sunucu bu kimliği zaten kaydetmiş; yeni bir kayıt
          oluşturulmadı. Liste yenilenebilir.
        </p>
      )}

      {startError !== null && (
        <p data-testid="paper-start-error" role="alert" className="text-sm text-rose-300">
          Oturum kaydedilemedi: {startError}
        </p>
      )}

      {notice !== null && (
        <p
          ref={noticeRef}
          tabIndex={-1}
          data-testid="paper-start-notice"
          role="status"
          className="rounded-sm border border-accent/40 bg-accent/10 px-3 py-2 text-xs text-slate-200"
        >
          {notice}
        </p>
      )}
    </form>
  );
}

// --- page --------------------------------------------------------------------

type TableKey = "positions" | "orders" | "history" | "equity";

type TableOffsets = Record<TableKey, number>;

const ZERO_OFFSETS: TableOffsets = { positions: 0, orders: 0, history: 0, equity: 0 };

interface LedgerTables {
  positions: PaperPage<PaperPosition> | null;
  orders: PaperPage<PaperOrder> | null;
  history: PaperPage<PaperTrade> | null;
  equity: PaperPage<PaperEquityPoint> | null;
}

function emptyTables(): LedgerTables {
  return { positions: null, orders: null, history: null, equity: null };
}

export default function PaperPage() {
  const [sessionsPage, setSessionsPage] = useState<PaperPage<PaperSessionItem> | null>(null);
  const [sessionsOffset, setSessionsOffset] = useState(0);
  const [sessionsToken, setSessionsToken] = useState(0);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [summary, setSummary] = useState<PaperSessionSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [summaryMissing, setSummaryMissing] = useState(false);
  const [detailToken, setDetailToken] = useState(0);

  const [tables, setTables] = useState<LedgerTables>(emptyTables);
  const [tablesError, setTablesError] = useState<string | null>(null);
  const [offsets, setOffsets] = useState<TableOffsets>(ZERO_OFFSETS);

  const [confirming, setConfirming] = useState(false);
  const [closing, setClosing] = useState(false);
  const [closeError, setCloseError] = useState<string | null>(null);
  const [closeNotice, setCloseNotice] = useState<string | null>(null);

  const confirmRef = useRef<HTMLButtonElement>(null);
  const closeTriggerRef = useRef<HTMLButtonElement>(null);
  const detailHeadingRef = useRef<HTMLHeadingElement>(null);
  /** Where focus goes when the inline confirm step closes. */
  const restoreFocusRef = useRef<"trigger" | "heading" | null>(null);

  const refreshSessions = useCallback(() => setSessionsToken((token) => token + 1), []);
  const refreshDetail = useCallback(() => setDetailToken((token) => token + 1), []);

  // -- session list ---------------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setSessionsLoading(true);

    void api
      .paperSessions(PAGE_SIZE, sessionsOffset)
      .then((page) => {
        if (cancelled) return;
        setSessionsPage(page);
        setSessionsError(null);
        // The newest session is the interesting one; a manual choice wins.
        setSelectedId((current) => current ?? page.items[0]?.session_id ?? null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setSessionsPage(null);
        setSessionsError(errorDetail(error));
      })
      .finally(() => {
        if (!cancelled) setSessionsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [sessionsOffset, sessionsToken]);

  // -- one session: summary + four pages ------------------------------------
  useEffect(() => {
    if (selectedId === null) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const load = async (): Promise<void> => {
      const results = await Promise.allSettled([
        api.paperSession(selectedId),
        api.paperPositions(selectedId, PAGE_SIZE, offsets.positions),
        api.paperOrders(selectedId, PAGE_SIZE, offsets.orders),
        api.paperHistory(selectedId, PAGE_SIZE, offsets.history),
        api.paperEquity(selectedId, PAGE_SIZE, offsets.equity),
      ]);
      if (cancelled) return;

      const [summaryResult, positionsResult, ordersResult, historyResult, equityResult] = results;

      if (summaryResult.status === "rejected") {
        setSummaryLoading(false);
        if (isNotFound(summaryResult.reason)) {
          setSummary(null);
          setTables(emptyTables());
          setTablesError(null);
          setSummaryError(null);
          setSummaryMissing(true);
        } else {
          // A failed poll stops the loop; the panel keeps a retry instead of
          // showing a stale balance as if it were current.
          setSummaryError(errorDetail(summaryResult.reason));
        }
        return;
      }

      const detail = summaryResult.value;
      setSummary(detail);
      setSummaryMissing(false);
      setSummaryError(null);

      const failures: string[] = [];
      const pageOf = <T,>(
        result: PromiseSettledResult<PaperPage<T>>,
        label: string,
      ): PaperPage<T> | null => {
        if (result.status === "fulfilled") return result.value;
        failures.push(`${label} (${errorDetail(result.reason)})`);
        return null;
      };

      setTables({
        positions: pageOf(positionsResult, "Pozisyonlar"),
        orders: pageOf(ordersResult, "Emirler"),
        history: pageOf(historyResult, "Geçmiş"),
        equity: pageOf(equityResult, "Özkaynak"),
      });
      setTablesError(failures.length > 0 ? `Okunamayan tablo: ${failures.join(", ")}` : null);
      setSummaryLoading(false);

      if (detail.status === "running") {
        timer = setTimeout(() => void load(), POLL_MS);
      }
    };

    setSummaryLoading(true);
    void load();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [selectedId, detailToken, offsets.positions, offsets.orders, offsets.history, offsets.equity]);

  // A session that left `running` can no longer be closed; drop a stale confirm.
  useEffect(() => {
    if (summary !== null && summary.status !== "running") setConfirming(false);
  }, [summary]);

  // The confirm step is inline, not a modal: move focus to the confirm button,
  // then return it deliberately. Without this a keyboard user lands on <body>
  // when the confirm group unmounts.
  useEffect(() => {
    if (confirming) {
      confirmRef.current?.focus();
      return;
    }
    const target = restoreFocusRef.current;
    if (target === null) return;
    restoreFocusRef.current = null;

    if (target === "trigger") {
      const trigger = closeTriggerRef.current;
      if (trigger !== null && !trigger.disabled) {
        trigger.focus();
        return;
      }
    }
    detailHeadingRef.current?.focus();
  }, [confirming]);

  /** Reset every per-session read so the detail panel cannot show the old row. */
  const showSession = (sessionId: string): void => {
    setSelectedId(sessionId);
    setSummary(null);
    setSummaryError(null);
    setSummaryMissing(false);
    setTables(emptyTables());
    setTablesError(null);
    setOffsets(ZERO_OFFSETS);
    setConfirming(false);
    setCloseError(null);
    setCloseNotice(null);
  };

  const selectSession = (sessionId: string): void => {
    if (sessionId === selectedId) return;
    showSession(sessionId);
  };

  // After a 201 the row exists, so select it and re-read the list rather than
  // patching a session into the page that the ledger has not returned yet.
  const handleSessionCreated = (created: PaperSessionCreated): void => {
    showSession(created.session_id);
    refreshSessions();
  };

  const pageTable = (key: TableKey, offset: number): void => {
    const page = tables[key];
    if (page === null || offset < 0 || offset === page.offset) return;
    if (offset > 0 && offset >= page.total) return;
    setOffsets((current) => ({ ...current, [key]: offset }));
  };

  const handleClose = (): void => {
    if (selectedId === null || summary === null || summary.status !== "running" || closing) {
      return;
    }
    setClosing(true);
    setCloseError(null);
    setCloseNotice(null);

    void api
      .closePaperSession(selectedId)
      .then((updated) => {
        setSummary(updated);
        setSummaryMissing(false);
        setSummaryError(null);
        setCloseNotice(`Oturum kapatıldı. Bitiş: ${formatTs(updated.ended_at)}`);
        restoreFocusRef.current = "heading";
        setConfirming(false);
        refreshDetail();
        refreshSessions();
      })
      .catch((error: unknown) => {
        if (isNotFound(error)) {
          setSummary(null);
          setSummaryMissing(true);
          setCloseError(null);
          refreshDetail();
          return;
        }
        if (error instanceof ApiError && error.status === 409) {
          // The ledger already closed it; refreshing shows the settled record.
          setCloseNotice("Oturum zaten kapalı; ikinci kapatma uygulanmadı.");
          restoreFocusRef.current = "heading";
          setConfirming(false);
          refreshDetail();
          refreshSessions();
          return;
        }
        setCloseError(errorDetail(error));
      })
      .finally(() => {
        setClosing(false);
      });
  };

  // -- derived reading ------------------------------------------------------
  const running = summary !== null && summary.status === "running";

  const extra = summary === null ? null : recordField(summary.config, "extra");
  const dsl = recordField(extra, "dsl");
  const specHash = textField(dsl, "spec_hash");
  const specVersion = finiteNumber(dsl?.["spec_version"]);
  const datasetId = textField(extra, "dataset_id");
  const origin = summary === null ? null : textField(summary.config, "origin");

  const equityItems = tables.equity === null ? [] : tables.equity.items;
  const equityStats = equityWindow(equityItems);
  const equityChartTitle =
    equityStats === null
      ? "Özkaynak eğrisi — geçerli işaret yok"
      : equityStats.points.length === 1
        ? `Özkaynak eğrisi — tek işaret ${formatTs(
            new Date(equityStats.points[0].time).toISOString(),
          )}, ${formatMoney(equityStats.points[0].value)}`
        : `Özkaynak eğrisi — ${formatTs(
            new Date(equityStats.points[0].time).toISOString(),
          )} ile ${formatTs(
            new Date(equityStats.points[equityStats.points.length - 1].time).toISOString(),
          )} arası, ${formatNumber(equityStats.points.length, 0)} işaret, pencerede değişim ${signedMoney(
            equityStats.change,
          )}`;

  const closeExplanation =
    summary === null
      ? "Kapatmak için bir oturum seçin."
      : summary.status !== "running"
        ? "Bu oturum çalışmıyor; kapatma yalnızca çalışan bir oturumda kullanılabilir."
        : closing
          ? "Kapatma isteği gönderiliyor…"
          : "Kapatma oturumu `closed` yapar ve bitiş zamanını yazar.";

  const statusSentence =
    selectedId === null
      ? sessionsLoading
        ? "Oturum listesi yükleniyor."
        : sessionsPage !== null && sessionsPage.total === 0
          ? "Paper oturumu yok."
          : "Bir oturum seçin."
      : summary === null
        ? summaryMissing
          ? "Seçili oturum bulunamadı."
          : "Oturum ayrıntısı yükleniyor."
        : `${summary.session_id}: ${STATUS_LABEL[summary.status] ?? summary.status}`;

  return (
    <section aria-labelledby="paper-heading" className="flex flex-col gap-5">
      <div>
        <h1 id="paper-heading" className="text-xl font-semibold text-slate-100">
          Paper Trading
        </h1>
        <p className="text-xs text-slate-400">
          Yerel paper defteri — oturumlar, pozisyonlar, emirler, işlemler ve özkaynak.
        </p>
      </div>

      <p
        role="note"
        data-testid="paper-notice"
        className="rounded-sm border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
      >
        <strong>Yalnızca paper: gerçek para yok.</strong> Bu sayfa `/api/paper` defterini okur;
        yazma işlemleri oturum kaydı ve kapatmadır. Hiçbir emir borsaya, broker’a veya MT5
        terminaline gitmez.
      </p>

      <p className="sr-only" role="status">
        {statusSentence}
      </p>

      <StartSessionForm onCreated={handleSessionCreated} />

      {/* -- sessions ------------------------------------------------------ */}
      <section aria-labelledby="paper-sessions-heading" className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="paper-sessions-heading" className="text-sm font-semibold text-slate-100">
            Oturumlar
          </h2>
          <button
            type="button"
            onClick={refreshSessions}
            disabled={sessionsLoading}
            aria-describedby="paper-sessions-note"
            className={SECONDARY_BUTTON}
          >
            Listeyi yenile
          </button>
        </div>
        <p id="paper-sessions-note" className="text-xs text-slate-400">
          Defter yoksa liste boştur. Kayıt yukarıdaki formla yazılır; form bir runner
          başlatmaz. Seçili oturum çalışırken 5 sn’de bir yenilenir.
        </p>

        {sessionsError !== null ? (
          <p data-testid="paper-sessions-error" role="alert" className="text-sm text-rose-300">
            Oturum listesi alınamadı: {sessionsError}
          </p>
        ) : sessionsPage === null ? (
          <p className="text-sm text-slate-400">Yükleniyor…</p>
        ) : sessionsPage.total === 0 ? (
          <p data-testid="paper-sessions-empty" role="status" className="text-sm text-slate-400">
            Henüz paper oturumu yok. Yukarıdaki formla ilk oturum kaydını oluşturabilirsiniz;
            kayıt bir runner başlatmaz.
          </p>
        ) : sessionsPage.items.length === 0 ? (
          <p data-testid="paper-sessions-page-empty" role="status" className="text-xs text-slate-400">
            İstenen sayfa boş; kayıt sayısı azalmış olabilir.{" "}
            <button
              type="button"
              onClick={() => setSessionsOffset(0)}
              className="underline underline-offset-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              İlk sayfaya dön
            </button>
          </p>
        ) : (
          <LedgerTable
            captionId="paper-sessions-caption"
            caption="Paper oturumları — en yeni önce"
            page={sessionsPage}
            error={null}
            loading={sessionsLoading}
            empty="Bu sayfada oturum yok."
            onPage={setSessionsOffset}
          >
            <thead className="text-slate-400">
              <tr>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Oturum
                </th>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Durum
                </th>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Sembol / TF
                </th>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Strateji
                </th>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Model sürümü
                </th>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Başladı
                </th>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Bitti
                </th>
              </tr>
            </thead>
            <tbody>
              {sessionsPage.items.map((item) => {
                const selected = item.session_id === selectedId;
                return (
                  <tr
                    key={item.session_id}
                    data-testid={`paper-session-row-${item.session_id}`}
                    className={selected ? "bg-slate-800/50" : undefined}
                  >
                    <th scope="row" className="px-3 py-1.5 font-normal">
                      <button
                        type="button"
                        aria-pressed={selected}
                        onClick={() => selectSession(item.session_id)}
                        className="break-all font-mono text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                      >
                        {item.session_id}
                      </button>
                    </th>
                    <td className="px-3 py-1.5">
                      <PaperStatusChip status={item.status} testId="paper-row-status" />
                    </td>
                    <td className="px-3 py-1.5 text-slate-300">
                      {item.symbol} / {item.timeframe}
                    </td>
                    <td className="px-3 py-1.5 text-slate-300">
                      <span className="flex flex-wrap items-center gap-1">
                        {item.strategy}
                        {isLegacyStrategy(item.strategy) && (
                          <LegacyBadge testId="paper-row-legacy" />
                        )}
                      </span>
                    </td>
                    <td className="break-all px-3 py-1.5 font-mono text-slate-300">
                      {item.model_version ?? "—"}
                    </td>
                    <td className="px-3 py-1.5 tabular-nums text-slate-300">
                      {formatTs(item.started_at)}
                    </td>
                    <td className="px-3 py-1.5 tabular-nums text-slate-300">
                      {formatTs(item.ended_at)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </LedgerTable>
        )}
      </section>

      {/* -- one session --------------------------------------------------- */}
      {selectedId === null ? (
        <p className="text-sm text-slate-400">
          Ayrıntı için listeden bir oturum seçin; yeni kayıt yukarıdaki formla yapılır.
        </p>
      ) : (
        <section
          aria-labelledby="paper-detail-heading"
          aria-busy={summaryLoading}
          className="flex flex-col gap-4 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex min-w-0 flex-col gap-1">
              <h2
                id="paper-detail-heading"
                ref={detailHeadingRef}
                tabIndex={-1}
                className="text-sm font-semibold text-slate-100"
              >
                Oturum ayrıntısı
              </h2>
              <p className="break-all font-mono text-sm text-slate-200">{selectedId}</p>
              {summary !== null && (
                <div className="flex flex-wrap items-center gap-2">
                  <PaperStatusChip status={summary.status} testId="paper-detail-status" />
                  {running && (
                    <span className="text-xs text-slate-400">5 sn’de bir yenileniyor</span>
                  )}
                </div>
              )}
            </div>

            <div className="flex flex-col items-end gap-1">
              {confirming ? (
                <div
                  role="group"
                  aria-labelledby="paper-close-confirm-heading"
                  className="flex flex-col items-end gap-1 text-right"
                >
                  <p id="paper-close-confirm-heading" className="text-xs text-amber-200">
                    Bu oturum kapatılacak; kapanış geri alınamaz.
                  </p>
                  <div className="flex gap-2">
                    <button
                      ref={confirmRef}
                      type="button"
                      data-testid="paper-close-confirm"
                      onClick={handleClose}
                      disabled={closing}
                      className={PRIMARY_BUTTON}
                    >
                      {closing ? "Kapatılıyor…" : "Kapatmayı onayla"}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        restoreFocusRef.current = "trigger";
                        setConfirming(false);
                      }}
                      disabled={closing}
                      className={SECONDARY_BUTTON}
                    >
                      Vazgeç
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  ref={closeTriggerRef}
                  type="button"
                  data-testid="paper-close-button"
                  onClick={() => {
                    setConfirming(true);
                    setCloseNotice(null);
                    setCloseError(null);
                  }}
                  disabled={!running || closing}
                  aria-describedby="paper-close-explanation"
                  className={SECONDARY_BUTTON}
                >
                  Oturumu kapat
                </button>
              )}
              <p id="paper-close-explanation" className="text-right text-xs text-slate-400">
                {closeExplanation}
              </p>
            </div>
          </div>

          {closeNotice !== null && (
            <p data-testid="paper-close-notice" role="status" className="text-xs text-amber-200">
              {closeNotice}
            </p>
          )}
          {closeError !== null && (
            <p data-testid="paper-close-error" role="alert" className="text-xs text-rose-300">
              Kapatma başarısız: {closeError}
            </p>
          )}

          {summaryMissing && (
            <div className="flex flex-wrap items-center gap-3">
              <p data-testid="paper-session-missing" role="status" className="text-sm text-amber-200">
                Bu oturum sunucuda bulunamadı (404). Defter taşınmış veya oturum silinmiş olabilir;
                sayfa değer uydurmaz.
              </p>
              <button type="button" onClick={refreshSessions} className={SECONDARY_BUTTON}>
                Listeyi yenile
              </button>
            </div>
          )}

          {summaryError !== null && (
            <div className="flex flex-wrap items-center gap-3">
              <p data-testid="paper-detail-error" role="alert" className="text-sm text-rose-300">
                Oturum ayrıntısı alınamadı: {summaryError}
              </p>
              <button type="button" onClick={refreshDetail} className={SECONDARY_BUTTON}>
                Yenile
              </button>
            </div>
          )}

          {summary === null && !summaryMissing && summaryError === null && (
            <p className="text-sm text-slate-400">Oturum ayrıntısı yükleniyor…</p>
          )}

          {summary !== null && (
            <>
              <dl
                data-testid="paper-session-facts"
                className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2 lg:grid-cols-3"
              >
                <Fact label="Bakiye" value={formatMoney(summary.balance)} testId="paper-balance" />
                <Fact label="Özkaynak" value={formatMoney(summary.equity)} testId="paper-equity" />
                <Fact
                  label="Kapanmış işlem"
                  value={formatNumber(summary.trade_count, 0)}
                  testId="paper-trade-count"
                />
                <Fact
                  label="Son bar"
                  value={formatTs(summary.last_bar_ts)}
                  testId="paper-last-bar"
                />
                <Fact label="Sembol / TF" value={`${summary.symbol} / ${summary.timeframe}`} />
                <Fact label="Başladı" value={formatTs(summary.started_at)} />
                <Fact label="Bitti" value={formatTs(summary.ended_at)} />
              </dl>

              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                <section
                  aria-labelledby="paper-position-heading"
                  className="rounded-md border border-slate-800 px-3 py-2"
                >
                  <h3
                    id="paper-position-heading"
                    className="mb-1 text-xs font-semibold text-slate-300"
                  >
                    Açık pozisyon
                  </h3>
                  {summary.open_position === null ? (
                    <p data-testid="paper-open-position" className="text-sm text-slate-400">
                      Açık pozisyon yok.
                    </p>
                  ) : (
                    <div data-testid="paper-open-position">
                      <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
                        <Fact label="Yön" value={summary.open_position.side} />
                        <Fact label="Lot" value={formatNumber(summary.open_position.lot, 2)} />
                        <Fact label="Giriş" value={formatTs(summary.open_position.entry_ts)} />
                        <Fact
                          label="Giriş fiyatı"
                          value={formatNumber(summary.open_position.entry_price, 2)}
                        />
                        <Fact label="SL" value={formatNumber(summary.open_position.sl, 2)} />
                        <Fact label="TP" value={formatNumber(summary.open_position.tp, 2)} />
                        <Fact label="Durum" value={summary.open_position.status} />
                        <Fact
                          label="Kapanış"
                          value={formatTs(summary.open_position.closed_ts)}
                        />
                      </dl>
                    </div>
                  )}
                </section>

                <section
                  aria-labelledby="paper-attribution-heading"
                  className="rounded-md border border-slate-800 px-3 py-2"
                >
                  <h3
                    id="paper-attribution-heading"
                    className="mb-1 text-xs font-semibold text-slate-300"
                  >
                    Defter atfı
                  </h3>
                  <dl
                    data-testid="paper-attribution"
                    className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs"
                  >
                    <Fact
                      label="Strateji"
                      value={
                        <span className="flex flex-wrap items-center gap-1">
                          {summary.strategy}
                          {isLegacyStrategy(summary.strategy) && (
                            <LegacyBadge testId="paper-detail-legacy" />
                          )}
                        </span>
                      }
                      testId="paper-strategy"
                    />
                    <Fact
                      label="Model sürümü"
                      value={summary.model_version ?? "—"}
                      testId="paper-model-version"
                      mono
                    />
                    <Fact
                      label="DSL spec hash"
                      value={specHash ?? "—"}
                      testId="paper-spec-hash"
                      mono
                    />
                    <Fact
                      label="DSL spec sürümü"
                      value={formatNumber(specVersion, 0)}
                      testId="paper-spec-version"
                    />
                    <Fact
                      label="Veri kümesi"
                      value={datasetId ?? "—"}
                      testId="paper-dataset"
                      mono
                    />
                    <Fact label="Kayıt kaynağı" value={origin ?? "—"} />
                  </dl>
                  {specHash === null && (
                    <p className="mt-1 text-xs text-slate-400">
                      Bu oturumda DSL spec hash’i yok; atıf yalnızca strateji ve model sürümü
                      alanlarıyla sınırlı.
                    </p>
                  )}
                </section>
              </div>

              {/* -- equity curve ------------------------------------------- */}
              <figure
                aria-labelledby="paper-equity-figure-caption"
                data-testid="paper-equity-figure"
                className="min-w-0 rounded-md border border-slate-800 px-3 py-2"
              >
                <figcaption
                  id="paper-equity-figure-caption"
                  className="mb-1 text-xs font-semibold text-slate-300"
                >
                  Özkaynak eğrisi — yüklenen pencere
                </figcaption>

                {tables.equity === null ? (
                  <p
                    data-testid="paper-equity-chart-unread"
                    className="text-xs text-slate-400"
                  >
                    Özkaynak verisi okunamadı; eğri çizilmedi.
                  </p>
                ) : equityItems.length === 0 ? (
                  <p
                    data-testid="paper-equity-chart-empty"
                    className="text-xs text-slate-400"
                  >
                    Bu oturumda özkaynak işareti yok; eğri çizilmez.
                  </p>
                ) : equityStats === null ? (
                  <p
                    data-testid="paper-equity-chart-invalid"
                    className="text-xs text-slate-400"
                  >
                    Özkaynak işaretleri geçersiz zaman veya değer taşıyor; eğri çizilmedi.
                  </p>
                ) : (
                  <>
                    <LineChart
                      title={equityChartTitle}
                      points={equityStats.points}
                      color="var(--color-accent)"
                    />
                    <dl
                      data-testid="paper-equity-summary"
                      className="mt-1 grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2 lg:grid-cols-5"
                    >
                      <Fact
                        label="İlk"
                        value={formatMoney(equityStats.first)}
                        testId="paper-equity-first"
                      />
                      <Fact
                        label="Son"
                        value={formatMoney(equityStats.last)}
                        testId="paper-equity-last"
                      />
                      <Fact
                        label="Pencere değişimi"
                        value={signedMoney(equityStats.change)}
                        testId="paper-equity-change"
                      />
                      <Fact
                        label="Değişim %"
                        value={formatPct(equityStats.changePct)}
                        testId="paper-equity-change-pct"
                      />
                      <Fact
                        label="İşaret"
                        value={formatNumber(equityStats.points.length, 0)}
                        testId="paper-equity-points"
                      />
                    </dl>
                    <p className="mt-1 text-xs text-slate-400">
                      Değişim, yüklenen pencerenin ilk ve son özkaynak işareti arasındaki farktır;
                      hesabın başlangıç bakiyesine göre toplam P&amp;L değildir. Aşağıdaki
                      özkaynak tablosu aynı veriyi metin olarak verir.
                    </p>
                  </>
                )}
              </figure>

              {/* -- tables ------------------------------------------------- */}
              <div className="flex flex-col gap-4">
                <h3 className="text-xs font-semibold text-slate-300">Defter tabloları</h3>

                {tablesError !== null && (
                  <p data-testid="paper-tables-error" role="alert" className="text-xs text-rose-300">
                    {tablesError}
                  </p>
                )}

                <LedgerTable
                  captionId="paper-positions-caption"
                  caption="Pozisyonlar — en eski önce"
                  page={tables.positions}
                  error={null}
                  loading={summaryLoading}
                  empty="Bu oturumda pozisyon yok."
                  onPage={(offset) => pageTable("positions", offset)}
                >
                  <thead className="text-slate-400">
                    <tr>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Pozisyon
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Yön
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        Lot
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Giriş
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        Giriş fiyatı
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        SL
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        TP
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Durum
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Kapanış
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.positions?.items.map((row) => (
                      <tr key={row.position_id} className="border-t border-slate-800/60">
                        <th scope="row" className="px-3 py-1.5 font-mono font-normal text-slate-300">
                          #{row.position_id}
                        </th>
                        <td className="px-3 py-1.5 text-slate-300">{row.side}</td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.lot, 2)}
                        </td>
                        <td className="px-3 py-1.5 tabular-nums text-slate-300">
                          {formatTs(row.entry_ts)}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.entry_price, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.sl, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.tp, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-slate-300">{row.status}</td>
                        <td className="px-3 py-1.5 tabular-nums text-slate-300">
                          {row.closed_ts === null
                            ? "—"
                            : `${formatTs(row.closed_ts)} · ${formatNumber(row.closed_price, 2)}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </LedgerTable>

                <LedgerTable
                  captionId="paper-orders-caption"
                  caption="Emir denemeleri — en eski önce (ret nedenleriyle)"
                  page={tables.orders}
                  error={null}
                  loading={summaryLoading}
                  empty="Bu oturumda emir denemesi yok."
                  onPage={(offset) => pageTable("orders", offset)}
                >
                  <thead className="text-slate-400">
                    <tr>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Emir
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Zaman
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Yön
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        Lot
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        İstenen fiyat
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Durum
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Ret nedeni
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.orders?.items.map((row) => (
                      <tr key={row.order_id} className="border-t border-slate-800/60">
                        <th scope="row" className="px-3 py-1.5 font-mono font-normal text-slate-300">
                          #{row.order_id}
                        </th>
                        <td className="px-3 py-1.5 tabular-nums text-slate-300">
                          {formatTs(row.ts)}
                        </td>
                        <td className="px-3 py-1.5 text-slate-300">{row.side}</td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.lot, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.requested_price, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-slate-300">{row.status}</td>
                        <td className="px-3 py-1.5 text-rose-200">
                          {row.rejection_reason ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </LedgerTable>

                <LedgerTable
                  captionId="paper-history-caption"
                  caption="Kapanmış işlemler — en eski önce"
                  page={tables.history}
                  error={null}
                  loading={summaryLoading}
                  empty="Bu oturumda kapanmış işlem yok."
                  onPage={(offset) => pageTable("history", offset)}
                >
                  <thead className="text-slate-400">
                    <tr>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        İşlem
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Giriş
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Çıkış
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Yön
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        Lot
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        P&amp;L
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        R
                      </th>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Çıkış nedeni
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.history?.items.map((row) => (
                      <tr key={row.trade_id} className="border-t border-slate-800/60">
                        <th scope="row" className="px-3 py-1.5 font-mono font-normal text-slate-300">
                          #{row.trade_id}
                        </th>
                        <td className="px-3 py-1.5 tabular-nums text-slate-300">
                          {formatTs(row.entry_ts)} · {formatNumber(row.entry_price, 2)}
                        </td>
                        <td className="px-3 py-1.5 tabular-nums text-slate-300">
                          {formatTs(row.exit_ts)} · {formatNumber(row.exit_price, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-slate-300">{row.side}</td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.lot, 2)}
                        </td>
                        <td
                          className={`px-3 py-1.5 text-right tabular-nums ${
                            row.pnl < 0 ? "text-rose-300" : "text-slate-200"
                          }`}
                        >
                          {formatMoney(row.pnl)}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatNumber(row.r_multiple, 2)}
                        </td>
                        <td className="px-3 py-1.5 text-slate-300">{row.exit_reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </LedgerTable>

                <LedgerTable
                  captionId="paper-equity-caption"
                  caption="Özkaynak işaretleri — offset 0 en yeni pencere, artan zamanla"
                  page={tables.equity}
                  error={null}
                  loading={summaryLoading}
                  empty="Bu oturumda özkaynak işareti yok."
                  prevLabel="Daha yeni"
                  nextLabel="Daha eski"
                  onPage={(offset) => pageTable("equity", offset)}
                >
                  <thead className="text-slate-400">
                    <tr>
                      <th scope="col" className="px-3 py-1.5 font-medium">
                        Zaman
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        Bakiye
                      </th>
                      <th scope="col" className="px-3 py-1.5 text-right font-medium">
                        Özkaynak
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {tables.equity?.items.map((row) => (
                      <tr key={row.ts} className="border-t border-slate-800/60">
                        <th scope="row" className="px-3 py-1.5 font-normal tabular-nums text-slate-300">
                          {formatTs(row.ts)}
                        </th>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatMoney(row.balance)}
                        </td>
                        <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                          {formatMoney(row.equity)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </LedgerTable>
              </div>
            </>
          )}
        </section>
      )}
    </section>
  );
}
