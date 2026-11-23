"use client";

/**
 * Classical ML lab: start a real training run, watch it, and trade one split.
 *
 * Every number on this page comes from `/api/ml/*`. Nothing is estimated in the
 * browser: a missing metric is a dash, an absent split says it was not read,
 * and an empty run list says it is empty. Prediction metrics (accuracy, log
 * loss, …) are explicitly labelled as not a trading result; only `/evaluate`
 * runs the simulator.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

import { ApiError, api } from "@/lib/api";
import { formatMoney, formatNumber, formatTs } from "@/lib/format";
import { formatFractionPct } from "@/lib/metrics";
import type {
  MLEvaluateResponse,
  MLRegimeRow,
  MLRunDetail,
  MLRunSummary,
  MLSplit,
  MLTimeframe,
  MLTradingMetrics,
  SymbolSpec,
} from "@/lib/types";

const POLL_MS = 2_000;

/**
 * Statuses that can still change, so the page keeps asking. `preparing` is a
 * defensive case: the endpoint returns its 202 only after preparation, but a
 * run written by another process may carry it.
 */
const POLLABLE: ReadonlySet<string> = new Set(["preparing", "running", "cancelling"]);

const STATUS_LABEL: Record<string, string> = {
  preparing: "Hazırlanıyor",
  running: "Çalışıyor",
  cancelling: "İptal ediliyor",
  cancelled: "İptal edildi",
  completed: "Bitti",
  failed: "Hata",
  unknown: "Bilinmiyor",
};

const STATUS_CLASS: Record<string, string> = {
  preparing: "border-slate-700 text-slate-300",
  running: "border-accent/50 bg-accent/10 text-accent",
  cancelling: "border-amber-500/50 bg-amber-500/10 text-amber-300",
  cancelled: "border-slate-700 text-slate-300",
  completed: "border-profit/50 bg-profit/10 text-profit",
  failed: "border-loss/50 bg-loss/10 text-loss",
};

const SPLIT_LABEL: Record<MLSplit, string> = {
  train: "Eğitim",
  validation: "Doğrulama",
  test: "Test",
};

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent aria-[invalid=true]:border-rose-500/70";

const LABEL_CLASS = "text-xs text-slate-400";
const HINT_CLASS = "text-xs text-slate-500";
const ERROR_CLASS = "text-xs text-rose-300";

const BUTTON_BASE =
  "rounded-sm border px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const PRIMARY_BUTTON = `${BUTTON_BASE} border-accent bg-accent/15 font-medium text-slate-100 hover:bg-accent/25 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;
const SECONDARY_BUTTON = `${BUTTON_BASE} border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;

// --- form values and validation ---------------------------------------------

type FieldName =
  | "symbol"
  | "timeframe"
  | "model"
  | "horizon"
  | "trainFraction"
  | "validationFraction"
  | "purge"
  | "embargo";

interface FormValues {
  symbol: string;
  timeframe: string;
  model: string;
  horizon: string;
  trainFraction: string;
  validationFraction: string;
  purge: string;
  embargo: string;
  includeFlat: boolean;
}

type FormErrors = Partial<Record<FieldName, string>>;

const FIELD_ID: Record<FieldName, string> = {
  symbol: "ml-symbol",
  timeframe: "ml-timeframe",
  model: "ml-model",
  horizon: "ml-horizon",
  trainFraction: "ml-train-fraction",
  validationFraction: "ml-validation-fraction",
  purge: "ml-purge",
  embargo: "ml-embargo",
};

const FIELD_ORDER: readonly FieldName[] = [
  "symbol",
  "timeframe",
  "model",
  "horizon",
  "trainFraction",
  "validationFraction",
  "purge",
  "embargo",
];

/** Turkish keyboards write a decimal comma; the wire wants a dot. */
function toNumber(raw: string): number {
  return Number(raw.trim().replace(",", "."));
}

function integerValue(raw: string): number | null {
  const value = toNumber(raw);
  return Number.isInteger(value) ? value : null;
}

function validateForm(values: FormValues): FormErrors {
  const errors: FormErrors = {};
  if (values.symbol.trim() === "") errors.symbol = "Sembol seçin.";
  if (values.timeframe.trim() === "") errors.timeframe = "Zaman dilimi seçin.";
  if (values.model.trim() === "") errors.model = "Model seçin.";

  const horizon = integerValue(values.horizon);
  if (horizon === null || horizon < 1 || horizon > 1_000) {
    errors.horizon = "Ufuk 1 ile 1000 arasında bir tam sayı olmalı.";
  }

  const train = toNumber(values.trainFraction);
  const trainValid = Number.isFinite(train) && train > 0 && train < 1;
  if (!trainValid) {
    errors.trainFraction = "Eğitim oranı 0 ile 1 arasında olmalı (örn. 0.70).";
  }

  const validation = toNumber(values.validationFraction);
  const validationValid = Number.isFinite(validation) && validation >= 0 && validation < 1;
  if (!validationValid) {
    errors.validationFraction = "Doğrulama oranı 0 veya daha büyük, 1'den küçük olmalı.";
  } else if (trainValid && train + validation >= 1) {
    // Same rule as the API schema, checked here so the message is immediate.
    errors.validationFraction =
      "Eğitim + doğrulama oranı 1'den küçük olmalı; en az test için yer kalmalı.";
  }

  for (const [field, raw] of [
    ["purge", values.purge],
    ["embargo", values.embargo],
  ] as const) {
    const value = integerValue(raw);
    if (value === null || value < 0 || value > 100_000) {
      errors[field] = "0 ile 100000 arasında bir tam sayı olmalı.";
    }
  }

  return errors;
}

/** Split the stress field on commas or whitespace; dots are decimals. */
function parseStress(raw: string): { values: number[]; error: string | null } {
  const tokens = raw
    .split(/[,\s]+/)
    .map((token) => token.trim())
    .filter((token) => token !== "");
  if (tokens.length > 5) return { values: [], error: "En fazla 5 maliyet çarpanı girin." };

  const values: number[] = [];
  for (const token of tokens) {
    const value = Number(token);
    if (!Number.isFinite(value) || value <= 0 || value > 10) {
      return {
        values: [],
        error: "Her çarpan 0'dan büyük, 10'dan küçük veya eşit olmalı (ondalık için nokta).",
      };
    }
    values.push(value);
  }
  return { values, error: null };
}

// --- small reading helpers ---------------------------------------------------

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "İstek başarısız oldu.";
}

/** A wire number or `null`; NaN and infinities never reach the screen. */
function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function specNumber(spec: Record<string, unknown>, key: string): number | null {
  return finiteNumber(spec[key]);
}

function specText(spec: Record<string, unknown>, key: string): string | null {
  const value = spec[key];
  return typeof value === "string" && value !== "" ? value : null;
}

function specGroups(spec: Record<string, unknown>): string | null {
  const value = spec["groups"];
  if (!Array.isArray(value)) return null;
  const groups = value.filter((item): item is string => typeof item === "string");
  return groups.length === 0 ? null : groups.join(", ");
}

function MLStatusChip({ status, testId }: { status: string; testId?: string }) {
  const key = status.toLowerCase();
  const label = STATUS_LABEL[key] ?? status;
  const tone = STATUS_CLASS[key] ?? "border-slate-700 text-slate-300";
  return (
    <span
      data-testid={testId}
      data-status={key}
      className={`inline-block rounded-sm border px-2 py-0.5 text-xs ${tone}`}
    >
      {label}
    </span>
  );
}

/** One row of a prediction-metric table; the value is already formatted. */
interface MetricRow {
  label: string;
  hint?: string;
  value: string;
}

function formatFraction(value: unknown, digits = 2): string {
  return formatFractionPct(finiteNumber(value), digits);
}

function formatMetric(value: unknown, digits = 4): string {
  return formatNumber(finiteNumber(value), digits);
}

function predictionRows(detail: MLRunDetail): MetricRow[] {
  const validation = detail.validation;
  const validationEce = finiteNumber(validation?.calibration?.ece) ?? detail.calibration?.ece;
  const validationMce = finiteNumber(validation?.calibration?.mce) ?? detail.calibration?.mce;

  return [
    { label: "Satır", value: formatMetric(validation?.rows, 0) },
    { label: "Doğruluk", value: formatFraction(validation?.accuracy) },
    { label: "Dengeli doğruluk", value: formatFraction(validation?.balanced_accuracy) },
    { label: "Log loss", value: formatMetric(validation?.log_loss) },
    { label: "Brier", value: formatMetric(validation?.brier) },
    { label: "ECE", hint: "beklenen kalibrasyon hatası", value: formatMetric(validationEce) },
    { label: "MCE", hint: "korumalı en büyük hata", value: formatMetric(validationMce) },
  ];
}

// --- page --------------------------------------------------------------------

export default function MLLabPage() {
  const [values, setValues] = useState<FormValues>({
    symbol: "",
    timeframe: "H1",
    model: "logistic",
    horizon: "5",
    trainFraction: "0.70",
    validationFraction: "0.15",
    purge: "5",
    embargo: "5",
    includeFlat: false,
  });
  const [touched, setTouched] = useState<Partial<Record<FieldName, boolean>>>({});
  const [submitted, setSubmitted] = useState(false);

  const [symbols, setSymbols] = useState<SymbolSpec[]>([]);
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [registriesError, setRegistriesError] = useState<string | null>(null);
  const [registriesLoading, setRegistriesLoading] = useState(true);

  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [busyDetail, setBusyDetail] = useState<string | null>(null);

  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<MLRunDetail | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [runMissing, setRunMissing] = useState(false);
  const [pollToken, setPollToken] = useState(0);

  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelNotice, setCancelNotice] = useState<string | null>(null);

  const [split, setSplit] = useState<MLSplit>("validation");
  const [stressInput, setStressInput] = useState("2, 3");
  const [regime, setRegime] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [evaluation, setEvaluation] = useState<MLEvaluateResponse | null>(null);
  const [evaluateError, setEvaluateError] = useState<string | null>(null);
  const [stressError, setStressError] = useState<string | null>(null);

  const [runs, setRuns] = useState<MLRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsMissing, setRunsMissing] = useState(false);

  const formRef = useRef<HTMLFormElement>(null);

  const errors = validateForm(values);
  const visibleErrors: FormErrors = submitted
    ? errors
    : Object.fromEntries(
        Object.entries(errors).filter(([field]) => touched[field as FieldName]),
      );

  const refreshRuns = useCallback(async (): Promise<void> => {
    setRunsLoading(true);
    try {
      const rows = await api.getMLRuns();
      setRuns(rows);
      setRunsError(null);
      setRunsMissing(false);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setRunsMissing(true);
        setRunsError(null);
      } else {
        setRunsError(errorDetail(error));
      }
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      const [symbolsResult, timeframesResult, modelsResult] = await Promise.allSettled([
        api.getSymbols(),
        api.getTimeframes(),
        api.mlModels(),
      ]);
      if (cancelled) return;

      // Every failure is kept: one broken list must not hide another, because
      // the form cannot be submitted without all three.
      const failures: string[] = [];

      if (symbolsResult.status === "fulfilled") {
        setSymbols(symbolsResult.value);
        const list = symbolsResult.value;
        if (list.length > 0) {
          setValues((current) =>
            current.symbol === "" ? { ...current, symbol: list[0].symbol } : current,
          );
        }
      } else {
        failures.push(`Sembol listesi alınamadı (${errorDetail(symbolsResult.reason)}).`);
      }

      if (timeframesResult.status === "fulfilled") {
        const names = timeframesResult.value.map((item) => item.timeframe);
        setTimeframes(names);
        setValues((current) =>
          names.includes(current.timeframe)
            ? current
            : { ...current, timeframe: names[0] ?? "" },
        );
      } else {
        failures.push(
          `Zaman dilimi listesi alınamadı (${errorDetail(timeframesResult.reason)}).`,
        );
      }

      if (modelsResult.status === "fulfilled") {
        setModels(modelsResult.value);
        setValues((current) =>
          modelsResult.value.includes(current.model)
            ? current
            : { ...current, model: modelsResult.value[0] ?? "" },
        );
      } else {
        failures.push(`Model listesi alınamadı (${errorDetail(modelsResult.reason)}).`);
      }

      setRegistriesError(failures.length > 0 ? failures.join(" ") : null);
      setRegistriesLoading(false);
    })();

    void refreshRuns();

    return () => {
      cancelled = true;
    };
  }, [refreshRuns]);

  // Poll one run every 2 s while it can still change. Cleanup clears the timer
  // and marks the loop stopped, so an unmount cannot set state.
  useEffect(() => {
    if (runId === null) return;

    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async (): Promise<void> => {
      try {
        const detail = await api.getMLRun(runId);
        if (stopped) return;
        setRun(detail);
        setRunError(null);
        setRunMissing(false);
        if (POLLABLE.has(detail.status)) {
          timer = setTimeout(() => void tick(), POLL_MS);
        } else {
          void refreshRuns();
        }
      } catch (error) {
        if (stopped) return;
        if (error instanceof ApiError && error.status === 404) {
          setRun(null);
          setRunMissing(true);
          setRunError(null);
        } else {
          // A failed poll stops the loop; the run may still be alive, so the
          // panel offers an explicit retry instead of a silent gap.
          setRunError(errorDetail(error));
        }
      }
    };

    void tick();

    return () => {
      stopped = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [runId, pollToken, refreshRuns]);

  const handleChange = <K extends keyof FormValues>(
    key: K,
    value: FormValues[K],
    field?: FieldName,
  ): void => {
    setValues((current) => ({ ...current, [key]: value }));
    if (field !== undefined) {
      setTouched((current) => ({ ...current, [field]: false }));
    }
  };

  const focusFirstError = (found: FormErrors): void => {
    const first = FIELD_ORDER.find((field) => found[field] !== undefined);
    if (first === undefined) return;
    // The control is a child of this form, so a scoped query is enough.
    formRef.current?.querySelector<HTMLElement>(`#${FIELD_ID[first]}`)?.focus();
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (starting) return;

    setSubmitted(true);
    const found = validateForm(values);
    if (Object.keys(found).length > 0) {
      focusFirstError(found);
      return;
    }

    setStarting(true);
    setStartError(null);
    setBusyDetail(null);
    setRunId(null);
    setRun(null);
    setRunError(null);
    setRunMissing(false);
    setCancelError(null);
    setCancelNotice(null);
    setEvaluation(null);
    setEvaluateError(null);

    void api
      .mlTrain({
        symbol: values.symbol,
        timeframe: values.timeframe as MLTimeframe,
        model: values.model,
        horizon: integerValue(values.horizon) ?? 0,
        train_fraction: toNumber(values.trainFraction),
        validation_fraction: toNumber(values.validationFraction),
        purge_bars: integerValue(values.purge) ?? 0,
        embargo_bars: integerValue(values.embargo) ?? 0,
        include_flat: values.includeFlat,
      })
      .then((result) => {
        setRunId(result.run_id);
      })
      .catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 409) {
          setBusyDetail(error.detail);
        } else {
          setStartError(errorDetail(error));
        }
      })
      .finally(() => {
        setStarting(false);
      });
  };

  const handleSelectRun = (id: string): void => {
    if (id === runId) return;
    setRunId(id);
    setRun(null);
    setRunError(null);
    setRunMissing(false);
    setCancelError(null);
    setCancelNotice(null);
    setEvaluation(null);
    setEvaluateError(null);
    setPollToken((token) => token + 1);
  };

  const handleCancel = (): void => {
    if (runId === null || run?.status !== "running") return;
    setCancelling(true);
    setCancelError(null);
    setCancelNotice(null);
    void api
      .cancelMLRun(runId)
      .then(() => {
        setCancelNotice(
          "İptal istendi. İşçi bir sonraki kontrol noktasında durur; uzun bir fit sırasında durum bir süre “İptal ediliyor” kalabilir.",
        );
        setPollToken((token) => token + 1);
      })
      .catch((error: unknown) => {
        setCancelError(errorDetail(error));
      })
      .finally(() => {
        setCancelling(false);
      });
  };

  const handleEvaluate = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (runId === null || run?.status !== "completed" || evaluating) return;

    const parsed = parseStress(stressInput);
    if (parsed.error !== null) {
      // A field-level message, kept apart from the server error so the two
      // never overwrite each other.
      setStressError(parsed.error);
      setEvaluateError(null);
      return;
    }

    setEvaluating(true);
    setStressError(null);
    setEvaluateError(null);
    setEvaluation(null);
    void api
      .mlEvaluate({ run_id: runId, split, stress: parsed.values, regime })
      .then((result) => {
        setEvaluation(result);
      })
      .catch((error: unknown) => {
        setEvaluateError(errorDetail(error));
      })
      .finally(() => {
        setEvaluating(false);
      });
  };

  const live = run !== null && POLLABLE.has(run.status);
  const canCancel = run !== null && run.status === "running" && !cancelling;
  const cancelExplanation =
    run === null
      ? "İptal için önce bir koşu seçin."
      : run.status === "running"
        ? cancelling
          ? "İptal isteği gönderiliyor…"
          : "İptal, işçiye birlikte çalışan bir bayrak bırakır."
        : run.status === "cancelling"
          ? "İptal zaten istendi; işçi kontrol noktasında duracak."
          : "Bu koşu bitti; iptal edilemez.";

  const canEvaluate = run !== null && run.status === "completed" && !evaluating;
  const evaluateExplanation =
    run === null
      ? "Önce bitmiş bir koşu seçin."
      : run.status !== "completed"
        ? "Yalnızca bitmiş bir koşu değerlendirilebilir."
        : evaluating
          ? "Değerlendirme sürüyor…"
          : null;

  const errorEntries = FIELD_ORDER.filter((field) => visibleErrors[field] !== undefined).map(
    (field) => ({ field, message: visibleErrors[field] as string }),
  );

  const statusSentence =
    run === null
      ? runError !== null
        ? `Koşu durumu alınamadı: ${runError}`
        : runMissing
          ? "Koşu bulunamadı."
          : runId !== null
            ? "Koşu durumu yükleniyor."
            : ""
      : `${run.run_id}: ${STATUS_LABEL[run.status] ?? run.status}`;

  return (
    <section aria-labelledby="ml-lab-heading" className="flex flex-col gap-5">
      <div>
        <h1 id="ml-lab-heading" className="text-xl font-semibold text-slate-100">
          ML Laboratuvarı
        </h1>
        <p className="text-xs text-slate-400">
          Klasik sınıflandırıcı denemeleri — tahmin metrikleri alım satım sonucu
          değildir.
        </p>
      </div>

      <p className="sr-only" role="status">
        {statusSentence}
      </p>

      {registriesError !== null && (
        <p data-testid="ml-registries-error" role="alert" className="text-sm text-rose-300">
          {registriesError} Form, sunucunun listeleri olmadan gönderilemez.
        </p>
      )}

      {/* -- training form -------------------------------------------------- */}
      <form
        ref={formRef}
        data-testid="ml-train-form"
        aria-labelledby="ml-train-heading"
        noValidate
        onSubmit={handleSubmit}
        aria-busy={starting}
        className="flex flex-col gap-4 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="ml-train-heading" className="text-sm font-semibold text-slate-100">
            Yeni koşu
          </h2>
          <span className="text-xs text-slate-500">
            {registriesLoading ? "Listeler yükleniyor…" : "Sunucu listeleri yüklendi"}
          </span>
        </div>

        <fieldset className="flex flex-col gap-3">
          <legend className="mb-1 text-xs font-medium text-slate-300">Veri ve model</legend>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.symbol} className={LABEL_CLASS}>
                Sembol
              </label>
              <select
                id={FIELD_ID.symbol}
                required
                value={values.symbol}
                disabled={starting}
                aria-invalid={visibleErrors.symbol !== undefined}
                aria-describedby={
                  visibleErrors.symbol !== undefined ? `${FIELD_ID.symbol}-error` : undefined
                }
                onChange={(event) => handleChange("symbol", event.target.value, "symbol")}
                onBlur={() => setTouched((current) => ({ ...current, symbol: true }))}
                className={FIELD_CLASS}
              >
                <option value="">Sembol seçin</option>
                {symbols.map((item) => (
                  <option key={item.symbol} value={item.symbol}>
                    {item.symbol} — {item.description}
                  </option>
                ))}
              </select>
              {visibleErrors.symbol !== undefined && (
                <p id={`${FIELD_ID.symbol}-error`} className={ERROR_CLASS}>
                  {visibleErrors.symbol}
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.timeframe} className={LABEL_CLASS}>
                Zaman dilimi
              </label>
              <select
                id={FIELD_ID.timeframe}
                required
                value={values.timeframe}
                disabled={starting}
                aria-invalid={visibleErrors.timeframe !== undefined}
                aria-describedby={
                  visibleErrors.timeframe !== undefined
                    ? `${FIELD_ID.timeframe}-error`
                    : undefined
                }
                onChange={(event) => handleChange("timeframe", event.target.value, "timeframe")}
                onBlur={() => setTouched((current) => ({ ...current, timeframe: true }))}
                className={FIELD_CLASS}
              >
                <option value="">Zaman dilimi seçin</option>
                {timeframes.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              {visibleErrors.timeframe !== undefined && (
                <p id={`${FIELD_ID.timeframe}-error`} className={ERROR_CLASS}>
                  {visibleErrors.timeframe}
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.model} className={LABEL_CLASS}>
                Model
              </label>
              <select
                id={FIELD_ID.model}
                required
                value={values.model}
                disabled={starting}
                aria-invalid={visibleErrors.model !== undefined}
                aria-describedby={
                  visibleErrors.model !== undefined ? `${FIELD_ID.model}-error` : undefined
                }
                onChange={(event) => handleChange("model", event.target.value, "model")}
                onBlur={() => setTouched((current) => ({ ...current, model: true }))}
                className={FIELD_CLASS}
              >
                <option value="">Model seçin</option>
                {models.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              {visibleErrors.model !== undefined && (
                <p id={`${FIELD_ID.model}-error`} className={ERROR_CLASS}>
                  {visibleErrors.model}
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.horizon} className={LABEL_CLASS}>
                Ufuk (bar)
              </label>
              <input
                id={FIELD_ID.horizon}
                type="number"
                inputMode="numeric"
                min={1}
                max={1000}
                step={1}
                required
                value={values.horizon}
                disabled={starting}
                aria-describedby={
                  visibleErrors.horizon !== undefined
                    ? `ml-horizon-hint ${FIELD_ID.horizon}-error`
                    : "ml-horizon-hint"
                }
                aria-invalid={visibleErrors.horizon !== undefined}
                onChange={(event) => handleChange("horizon", event.target.value, "horizon")}
                onBlur={() => setTouched((current) => ({ ...current, horizon: true }))}
                className={FIELD_CLASS}
              />
              <p id="ml-horizon-hint" className={HINT_CLASS}>
                Etiket, bu kadar bar sonraki yön.
              </p>
              {visibleErrors.horizon !== undefined && (
                <p id={`${FIELD_ID.horizon}-error`} className={ERROR_CLASS}>
                  {visibleErrors.horizon}
                </p>
              )}
            </div>
          </div>
        </fieldset>

        <fieldset className="flex flex-col gap-3">
          <legend className="mb-1 text-xs font-medium text-slate-300">Bölünme ve boşluk</legend>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.trainFraction} className={LABEL_CLASS}>
                Eğitim oranı
              </label>
              <input
                id={FIELD_ID.trainFraction}
                type="number"
                inputMode="decimal"
                min={0.01}
                max={0.99}
                step={0.01}
                required
                value={values.trainFraction}
                disabled={starting}
                aria-invalid={visibleErrors.trainFraction !== undefined}
                aria-describedby={
                  visibleErrors.trainFraction !== undefined
                    ? `${FIELD_ID.trainFraction}-error`
                    : undefined
                }
                onChange={(event) =>
                  handleChange("trainFraction", event.target.value, "trainFraction")
                }
                onBlur={() =>
                  setTouched((current) => ({ ...current, trainFraction: true }))
                }
                className={FIELD_CLASS}
              />
              {visibleErrors.trainFraction !== undefined && (
                <p id={`${FIELD_ID.trainFraction}-error`} className={ERROR_CLASS}>
                  {visibleErrors.trainFraction}
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.validationFraction} className={LABEL_CLASS}>
                Doğrulama oranı
              </label>
              <input
                id={FIELD_ID.validationFraction}
                type="number"
                inputMode="decimal"
                min={0}
                max={0.99}
                step={0.01}
                required
                value={values.validationFraction}
                disabled={starting}
                aria-invalid={visibleErrors.validationFraction !== undefined}
                aria-describedby={
                  visibleErrors.validationFraction !== undefined
                    ? `${FIELD_ID.validationFraction}-error`
                    : undefined
                }
                onChange={(event) =>
                  handleChange("validationFraction", event.target.value, "validationFraction")
                }
                onBlur={() =>
                  setTouched((current) => ({ ...current, validationFraction: true }))
                }
                className={FIELD_CLASS}
              />
              {visibleErrors.validationFraction !== undefined && (
                <p id={`${FIELD_ID.validationFraction}-error`} className={ERROR_CLASS}>
                  {visibleErrors.validationFraction}
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.purge} className={LABEL_CLASS}>
                Purge (bar)
              </label>
              <input
                id={FIELD_ID.purge}
                type="number"
                inputMode="numeric"
                min={0}
                max={100000}
                step={1}
                required
                value={values.purge}
                disabled={starting}
                aria-invalid={visibleErrors.purge !== undefined}
                aria-describedby={
                  visibleErrors.purge !== undefined ? `${FIELD_ID.purge}-error` : undefined
                }
                onChange={(event) => handleChange("purge", event.target.value, "purge")}
                onBlur={() => setTouched((current) => ({ ...current, purge: true }))}
                className={FIELD_CLASS}
              />
              {visibleErrors.purge !== undefined && (
                <p id={`${FIELD_ID.purge}-error`} className={ERROR_CLASS}>
                  {visibleErrors.purge}
                </p>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor={FIELD_ID.embargo} className={LABEL_CLASS}>
                Embargo (bar)
              </label>
              <input
                id={FIELD_ID.embargo}
                type="number"
                inputMode="numeric"
                min={0}
                max={100000}
                step={1}
                required
                value={values.embargo}
                disabled={starting}
                aria-invalid={visibleErrors.embargo !== undefined}
                aria-describedby={
                  visibleErrors.embargo !== undefined
                    ? `${FIELD_ID.embargo}-error`
                    : undefined
                }
                onChange={(event) => handleChange("embargo", event.target.value, "embargo")}
                onBlur={() => setTouched((current) => ({ ...current, embargo: true }))}
                className={FIELD_CLASS}
              />
              {visibleErrors.embargo !== undefined && (
                <p id={`${FIELD_ID.embargo}-error`} className={ERROR_CLASS}>
                  {visibleErrors.embargo}
                </p>
              )}
            </div>
          </div>

          <div className="flex items-start gap-2">
            <input
              id="ml-include-flat"
              type="checkbox"
              checked={values.includeFlat}
              disabled={starting}
              onChange={(event) => handleChange("includeFlat", event.target.checked)}
              className="mt-0.5 accent-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            />
            <div className="flex flex-col">
              <label htmlFor="ml-include-flat" className="text-xs text-slate-300">
                Düz (sıfır getiri) satırları dahil et
              </label>
              <p className={HINT_CLASS}>
                Varsayılan olarak düz satırlar etiketlenmez; sayıları bölünme
                tablosunda ayrı görünür.
              </p>
            </div>
          </div>
        </fieldset>

        {submitted && errorEntries.length > 0 && (
          <div
            data-testid="ml-form-errors"
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
                    href={`#${FIELD_ID[entry.field]}`}
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
          <button
            type="submit"
            disabled={starting}
            aria-describedby={starting ? "ml-start-busy" : undefined}
            className={PRIMARY_BUTTON}
          >
            Koşuyu başlat
          </button>
          {starting && (
            <span id="ml-start-busy" className="text-xs text-slate-400">
              Hazırlanıyor: doğrulama, depo okuması ve özellik üretimi 202’den
              önce çalışır.
            </span>
          )}
          {!starting && (
            <span className="text-xs text-slate-500">
              Bir koşu süresince ikinci bir iş 409 döner; istek kuyruğa alınmaz.
            </span>
          )}
        </div>

        {busyDetail !== null && (
          <div
            data-testid="ml-busy"
            role="alert"
            className="rounded-sm border border-amber-500/40 bg-amber-500/10 px-3 py-2"
          >
            <p className="text-sm font-medium text-amber-200">Başka bir ML işi çalışıyor</p>
            <p className="mt-1 text-xs text-amber-200">
              {busyDetail}. Bu istek kuyruğa alınmadı; iş bitince yeniden deneyin.
            </p>
          </div>
        )}

        {startError !== null && (
          <p data-testid="ml-start-error" role="alert" className="text-sm text-rose-300">
            Koşu başlatılamadı: {startError}
          </p>
        )}
      </form>

      {/* -- one run -------------------------------------------------------- */}
      {runId !== null && (
        <section
          aria-labelledby="ml-run-heading"
          className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 id="ml-run-heading" className="text-sm font-semibold text-slate-100">
                Koşu
              </h2>
              <span data-testid="ml-run-id" className="font-mono text-sm text-slate-100">
                {runId}
              </span>
              {run !== null && <MLStatusChip status={run.status} testId="ml-run-status" />}
              {live && <span className="text-xs text-slate-500">2 sn’de bir sorgulanıyor</span>}
            </div>

            <div className="flex flex-col items-end gap-1">
              <button
                type="button"
                onClick={handleCancel}
                disabled={!canCancel}
                aria-describedby="ml-cancel-explanation"
                className={SECONDARY_BUTTON}
              >
                İptal et
              </button>
              <p id="ml-cancel-explanation" className="text-xs text-slate-500">
                {cancelExplanation}
              </p>
            </div>
          </div>

          {cancelNotice !== null && (
            <p role="status" className="text-xs text-amber-200">
              {cancelNotice}
            </p>
          )}
          {cancelError !== null && (
            <p role="alert" className="text-xs text-rose-300">
              İptal isteği başarısız: {cancelError}
            </p>
          )}

          {runError !== null && (
            <div data-testid="ml-run-error" role="alert" className="flex flex-wrap items-center gap-3">
              <p className="text-sm text-rose-300">Koşu durumu alınamadı: {runError}</p>
              <button
                type="button"
                onClick={() => setPollToken((token) => token + 1)}
                className={SECONDARY_BUTTON}
              >
                Yenile
              </button>
            </div>
          )}

          {runMissing && (
            <p data-testid="ml-run-missing" role="status" className="text-sm text-amber-200">
              Bu koşu sunucuda bulunamadı (404). Koşu silinmiş veya başka bir
              makinede oluşturulmuş olabilir; sayfa metrik uydurmaz.
            </p>
          )}

          {run === null && runError === null && !runMissing && (
            <p className="text-sm text-slate-400">Koşu durumu yükleniyor…</p>
          )}

          {run !== null && run.status === "failed" && (
            <div data-testid="ml-run-failed" role="alert" className="flex flex-col gap-1">
              <p className="text-sm font-medium text-rose-300">Koşu başarısız oldu.</p>
              <p className="text-xs text-rose-200">
                {run.error ?? "Sunucu bir hata metni döndürmedi."}
              </p>
              <p className="text-xs text-slate-500">
                Tek iş yuvası bu koşuyla serbest kaldı; yeni bir deneme
                başlatılabilir.
              </p>
            </div>
          )}

          {run !== null && run.status === "cancelled" && (
            <p data-testid="ml-run-cancelled" role="status" className="text-sm text-slate-300">
              Koşu iptal edildi{run.error !== null ? ` — ${run.error}` : ""}. Kısmi
              metrik yoktur; iptal edilen bir koşu sonuç üretmez.
            </p>
          )}

          {run !== null && run.status === "cancelling" && (
            <p data-testid="ml-run-cancelling" role="status" className="text-sm text-amber-200">
              İptal istendi. İşçi fit veya simülatör çağrısının içindeyse durum
              “İptal ediliyor” kalır; bu takılı bir iş değildir.
            </p>
          )}

          {run !== null && (run.status === "running" || run.status === "preparing") && (
            <p data-testid="ml-run-running" role="status" className="text-sm text-slate-300">
              Koşu {run.status === "running" ? "çalışıyor" : "hazırlanıyor"}. Metrikler
              bittiğinde tek seferde yazılır; ara değer gösterilmez.
            </p>
          )}

          {run !== null && run.status === "completed" && <RunResults detail={run} />}
        </section>
      )}

      {/* -- trading evaluation --------------------------------------------- */}
      <form
        data-testid="ml-evaluate-form"
        aria-labelledby="ml-evaluate-heading"
        onSubmit={handleEvaluate}
        aria-busy={evaluating}
        className="flex flex-col gap-3 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="ml-evaluate-heading" className="text-sm font-semibold text-slate-100">
            Alım satım değerlendirmesi
          </h2>
          <span className="text-xs text-slate-500">Simülatör maliyetleri dahil</span>
        </div>
        <p className="text-xs text-slate-400">
          Bitmiş bir koşunun artefaktı tek bir kronolojik bölünmede işlem yapar.
          Sembol ve zaman dilimi koşudan okunur, bu formdan değil.
        </p>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <div className="flex flex-col gap-1">
            <label htmlFor="ml-split" className={LABEL_CLASS}>
              Bölünme
            </label>
            <select
              id="ml-split"
              value={split}
              disabled={evaluating}
              onChange={(event) => setSplit(event.target.value as MLSplit)}
              className={FIELD_CLASS}
            >
              {(["validation", "train", "test"] as const).map((name) => (
                <option key={name} value={name}>
                  {SPLIT_LABEL[name]}
                </option>
              ))}
            </select>
            <p className={HINT_CLASS}>
              Test bölünmesi aday başına yalnızca bir kez okunmalı.
            </p>
          </div>

          <div className="flex flex-col gap-1">
            <label htmlFor="ml-stress" className={LABEL_CLASS}>
              Maliyet çarpanları
            </label>
            <input
              id="ml-stress"
              type="text"
              inputMode="decimal"
              value={stressInput}
              disabled={evaluating}
              onChange={(event) => {
                setStressInput(event.target.value);
                setStressError(null);
              }}
              aria-invalid={stressError !== null}
              aria-describedby={
                stressError !== null ? "ml-stress-hint ml-stress-error" : "ml-stress-hint"
              }
              autoComplete="off"
              spellCheck={false}
              className={FIELD_CLASS}
            />
            <p id="ml-stress-hint" className={HINT_CLASS}>
              Virgül veya boşlukla ayırın (en fazla 5); boş bırakılırsa stres
              tablosu olmaz. Ondalık için nokta kullanın.
            </p>
            {stressError !== null && (
              <p id="ml-stress-error" data-testid="ml-stress-error" className={ERROR_CLASS}>
                {stressError}
              </p>
            )}
          </div>

          <div className="flex items-start gap-2 sm:items-end">
            <input
              id="ml-regime"
              type="checkbox"
              checked={regime}
              disabled={evaluating}
              onChange={(event) => setRegime(event.target.checked)}
              className="mb-2 accent-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            />
            <div className="mb-1 flex flex-col">
              <label htmlFor="ml-regime" className="text-xs text-slate-300">
                Rejim kırılımını hesapla
              </label>
              <p className={HINT_CLASS}>Giriş barındaki kural etiketiyle eşlenir.</p>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={!canEvaluate}
            aria-describedby="ml-evaluate-explanation"
            className={PRIMARY_BUTTON}
          >
            Değerlendir
          </button>
          <span id="ml-evaluate-explanation" className="text-xs text-slate-500">
            {evaluateExplanation ?? "Bölünme bir kez simülatörde çalıştırılır."}
          </span>
        </div>

        {evaluateError !== null && (
          <p data-testid="ml-evaluate-error" role="alert" className="text-sm text-rose-300">
            Değerlendirme başarısız: {evaluateError}
          </p>
        )}

        {evaluation !== null && <EvaluationResults result={evaluation} />}
      </form>

      {/* -- runs ----------------------------------------------------------- */}
      <section aria-labelledby="ml-runs-heading" className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="ml-runs-heading" className="text-sm font-semibold text-slate-100">
            Koşular
          </h2>
          <button
            type="button"
            onClick={() => void refreshRuns()}
            disabled={runsLoading}
            aria-describedby="ml-runs-refresh-note"
            className={SECONDARY_BUTTON}
          >
            Listeyi yenile
          </button>
        </div>
        <p id="ml-runs-refresh-note" className="text-xs text-slate-500">
          Liste `runs/ml/meta.json` dosyalarından gelir; boş liste gerçekten boş
          demektir.
        </p>

        {runsLoading && runs.length === 0 ? (
          <p className="text-sm text-slate-400">Yükleniyor…</p>
        ) : runsMissing ? (
          <p data-testid="ml-runs-missing" role="status" className="text-sm text-amber-200">
            Sunucuda ML koşu listesi yok (404). Bu uç nokta olmadan eski koşular
            listelenemez.
          </p>
        ) : runsError !== null ? (
          <p data-testid="ml-runs-error" role="alert" className="text-sm text-rose-300">
            Koşu listesi alınamadı: {runsError}
          </p>
        ) : runs.length === 0 ? (
          <p data-testid="ml-runs-empty" role="status" className="text-sm text-slate-400">
            Henüz ML koşusu yok. Yukarıdaki formdan ilk denemeyi başlatın.
          </p>
        ) : (
          <RunsTable runs={runs} selectedId={runId} onSelect={handleSelectRun} />
        )}
      </section>
    </section>
  );
}

// --- results -----------------------------------------------------------------

function RunResults({ detail }: { detail: MLRunDetail }) {
  const rows = predictionRows(detail);
  const testValues = [
    formatMetric(detail.test?.rows, 0),
    formatFraction(detail.test?.accuracy),
    formatFraction(detail.test?.balanced_accuracy),
    formatMetric(detail.test?.log_loss),
    formatMetric(detail.test?.brier),
    formatMetric(detail.test?.calibration?.ece),
    formatMetric(detail.test?.calibration?.mce),
  ];

  const splitCounts: Array<{ key: string; label: string }> = [
    { key: "train", label: "Eğitim" },
    { key: "validation", label: "Doğrulama" },
    { key: "test", label: "Test" },
    { key: "purged", label: "Purged" },
    { key: "flat_dropped", label: "Düz (atılan)" },
  ];

  const baselineAccuracy = finiteNumber(detail.baseline?.["accuracy"]);
  const baselineLogLoss = finiteNumber(detail.baseline?.["log_loss"]);

  const specFacts: Array<{ label: string; value: string }> = [];
  const specModel = specText(detail.spec, "model");
  if (specModel !== null) specFacts.push({ label: "Model", value: specModel });
  const specHorizon = specNumber(detail.spec, "horizon");
  if (specHorizon !== null) {
    specFacts.push({ label: "Ufuk", value: `${formatNumber(specHorizon, 0)} bar` });
  }
  const trainFraction = specNumber(detail.spec, "train_fraction");
  const validationFraction = specNumber(detail.spec, "validation_fraction");
  if (trainFraction !== null && validationFraction !== null) {
    specFacts.push({
      label: "Eğitim / doğrulama",
      value: `${formatFractionPct(trainFraction, 0)} / ${formatFractionPct(validationFraction, 0)}`,
    });
  }
  const purge = specNumber(detail.spec, "purge_bars");
  const embargo = specNumber(detail.spec, "embargo_bars");
  if (purge !== null && embargo !== null) {
    specFacts.push({
      label: "Purge / embargo",
      value: `${formatNumber(purge, 0)} / ${formatNumber(embargo, 0)} bar`,
    });
  }
  const includeFlat = detail.spec["include_flat"];
  if (typeof includeFlat === "boolean") {
    specFacts.push({
      label: "Düz satırlar",
      value: includeFlat ? "dahil" : "hariç",
    });
  }
  const groups = specGroups(detail.spec);
  if (groups !== null) specFacts.push({ label: "Gruplar", value: groups });

  return (
    <div className="flex flex-col gap-4">
      <dl
        data-testid="ml-run-facts"
        className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2"
      >
        <div className="flex gap-2">
          <dt className="text-slate-500">Veri kümesi</dt>
          <dd className="truncate font-mono text-slate-300">{detail.dataset_id ?? "—"}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="text-slate-500">Sembol / zaman dilimi</dt>
          <dd className="text-slate-300">
            {detail.symbol ?? "—"} / {detail.timeframe ?? "—"}
          </dd>
        </div>
        <div className="flex gap-2">
          <dt className="text-slate-500">Özellik sütunu</dt>
          <dd className="tabular-nums text-slate-300">
            {formatNumber(detail.columns.length, 0)}
          </dd>
        </div>
        <div className="flex gap-2">
          <dt className="text-slate-500">Bitiş</dt>
          <dd className="text-slate-300">{formatTs(detail.finished_at)}</dd>
        </div>
      </dl>

      {specFacts.length > 0 && (
        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {specFacts.map((fact) => (
            <div key={fact.label} className="flex gap-1">
              <dt className="text-slate-500">{fact.label}:</dt>
              <dd className="text-slate-300">{fact.value}</dd>
            </div>
          ))}
        </dl>
      )}

      <div
        role="region"
        aria-labelledby="ml-prediction-caption"
        tabIndex={0}
        className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        <table className="w-full min-w-[26rem] border-collapse text-left text-xs">
          <caption id="ml-prediction-caption" className="px-3 py-2 text-left text-slate-400">
            Tahmin metrikleri — alım satım sonucu değil
          </caption>
          <thead className="bg-slate-900/60 text-slate-400">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-medium">
                Metrik
              </th>
              <th scope="col" className="px-3 py-1.5 text-right font-medium">
                Doğrulama
              </th>
              <th scope="col" className="px-3 py-1.5 text-right font-medium">
                Test
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={row.label} className="border-t border-slate-800/60">
                <th scope="row" className="px-3 py-1.5 font-normal text-slate-300">
                  {row.label}
                  {row.hint !== undefined && (
                    <span className="ml-1 text-slate-500">({row.hint})</span>
                  )}
                </th>
                <td
                  data-testid={`ml-validation-${row.label}`}
                  className="px-3 py-1.5 text-right tabular-nums text-slate-200"
                >
                  {row.value}
                </td>
                <td
                  data-testid={`ml-test-${row.label}`}
                  className="px-3 py-1.5 text-right tabular-nums text-slate-200"
                >
                  {testValues[index]}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {detail.test === null && (
        <p data-testid="ml-test-absent" className="text-xs text-slate-500">
          Test bölünmesi okunmadı: koşu `evaluate_test` olmadan çalıştı. Test
          sütunu bu yüzden tire gösterir.
        </p>
      )}

      {baselineAccuracy !== null && (
        <p data-testid="ml-baseline" className="text-xs text-slate-400">
          Baseline (çoğunluk sınıfı): doğruluk {formatFractionPct(baselineAccuracy, 1)}
          {baselineLogLoss !== null
            ? ` · prior log loss ${formatNumber(baselineLogLoss, 4)}`
            : ""}
          . Baseline’ı geçmeyen bir model önerilmez.
        </p>
      )}

      <div className="flex flex-col gap-1">
        <h3 className="text-xs font-semibold text-slate-300">Bölünme satır sayıları</h3>
        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {splitCounts.map((item) => (
            <div key={item.key} className="flex gap-1">
              <dt className="text-slate-500">{item.label}:</dt>
              <dd
                data-testid={`ml-count-${item.key}`}
                className="tabular-nums text-slate-300"
              >
                {formatNumber(finiteNumber(detail.split_counts[item.key]), 0)}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <p data-testid="ml-not-trading" className="text-xs text-amber-200">
        Bu tablo tahmin kalitesidir, alım satım sonucu değildir. Kâr/zarar için
        aşağıdaki değerlendirme gerekir; o da bir promosyon kararı değildir.
      </p>
    </div>
  );
}

function EvaluationResults({ result }: { result: MLEvaluateResponse }) {
  const trading: Array<{ label: string; key: keyof MLTradingMetrics; kind: string }> = [
    { label: "İşlem", key: "trades", kind: "count" },
    { label: "Net kâr", key: "net_profit", kind: "money" },
    { label: "Profit Factor", key: "profit_factor", kind: "number" },
    { label: "Win rate", key: "win_rate", kind: "fraction" },
    { label: "Expectancy (R)", key: "expectancy_r", kind: "number" },
    { label: "Max DD", key: "max_drawdown", kind: "drawdown" },
    { label: "Toplam maliyet", key: "total_costs", kind: "money" },
  ];

  const formatTrading = (
    metrics: MLTradingMetrics,
    key: keyof MLTradingMetrics,
    kind: string,
  ): string => {
    const value = finiteNumber(metrics[key]);
    if (kind === "count") return formatNumber(value, 0);
    if (kind === "money") return formatMoney(value);
    if (kind === "fraction") return formatFractionPct(value, 1);
    if (kind === "drawdown") {
      return value === null ? formatNumber(null) : formatFractionPct(-Math.abs(value), 1);
    }
    return formatNumber(value, 2);
  };

  const stressKeys = Object.keys(result.stress);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <p data-testid="ml-evaluation-caption" className="text-xs text-slate-400">
          {result.symbol} {result.timeframe} · {SPLIT_LABEL[result.split as MLSplit] ?? result.split}{" "}
          bölünmesi · eşik {formatNumber(result.threshold, 2)} /{" "}
          {formatNumber(result.exit_threshold, 2)} · veri kümesi{" "}
          <span className="font-mono">{result.dataset_id}</span>
        </p>
      </div>

      <div
        role="region"
        aria-labelledby="ml-headline-caption"
        tabIndex={0}
        className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        <table className="w-full min-w-[24rem] border-collapse text-left text-xs">
          <caption id="ml-headline-caption" className="px-3 py-2 text-left text-slate-400">
            Başlık ve tut (hold) sonucu
          </caption>
          <thead className="bg-slate-900/60 text-slate-400">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-medium">
                Metrik
              </th>
              <th scope="col" className="px-3 py-1.5 text-right font-medium">
                Model
              </th>
              <th scope="col" className="px-3 py-1.5 text-right font-medium">
                Tut
              </th>
            </tr>
          </thead>
          <tbody>
            {trading.map((row) => (
              <tr key={row.key} className="border-t border-slate-800/60">
                <th scope="row" className="px-3 py-1.5 font-normal text-slate-300">
                  {row.label}
                </th>
                <td
                  data-testid={`ml-eval-${row.key}`}
                  className="px-3 py-1.5 text-right tabular-nums text-slate-200"
                >
                  {formatTrading(result.metrics, row.key, row.kind)}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums text-slate-400">
                  {formatTrading(result.hold, row.key, row.kind)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {stressKeys.length > 0 ? (
        <div
          role="region"
          aria-labelledby="ml-stress-caption"
          tabIndex={0}
          className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <table className="w-full min-w-[30rem] border-collapse text-left text-xs">
            <caption id="ml-stress-caption" className="px-3 py-2 text-left text-slate-400">
              Maliyet stresi (başlık dışı, yalnızca bu tablo)
            </caption>
            <thead className="bg-slate-900/60 text-slate-400">
              <tr>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Çarpan
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  İşlem
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  Net
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  PF
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  Expectancy
                </th>
              </tr>
            </thead>
            <tbody>
              {stressKeys.map((key) => {
                const row = result.stress[key];
                return (
                  <tr key={key} data-testid={`ml-stress-${key}`} className="border-t border-slate-800/60">
                    <th scope="row" className="px-3 py-1.5 font-normal text-slate-300">
                      {key}
                    </th>
                    <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                      {formatNumber(finiteNumber(row.trades), 0)}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                      {formatMoney(finiteNumber(row.net_profit))}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                      {formatNumber(finiteNumber(row.profit_factor), 2)}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                      {formatNumber(finiteNumber(row.expectancy_r), 2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p data-testid="ml-stress-empty" className="text-xs text-slate-500">
          Maliyet stresi istenmedi; tablo yok.
        </p>
      )}

      {result.regime.rows.length > 0 ? (
        <div
          role="region"
          aria-labelledby="ml-regime-caption"
          tabIndex={0}
          className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          <table
            data-testid="ml-regime-table"
            className="w-full min-w-[36rem] border-collapse text-left text-xs"
          >
            <caption id="ml-regime-caption" className="px-3 py-2 text-left text-slate-400">
              Rejim kırılımı — giriş barındaki etiket; promosyon kapısı değil
            </caption>
            <thead className="bg-slate-900/60 text-slate-400">
              <tr>
                <th scope="col" className="px-3 py-1.5 font-medium">
                  Rejim
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  İşlem
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  Win rate
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  Expectancy (R)
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  Toplam R
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  PnL
                </th>
                <th scope="col" className="px-3 py-1.5 text-right font-medium">
                  PF
                </th>
              </tr>
            </thead>
            <tbody>
              {result.regime.rows.map((row: MLRegimeRow) => (
                <tr key={row.regime} className="border-t border-slate-800/60">
                  <th scope="row" className="px-3 py-1.5 font-normal text-slate-300">
                    {row.regime}
                  </th>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                    {formatNumber(finiteNumber(row.trades), 0)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                    {formatFractionPct(finiteNumber(row.win_rate), 1)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                    {formatNumber(finiteNumber(row.expectancy_r), 2)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                    {formatNumber(finiteNumber(row.total_r), 2)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                    {formatMoney(finiteNumber(row.pnl))}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                    {formatNumber(finiteNumber(row.profit_factor), 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p data-testid="ml-regime-empty" className="text-sm text-slate-400">
          Rejim tablosu boş: bu bölünmede sınıflandırılabilir kapalı işlem yok
          (veya rejim kırılımı istenmedi).
        </p>
      )}

      <p className="text-xs text-slate-500">
        Rejim satırlarına girmeyen işlemler:{" "}
        <span data-testid="ml-regime-unmatched" className="tabular-nums text-slate-300">
          {formatNumber(result.regime.unmatched_trades, 0)}
        </span>
        . Bunlar başlık işlem sayısına dahildir; sahte bir kovaya konmaz.
      </p>
    </div>
  );
}

function RunsTable({
  runs,
  selectedId,
  onSelect,
}: {
  runs: readonly MLRunSummary[];
  selectedId: string | null;
  onSelect: (runId: string) => void;
}) {
  return (
    <div
      role="region"
      aria-labelledby="ml-runs-caption"
      tabIndex={0}
      className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <table className="w-full min-w-[38rem] border-collapse text-left text-xs">
        <caption id="ml-runs-caption" className="px-3 py-2 text-left text-slate-400">
          ML koşuları — seçmek için koşu kimliğine basın
        </caption>
        <thead className="bg-slate-900/60 text-slate-400">
          <tr>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Koşu
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Durum
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Model
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Sembol / TF
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Tarih
            </th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const selected = run.run_id === selectedId;
            return (
              <tr
                key={run.run_id}
                aria-selected={selected}
                className={`border-t border-slate-800/60 ${selected ? "bg-slate-800/40" : ""}`}
              >
                <th scope="row" className="px-3 py-1.5 font-normal">
                  <button
                    type="button"
                    onClick={() => onSelect(run.run_id)}
                    aria-current={selected ? "true" : undefined}
                    className="font-mono text-slate-200 underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  >
                    {run.run_id}
                  </button>
                </th>
                <td className="px-3 py-1.5">
                  <MLStatusChip status={run.status} />
                </td>
                <td className="px-3 py-1.5 text-slate-300">{run.model ?? "—"}</td>
                <td className="px-3 py-1.5 text-slate-300">
                  {run.symbol ?? "—"} / {run.timeframe ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-slate-400">{formatTs(run.created_at)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
