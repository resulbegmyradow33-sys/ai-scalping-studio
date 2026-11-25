"use client";

/**
 * Backtesting screen: replay one completed run through `POST /api/backtest`.
 *
 * Every number on this page comes from a response. The endpoint accepts only
 * `run_id` and `symbol`, so there is no cost-stress input here; it also returns
 * metrics and a report path, not an equity curve. When a response carries no
 * equity array the page says so instead of reconstructing one from the metrics,
 * and Monte Carlo stays an explicit CLI-only note rather than a fake chart.
 *
 * The run selector is fed by `GET /api/runs` and keeps only `completed` rows:
 * the backtest route answers 409 for a live or failed run, so offering those
 * would only produce a predictable error.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

import { LineChart } from "@/components/line-chart";
import { MetricGrid } from "@/components/metric-grid";
import { RunStatusChip } from "@/components/run-table";
import { ApiError, api } from "@/lib/api";
import { completedRuns, equitySeries, hasMetrics } from "@/lib/backtest";
import { formatMoney, formatNumber, formatTs } from "@/lib/format";
import type { BacktestResponse, RunSummary } from "@/lib/types";

/** Mirrors the API's `SymbolId` constraint so the message is immediate. */
const SYMBOL_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SYMBOL_MAX = 32;

const FIELD_CLASS =
  "rounded-sm border border-slate-700 bg-slate-950/60 px-2 py-1.5 text-sm text-slate-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent aria-[invalid=true]:border-rose-500/70";

const LABEL_CLASS = "text-xs text-slate-400";
const HINT_CLASS = "text-xs text-slate-400";
const ERROR_CLASS = "text-xs text-rose-300";

const BUTTON_BASE =
  "rounded-sm border px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";

const PRIMARY_BUTTON = `${BUTTON_BASE} border-accent bg-accent/15 font-medium text-slate-100 hover:bg-accent/25 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;
const SECONDARY_BUTTON = `${BUTTON_BASE} border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "İstek başarısız oldu.";
}

/** Map the route's documented statuses to a sentence that keeps the code. */
function backtestMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 404) return `Koşu bulunamadı (404): ${error.detail}`;
    if (error.status === 409) return `Koşu yeniden oynatılamıyor (409): ${error.detail}`;
    if (error.status === 422) return `İstek reddedildi (422): ${error.detail}`;
    if (error.status === 0) return `Sunucuya ulaşılamadı: ${error.detail}`;
    return `Backtest başarısız (HTTP ${error.status}): ${error.detail}`;
  }
  return errorDetail(error);
}

function symbolProblem(raw: string): string | null {
  const value = raw.trim();
  if (value === "") return "Sembol girin.";
  if (value.length > SYMBOL_MAX) return `Sembol en fazla ${SYMBOL_MAX} karakter olabilir.`;
  if (!SYMBOL_RE.test(value)) {
    return "Sembol harf veya rakamla başlamalı; yalnızca harf, rakam, nokta, alt çizgi ve tire içerebilir.";
  }
  return null;
}

/** The response's optional equity array, or a truthful reason it is absent. */
function EquityBlock({ value }: { value: unknown }) {
  const points = equitySeries(value);

  if (points.length < 2) {
    const message = Array.isArray(value)
      ? "API bir equity dizisi döndürdü ama çizilecek en az iki sonlu nokta yok. Eğri uydurulmadı."
      : "API bu yanıtta equity eğrisi döndürmüyor. Eğri uydurulmadı; yalnızca metrikler gösteriliyor.";
    return (
      <section aria-labelledby="backtest-equity-heading" className="flex flex-col gap-1">
        <h3 id="backtest-equity-heading" className="text-xs font-semibold text-slate-100">
          Equity eğrisi
        </h3>
        <p
          data-testid="backtest-equity-absent"
          role="status"
          className="rounded-sm border border-slate-800 bg-slate-900/40 px-3 py-2 text-xs text-slate-400"
        >
          {message}
        </p>
      </section>
    );
  }

  const first = points[0].value;
  const last = points[points.length - 1].value;
  let low = first;
  let high = first;
  for (const point of points) {
    if (point.value < low) low = point.value;
    if (point.value > high) high = point.value;
  }

  const summary = `Equity eğrisi: ${formatNumber(points.length, 0)} işaret; ilk ${formatMoney(
    first,
  )}, son ${formatMoney(last)}, en düşük ${formatMoney(low)}, en yüksek ${formatMoney(high)}.`;

  return (
    <figure data-testid="backtest-equity" className="flex flex-col gap-1">
      <figcaption id="backtest-equity-caption" className="text-xs text-slate-400">
        Equity eğrisi — {formatNumber(points.length, 0)} işaret
      </figcaption>
      <div
        role="region"
        aria-labelledby="backtest-equity-caption"
        tabIndex={0}
        className="rounded-sm border border-slate-800 bg-slate-900/40 p-3 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        <LineChart title={summary} points={points} color="#38bdf8" />
        <p className="mt-1 text-xs text-slate-400">{summary}</p>
      </div>
    </figure>
  );
}

export default function BacktestingPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsMissing, setRunsMissing] = useState(false);

  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [symbol, setSymbol] = useState("XAUUSD");
  const [submitted, setSubmitted] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const [running, setRunning] = useState(false);
  const [backtestError, setBacktestError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [executed, setExecuted] = useState<{ runId: string; symbol: string } | null>(null);

  const formRef = useRef<HTMLFormElement>(null);
  const symbolError = symbolProblem(symbol);
  const available = completedRuns(runs);

  const refreshRuns = useCallback(async (): Promise<void> => {
    setRunsLoading(true);
    try {
      const rows = await api.getRuns();
      setRuns(rows);
      setRunsError(null);
      setRunsMissing(false);
      // Keep a still-valid selection; otherwise take the newest completed run.
      setSelectedRunId((current) => {
        const completed = completedRuns(rows);
        if (current !== null && completed.some((run) => run.run_id === current)) {
          return current;
        }
        return completed[0]?.run_id ?? null;
      });
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setRunsMissing(true);
        setRunsError(null);
      } else {
        setRunsError(errorDetail(error));
      }
      setRuns([]);
      setSelectedRunId(null);
    } finally {
      setRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (running) return;

    setSubmitted(true);
    const problem = symbolProblem(symbol);
    if (problem !== null) {
      formRef.current?.querySelector<HTMLElement>("#backtest-symbol")?.focus();
      return;
    }
    if (selectedRunId === null) {
      setFormError("Önce tamamlanmış bir koşu seçin.");
      return;
    }

    const body = { run_id: selectedRunId, symbol: symbol.trim() };
    setFormError(null);
    setBacktestError(null);
    setResult(null);
    setExecuted(null);
    setRunning(true);

    void api
      .backtest(body)
      .then((response) => {
        setResult(response);
        setExecuted({ runId: body.run_id, symbol: body.symbol });
      })
      .catch((error: unknown) => {
        setBacktestError(backtestMessage(error));
      })
      .finally(() => {
        setRunning(false);
      });
  };

  const trades = result !== null && hasMetrics(result.metrics) ? result.metrics.trades : null;
  const netProfit =
    result !== null && hasMetrics(result.metrics) ? result.metrics.net_profit : null;
  const announcement =
    backtestError !== null
      ? `Backtest hatası: ${backtestError}`
      : result !== null
        ? `Backtest tamamlandı. ${
            trades === null ? "Metrik nesnesi yok." : `${formatNumber(trades, 0)} işlem.`
          } Net kâr ${formatMoney(netProfit)}.`
        : "";

  const canRun = selectedRunId !== null && !running;
  const runExplanation =
    selectedRunId === null
      ? runsLoading
        ? "Koşu listesi yükleniyor…"
        : "Önce tamamlanmış bir koşu seçin."
      : running
        ? "Koşu simülatörde yeniden oynatılıyor…"
        : symbolError !== null
          ? "Sembol biçimi geçersiz; ayrıntı alanın altında."
          : "İstek yalnızca koşu kimliği ve sembol içerir.";

  return (
    <section aria-labelledby="backtesting-heading" className="flex flex-col gap-5">
      <div>
        <h1 id="backtesting-heading" className="text-xl font-semibold text-slate-100">
          Backtesting
        </h1>
        <p className="text-xs text-slate-400">
          Tamamlanmış bir koşuyu saklanan barlar üzerinde yeniden oynatır — gerçek
          para değil, uydurulmuş metrik yok.
        </p>
      </div>

      <p data-testid="backtest-live" className="sr-only" role="status">
        {announcement}
      </p>

      <form
        ref={formRef}
        data-testid="backtest-form"
        aria-labelledby="backtest-form-heading"
        noValidate
        onSubmit={handleSubmit}
        aria-busy={running}
        className="flex flex-col gap-4 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="backtest-form-heading" className="text-sm font-semibold text-slate-100">
            Koşu seç ve çalıştır
          </h2>
          <span className="text-xs text-slate-400">
            {runsLoading
              ? "Koşular yükleniyor…"
              : `${formatNumber(available.length, 0)} tamamlanmış koşu`}
          </span>
        </div>

        <fieldset className="flex flex-col gap-3">
          <legend className="mb-1 text-xs font-medium text-slate-300">Girdiler</legend>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="flex flex-col gap-1">
              <label htmlFor="backtest-run" className={LABEL_CLASS}>
                Koşu
              </label>
              <select
                id="backtest-run"
                required
                value={selectedRunId ?? ""}
                disabled={running || runsLoading || available.length === 0}
                aria-describedby="backtest-run-hint"
                onChange={(event) => {
                  setSelectedRunId(event.target.value === "" ? null : event.target.value);
                  setFormError(null);
                }}
                className={FIELD_CLASS}
              >
                <option value="">
                  {available.length === 0 ? "Tamamlanmış koşu yok" : "Koşu seçin"}
                </option>
                {available.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id} — {formatTs(run.created_at)}
                  </option>
                ))}
              </select>
              <p id="backtest-run-hint" className={HINT_CLASS}>
                Yalnızca <code className="font-mono">completed</code> durumundaki koşular
                listelenir; canlı veya başarısız bir koşu 409 döner.
              </p>
            </div>

            <div className="flex flex-col gap-1">
              <label htmlFor="backtest-symbol" className={LABEL_CLASS}>
                Sembol
              </label>
              <input
                id="backtest-symbol"
                type="text"
                required
                value={symbol}
                disabled={running}
                autoComplete="off"
                spellCheck={false}
                aria-invalid={submitted && symbolError !== null}
                aria-describedby={
                  submitted && symbolError !== null
                    ? "backtest-symbol-hint backtest-symbol-error"
                    : "backtest-symbol-hint"
                }
                onChange={(event) => {
                  setSymbol(event.target.value);
                  setFormError(null);
                }}
                className={FIELD_CLASS}
              />
              <p id="backtest-symbol-hint" className={HINT_CLASS}>
                Depodaki barlar bu sembolden okunur; koşunun eğitildiği veriyle
                eşleşmezse sunucu 409 döner.
              </p>
              {submitted && symbolError !== null && (
                <p id="backtest-symbol-error" className={ERROR_CLASS}>
                  {symbolError}
                </p>
              )}
            </div>
          </div>
        </fieldset>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={!canRun}
            aria-describedby="backtest-run-explanation"
            className={PRIMARY_BUTTON}
          >
            Backtest çalıştır
          </button>
          <span id="backtest-run-explanation" className="text-xs text-slate-400">
            {runExplanation}
          </span>
        </div>

        {formError !== null && (
          <p data-testid="backtest-form-error" role="alert" className="text-sm text-rose-300">
            {formError}
          </p>
        )}

        {backtestError !== null && (
          <p data-testid="backtest-error" role="alert" className="text-sm text-rose-300">
            {backtestError}
          </p>
        )}
      </form>

      <section aria-labelledby="backtest-runs-heading" className="flex flex-col gap-2">
        <h2 id="backtest-runs-heading" className="text-sm font-semibold text-slate-100">
          Koşular
        </h2>

        {runsMissing && (
          <p data-testid="backtest-runs-missing" role="status" className="text-sm text-amber-200">
            Sunucuda koşu listesi yok (404). Bu uç nokta olmadan backtest için koşu
            seçilemez; sayfa metrik uydurmaz.
          </p>
        )}

        {runsError !== null && (
          <p data-testid="backtest-runs-error" role="alert" className="text-sm text-rose-300">
            Koşu listesi alınamadı: {runsError}
          </p>
        )}

        {runsLoading && runs.length === 0 && !runsMissing && runsError === null ? (
          <p className="text-sm text-slate-400">Yükleniyor…</p>
        ) : runs.length === 0 && !runsMissing && runsError === null ? (
          <p data-testid="backtest-runs-empty" role="status" className="text-sm text-slate-400">
            Henüz koşu yok. Eğitim ekranından bir koşu tamamlandığında burada
            listelenir.
          </p>
        ) : available.length === 0 && runs.length > 0 ? (
          <p data-testid="backtest-runs-none-completed" role="status" className="text-sm text-amber-200">
            {formatNumber(runs.length, 0)} koşu var ama hiçbiri{" "}
            <code className="font-mono">completed</code> değil. Backtest yalnızca
            tamamlanmış bir koşuyu yeniden oynatabilir.
          </p>
        ) : available.length > 0 ? (
          <div
            role="region"
            aria-labelledby="backtest-runs-caption"
            tabIndex={0}
            className="overflow-x-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <table className="w-full min-w-[36rem] border-collapse text-left text-xs">
              <caption id="backtest-runs-caption" className="sr-only">
                Backtest için kullanılabilir tamamlanmış koşular
              </caption>
              <thead className="bg-slate-900/60 text-slate-400">
                <tr>
                  <th scope="col" className="px-2 py-1.5 font-medium">
                    Koşu
                  </th>
                  <th scope="col" className="px-2 py-1.5 font-medium">
                    Durum
                  </th>
                  <th scope="col" className="px-2 py-1.5 text-right font-medium">
                    Adım
                  </th>
                  <th scope="col" className="px-2 py-1.5 font-medium">
                    Oluşturuldu
                  </th>
                  <th scope="col" className="px-2 py-1.5 text-right font-medium">
                    Seç
                  </th>
                </tr>
              </thead>
              <tbody>
                {available.map((run) => {
                  const selected = run.run_id === selectedRunId;
                  return (
                    <tr
                      key={run.run_id}
                      className={selected ? "bg-slate-800/70" : "hover:bg-slate-900/60"}
                    >
                      <th scope="row" className="px-2 py-1.5 font-normal">
                        <span className="font-mono text-slate-100">{run.run_id}</span>
                      </th>
                      <td className="px-2 py-1.5">
                        <RunStatusChip status={run.status} />
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">
                        {formatNumber(run.steps, 0)}
                      </td>
                      <td className="px-2 py-1.5 tabular-nums text-slate-400">
                        {formatTs(run.created_at)}
                      </td>
                      <td className="px-2 py-1.5 text-right">
                        <button
                          type="button"
                          aria-pressed={selected}
                          aria-label={`Seç: ${run.run_id}`}
                          disabled={running}
                          onClick={() => {
                            setSelectedRunId(run.run_id);
                            setFormError(null);
                          }}
                          className={SECONDARY_BUTTON}
                        >
                          {selected ? "Seçili" : "Seç"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      {result !== null && executed !== null && (
        <section
          data-testid="backtest-result"
          aria-labelledby="backtest-result-heading"
          className="flex flex-col gap-4 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 id="backtest-result-heading" className="text-sm font-semibold text-slate-100">
              Sonuç
            </h2>
            <span className="truncate font-mono text-xs text-slate-300">
              {executed.runId} · {executed.symbol}
            </span>
          </div>

          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
            <div className="flex min-w-0 gap-2">
              <dt className="shrink-0 text-slate-400">Rapor</dt>
              <dd className="truncate font-mono text-slate-300" title={result.report_path}>
                {typeof result.report_path === "string" && result.report_path !== ""
                  ? result.report_path
                  : "—"}
              </dd>
            </div>
            <div className="flex gap-2">
              <dt className="text-slate-400">Yanıt</dt>
              <dd className="text-slate-300">HTTP 200 — rapor yazıldı</dd>
            </div>
          </dl>

          {hasMetrics(result.metrics) ? (
            <section aria-labelledby="backtest-metrics-heading">
              <h3 id="backtest-metrics-heading" className="mb-2 text-xs font-semibold text-slate-100">
                Metrikler
              </h3>
              <MetricGrid metrics={result.metrics} />
            </section>
          ) : (
            <p data-testid="backtest-metrics-absent" role="status" className="text-sm text-amber-200">
              API bir metrik nesnesi döndürmedi. Hiçbir metrik gösterilmiyor.
            </p>
          )}

          <EquityBlock value={result.equity} />
        </section>
      )}

      <section
        aria-labelledby="backtest-scope-heading"
        className="rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
      >
        <h2 id="backtest-scope-heading" className="text-sm font-semibold text-slate-100">
          Bu sayfada olmayan analizler
        </h2>
        <ul className="mt-2 list-inside list-disc text-xs text-slate-400">
          <li>
            Monte Carlo işlem sırası analizi yalnızca CLI’da var
            (<code className="font-mono">scalping.cli backtest montecarlo</code>);
            HTTP ucu yok. Bu sayfa onun yerine grafik veya dağılım göstermez.
          </li>
          <li>
            Walk-forward ve maliyet stresi ayrı bir uçta
            (<code className="font-mono">POST /api/evaluate</code>) bulunur; bu sayfa
            yalnızca <code className="font-mono">/api/backtest</code> çağırır ve
            maliyet çarpanı göndermez.
          </li>
          <li>
            <code className="font-mono">/api/backtest</code> gövdesi yalnızca{" "}
            <code className="font-mono">run_id</code> ve{" "}
            <code className="font-mono">symbol</code> kabul eder. Başka bir alan
            gönderilmez.
          </li>
          <li>
            Equity eğrisi yanıtta yoksa çizilmez; yöntem ve sınırlar için{" "}
            <code className="font-mono">docs/backtesting-methodology.md</code>.
          </li>
        </ul>
      </section>
    </section>
  );
}
