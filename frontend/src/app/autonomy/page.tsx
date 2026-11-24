"use client";

/**
 * Autonomy: the read-only `/api/autonomy/*` view of the learning loop.
 *
 * Every value is a response field. A store that is absent (`configured: false`)
 * or present but unreadable (`readable: false`) renders an explicit sentence;
 * the page never shows a zero it did not read. Jobs and events are bounded
 * windows: the API exposes only the newest N rows and no offset, so the page
 * offers a window size instead of inventing older pages.
 *
 * Nothing here writes. No control enqueues, cancels, pauses, promotes or starts
 * a daemon, and no champion can be selected from this screen.
 */

import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";

import { StatCard } from "@/components/stat-card";
import { ApiError, api } from "@/lib/api";
import { formatNumber, formatTs } from "@/lib/format";
import type {
  AutonomyChampion,
  AutonomyDataStatus,
  AutonomyDiscoveries,
  AutonomyEventItem,
  AutonomyEventPage,
  AutonomyJobItem,
  AutonomyJobPage,
  AutonomyOrchestratorStatus,
  AutonomyPaperStatus,
  AutonomyStatus,
} from "@/lib/types";

/** The API caps jobs/events at 500; a dashboard only needs a recent glance. */
const WINDOW_SIZES = [25, 50, 100] as const;
const DEFAULT_WINDOW = 25;

const BUTTON_BASE =
  "rounded-sm border px-3 py-1.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent";
const SECONDARY_BUTTON = `${BUTTON_BASE} border-slate-700 text-slate-300 hover:border-slate-600 hover:text-slate-100 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-600`;

const JOB_STATUS_LABEL: Record<string, string> = {
  queued: "Sırada",
  running: "Çalışıyor",
  succeeded: "Tamamlandı",
  failed: "Hata",
  cancelled: "İptal",
  skipped: "Atlandı",
};

const JOB_STATUS_TONE: Record<string, string> = {
  queued: "border-slate-700 bg-slate-800/40 text-slate-300",
  running: "border-accent/50 bg-accent/10 text-accent",
  succeeded: "border-profit/50 bg-profit/10 text-profit",
  failed: "border-loss/50 bg-loss/10 text-rose-300",
  cancelled: "border-slate-700 bg-slate-800/40 text-slate-400",
  skipped: "border-slate-700 bg-slate-800/40 text-slate-400",
};

// --- helpers -----------------------------------------------------------------

function errorDetail(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return "İstek başarısız oldu.";
}

/**
 * `null` when the store answered, otherwise the sentence that replaces every
 * value in that section. Missing and unreadable are deliberately different.
 */
function storeState(
  configured: boolean,
  readable: boolean,
  what: string,
): string | null {
  if (!configured) {
    return `${what} yapılandırılmadı: store mevcut değil, hiçbir değer okunmadı.`;
  }
  if (!readable) {
    return `${what} okunamıyor: dosya var ama şeması bu görünüm için yeterli değil; sayı gösterilmiyor.`;
  }
  return null;
}

/** A count or a dash. An absent field is never turned into `0`. */
function count(value: number | null | undefined): string {
  return formatNumber(typeof value === "number" && Number.isFinite(value) ? value : null, 0);
}

/** A persisted control that can genuinely be unknown, not "off". */
function flag(
  value: boolean | null | undefined,
  on: string,
  off: string,
  unknown: string,
): string {
  if (value === true) return on;
  if (value === false) return off;
  return unknown;
}

// --- champion mode -----------------------------------------------------------

type ModeKind = "evidence" | "exploratory" | "manual" | "unknown";

const MODE_STYLE: Record<ModeKind, string> = {
  evidence: "border-profit/50 bg-profit/10 text-profit",
  exploratory: "border-amber-500/50 bg-amber-500/10 text-amber-200",
  manual: "border-slate-600 bg-slate-800/50 text-slate-200",
  unknown: "border-loss/40 bg-loss/10 text-rose-300",
};

function modeKind(mode: string | null | undefined): ModeKind {
  if (mode === "evidence") return "evidence";
  if (mode === "exploratory") return "exploratory";
  if (mode === "manual") return "manual";
  return "unknown";
}

function modeText(mode: string | null | undefined): string {
  const kind = modeKind(mode);
  if (kind === "evidence") return "kanıt · evidence";
  if (kind === "exploratory") return "keşif · exploratory";
  if (kind === "manual") return "manuel · manual";
  return typeof mode === "string" && mode !== "" ? `${mode} · kanıtsız` : "mod yok · kanıtsız";
}

function ModeBadge({
  mode,
  scope,
}: {
  mode: string | null | undefined;
  scope: string;
}) {
  const kind = modeKind(mode);
  return (
    <span
      data-testid={`autonomy-champion-mode-${scope}`}
      data-mode={kind}
      className={`inline-block rounded-sm border px-2 py-0.5 text-xs ${MODE_STYLE[kind]}`}
    >
      {modeText(mode)}
    </span>
  );
}

// --- small building blocks ---------------------------------------------------

function JobStatusChip({ status, jobId }: { status: string; jobId: number }) {
  const key = status.toLowerCase();
  return (
    <span
      data-testid={`autonomy-job-status-${jobId}`}
      data-status={key}
      className={`inline-block rounded-sm border px-2 py-0.5 text-xs ${
        JOB_STATUS_TONE[key] ?? "border-slate-700 text-slate-300"
      }`}
    >
      {JOB_STATUS_LABEL[key] ?? status}
    </span>
  );
}

/** A section-level sentence that replaces values rather than guessing them. */
function StoreNotice({ state, testId }: { state: string; testId: string }) {
  return (
    <p data-testid={testId} className="text-sm text-amber-200">
      {state}
    </p>
  );
}

function ErrorNotice({
  children,
  onRetry,
  retryLabel,
  testId,
}: {
  children: ReactNode;
  onRetry: () => void;
  retryLabel: string;
  testId: string;
}) {
  return (
    <div
      data-testid={testId}
      role="alert"
      className="flex flex-wrap items-center gap-3 rounded-sm border border-loss/40 bg-loss/10 px-3 py-2 text-xs text-rose-200"
    >
      <p>{children}</p>
      <button type="button" onClick={onRetry} className={SECONDARY_BUTTON}>
        {retryLabel}
      </button>
    </div>
  );
}

/**
 * A bounded table: focusable scroll region named by its caption, sticky header
 * and row-header cells, and a window control that only ever asks for the newest
 * N rows. There is no offset in the API, so there is no "previous page" button.
 */
function WindowTable({
  id,
  caption,
  regionLabel,
  limit,
  onLimit,
  limitLabel,
  limitNote,
  children,
}: {
  id: string;
  caption: string;
  regionLabel: string;
  limit: number;
  onLimit: (limit: number) => void;
  limitLabel: string;
  limitNote: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div
        role="region"
        aria-labelledby={`${id}-caption`}
        tabIndex={0}
        className="max-h-80 overflow-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent [&_tbody_th]:sticky [&_tbody_th]:left-0 [&_tbody_th]:bg-slate-900/95 [&_thead_th]:sticky [&_thead_th]:top-0 [&_thead_th]:z-10 [&_thead_th]:bg-slate-900"
      >
        <table className="w-full min-w-[52rem] border-collapse text-left text-xs">
          <caption id={`${id}-caption`} className="px-3 py-2 text-left text-slate-400">
            {caption}
          </caption>
          {children}
        </table>
      </div>

      <div
        role="group"
        aria-label={regionLabel}
        className="flex flex-wrap items-center justify-end gap-2"
      >
        <label htmlFor={`${id}-limit`} className="text-xs text-slate-400">
          {limitLabel}
        </label>
        <select
          id={`${id}-limit`}
          data-testid={`${id}-limit`}
          value={limit}
          onChange={(event) => onLimit(Number(event.target.value))}
          aria-describedby={`${id}-limit-note`}
          className="rounded-sm border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          {WINDOW_SIZES.map((size) => (
            <option key={size} value={size}>
              {formatNumber(size, 0)} kayıt
            </option>
          ))}
        </select>
      </div>
      <p id={`${id}-limit-note`} className="text-right text-xs text-slate-400">
        {limitNote}
      </p>
    </div>
  );
}

// --- page --------------------------------------------------------------------

export default function AutonomyPage() {
  const [status, setStatus] = useState<AutonomyStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [statusToken, setStatusToken] = useState(0);
  const [lastReadAt, setLastReadAt] = useState<string | null>(null);

  const [jobs, setJobs] = useState<AutonomyJobPage | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [jobsToken, setJobsToken] = useState(0);
  const [jobLimit, setJobLimit] = useState<number>(DEFAULT_WINDOW);

  const [events, setEvents] = useState<AutonomyEventPage | null>(null);
  const [eventsError, setEventsError] = useState<string | null>(null);
  const [eventsLoading, setEventsLoading] = useState(true);
  const [eventsToken, setEventsToken] = useState(0);
  const [eventLimit, setEventLimit] = useState<number>(DEFAULT_WINDOW);

  // Derived before the callback so the inflight guard reads the same value the
  // button exposes through `aria-disabled`.
  const refreshing = statusLoading || jobsLoading || eventsLoading;

  const refreshAll = useCallback(() => {
    if (refreshing) return;
    setStatusToken((token) => token + 1);
    setJobsToken((token) => token + 1);
    setEventsToken((token) => token + 1);
  }, [refreshing]);

  // -- aggregate status -----------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setStatusLoading(true);

    void api
      .autonomyStatus()
      .then((body) => {
        if (cancelled) return;
        setStatus(body);
        setStatusError(null);
        setLastReadAt(new Date().toISOString());
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        // Keep the last successful snapshot; the alert marks it as stale.
        setStatusError(errorDetail(error));
      })
      .finally(() => {
        if (!cancelled) setStatusLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [statusToken]);

  // -- bounded jobs window --------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setJobsLoading(true);

    void api
      .autonomyJobs(jobLimit)
      .then((page) => {
        if (cancelled) return;
        setJobs(page);
        setJobsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setJobsError(errorDetail(error));
      })
      .finally(() => {
        if (!cancelled) setJobsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [jobLimit, jobsToken]);

  // -- bounded events window ------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    setEventsLoading(true);

    void api
      .autonomyEvents(eventLimit)
      .then((page) => {
        if (cancelled) return;
        setEvents(page);
        setEventsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setEventsError(errorDetail(error));
      })
      .finally(() => {
        if (!cancelled) setEventsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [eventLimit, eventsToken]);

  const liveMessage =
    lastReadAt === null ? "" : `Otonomi durumu güncellendi: ${formatTs(lastReadAt)}`;

  const paper = status?.paper;
  const orchestrator = status?.orchestrator;
  const discoveries = status?.discoveries;
  const data = status?.data;
  const champions = status?.champions ?? [];
  const championsReadable = status?.champions_readable ?? false;

  return (
    <section aria-labelledby="autonomy-heading" aria-busy={refreshing} className="flex flex-col gap-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 id="autonomy-heading" className="text-xl font-semibold text-slate-100">
            Otonomi
          </h1>
          <p className="text-xs text-slate-400">
            Öğrenme döngüsünün salt-okunur durumu — paper defteri, orkestratör kuyruğu,
            şampiyonlar, keşif arşivi ve veri.
          </p>
        </div>

        <div className="flex flex-col items-end gap-1">
          <button
            type="button"
            data-testid="autonomy-refresh"
            onClick={refreshAll}
            aria-disabled={refreshing}
            aria-describedby="autonomy-refresh-note"
            className={`${SECONDARY_BUTTON} aria-disabled:cursor-not-allowed aria-disabled:opacity-60`}
          >
            {refreshing ? "Yenileniyor…" : "Yenile"}
          </button>
          <p id="autonomy-refresh-note" className="text-right text-xs text-slate-400">
            {lastReadAt === null
              ? "Henüz başarılı okuma yok."
              : `Son okuma: ${formatTs(lastReadAt)}`}
          </p>
        </div>
      </div>

      <p
        role="note"
        data-testid="autonomy-paper-notice"
        className="rounded-sm border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200"
      >
        <strong>Yalnızca paper: gerçek para yok.</strong> Bu sayfa `/api/autonomy/*` uçlarını
        okur; hiçbir düğme iş kuyruğa almaz, duraklatmaz, iptal etmez veya şampiyon terfi
        ettirmez. Gerçek para yönlendirmesi kapsam dışıdır ve kapalı kalır.
      </p>

      <p className="sr-only" role="status" data-testid="autonomy-status-live">
        {liveMessage}
      </p>

      {statusError !== null && (
        <ErrorNotice
          testId="autonomy-status-error"
          onRetry={() => setStatusToken((token) => token + 1)}
          retryLabel="Durumu yenile"
        >
          Durum okunamadı: {statusError}
          {status !== null && " — aşağıdaki değerler son başarılı okumaya ait olabilir."}
        </ErrorNotice>
      )}

      {status === null && statusLoading && (
        <p className="text-sm text-slate-400">Durum okunuyor…</p>
      )}

      {status !== null && (
        <>
          {/* -- paper ------------------------------------------------------ */}
          <section
            aria-labelledby="autonomy-paper-heading"
            className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
          >
            <h2 id="autonomy-paper-heading" className="text-sm font-semibold text-slate-100">
              Paper defteri
            </h2>
            <PaperSection paper={paper} />
          </section>

          {/* -- orchestrator ---------------------------------------------- */}
          <section
            aria-labelledby="autonomy-orchestrator-heading"
            className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
          >
            <h2
              id="autonomy-orchestrator-heading"
              className="text-sm font-semibold text-slate-100"
            >
              Orkestratör
            </h2>
            <OrchestratorSection orchestrator={orchestrator} />
          </section>

          {/* -- champions ------------------------------------------------- */}
          <section
            aria-labelledby="autonomy-champions-heading"
            className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
          >
            <h2 id="autonomy-champions-heading" className="text-sm font-semibold text-slate-100">
              Şampiyonlar
            </h2>
            <p role="note" className="text-xs text-amber-200">
              Yalnızca <code>evidence</code> modundaki şampiyon paper işlem yapabilir.
              <code>exploratory</code> ve <code>manual</code> kayıtlar paper’a alınmaz; bu
              sayfa hiçbir şampiyonu seçmez veya değiştirmez.
            </p>
            <ChampionsSection
              champions={champions}
              readable={championsReadable}
              configured={discoveries?.configured ?? false}
            />
          </section>

          {/* -- discovery and data ---------------------------------------- */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <section
              aria-labelledby="autonomy-discoveries-heading"
              className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
            >
              <h2
                id="autonomy-discoveries-heading"
                className="text-sm font-semibold text-slate-100"
              >
                Keşif arşivi
              </h2>
              <DiscoveriesSection discoveries={discoveries} />
            </section>

            <section
              aria-labelledby="autonomy-data-heading"
              className="flex flex-col gap-2 rounded-md border border-slate-800 bg-slate-900/40 px-4 py-3"
            >
              <h2 id="autonomy-data-heading" className="text-sm font-semibold text-slate-100">
                Veri
              </h2>
              <DataSection data={data} />
            </section>
          </div>
        </>
      )}

      {/* -- jobs ---------------------------------------------------------- */}
      <section aria-labelledby="autonomy-jobs-heading" className="flex flex-col gap-2">
        <h2 id="autonomy-jobs-heading" className="text-sm font-semibold text-slate-100">
          İşler
        </h2>
        <JobsSection
          page={jobs}
          loading={jobsLoading}
          error={jobsError}
          limit={jobLimit}
          onLimit={setJobLimit}
          onRetry={() => setJobsToken((token) => token + 1)}
        />
      </section>

      {/* -- events -------------------------------------------------------- */}
      <section aria-labelledby="autonomy-events-heading" className="flex flex-col gap-2">
        <h2 id="autonomy-events-heading" className="text-sm font-semibold text-slate-100">
          Denetim olayları
        </h2>
        <EventsSection
          page={events}
          loading={eventsLoading}
          error={eventsError}
          limit={eventLimit}
          onLimit={setEventLimit}
          onRetry={() => setEventsToken((token) => token + 1)}
        />
      </section>
    </section>
  );
}

// --- sections ----------------------------------------------------------------

function PaperSection({ paper }: { paper: AutonomyPaperStatus | undefined }) {
  if (paper === undefined) {
    return <p className="text-sm text-slate-400">Paper bölümü okunuyor…</p>;
  }

  const state = storeState(paper.configured, paper.readable, "Paper defteri");
  if (state !== null) {
    return <StoreNotice state={state} testId="autonomy-paper-state" />;
  }

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
      <StatCard
        label="Oturumlar"
        value={count(paper.sessions)}
        testId="autonomy-paper-sessions"
      />
      <StatCard
        label="Açık pozisyonlar"
        value={count(paper.open_positions)}
        testId="autonomy-paper-open-positions"
      />
      <StatCard
        label="Kapanmış işlemler"
        value={count(paper.closed_trades)}
        testId="autonomy-paper-closed-trades"
      />
    </div>
  );
}

function OrchestratorSection({
  orchestrator,
}: {
  orchestrator: AutonomyOrchestratorStatus | undefined;
}) {
  if (orchestrator === undefined) {
    return <p className="text-sm text-slate-400">Orkestratör bölümü okunuyor…</p>;
  }

  const state = storeState(orchestrator.configured, orchestrator.readable, "Orkestratör kuyruğu");
  if (state !== null) {
    return <StoreNotice state={state} testId="autonomy-orchestrator-state" />;
  }

  const lastEvent = orchestrator.last_event ?? null;

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard
          label="Sırada"
          value={count(orchestrator.queued)}
          testId="autonomy-orchestrator-queued"
        />
        <StatCard
          label="Çalışıyor"
          value={count(orchestrator.running)}
          testId="autonomy-orchestrator-running"
        />
        <StatCard
          label="Hata"
          value={
            typeof orchestrator.failed === "number" && orchestrator.failed > 0 ? (
              <span className="text-rose-300">{count(orchestrator.failed)}</span>
            ) : (
              count(orchestrator.failed)
            )
          }
          testId="autonomy-orchestrator-failed"
        />
        <StatCard
          label="Duraklatıldı"
          value={flag(orchestrator.paused, "Evet", "Hayır", "bilinmiyor")}
          testId="autonomy-orchestrator-paused"
        >
          {(orchestrator.paused === null || orchestrator.paused === undefined) && (
            <p className="mt-0.5 text-xs text-slate-400">state tablosu yok</p>
          )}
        </StatCard>
        <StatCard
          label="Güvenli mod"
          value={flag(orchestrator.safe_mode, "Etkin", "Kapalı", "bilinmiyor")}
          testId="autonomy-orchestrator-safe-mode"
        >
          {(orchestrator.safe_mode === null || orchestrator.safe_mode === undefined) && (
            <p className="mt-0.5 text-xs text-slate-400">state tablosu yok</p>
          )}
        </StatCard>
        <StatCard
          label="Araştırma"
          value={flag(orchestrator.research_disabled, "Kapalı", "Açık", "bilinmiyor")}
          testId="autonomy-orchestrator-research-disabled"
        >
          {(orchestrator.research_disabled === null ||
            orchestrator.research_disabled === undefined) && (
            <p className="mt-0.5 text-xs text-slate-400">state tablosu yok</p>
          )}
        </StatCard>
      </div>

      <p className="text-xs text-slate-400" data-testid="autonomy-orchestrator-last-event">
        {lastEvent === null
          ? "Henüz denetim olayı yok."
          : `Son olay: ${lastEvent.event} · ${formatTs(lastEvent.ts)}${
              lastEvent.job_id === null || lastEvent.job_id === undefined
                ? " · kontrol"
                : ` · iş #${lastEvent.job_id}`
            }`}
      </p>
    </div>
  );
}

function ChampionsSection({
  champions,
  readable,
  configured,
}: {
  champions: AutonomyChampion[];
  readable: boolean;
  /** Same archive as the discovery section; tells "absent" from "unreadable". */
  configured: boolean;
}) {
  if (!configured) {
    return (
      <StoreNotice
        state="Şampiyon arşivi yapılandırılmadı: keşif arşivi yok, hiçbir şampiyon okunmadı."
        testId="autonomy-champions-state"
      />
    );
  }

  if (!readable) {
    return (
      <StoreNotice
        state="Şampiyon tablosu okunamıyor: arşiv var ama şeması bu görünüm için yeterli değil; şampiyon listesi gösterilmiyor."
        testId="autonomy-champions-state"
      />
    );
  }

  if (champions.length === 0) {
    return (
      <p data-testid="autonomy-champions-empty" className="text-sm text-slate-400">
        Kayıtlı şampiyon yok. NO TRADE geçerli bir sonuçtur; sayfa şampiyon uydurmaz.
      </p>
    );
  }

  return (
    <div
      role="region"
      aria-labelledby="autonomy-champions-caption"
      tabIndex={0}
      className="overflow-auto rounded-md border border-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <table className="w-full min-w-[44rem] border-collapse text-left text-xs">
        <caption id="autonomy-champions-caption" className="px-3 py-2 text-left text-slate-400">
          Kapsam başına şampiyon — {formatNumber(champions.length, 0)} kayıt
        </caption>
        <thead className="text-slate-400">
          <tr>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Kapsam
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Aday
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Mod
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Politika
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Politika özeti
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Terfi
            </th>
          </tr>
        </thead>
        <tbody>
          {champions.map((champion) => (
            <tr
              key={champion.scope}
              data-testid={`autonomy-champion-row-${champion.scope}`}
              className="border-t border-slate-800/60 align-top"
            >
              <th scope="row" className="px-3 py-1.5 font-medium text-slate-200">
                {champion.scope}
              </th>
              <td
                data-testid={`autonomy-champion-candidate-${champion.scope}`}
                className="break-all px-3 py-1.5 font-mono text-slate-300"
              >
                {champion.candidate_id}
              </td>
              <td className="px-3 py-1.5">
                <ModeBadge mode={champion.mode} scope={champion.scope} />
                {modeKind(champion.mode) !== "evidence" && (
                  <span className="ml-1 text-xs text-amber-300">paper dışı</span>
                )}
              </td>
              <td className="break-all px-3 py-1.5 font-mono text-slate-300">
                {champion.policy_id ?? "—"}
              </td>
              <td className="break-all px-3 py-1.5 font-mono text-slate-400">
                {champion.policy_digest ?? "—"}
              </td>
              <td className="px-3 py-1.5 tabular-nums text-slate-300">
                {formatTs(champion.promoted_at ?? null)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiscoveriesSection({ discoveries }: { discoveries: AutonomyDiscoveries | undefined }) {
  if (discoveries === undefined) {
    return <p className="text-sm text-slate-400">Keşif bölümü okunuyor…</p>;
  }

  const state = storeState(discoveries.configured, discoveries.readable, "Keşif arşivi");
  if (state !== null) {
    return <StoreNotice state={state} testId="autonomy-discoveries-state" />;
  }

  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      <StatCard
        label="Arşivlenmiş değerlendirme (runs)"
        value={count(discoveries.runs)}
        testId="autonomy-discoveries-runs"
      />
      <StatCard
        label="Kapı kararı (decisions)"
        value={count(discoveries.decisions)}
        testId="autonomy-discoveries-decisions"
      />
    </div>
  );
}

function DataSection({ data }: { data: AutonomyDataStatus | undefined }) {
  if (data === undefined) {
    return <p className="text-sm text-slate-400">Veri bölümü okunuyor…</p>;
  }

  const state = storeState(data.configured, data.readable, "Veri kökü");
  if (state !== null) {
    return <StoreNotice state={state} testId="autonomy-data-state" />;
  }

  const symbols = data.symbols ?? [];
  if (symbols.length === 0) {
    return (
      <p data-testid="autonomy-data-empty" className="text-sm text-slate-400">
        Ay partition’ı olan sembol yok.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <p className="text-xs text-slate-400">
        Ay partition’ı olan semboller — {formatNumber(symbols.length, 0)} adet
      </p>
      <ul
        data-testid="autonomy-data-symbols"
        aria-label="Ay partition'ı olan semboller"
        className="flex flex-wrap gap-1"
      >
        {symbols.map((symbol) => (
          <li
            key={symbol}
            className="rounded-sm border border-slate-700 bg-slate-800/40 px-2 py-0.5 font-mono text-xs text-slate-200"
          >
            {symbol}
          </li>
        ))}
      </ul>
    </div>
  );
}

function JobsSection({
  page,
  loading,
  error,
  limit,
  onLimit,
  onRetry,
}: {
  page: AutonomyJobPage | null;
  loading: boolean;
  error: string | null;
  limit: number;
  onLimit: (limit: number) => void;
  onRetry: () => void;
}) {
  if (error !== null) {
    return (
      <ErrorNotice testId="autonomy-jobs-error" onRetry={onRetry} retryLabel="İşleri yenile">
        İşler okunamadı: {error}
      </ErrorNotice>
    );
  }

  if (page === null) {
    return <p className="text-sm text-slate-400">{loading ? "İşler okunuyor…" : "İşler okunamadı."}</p>;
  }

  const state = storeState(page.configured, page.readable, "Orkestratör kuyruğu");
  if (state !== null) {
    return <StoreNotice state={state} testId="autonomy-jobs-state" />;
  }

  const items: AutonomyJobItem[] = page.items ?? [];
  const caption = `İşler — en yeni önce, en fazla ${formatNumber(page.limit ?? limit, 0)} kayıt`;

  return (
    <div data-testid="autonomy-jobs-table" className="flex flex-col gap-2">
      <WindowTable
        id="autonomy-jobs"
        caption={caption}
        regionLabel="İş tablosu penceresi"
        limit={limit}
        onLimit={onLimit}
        limitLabel="Gösterilecek iş sayısı"
        limitNote="API yalnızca en yeni N kaydı döndürür; daha eski sayfa yoktur."
      >
        <thead className="text-slate-400">
          <tr>
            <th scope="col" className="px-3 py-1.5 font-medium">
              İş
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Tür
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Durum
            </th>
            <th scope="col" className="px-3 py-1.5 text-right font-medium">
              Deneme
            </th>
            <th scope="col" className="px-3 py-1.5 text-right font-medium">
              Erteleme
            </th>
            <th scope="col" className="px-3 py-1.5 text-right font-medium">
              Öncelik
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Oluşturuldu
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Planlandı
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
          {items.map((job) => (
            <tr key={job.id} className="border-t border-slate-800/60">
              <th scope="row" className="px-3 py-1.5 font-mono font-normal text-slate-300">
                #{job.id}
              </th>
              <td className="px-3 py-1.5 font-mono text-slate-300">{job.kind}</td>
              <td className="px-3 py-1.5">
                <JobStatusChip status={job.status} jobId={job.id} />
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                {formatNumber(job.attempts, 0)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                {formatNumber(job.defer_count, 0)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums text-slate-300">
                {formatNumber(job.priority, 0)}
              </td>
              <td className="px-3 py-1.5 tabular-nums text-slate-300">
                {formatTs(job.created_at)}
              </td>
              <td className="px-3 py-1.5 tabular-nums text-slate-300">
                {formatTs(job.scheduled_at)}
              </td>
              <td className="px-3 py-1.5 tabular-nums text-slate-300">
                {formatTs(job.started_at ?? null)}
              </td>
              <td className="px-3 py-1.5 tabular-nums text-slate-300">
                {formatTs(job.finished_at ?? null)}
              </td>
            </tr>
          ))}
        </tbody>
      </WindowTable>

      {items.length === 0 && (
        <p data-testid="autonomy-jobs-empty" className="text-xs text-slate-400">
          Bu pencerede iş yok. Kuyruk boş olabilir ya da bu aralıkta iş oluşmamıştır.
        </p>
      )}
    </div>
  );
}

function EventsSection({
  page,
  loading,
  error,
  limit,
  onLimit,
  onRetry,
}: {
  page: AutonomyEventPage | null;
  loading: boolean;
  error: string | null;
  limit: number;
  onLimit: (limit: number) => void;
  onRetry: () => void;
}) {
  if (error !== null) {
    return (
      <ErrorNotice testId="autonomy-events-error" onRetry={onRetry} retryLabel="Olayları yenile">
        Olaylar okunamadı: {error}
      </ErrorNotice>
    );
  }

  if (page === null) {
    return (
      <p className="text-sm text-slate-400">
        {loading ? "Olaylar okunuyor…" : "Olaylar okunamadı."}
      </p>
    );
  }

  const state = storeState(page.configured, page.readable, "Orkestratör olay günlüğü");
  if (state !== null) {
    return <StoreNotice state={state} testId="autonomy-events-state" />;
  }

  const items: AutonomyEventItem[] = page.items ?? [];
  const caption = `Denetim olayları — en yeni önce, en fazla ${formatNumber(
    page.limit ?? limit,
    0,
  )} kayıt`;

  return (
    <div data-testid="autonomy-events-table" className="flex flex-col gap-2">
      <WindowTable
        id="autonomy-events"
        caption={caption}
        regionLabel="Olay tablosu penceresi"
        limit={limit}
        onLimit={onLimit}
        limitLabel="Gösterilecek olay sayısı"
        limitNote="API yalnızca en yeni N kaydı döndürür; daha eski sayfa yoktur."
      >
        <thead className="text-slate-400">
          <tr>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Olay
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Zaman
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              Tür
            </th>
            <th scope="col" className="px-3 py-1.5 font-medium">
              İş
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((event) => (
            <tr key={event.id} className="border-t border-slate-800/60">
              <th scope="row" className="px-3 py-1.5 font-mono font-normal text-slate-300">
                #{event.id}
              </th>
              <td className="px-3 py-1.5 tabular-nums text-slate-300">{formatTs(event.ts)}</td>
              <td className="px-3 py-1.5 font-mono text-slate-300">{event.event}</td>
              <td className="px-3 py-1.5 text-slate-300">
                {event.job_id === null || event.job_id === undefined ? "kontrol" : `#${event.job_id}`}
              </td>
            </tr>
          ))}
        </tbody>
      </WindowTable>

      {items.length === 0 && (
        <p data-testid="autonomy-events-empty" className="text-xs text-slate-400">
          Bu pencerede denetim olayı yok.
        </p>
      )}
    </div>
  );
}
