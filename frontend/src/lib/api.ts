import type {
  CalibrateRequest,
  CalibrateResponse,
  DataStatus,
  DownloadRequest,
  DownloadResponse,
  Health,
  MLEvaluateRequest,
  MLEvaluateResponse,
  MLRunDetail,
  MLRunSummary,
  MLTrainRequest,
  MLTrainResponse,
  OkResponse,
  PaperEquityPoint,
  PaperOrder,
  PaperPage,
  PaperPosition,
  PaperSessionItem,
  PaperSessionSummary,
  PaperTrade,
  RegimeResponse,
  RunControlAction,
  RunDetail,
  RunSummary,
  SimControl,
  SimMessage,
  SimStartRequest,
  StartResponse,
  StrategyCard,
  SymbolSpec,
  TimeframeSpec,
} from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/** Default per-call budget; downloads are synchronous and get more. */
const DEFAULT_TIMEOUT_MS = 5_000;

/** `/api/data/download` blocks on the provider, so it outlasts the default. */
const DOWNLOAD_TIMEOUT_MS = 30_000;

/** Recalibration labels years of bars; a 5-year H1 grid takes minutes. */
const CALIBRATE_TIMEOUT_MS = 600_000;

/** `/api/ml/train` reads the store and builds features before its 202. */
const ML_PREPARE_TIMEOUT_MS = 60_000;

/** `/api/ml/evaluate` runs the simulator once per stress multiplier. */
const ML_EVALUATE_TIMEOUT_MS = 300_000;

/** `/api/paper/.../close` takes a short `BEGIN IMMEDIATE`; still not a cache hit. */
const PAPER_CLOSE_TIMEOUT_MS = 15_000;

/**
 * A failed API call. `status` is the HTTP status, or `0` when the request never
 * produced a response (backend down, DNS failure, timeout).
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function failureDetail(error: unknown): string {
  // `AbortSignal.timeout` aborts with `TimeoutError`; some fetch stacks surface
  // the abort as `AbortError` instead. Both mean the same thing to the caller.
  if (
    error instanceof DOMException &&
    (error.name === "TimeoutError" || error.name === "AbortError")
  ) {
    return "request timed out";
  }
  return error instanceof Error && error.message ? error.message : "network request failed";
}

/**
 * Read FastAPI's `detail`. A body that is not JSON (a proxy's HTML page, an
 * empty 502) must yield a generic detail instead of a JSON parse error.
 */
async function errorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null) {
      const detail = (body as { detail?: unknown }).detail;
      if (typeof detail === "string" && detail.length > 0) return detail;
      // Pydantic validation errors send `detail` as [{loc, msg, type}].
      if (Array.isArray(detail)) {
        const messages = detail
          .map((item) =>
            typeof item === "object" && item !== null ? (item as { msg?: unknown }).msg : undefined,
          )
          .filter((msg): msg is string => typeof msg === "string" && msg.length > 0);
        if (messages.length > 0) return messages.join("; ");
      }
    }
  } catch {
    // Not JSON: fall through to the status line.
  }
  return `HTTP ${response.status}`;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body !== undefined) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    throw new ApiError(0, failureDetail(error));
  }

  if (!response.ok) {
    throw new ApiError(response.status, await errorDetail(response));
  }

  // Read text first: a 204/empty 200 must not become a JSON parse crash.
  const text = await response.text();
  if (text.length === 0) return undefined as T;

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(response.status, "invalid JSON response");
  }
}

function post<T>(path: string, body: unknown, timeoutMs?: number): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) }, timeoutMs);
}

export const api = {
  getHealth(): Promise<Health> {
    return request<Health>("/api/health");
  },

  getDataStatus(symbol: string): Promise<DataStatus> {
    return request<DataStatus>(`/api/data/status?symbol=${encodeURIComponent(symbol)}`);
  },

  getTimeframes(): Promise<TimeframeSpec[]> {
    return request<TimeframeSpec[]>("/api/timeframes");
  },

  getSymbols(): Promise<SymbolSpec[]> {
    return request<SymbolSpec[]>("/api/symbols");
  },

  getRegime(symbol: string, timeframe: string, limit = 500): Promise<RegimeResponse> {
    const params = new URLSearchParams({ symbol, timeframe, limit: String(limit) });
    return request<RegimeResponse>(`/api/regime?${params.toString()}`);
  },

  downloadData(body: DownloadRequest): Promise<DownloadResponse> {
    return post<DownloadResponse>("/api/data/download", body, DOWNLOAD_TIMEOUT_MS);
  },

  startSim(body: SimStartRequest): Promise<StartResponse> {
    return post<StartResponse>("/api/sim/start", body);
  },

  control(body: SimControl): Promise<OkResponse> {
    return post<OkResponse>("/api/sim/control", body);
  },

  getRuns(): Promise<RunSummary[]> {
    return request<RunSummary[]>("/api/runs");
  },

  getRun(id: string): Promise<RunDetail> {
    return request<RunDetail>(`/api/runs/${encodeURIComponent(id)}`);
  },

  controlRun(id: string, action: RunControlAction): Promise<OkResponse> {
    return post<OkResponse>(`/api/runs/${encodeURIComponent(id)}/control`, { action });
  },

  getStrategyCard(): Promise<StrategyCard> {
    return request<StrategyCard>("/api/strategy/card");
  },

  calibrateStrategy(body: CalibrateRequest = {}): Promise<CalibrateResponse> {
    // The grid labels years of bars on the backend, so this outlasts the rest.
    return post<CalibrateResponse>("/api/strategy/calibrate", body, CALIBRATE_TIMEOUT_MS);
  },

  // --- classical ML ---------------------------------------------------------

  mlModels(): Promise<string[]> {
    return request<string[]>("/api/ml/models");
  },

  mlTrain(body: MLTrainRequest): Promise<MLTrainResponse> {
    // Preparation (validation, store read, feature build) runs before the 202.
    return post<MLTrainResponse>("/api/ml/train", body, ML_PREPARE_TIMEOUT_MS);
  },

  getMLRuns(): Promise<MLRunSummary[]> {
    return request<MLRunSummary[]>("/api/ml/runs");
  },

  getMLRun(runId: string): Promise<MLRunDetail> {
    return request<MLRunDetail>(`/api/ml/runs/${encodeURIComponent(runId)}`);
  },

  cancelMLRun(runId: string): Promise<OkResponse> {
    return post<OkResponse>(`/api/ml/runs/${encodeURIComponent(runId)}/control`, {
      action: "cancel",
    });
  },

  mlEvaluate(body: MLEvaluateRequest): Promise<MLEvaluateResponse> {
    // A split plus one simulator pass per stress multiplier; not a 5 s call.
    return post<MLEvaluateResponse>("/api/ml/evaluate", body, ML_EVALUATE_TIMEOUT_MS);
  },

  // --- paper ledger ----------------------------------------------------------
  // Reads are `mode=ro` on the backend and stay cheap; `close` is the only
  // write this client performs. Every list is a bounded `{items,total,...}`.

  paperSessions(limit = 100, offset = 0): Promise<PaperPage<PaperSessionItem>> {
    return request<PaperPage<PaperSessionItem>>(paperPagePath("/api/paper/sessions", limit, offset));
  },

  paperSession(sessionId: string): Promise<PaperSessionSummary> {
    return request<PaperSessionSummary>(paperSessionBase(sessionId));
  },

  closePaperSession(sessionId: string): Promise<PaperSessionSummary> {
    // The ledger takes `BEGIN IMMEDIATE` for one row; still give it headroom.
    return request<PaperSessionSummary>(
      `${paperSessionBase(sessionId)}/close`,
      { method: "POST" },
      PAPER_CLOSE_TIMEOUT_MS,
    );
  },

  paperPositions(
    sessionId: string,
    limit = 100,
    offset = 0,
  ): Promise<PaperPage<PaperPosition>> {
    return request<PaperPage<PaperPosition>>(
      paperPagePath(`${paperSessionBase(sessionId)}/positions`, limit, offset),
    );
  },

  paperOrders(sessionId: string, limit = 100, offset = 0): Promise<PaperPage<PaperOrder>> {
    return request<PaperPage<PaperOrder>>(
      paperPagePath(`${paperSessionBase(sessionId)}/orders`, limit, offset),
    );
  },

  paperHistory(sessionId: string, limit = 100, offset = 0): Promise<PaperPage<PaperTrade>> {
    return request<PaperPage<PaperTrade>>(
      paperPagePath(`${paperSessionBase(sessionId)}/history`, limit, offset),
    );
  },

  paperEquity(
    sessionId: string,
    limit = 100,
    offset = 0,
  ): Promise<PaperPage<PaperEquityPoint>> {
    // `offset=0` is the newest window; raising it pages into older marks.
    return request<PaperPage<PaperEquityPoint>>(
      paperPagePath(`${paperSessionBase(sessionId)}/equity`, limit, offset),
    );
  },
};

/** Session ids are opaque tokens; encoding keeps them a single path segment. */
function paperSessionBase(sessionId: string): string {
  return `/api/paper/sessions/${encodeURIComponent(sessionId)}`;
}

function paperPagePath(base: string, limit: number, offset: number): string {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return `${base}?${params.toString()}`;
}

/**
 * Runtime mirror of the `SimMessage["type"]` union. `ReadonlySet<string>`
 * keeps `has` cheap while the constructor still type-checks every literal.
 */
const SIM_MESSAGE_TYPES: ReadonlySet<string> = new Set<SimMessage["type"]>([
  "status",
  "bar",
  "decision",
  "trade_open",
  "trade_close",
  "equity",
  "risk_blocked",
  "episode_end",
]);

/** Validate an envelope in one place so the socket hook stays a switch. */
export function isSimMessage(value: unknown): value is SimMessage {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as { seq?: unknown; type?: unknown; payload?: unknown };
  return (
    typeof candidate.seq === "number" &&
    Number.isInteger(candidate.seq) &&
    candidate.seq >= 0 &&
    typeof candidate.type === "string" &&
    SIM_MESSAGE_TYPES.has(candidate.type) &&
    typeof candidate.payload === "object" &&
    candidate.payload !== null
  );
}
