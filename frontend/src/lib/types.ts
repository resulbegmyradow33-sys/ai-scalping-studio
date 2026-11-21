/**
 * Wire contracts shared with the Python backend.
 *
 * Source of truth: `backend/src/scalping/api/schemas.py` (HTTP bodies) and
 * `backend/src/scalping/api/runner.py` (WebSocket payloads). Keep every field
 * name identical to the serialized JSON, not to the Python attribute it comes
 * from: `trade_open.entry` is `Position.entry_price` on the wire.
 */

// --- HTTP --------------------------------------------------------------------

export type ProviderName = "synthetic" | "csv" | "dukascopy";
export type SimMode = "replay";
export type ControlAction = "pause" | "resume" | "stop" | "reset";

export interface Health {
  status: string;
  version: string;
  device: string;
}

export interface DataStatus {
  symbol: string;
  rows: number;
  first_ts: string | null;
  last_ts: string | null;
  providers: string[];
}

export interface DownloadRequest {
  provider: ProviderName;
  symbol: string;
  start: string;
  end: string;
  seed?: number | null;
  path?: string | null;
  tz_offset?: number | null;
}

export interface DownloadResponse {
  rows: number;
}

export interface SimStartRequest {
  speed: number;
  mode: SimMode;
}

export interface StartResponse {
  session_id: string;
}

export interface SimControl {
  action: ControlAction;
}

export interface OkResponse {
  ok: boolean;
}

// --- Trades and metrics ------------------------------------------------------

/** `1` is long, `-1` is short; the backend sends `int(Side)`. */
export type TradeSide = 1 | -1;

/** `0` hold, `1` buy, `2` sell, `3` close. */
export type DecisionAction = 0 | 1 | 2 | 3;

/** The `trade_open` payload: an open position, not yet a closed trade. */
export interface TradeOpenPayload {
  side: TradeSide;
  lot: number;
  entry: number;
  entry_ts: string;
  sl: number;
  tp: number;
  strategy: string;
}

/** A closed trade, matching `scalping.types.Trade` serialization. */
export interface TradeSummary {
  entry_ts: string;
  exit_ts: string;
  side: TradeSide;
  lot: number;
  entry: number;
  exit: number;
  sl: number;
  tp: number;
  pnl: number;
  costs: number;
  r_multiple: number;
  reason: string;
  strategy: string;
}

/** `scalping.backtest.metrics.Metrics`; non-finite floats arrive as `null`. */
export interface Metrics {
  net_profit: number;
  profit_factor: number | null;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  expectancy_r: number;
  trades: number;
  total_costs: number;
  cost_ratio: number;
  avg_bars: number;
}

/** One entry of `GET /api/runs`. */
export interface RunSummary {
  run_id: string;
  status: string;
  created_at: string;
  steps: number;
  reward_mean: number | null;
}

/** `GET /api/runs/{id}`; config is free-form run metadata. */
export interface RunDetail {
  config: Record<string, unknown>;
  metrics: Metrics;
  status: string;
}

// --- WebSocket ---------------------------------------------------------------

export type SimState =
  | "idle"
  | "running"
  | "paused"
  | "stopped"
  | "finished"
  | "error";

export interface StatusPayload {
  state: SimState;
  symbol: string;
  /** Null until the first start/reset installs a session. */
  session_id: string | null;
  head_seq: number;
}

export interface BarPayload {
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface DecisionPayload {
  action: DecisionAction;
  strategy: string;
  blocked: string | null;
}

export interface EquityPayload {
  equity: number;
  balance: number;
}

export interface RiskBlockedPayload {
  reason: string;
}

export interface EpisodeEndPayload {
  reason: string;
}

/** Common envelope; `seq` restarts at 1 for every start/reset (`status` is 0). */
export interface SimMessageOf<T extends string, P> {
  seq: number;
  type: T;
  payload: P;
}

export type SimMessage =
  | SimMessageOf<"status", StatusPayload>
  | SimMessageOf<"bar", BarPayload>
  | SimMessageOf<"decision", DecisionPayload>
  | SimMessageOf<"trade_open", TradeOpenPayload>
  | SimMessageOf<"trade_close", TradeSummary>
  | SimMessageOf<"equity", EquityPayload>
  | SimMessageOf<"risk_blocked", RiskBlockedPayload>
  | SimMessageOf<"episode_end", EpisodeEndPayload>;

export type SimMessageType = SimMessage["type"];
