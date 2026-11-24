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
  /** Absent on servers older than the provenance endpoint. */
  provenance?: Provenance | null;
}

/** Sidecar metadata written next to a store partition. */
export interface Provenance {
  dataset_id: string;
  symbol: string;
  timeframe: string;
  source: string;
  synthetic: boolean;
  rows: number;
  first_ts: string | null;
  last_ts: string | null;
  checksum: string;
  created_at: string;
  notes: string;
}


/** One supported bar size from `GET /api/timeframes`. */
export interface TimeframeSpec {
  timeframe: string;
  rule: string;
}

export interface RegimePoint {
  ts: string;
  regime: string;
  trend: string;
  volatility: string;
}

export interface RegimeCount {
  regime: string;
  bars: number;
  share: number;
}

export interface RegimeCurrent {
  ts: string;
  trend: string;
  trend_strength: string;
  volatility: string;
  expansion: string;
  breakout: string;
  unusual: boolean;
  regime: string;
}

/** Descriptive regime state; labels never predict the next bar. */
export interface RegimeResponse {
  symbol: string;
  timeframe: string;
  rows: number;
  current: RegimeCurrent | null;
  counts: RegimeCount[];
  series: RegimePoint[];
}

export interface DownloadRequest {
  provider: ProviderName;
  symbol: string;
  start: string;
  end: string;
  seed?: number | null;
  /** CSV only: a bare file name in the backend's `<data_root>/imports`. */
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

/** One checkpoint of a training curve; `step` is the x axis. */
export interface TrainingPoint {
  step: number;
  value: number;
}

/** Training curves; every series is optional until the RL API fills them. */
export interface TrainingSeries {
  reward: TrainingPoint[];
  episode_length: TrainingPoint[];
  entropy: TrainingPoint[];
  loss: TrainingPoint[];
}

/** LinUCB arm weight; one row per strategy in `BANDIT_STRATEGIES`. */
export interface BanditWeight {
  strategy: string;
  weight: number;
}

/** `pause` / `resume` / `stop` for a training run. */
export type RunControlAction = "pause" | "resume" | "stop";

/** `GET /api/runs/{id}`; config is free-form run metadata. */
export interface RunDetail {
  config: Record<string, unknown>;
  metrics: Metrics;
  status: string;
  /** Present once the RL training API lands; absent on the current backend. */
  step?: number;
  total_steps?: number;
  series?: Partial<TrainingSeries>;
  bandit?: BanditWeight[];
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

// --- strategy card -----------------------------------------------------------

/** Parameters the promoted flow-momentum rule replays. */
export interface RuleParams {
  timeframe: string;
  flow_quantile: number;
  flow_window: number;
  sl_atr_mult: number;
  tp_atr_mult: number;
  max_bars_in_trade: number;
}

/** One slice's realised behaviour: training or holdout. */
export interface RuleMetrics {
  trades: number;
  expectancy_r: number;
  profit_factor: number;
  win_rate: number;
  max_drawdown: number;
  total_r: number;
}

/** The latest weekly forward test, attached to the card by the backend. */
export interface ForwardResult {
  at?: string;
  start?: string;
  end?: string;
  trades?: number;
  expectancy_r?: number;
  profit_factor?: number;
  net_profit?: number;
  max_drawdown?: number;
}

/** `GET /api/strategy/card`; optional fields are absent when inactive. */
export interface StrategyCard {
  active: boolean;
  strategy: string;
  timeframe?: string;
  symbol?: string;
  params?: RuleParams;
  train?: RuleMetrics;
  holdout?: RuleMetrics;
  digest?: string;
  created_at?: string;
  note?: string;
  version?: number;
  forward?: ForwardResult;
}

// --- classical ML ------------------------------------------------------------

/** `GET /api/symbols`; only the fields the form shows are named. */
export interface SymbolSpec {
  symbol: string;
  description: string;
  asset_class: string;
}

/** Bar sizes the ML endpoints accept (`MLTimeframe` in the API schema). */
export type MLTimeframe = "M1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1";

/** Split names `/api/ml/evaluate` accepts. */
export type MLSplit = "train" | "validation" | "test";

/** One of the runner's statuses; unknown strings are shown verbatim. */
export type MLRunStatus =
  | "preparing"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed"
  | "unknown";

/** `POST /api/ml/train`; `groups: null` means every registered group. */
export interface MLTrainRequest {
  symbol: string;
  timeframe: MLTimeframe;
  model: string;
  horizon: number;
  groups?: string[] | null;
  train_fraction: number;
  validation_fraction: number;
  purge_bars: number;
  embargo_bars: number;
  seed?: number;
  evaluate_test?: boolean;
  include_flat: boolean;
}

export interface MLTrainResponse {
  run_id: string;
}

/** One row of `GET /api/ml/runs`. */
export interface MLRunSummary {
  run_id: string;
  status: string;
  created_at: string | null;
  model: string | null;
  symbol: string | null;
  timeframe: string | null;
  dataset_id: string | null;
}

/**
 * Prediction metrics of one split. Fields are optional because a failed or
 * older report may omit them; a missing value is a dash, never a zero.
 */
export interface MLPredictionMetrics {
  rows?: number;
  accuracy?: number;
  balanced_accuracy?: number;
  log_loss?: number;
  brier?: number;
  calibration?: {
    ece?: number;
    mce?: number;
  };
}

/** Flattened calibration of the validation split (top level of the detail). */
export interface MLCalibration {
  brier: number | null;
  log_loss: number | null;
  ece: number | null;
  mce: number | null;
}

/** `GET /api/ml/runs/{id}`; everything is prediction metrics, not trading. */
export interface MLRunDetail {
  run_id: string;
  status: string;
  created_at: string | null;
  finished_at: string | null;
  model: string | null;
  symbol: string | null;
  timeframe: string | null;
  dataset_id: string | null;
  config: Record<string, unknown>;
  spec: Record<string, unknown>;
  columns: string[];
  validation: MLPredictionMetrics | null;
  test: MLPredictionMetrics | null;
  baseline: Record<string, unknown> | null;
  split_counts: Record<string, number>;
  calibration: MLCalibration | null;
  error: string | null;
}

/** The trading metrics of one split; same shape as `Metrics`, fewer fields. */
export interface MLTradingMetrics {
  trades: number;
  net_profit: number;
  profit_factor: number | null;
  win_rate: number;
  expectancy_r: number;
  max_drawdown: number;
  total_costs: number;
}

/** One per-regime bucket; `profit_factor` is `null` when not representable. */
export interface MLRegimeRow {
  regime: string;
  trades: number;
  win_rate: number;
  expectancy_r: number;
  total_r: number;
  pnl: number;
  profit_factor: number | null;
}

export interface MLRegimeTable {
  rows: MLRegimeRow[];
  unmatched_trades: number;
}

export interface MLEvaluateRequest {
  run_id: string;
  split: MLSplit;
  stress?: number[];
  regime?: boolean;
}

/** `POST /api/ml/evaluate`: headline, hold baseline, stress and regimes. */
export interface MLEvaluateResponse {
  run_id: string;
  symbol: string;
  timeframe: string;
  dataset_id: string;
  split: string;
  threshold: number;
  exit_threshold: number;
  metrics: MLTradingMetrics;
  hold: MLTradingMetrics;
  stress: Record<string, MLTradingMetrics>;
  regime: MLRegimeTable;
}

export interface CalibrateRequest {
  symbol?: string;
  timeframe?: "M1" | "M5" | "M15" | "H1" | "D1";
  holdout_fraction?: number;
  min_train_trades?: number;
  min_holdout_trades?: number;
  dry_run?: boolean;
}

/** `POST /api/strategy/calibrate`: the gate's verdict plus the evidence. */
export interface CalibrateResponse {
  promoted: boolean;
  reason: string;
  digest: string;
  evaluated: number;
  candidate: {
    params: RuleParams;
    train: RuleMetrics;
    holdout: RuleMetrics;
  };
  incumbent: RuleMetrics | null;
  card: StrategyCard;
}

// --- paper ledger (`/api/paper`) ---------------------------------------------

/** Bar sizes a paper session may be opened on (`PaperTimeframe` in the schema). */
export type PaperTimeframe = "M1" | "M5" | "M15" | "M30" | "H1" | "H4" | "D1";

/**
 * One row of `GET /api/paper/sessions`. `model_version` carries the DSL spec
 * hash when the autonomous runner opened the session; it is `null` otherwise.
 */
export interface PaperSessionItem {
  session_id: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  model_version: string | null;
  status: string;
  started_at: string;
  ended_at: string | null;
}

/** A position lifecycle; `closed_ts` is set when a trade settles it. */
export interface PaperPosition {
  position_id: number;
  side: string;
  lot: number;
  entry_ts: string;
  entry_price: number;
  sl: number;
  tp: number;
  status: string;
  closed_ts: string | null;
  closed_price: number | null;
}

/** One order attempt; a rejection always carries its reason. */
export interface PaperOrder {
  order_id: number;
  signal_id: number | null;
  ts: string;
  side: string;
  lot: number;
  requested_price: number | null;
  status: string;
  rejection_reason: string | null;
}

/** One closed trade, exactly as the broker settled it. */
export interface PaperTrade {
  trade_id: number;
  side: string;
  entry_ts: string;
  entry_price: number;
  exit_ts: string;
  exit_price: number;
  lot: number;
  pnl: number;
  r_multiple: number;
  exit_reason: string;
}

/** One account mark. */
export interface PaperEquityPoint {
  ts: string;
  balance: number;
  equity: number;
}

/**
 * `GET /api/paper/sessions/{id}`. `balance`/`equity` stay `null` until the
 * first equity mark; `trade_count` counts closed trades only.
 */
export interface PaperSessionSummary {
  session_id: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  model_version: string | null;
  status: string;
  started_at: string;
  ended_at: string | null;
  balance: number | null;
  equity: number | null;
  open_position: PaperPosition | null;
  trade_count: number;
  last_bar_ts: string | null;
  config: Record<string, unknown>;
}

/** A bounded page: `limit` defaults to 100 server-side and is capped at 1000. */
export interface PaperPage<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

