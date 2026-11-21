/**
 * Pure view state for the simulator stream.
 *
 * `applySimMessage` is the whole reducer the socket hook needs; keeping it out
 * of React lets the buffer caps and the session-epoch reset be tested with
 * plain data instead of a fake socket. It never mutates the input.
 */

import type {
  BarPayload,
  DecisionPayload,
  EquityPayload,
  SimMessage,
  SimState as SimStatus,
  TradeOpenPayload,
  TradeSummary,
} from "./types";

/** Oldest entries are dropped once these are reached. */
export const BAR_LIMIT = 5_000;
export const TRADE_LIMIT = 500;
export const EQUITY_LIMIT = 5_000;

export interface SimSnapshot {
  /** Wire status; `finished` is set locally on `episode_end`. */
  status: SimStatus;
  /** Epoch marker announced by the server; null before the first start. */
  sessionId: string | null;
  /** Highest sequence applied, or the server head from a status frame. */
  seq: number;
  symbol: string | null;
  bars: BarPayload[];
  position: TradeOpenPayload | null;
  equity: EquityPayload[];
  trades: TradeSummary[];
  lastDecision: DecisionPayload | null;
  /** Latest blocked action, mirrored from `decision.blocked` / `risk_blocked`. */
  riskBlocked: string | null;
  error: string | null;
}

export const initialSimSnapshot: SimSnapshot = {
  status: "idle",
  sessionId: null,
  seq: 0,
  symbol: null,
  bars: [],
  position: null,
  equity: [],
  trades: [],
  lastDecision: null,
  riskBlocked: null,
  error: null,
};

export type SimAction =
  | { type: "message"; message: SimMessage }
  | { type: "error"; error: string }
  | { type: "clearError" };

/**
 * A non-null session that differs from the one we are showing is a new epoch.
 * A null session is the backend before its first start: it must not wipe data
 * the client already holds.
 */
export function isNewSession(state: SimSnapshot, message: SimMessage): boolean {
  return (
    message.type === "status" &&
    message.payload.session_id !== null &&
    message.payload.session_id !== state.sessionId
  );
}

function appendCapped<T>(items: readonly T[], item: T, limit: number): T[] {
  if (items.length < limit) return [...items, item];
  return [...items.slice(items.length - limit + 1), item];
}

export function applySimMessage(state: SimSnapshot, message: SimMessage): SimSnapshot {
  if (message.type === "status") {
    const { payload } = message;

    if (isNewSession(state, message)) {
      // A start/reset rewound the stream: drop the previous episode entirely
      // and adopt the new head so low sequence numbers are not rejected.
      return {
        ...state,
        status: payload.state,
        symbol: payload.symbol,
        sessionId: payload.session_id,
        seq: payload.head_seq,
        bars: [],
        position: null,
        equity: [],
        trades: [],
        lastDecision: null,
        riskBlocked: null,
      };
    }

    // Same session (reconnect or control acknowledgement): update the status
    // and head but keep everything already drawn.
    return {
      ...state,
      status: payload.state,
      symbol: payload.symbol,
      seq: Math.max(state.seq, payload.head_seq),
    };
  }

  const seq = Math.max(state.seq, message.seq);

  switch (message.type) {
    case "bar":
      return { ...state, seq, bars: appendCapped(state.bars, message.payload, BAR_LIMIT) };
    case "trade_open":
      return { ...state, seq, position: message.payload };
    case "trade_close":
      return {
        ...state,
        seq,
        position: null,
        trades: appendCapped(state.trades, message.payload, TRADE_LIMIT),
      };
    case "equity":
      return { ...state, seq, equity: appendCapped(state.equity, message.payload, EQUITY_LIMIT) };
    case "decision":
      return {
        ...state,
        seq,
        lastDecision: message.payload,
        riskBlocked: message.payload.blocked,
      };
    case "risk_blocked":
      return { ...state, seq, riskBlocked: message.payload.reason };
    case "episode_end":
      // The backend keeps the episode on screen; only the status changes.
      return { ...state, seq, status: "finished" };
  }
}

export function simReducer(state: SimSnapshot, action: SimAction): SimSnapshot {
  switch (action.type) {
    case "message":
      return applySimMessage(state, action.message);
    case "error":
      return { ...state, error: action.error };
    case "clearError":
      return state.error === null ? state : { ...state, error: null };
  }
}
