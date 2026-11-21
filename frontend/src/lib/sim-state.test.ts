import { describe, expect, it } from "vitest";

import {
  BAR_LIMIT,
  EQUITY_LIMIT,
  TRADE_LIMIT,
  applySimMessage,
  initialSimSnapshot,
  isNewSession,
  simReducer,
  type SimSnapshot,
} from "./sim-state";
import type {
  BarPayload,
  EquityPayload,
  SimMessage,
  StatusPayload,
  TradeOpenPayload,
  TradeSummary,
} from "./types";

const STATUS: StatusPayload = {
  state: "running",
  symbol: "XAUUSD",
  session_id: "aaaaaaaa",
  head_seq: 0,
};

const BAR: BarPayload = { ts: "2026-01-01T00:00:00Z", open: 1, high: 2, low: 0.5, close: 1.5 };
const EQUITY: EquityPayload = { equity: 10_000, balance: 9_990 };
const POSITION: TradeOpenPayload = {
  side: 1,
  lot: 0.1,
  entry: 1.5,
  entry_ts: "2026-01-01T00:00:00Z",
  sl: 1.4,
  tp: 1.7,
  strategy: "breakout",
};
const TRADE: TradeSummary = {
  entry_ts: "2026-01-01T00:00:00Z",
  exit_ts: "2026-01-01T00:01:00Z",
  side: 1,
  lot: 0.1,
  entry: 1.5,
  exit: 1.6,
  sl: 1.4,
  tp: 1.7,
  pnl: 10,
  costs: 1,
  r_multiple: 0.8,
  reason: "tp",
  strategy: "breakout",
};

function status(payload: Partial<StatusPayload> = {}): SimMessage {
  return { seq: 0, type: "status", payload: { ...STATUS, ...payload } };
}

function bar(seq: number, ts = BAR.ts): SimMessage {
  return { seq, type: "bar", payload: { ...BAR, ts } };
}

function equity(seq: number, value = EQUITY.equity): SimMessage {
  return { seq, type: "equity", payload: { ...EQUITY, equity: value } };
}

/** Apply a run of messages, one reducer step each, like the socket does. */
function run(state: SimSnapshot, messages: SimMessage[]): SimSnapshot {
  return messages.reduce(applySimMessage, state);
}

describe("applySimMessage — status", () => {
  it("adopts the session and status", () => {
    const next = applySimMessage(initialSimSnapshot, status({ head_seq: 7 }));

    expect(next.status).toBe("running");
    expect(next.symbol).toBe("XAUUSD");
    expect(next.sessionId).toBe("aaaaaaaa");
    expect(next.seq).toBe(7);
  });

  it("clears data when a null session_id means the server restarted", () => {
    const started = run(initialSimSnapshot, [status(), bar(1), equity(2)]);

    const next = applySimMessage(started, status({ session_id: null, state: "idle", head_seq: 0 }));

    expect(next.sessionId).toBeNull();
    expect(next.bars).toEqual([]);
    expect(next.equity).toEqual([]);
    expect(next.status).toBe("idle");
    expect(next.seq).toBe(0);
  });

  it("keeps an unstarted null session a no-op", () => {
    const next = applySimMessage(initialSimSnapshot, status({ session_id: null, state: "idle" }));

    expect(next.sessionId).toBeNull();
    expect(next.bars).toEqual([]);
    expect(next.seq).toBe(0);
  });

  it("keeps the buffers when the same session re-announces itself", () => {
    const started = run(initialSimSnapshot, [status(), bar(1), equity(2)]);

    const next = applySimMessage(started, status({ head_seq: 2 }));

    expect(next.bars).toHaveLength(1);
    expect(next.equity).toHaveLength(1);
    expect(next.seq).toBe(2);
  });

  it("clears the session data when a different session_id arrives", () => {
    const first = run(initialSimSnapshot, [
      status(),
      bar(1),
      equity(2),
      { seq: 3, type: "trade_open", payload: POSITION },
      { seq: 4, type: "trade_close", payload: TRADE },
    ]);

    const next = applySimMessage(first, status({ session_id: "bbbbbbbb", head_seq: 0 }));

    expect(next.sessionId).toBe("bbbbbbbb");
    expect(next.bars).toEqual([]);
    expect(next.trades).toEqual([]);
    expect(next.equity).toEqual([]);
    expect(next.position).toBeNull();
    expect(next.seq).toBe(0);
    expect(next.status).toBe("running");
  });

  it("adopts a non-zero head_seq for a fresh session", () => {
    const next = applySimMessage(initialSimSnapshot, status({ session_id: "cccccccc", head_seq: 42 }));

    expect(next.seq).toBe(42);
    expect(next.bars).toEqual([]);
  });
});

describe("applySimMessage — data frames", () => {
  it("appends bars in order", () => {
    const next = run(initialSimSnapshot, [status(), bar(1, "a"), bar(2, "b")]);

    expect(next.bars.map((item) => item.ts)).toEqual(["a", "b"]);
    expect(next.seq).toBe(2);
  });

  it("caps bars at the limit, dropping the oldest", () => {
    const total = BAR_LIMIT + 2;
    let state: SimSnapshot = initialSimSnapshot;
    for (let seq = 0; seq < total; seq += 1) {
      state = applySimMessage(state, bar(seq, `t${seq}`));
    }

    expect(state.bars).toHaveLength(BAR_LIMIT);
    expect(state.bars[0].ts).toBe("t2");
    expect(state.bars.at(-1)?.ts).toBe(`t${total - 1}`);
  });

  it("tracks the open position and appends closed trades", () => {
    const opened = run(initialSimSnapshot, [status(), { seq: 1, type: "trade_open", payload: POSITION }]);
    expect(opened.position).toEqual(POSITION);

    const closed = applySimMessage(opened, { seq: 2, type: "trade_close", payload: TRADE });

    expect(closed.position).toBeNull();
    expect(closed.trades).toEqual([TRADE]);
  });

  it("caps trades at the limit, dropping the oldest", () => {
    let state: SimSnapshot = initialSimSnapshot;
    for (let seq = 0; seq < TRADE_LIMIT + 1; seq += 1) {
      state = applySimMessage(state, {
        seq,
        type: "trade_close",
        payload: { ...TRADE, exit_ts: `t${seq}` },
      });
    }

    expect(state.trades).toHaveLength(TRADE_LIMIT);
    expect(state.trades[0].exit_ts).toBe("t1");
  });

  it("appends equity points up to the limit", () => {
    let state: SimSnapshot = initialSimSnapshot;
    for (let seq = 0; seq < EQUITY_LIMIT + 1; seq += 1) {
      state = applySimMessage(state, equity(seq, 1_000 + seq));
    }

    expect(state.equity).toHaveLength(EQUITY_LIMIT);
    expect(state.equity[0].equity).toBe(1_001);
    expect(state.equity.at(-1)?.equity).toBe(1_000 + EQUITY_LIMIT);
  });

  it("records decisions and risk blocks", () => {
    const decided = applySimMessage(initialSimSnapshot, {
      seq: 1,
      type: "decision",
      payload: { action: 1, strategy: "breakout", blocked: null },
    });
    expect(decided.lastDecision).toEqual({ action: 1, strategy: "breakout", blocked: null });
    expect(decided.riskBlocked).toBeNull();

    const blocked = applySimMessage(decided, {
      seq: 2,
      type: "risk_blocked",
      payload: { reason: "max daily loss" },
    });
    expect(blocked.riskBlocked).toBe("max daily loss");
  });

  it("marks an episode finished and keeps the data", () => {
    const started = run(initialSimSnapshot, [status(), bar(1), equity(2)]);

    const ended = applySimMessage(started, {
      seq: 3,
      type: "episode_end",
      payload: { reason: "finished" },
    });

    expect(ended.status).toBe("finished");
    expect(ended.bars).toHaveLength(1);
    expect(ended.equity).toHaveLength(1);
  });

  it("does not mutate the input state", () => {
    const started = Object.freeze(run(initialSimSnapshot, [status()]));
    const next = applySimMessage(started, bar(1));

    expect(next).not.toBe(started);
    expect(next.bars).not.toBe(started.bars);
    expect(started.bars).toHaveLength(0);
  });
});

describe("isNewSession", () => {
  it("is true for any session that differs from the known one", () => {
    const state = applySimMessage(initialSimSnapshot, status());

    expect(isNewSession(state, status({ session_id: "bbbbbbbb" }))).toBe(true);
    expect(isNewSession(state, status({ session_id: "aaaaaaaa" }))).toBe(false);
    expect(isNewSession(state, bar(1))).toBe(false);
  });

  it("treats a null session as new only when a session was known", () => {
    const state = applySimMessage(initialSimSnapshot, status());

    expect(isNewSession(state, status({ session_id: null }))).toBe(true);
    expect(isNewSession(initialSimSnapshot, status({ session_id: null }))).toBe(false);
  });
});

describe("simReducer — sessionAdopted", () => {
  it("keeps the buffers when no previous session was known", () => {
    const state = run(initialSimSnapshot, [bar(1)]);

    const next = simReducer(state, { type: "sessionAdopted", sessionId: "sess-A" });

    expect(next.sessionId).toBe("sess-A");
    expect(next.status).toBe("running");
    expect(next.seq).toBe(0);
    expect(next.bars).toEqual(state.bars);
  });

  it("clears the previous episode when a different session was known", () => {
    const populated = run(initialSimSnapshot, [
      status({ session_id: "old" }),
      bar(1),
      equity(2),
      { seq: 3, type: "trade_open", payload: POSITION },
    ]);

    const next = simReducer(populated, { type: "sessionAdopted", sessionId: "sess-A" });

    expect(next.sessionId).toBe("sess-A");
    expect(next.status).toBe("running");
    expect(next.seq).toBe(0);
    expect(next.bars).toEqual([]);
    expect(next.equity).toEqual([]);
    expect(next.trades).toEqual([]);
    expect(next.position).toBeNull();
  });

  it("is idempotent for the same session", () => {
    const state = applySimMessage(initialSimSnapshot, status({ session_id: "sess-A" }));

    expect(simReducer(state, { type: "sessionAdopted", sessionId: "sess-A" })).toBe(state);
  });
});
