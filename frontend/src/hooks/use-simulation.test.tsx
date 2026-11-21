import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { API_BASE } from "@/lib/api";
import type { SimControl, SimMessage, SimStartRequest } from "@/lib/types";

import { useSimulation } from "./use-simulation";

/**
 * Minimal WebSocket double. The hook only uses `readyState`, `send`, `close`
 * and the four handler properties, so a plain class keeps the test readable.
 */
class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.CONNECTING;
  readonly sent: string[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({} as CloseEvent);
  }

  // --- test controls ---------------------------------------------------
  serverOpen(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  serverMessage(message: SimMessage): void {
    this.onmessage?.({ data: JSON.stringify(message) } as MessageEvent);
  }

  serverClose(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({} as CloseEvent);
  }

  frames(): unknown[] {
    return this.sent.map((frame) => JSON.parse(frame) as unknown);
  }
}

function latest(): FakeWebSocket {
  const socket = FakeWebSocket.instances.at(-1);
  if (!socket) throw new Error("no socket was opened");
  return socket;
}

function status(sessionId: string | null, head = 0): SimMessage {
  return {
    seq: 0,
    type: "status",
    payload: { state: "running", symbol: "XAUUSD", session_id: sessionId, head_seq: head },
  };
}

function bar(seq: number, ts = `t${seq}`): SimMessage {
  return { seq, type: "bar", payload: { ts, open: 1, high: 2, low: 0.5, close: 1.5 } };
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("useSimulation — socket lifecycle", () => {
  it("connects to the ws url derived from the api base and never resumes below zero", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    expect(socket.url).toBe(API_BASE.replace(/^http/, "ws") + "/ws/sim");
    expect(result.current.status).toBe("idle");

    act(() => socket.serverOpen());

    // No cursor is seeded yet, so there is nothing to resume from and the
    // route would reject a negative seq anyway.
    expect(socket.frames()).toEqual([]);
  });

  it("starts a new sequence epoch without freezing on low seq numbers", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status("aaaaaaaa"));
      socket.serverMessage(bar(1, "old"));
    });
    expect(result.current.bars).toHaveLength(1);

    act(() => {
      socket.serverMessage(status("bbbbbbbb", 0));
      socket.serverMessage(bar(1, "new"));
    });

    expect(result.current.bars).toHaveLength(1);
    expect(result.current.bars[0].ts).toBe("new");
  });

  it("adopts the session returned by start and keeps bars across a reconnect", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ session_id: "sess-A" }), { status: 200 })),
    );
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => socket.serverMessage(status(null)));

    await act(async () => {
      await result.current.start(1);
    });

    act(() => socket.serverMessage(bar(1, "a")));
    // Adoption seeded the gate at 0: seq 1 is contiguous, so no gap resume.
    expect(socket.frames()).toEqual([]);

    act(() => {
      socket.serverMessage(bar(2, "b"));
      socket.serverMessage(bar(3, "c"));
    });
    expect(result.current.bars).toHaveLength(3);

    vi.useFakeTimers();
    act(() => socket.serverClose());
    act(() => vi.advanceTimersByTime(500));

    const reconnected = latest();
    expect(reconnected).not.toBe(socket);
    act(() => reconnected.serverOpen());
    act(() => reconnected.serverMessage(status("sess-A", 3)));

    // sess-A is the adopted epoch, so the status must not clear the chart.
    expect(result.current.bars.map((item) => item.ts)).toEqual(["a", "b", "c"]);
    expect(reconnected.frames()).toEqual([{ type: "resume", seq: 3 }]);
  });

  it("treats a null status after a known session as a server restart", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status("sess-A"));
      socket.serverMessage(bar(1, "old"));
      socket.serverMessage(bar(2, "older"));
    });
    expect(result.current.bars).toHaveLength(2);

    act(() => socket.serverMessage(status(null, 0)));
    expect(result.current.bars).toEqual([]);

    // The gate was reset to 0, so the restarted session's seq 1 is contiguous.
    act(() => socket.serverMessage(bar(1, "new")));
    expect(result.current.bars.map((item) => item.ts)).toEqual(["new"]);
    expect(socket.frames()).toEqual([]);
  });

  it("requests a resume from the last contiguous sequence when a gap appears", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status("aaaaaaaa"));
      socket.serverMessage(bar(1));
      socket.serverMessage(bar(3));
    });

    // The unseeded gate sends nothing on open; the gap adds resume(1).
    expect(socket.frames()).toEqual([{ type: "resume", seq: 1 }]);
    // The out-of-order bar is still applied; the replay will fill seq 2.
    expect(result.current.bars.map((item) => item.ts)).toEqual(["t1", "t3"]);
  });

  it("requests a gap resume once and does not repeat the same cursor", () => {
    renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status(null));
      socket.serverMessage(bar(1));
      socket.serverMessage(bar(3));
    });
    expect(socket.frames()).toEqual([{ type: "resume", seq: 1 }]);

    // Another frame above the same hole must not re-send resume(1).
    act(() => socket.serverMessage(bar(4)));

    expect(socket.frames()).toEqual([{ type: "resume", seq: 1 }]);
  });

  it("ignores duplicate and stale frames", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status("aaaaaaaa"));
      socket.serverMessage(bar(1, "first"));
      socket.serverMessage(bar(1, "duplicate"));
      socket.serverMessage(bar(5, "high"));
      socket.serverMessage(bar(0, "stale"));
    });

    expect(result.current.bars.map((item) => item.ts)).toEqual(["first", "high"]);
  });

  it("sets status finished on episode_end and keeps the bars", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status("aaaaaaaa"));
      socket.serverMessage(bar(1));
      socket.serverMessage({ seq: 2, type: "episode_end", payload: { reason: "finished" } });
    });

    expect(result.current.status).toBe("finished");
    expect(result.current.bars).toHaveLength(1);
  });

  it("drops malformed frames without throwing", () => {
    const { result } = renderHook(() => useSimulation());
    const socket = latest();
    act(() => socket.serverOpen());

    act(() => {
      socket.onmessage?.({ data: "not json" } as MessageEvent);
      socket.onmessage?.({ data: JSON.stringify({ seq: 1, type: "bogus", payload: {} }) } as MessageEvent);
    });

    expect(result.current.bars).toEqual([]);
  });
});

describe("useSimulation — reconnect", () => {
  it("backs off 500, 1000, 2000 and caps at 8 s", () => {
    vi.useFakeTimers();
    renderHook(() => useSimulation());

    expect(FakeWebSocket.instances).toHaveLength(1);

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(499));
    expect(FakeWebSocket.instances).toHaveLength(1);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(2);

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3);

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(1_999));
    expect(FakeWebSocket.instances).toHaveLength(3);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(4);

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(3_999));
    expect(FakeWebSocket.instances).toHaveLength(4);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(5);

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(8_000));
    expect(FakeWebSocket.instances).toHaveLength(6);

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(8_000));
    expect(FakeWebSocket.instances).toHaveLength(7);
  });

  it("closes the socket and cancels the reconnect timer on unmount", () => {
    vi.useFakeTimers();
    const { unmount } = renderHook(() => useSimulation());
    const socket = latest();

    unmount();

    expect(socket.readyState).toBe(FakeWebSocket.CLOSED);
    act(() => vi.advanceTimersByTime(60_000));
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("resets the backoff only after a valid message", () => {
    vi.useFakeTimers();
    renderHook(() => useSimulation());

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(500));
    expect(FakeWebSocket.instances).toHaveLength(2);

    act(() => latest().serverOpen());
    act(() => latest().serverMessage(status(null)));
    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(500));
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("keeps backing off when the server closes a bare open", () => {
    vi.useFakeTimers();
    renderHook(() => useSimulation());

    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(500));
    expect(FakeWebSocket.instances).toHaveLength(2);

    // Open, no frame, close: the previous failure still counts.
    act(() => latest().serverOpen());
    act(() => latest().serverClose());
    act(() => vi.advanceTimersByTime(999));
    expect(FakeWebSocket.instances).toHaveLength(2);
    act(() => vi.advanceTimersByTime(1));
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("does not reconnect after a finished episode", () => {
    vi.useFakeTimers();
    renderHook(() => useSimulation());
    const socket = latest();

    act(() => socket.serverOpen());
    act(() => {
      socket.serverMessage(status("sess-A"));
      socket.serverMessage({ seq: 1, type: "episode_end", payload: { reason: "finished" } });
    });

    act(() => socket.serverClose());
    act(() => vi.advanceTimersByTime(60_000));

    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});

describe("useSimulation — commands", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("starts the sim through the api and clears a previous error", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "no data" }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ session_id: "aaaaaaaa" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.start(1);
    });
    expect(result.current.error).toBe("no data");

    await act(async () => {
      await result.current.start(100);
    });

    const [url, init] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
    expect(url).toBe(`${API_BASE}/api/sim/start`);
    expect(JSON.parse(String(init.body))).toEqual({
      speed: 100,
      mode: "replay",
    } satisfies SimStartRequest);
    expect(result.current.error).toBeNull();
  });

  it("surfaces a start failure as error instead of throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "no data" }), { status: 409 })),
    );
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await expect(result.current.start(1)).resolves.toBeUndefined();
    });

    await waitFor(() => expect(result.current.error).toBe("no data"));
  });

  it("sends control actions through the api", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.control("pause");
    });

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`${API_BASE}/api/sim/control`);
    expect(JSON.parse(String(init.body))).toEqual({ action: "pause" } satisfies SimControl);
  });

  it("surfaces a control failure as error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "no run" }), { status: 409 })),
    );
    const { result } = renderHook(() => useSimulation());

    await act(async () => {
      await result.current.control("resume");
    });

    await waitFor(() => expect(result.current.error).toBe("no run"));
  });
});
