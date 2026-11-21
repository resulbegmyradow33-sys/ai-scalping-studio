import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, isSimMessage } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("returns parsed json on 200", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ status: "ok" }), { status: 200 })),
    );
    await expect(api.getHealth()).resolves.toEqual({ status: "ok" });
  });

  it("throws ApiError with server detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "no data" }), { status: 409 })),
    );
    await expect(api.startSim({ speed: 1, mode: "replay" })).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      detail: "no data",
    });
  });

  it("maps network failure to ApiError 0", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("fetch failed");
      }),
    );
    const err = await api.getHealth().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
  });

  it("maps a non-json error body to a generic ApiError, not a parse crash", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response("<html><body>boom</body></html>", {
            status: 500,
            headers: { "content-type": "text/html" },
          }),
      ),
    );
    const err = await api.getHealth().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(500);
    expect(err.detail).toBe("HTTP 500");
  });

  it("gives downloads 30 s and every other call 5 s", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ rows: 1 }), { status: 200 })),
    );
    const timeout = vi.spyOn(AbortSignal, "timeout");

    await api.downloadData({
      provider: "synthetic",
      symbol: "XAUUSD",
      start: "2025-01-01T00:00:00Z",
      end: "2025-01-02T00:00:00Z",
    });
    await api.getHealth();

    expect(timeout).toHaveBeenNthCalledWith(1, 30_000);
    expect(timeout).toHaveBeenNthCalledWith(2, 5_000);
  });

  it("maps an aborted request to a timeout ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new DOMException("The operation was aborted.", "AbortError");
      }),
    );
    await expect(api.getHealth()).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
      detail: "request timed out",
    });
  });

  it("joins FastAPI validation messages from an array detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              detail: [
                { loc: ["body", "speed"], msg: "Input should be >= 0", type: "greater_than_equal" },
                { loc: ["body", "mode"], msg: "Input should be 'replay'", type: "literal_error" },
              ],
            }),
            { status: 422 },
          ),
      ),
    );
    await expect(api.startSim({ speed: -1, mode: "replay" })).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      detail: "Input should be >= 0; Input should be 'replay'",
    });
  });

  it("resolves an empty success body instead of throwing SyntaxError", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 204 })));
    await expect(api.getHealth()).resolves.toBeUndefined();
  });
});

describe("isSimMessage", () => {
  const valid = { seq: 1, type: "bar", payload: { ts: "2025-01-01T00:00:00Z", open: 1, high: 2, low: 0.5, close: 1.5 } };

  it("accepts a known frame with a non-negative integer seq", () => {
    expect(isSimMessage(valid)).toBe(true);
    expect(isSimMessage({ seq: 0, type: "status", payload: { state: "idle", symbol: "XAUUSD", session_id: null, head_seq: 0 } })).toBe(true);
  });

  it("rejects an unknown frame type", () => {
    expect(isSimMessage({ seq: 1, type: "heartbeat", payload: {} })).toBe(false);
  });

  it("rejects a negative or fractional seq", () => {
    expect(isSimMessage({ ...valid, seq: -1 })).toBe(false);
    expect(isSimMessage({ ...valid, seq: 1.5 })).toBe(false);
  });
});
