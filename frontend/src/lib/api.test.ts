import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./api";

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
});
