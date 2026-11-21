import { cleanup, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { SWRConfig } from "swr";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useDataStatus } from "./use-data-status";
import { useHealth } from "./use-health";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function wrapper({ children }: { children: ReactNode }) {
  return (
    <SWRConfig
      value={{ provider: () => new Map(), dedupingInterval: 0, errorRetryCount: 0 }}
    >
      {children}
    </SWRConfig>
  );
}

function backendDown() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw new TypeError("fetch failed");
    }),
  );
}

describe("backend status hooks", () => {
  it("surfaces a health failure in error and leaves data undefined", async () => {
    backendDown();
    const { result } = renderHook(() => useHealth(), { wrapper });

    await waitFor(() => expect(result.current.error).toBeInstanceOf(Error));
    expect(result.current.data).toBeUndefined();
  });

  it("surfaces a data status failure in error and leaves data undefined", async () => {
    backendDown();
    const { result } = renderHook(() => useDataStatus("XAUUSD"), { wrapper });

    await waitFor(() => expect(result.current.error).toBeInstanceOf(Error));
    expect(result.current.data).toBeUndefined();
  });

  it("does not fetch a data status without a symbol", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    renderHook(() => useDataStatus(""), { wrapper });

    expect(fetchMock).not.toHaveBeenCalled();
  });
});
