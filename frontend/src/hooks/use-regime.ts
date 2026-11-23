"use client";

import useSWR from "swr";

import { api } from "@/lib/api";
import type { RegimeResponse, TimeframeSpec } from "@/lib/types";

const REFRESH_MS = 30_000;

/**
 * The ladder comes from `GET /api/timeframes` so the panel never hard-codes a
 * list the backend can change. A missing endpoint is not fatal: the page falls
 * back to M1 (the only timeframe the current store writes by default).
 */
export function useTimeframes() {
  return useSWR<TimeframeSpec[]>("api/timeframes", api.getTimeframes, {
    refreshInterval: 0,
    shouldRetryOnError: false,
  });
}

/**
 * Rule-based regime labels for one stored slice. The key is null without a
 * symbol, so an empty input never fetches `/api/regime?symbol=`.
 */
export function useRegime(symbol: string, timeframe: string) {
  return useSWR<RegimeResponse>(
    symbol ? `api/regime:${symbol}:${timeframe}` : null,
    () => api.getRegime(symbol, timeframe),
    { refreshInterval: REFRESH_MS },
  );
}
