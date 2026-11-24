"use client";

import useSWR from "swr";

import { api } from "@/lib/api";
import type { DataQuality } from "@/lib/types";

/**
 * Quality report for one slice. The backend validates the whole frame, so this
 * does not poll: the key changes only with the symbol, timeframe or outlier
 * switch, and a fresh report follows a download by remount or refetch.
 */
export function useDataQuality(
  symbol: string,
  timeframe: string,
  includeOutliers = true,
) {
  return useSWR<DataQuality>(
    symbol ? `api/data/quality:${symbol}:${timeframe}:${includeOutliers}` : null,
    () => api.getDataQuality(symbol, timeframe, includeOutliers),
    { refreshInterval: 0 },
  );
}
