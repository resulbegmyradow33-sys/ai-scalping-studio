"use client";

import useSWR from "swr";

import { api } from "@/lib/api";
import type { DataStatus } from "@/lib/types";

const REFRESH_MS = 30_000;

/**
 * Coverage for one symbol. The key is null without a symbol, so SWR never
 * fetches `/api/data/status?symbol=`; a down backend surfaces in `error`.
 */
export function useDataStatus(symbol: string) {
  return useSWR<DataStatus>(
    symbol ? `api/data/status:${symbol}` : null,
    () => api.getDataStatus(symbol),
    { refreshInterval: REFRESH_MS },
  );
}
