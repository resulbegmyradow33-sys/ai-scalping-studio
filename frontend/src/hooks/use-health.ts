"use client";

import useSWR from "swr";

import { api } from "@/lib/api";
import type { Health } from "@/lib/types";

const REFRESH_MS = 5_000;

/**
 * Backend liveness and version. A rejected fetch lands in `error` while `data`
 * stays undefined, so the shell can show an offline badge instead of throwing.
 */
export function useHealth() {
  return useSWR<Health>("api/health", api.getHealth, {
    refreshInterval: REFRESH_MS,
  });
}
