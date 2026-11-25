"use client";

import useSWR from "swr";

import { api } from "@/lib/api";
import type { AutonomyChampionList } from "@/lib/types";

const REFRESH_MS = 60_000;

/**
 * Champions per scope, read-only. `readable: false` means the archive or its
 * champion table could not be read; it is not an empty champion list, so the
 * page must not render "no champion" for it.
 */
export function useAutonomyChampions() {
  return useSWR<AutonomyChampionList>("api/autonomy/champions", api.autonomyChampions, {
    refreshInterval: REFRESH_MS,
    shouldRetryOnError: false,
  });
}
