"use client";

import { useCallback, useState } from "react";
import useSWR from "swr";

import { api } from "@/lib/api";
import type { CalibrateResponse, StrategyCard } from "@/lib/types";

const REFRESH_MS = 60_000;

/**
 * The live rule card plus the nightly recalibration action.
 *
 * The card is fetched like any other status resource; recalibration is an
 * explicit POST because it can take minutes and writes the promoted card the
 * next simulator start will replay.
 */
export function useStrategyCard() {
  const { data, error, isLoading, mutate } = useSWR<StrategyCard>(
    "api/strategy/card",
    api.getStrategyCard,
    { refreshInterval: REFRESH_MS },
  );
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<CalibrateResponse | null>(null);
  const [calibrateError, setCalibrateError] = useState<string | null>(null);

  const recalibrate = useCallback(async (): Promise<void> => {
    setRunning(true);
    setCalibrateError(null);
    try {
      const response = await api.calibrateStrategy({});
      setResult(response);
      await mutate(response.card, { revalidate: false });
    } catch (cause) {
      setCalibrateError(cause instanceof Error ? cause.message : "kalibrasyon başarısız");
    } finally {
      setRunning(false);
    }
  }, [mutate]);

  return { card: data, error, isLoading, running, result, calibrateError, recalibrate };
}
