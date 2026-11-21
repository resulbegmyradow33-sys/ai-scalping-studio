"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import { API_BASE, ApiError, api, isSimMessage } from "@/lib/api";
import {
  initialSimSnapshot,
  isNewSession,
  simReducer,
  type SimAction,
} from "@/lib/sim-state";
import type { ControlAction, SimMessage } from "@/lib/types";
import { createSequenceGate, socketUrl } from "@/lib/ws";

const BACKOFF_BASE_MS = 500;
const BACKOFF_MAX_MS = 8_000;

function backoffDelay(attempt: number): number {
  return Math.min(BACKOFF_BASE_MS * 2 ** attempt, BACKOFF_MAX_MS);
}

function failureMessage(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  return error instanceof Error && error.message ? error.message : "unknown error";
}

/**
 * Live simulator feed over the backend WebSocket.
 *
 * The socket is a thin shell: a sequence gate decides which frames to apply,
 * the pure reducer builds the view state, and gap detection asks the server to
 * replay from the last contiguous sequence. Reconnects back off to 8 s and stop
 * on unmount. Command failures land in `error`, never in React's error path.
 */
export function useSimulation() {
  const [snapshot, reactDispatch] = useReducer(simReducer, initialSimSnapshot);

  // The socket handler needs the latest state to detect a session epoch before
  // React has re-rendered. Keep a ref reduced in lockstep with the dispatch.
  const snapshotRef = useRef(snapshot);
  const dispatch = useCallback((action: SimAction) => {
    snapshotRef.current = simReducer(snapshotRef.current, action);
    reactDispatch(action);
  }, []);

  useEffect(() => {
    if (typeof WebSocket === "undefined") return;

    const gate = createSequenceGate();
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;
    let disposed = false;
    let lastResumeSent = Number.NaN;

    const sendResume = (target: WebSocket, seq: number): void => {
      if (target.readyState !== WebSocket.OPEN) return;
      target.send(JSON.stringify({ type: "resume", seq }));
      lastResumeSent = seq;
    };

    const handleMessage = (target: WebSocket, message: SimMessage): void => {
      if (message.type === "status") {
        // `status` is seq 0 on every connection and must bypass the gate, or a
        // reconnect would be rejected as stale. A new session resets it to
        // `head_seq` so the new episode's low numbers are accepted.
        if (isNewSession(snapshotRef.current, message)) {
          gate.reset(message.payload.head_seq);
          lastResumeSent = Number.NaN;
        }
        dispatch({ type: "message", message });
        return;
      }

      if (!gate.accept(message.seq)) return;
      dispatch({ type: "message", message });

      // Out-of-order frame: ask for exactly the hole, once per cursor.
      if (gate.gap) {
        const resumeSeq = gate.resumeFrom - 1;
        if (resumeSeq !== lastResumeSent) sendResume(target, resumeSeq);
      }
    };

    const connect = (): void => {
      if (disposed) return;

      const ws = new WebSocket(socketUrl(API_BASE));
      socket = ws;

      ws.onopen = () => {
        attempt = 0;
        sendResume(ws, gate.resumeFrom - 1);
      };

      ws.onmessage = (event: MessageEvent) => {
        let raw: unknown;
        try {
          raw = JSON.parse(String(event.data));
        } catch {
          return;
        }
        if (!isSimMessage(raw)) return;
        handleMessage(ws, raw);
      };

      ws.onclose = () => {
        if (disposed || socket !== ws) return;
        const delay = backoffDelay(attempt);
        attempt += 1;
        timer = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposed = true;
      if (timer !== null) clearTimeout(timer);
      const ws = socket;
      socket = null;
      if (ws) {
        // Drop handlers first: closing must not schedule another reconnect.
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
      }
    };
  }, [dispatch]);

  const start = useCallback(
    async (speed: number): Promise<void> => {
      dispatch({ type: "clearError" });
      try {
        await api.startSim({ speed, mode: "replay" });
      } catch (error) {
        dispatch({ type: "error", error: failureMessage(error) });
      }
    },
    [dispatch],
  );

  const control = useCallback(
    async (action: ControlAction): Promise<void> => {
      dispatch({ type: "clearError" });
      try {
        await api.control({ action });
      } catch (error) {
        dispatch({ type: "error", error: failureMessage(error) });
      }
    },
    [dispatch],
  );

  return {
    status: snapshot.status,
    seq: snapshot.seq,
    bars: snapshot.bars,
    position: snapshot.position,
    equity: snapshot.equity,
    trades: snapshot.trades,
    lastDecision: snapshot.lastDecision,
    riskBlocked: snapshot.riskBlocked,
    error: snapshot.error,
    start,
    control,
  };
}
