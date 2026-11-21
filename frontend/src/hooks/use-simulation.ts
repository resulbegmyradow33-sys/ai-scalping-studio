"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

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

/** HTTP status of a failed call; `0` means no response, null means unknown. */
function failureStatus(error: unknown): number | null {
  return error instanceof ApiError ? error.status : null;
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

  // Shared with `start`: a session adopted from the API must rewind the gate
  // the socket handler is using. `useState` gives one stable gate per hook.
  const [gate] = useState(createSequenceGate);
  const lastResumeRef = useRef(Number.NaN);
  const socketRef = useRef<WebSocket | null>(null);
  const connectRef = useRef<() => void>(() => {});

  // The backend pump can emit the new session's first frames before the HTTP
  // start response resolves. Hold those raw frames until the session is
  // adopted, then replay them contiguously instead of gapping and duplicating.
  const startingRef = useRef(false);
  const bufferRef = useRef<unknown[]>([]);
  const flushBufferRef = useRef<() => void>(() => {});

  useEffect(() => {
    if (typeof WebSocket === "undefined") return;

    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;
    let disposed = false;

    const sendResume = (target: WebSocket, seq: number): void => {
      // The backend's reader rejects negative cursors outright.
      if (seq < 0) return;
      if (target.readyState !== WebSocket.OPEN) return;
      target.send(JSON.stringify({ type: "resume", seq }));
      lastResumeRef.current = seq;
    };

    const handleMessage = (target: WebSocket, message: SimMessage): void => {
      if (message.type === "status") {
        // `status` is seq 0 on every connection and must bypass the gate, or a
        // reconnect would be rejected as stale. A new session (including a
        // null session after a known one: the server restarted) resets the
        // cursor to the advertised head. A still-unseeded gate is seeded here
        // too, so the first data frame is contiguous instead of a gap.
        const incoming = message.payload.session_id;
        if (isNewSession(snapshotRef.current, message)) {
          gate.reset(incoming === null ? 0 : message.payload.head_seq);
          lastResumeRef.current = Number.NaN;
        } else if (gate.lastSeen < 0) {
          gate.reset(incoming === null ? 0 : message.payload.head_seq);
        }
        dispatch({ type: "message", message });
        return;
      }

      if (!gate.accept(message.seq)) return;
      dispatch({ type: "message", message });

      // Out-of-order frame: ask for exactly the hole, once per cursor.
      if (gate.gap) {
        const resumeSeq = gate.resumeFrom - 1;
        if (resumeSeq !== lastResumeRef.current) sendResume(target, resumeSeq);
      }
    };

    const applyRaw = (target: WebSocket, raw: unknown): void => {
      if (!isSimMessage(raw)) return;
      // A server that opens and closes immediately never resets the
      // backoff: only a real frame proves the connection is healthy.
      attempt = 0;
      if (startingRef.current) {
        // The live status path and the adoption own the epoch during a start;
        // replaying a pre-start status would revert it. Never buffer status.
        if (raw.type !== "status") bufferRef.current.push(raw);
        return;
      }
      handleMessage(target, raw);
    };

    const flushBuffer = (): void => {
      const frames = bufferRef.current;
      bufferRef.current = [];
      const target = socket;
      if (!target) return;
      for (const frame of frames) {
        if (isSimMessage(frame)) handleMessage(target, frame);
      }
    };
    flushBufferRef.current = flushBuffer;

    const connect = (): void => {
      if (disposed) return;

      const ws = new WebSocket(socketUrl(API_BASE));
      socket = ws;
      socketRef.current = ws;

      ws.onopen = () => {
        // Only a valid cursor is worth sending; an unseeded gate sends none.
        if (gate.resumeFrom > 0) sendResume(ws, gate.resumeFrom - 1);
      };

      ws.onmessage = (event: MessageEvent) => {
        let raw: unknown;
        try {
          raw = JSON.parse(String(event.data));
        } catch {
          return;
        }
        applyRaw(ws, raw);
      };

      ws.onclose = () => {
        if (disposed || socket !== ws) return;
        // A finished episode is a terminal state for this socket; retrying
        // every 8 s forever would be noise. `start` reconnects explicitly.
        if (snapshotRef.current.status === "finished") return;
        const delay = backoffDelay(attempt);
        attempt += 1;
        timer = setTimeout(connect, delay);
      };
    };

    connectRef.current = connect;
    connect();

    return () => {
      disposed = true;
      connectRef.current = () => {};
      flushBufferRef.current = () => {};
      if (timer !== null) clearTimeout(timer);
      const ws = socket;
      socket = null;
      socketRef.current = null;
      if (ws) {
        // Drop handlers first: closing must not schedule another reconnect.
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
      }
    };
  }, [dispatch, gate]);

  const start = useCallback(
    async (speed: number): Promise<void> => {
      // Defensive: the UI disables start while one is in flight. A second
      // call must not clear the buffer the first one is filling.
      if (startingRef.current) return;

      dispatch({ type: "clearError" });
      const wasFinished = snapshotRef.current.status === "finished";

      // Frames for a session that does not exist yet must not reach the view:
      // the pump may start before this promise resolves.
      startingRef.current = true;
      bufferRef.current = [];

      try {
        const { session_id } = await api.startSim({ speed, mode: "replay" });
        // The backend does not announce a start over the socket; adopt the
        // session it returned and rewind the gate to the new stream's head.
        dispatch({ type: "sessionAdopted", sessionId: session_id });
        gate.reset(0);
        lastResumeRef.current = Number.NaN;
        startingRef.current = false;
        flushBufferRef.current();
        if (
          wasFinished &&
          typeof WebSocket !== "undefined" &&
          socketRef.current?.readyState !== WebSocket.OPEN
        ) {
          connectRef.current();
        }
      } catch (error) {
        bufferRef.current = [];
        startingRef.current = false;
        dispatch({ type: "error", error: failureMessage(error), status: failureStatus(error) });
      }
    },
    [dispatch, gate],
  );

  const control = useCallback(
    async (action: ControlAction): Promise<void> => {
      dispatch({ type: "clearError" });
      try {
        await api.control({ action });
      } catch (error) {
        dispatch({ type: "error", error: failureMessage(error), status: failureStatus(error) });
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
    equityAnchor: snapshot.equityAnchor,
    trades: snapshot.trades,
    lastDecision: snapshot.lastDecision,
    riskBlocked: snapshot.riskBlocked,
    error: snapshot.error,
    errorStatus: snapshot.errorStatus,
    start,
    control,
  };
}
