/**
 * Sequence handling for the simulator socket.
 *
 * The backend numbers every event of a session: `status` is `seq: 0` and each
 * later event increments from 1. A reconnect can replay from a cursor, and a
 * start/reset rewinds to 0 under a new `session_id`. The gate decides which
 * frames are worth applying and from where a replay should start.
 */

export interface SequenceGate {
  /**
   * Accept a frame. Returns false for duplicates, for frames already covered
   * by the contiguous cursor, and for frames behind an accepted hole (those
   * must arrive through replay).
   */
  accept(seq: number): boolean;
  /** Rewind to a new session's head; every earlier sequence is forgotten. */
  reset(headSeq: number): void;
  /** True while an accepted frame sits above a hole in the sequence. */
  readonly gap: boolean;
  /** Sequence the next replay frame should start after (`lastSeen + 1`). */
  readonly resumeFrom: number;
  /** Highest contiguous sequence applied so far; `-1` before the first. */
  readonly lastSeen: number;
}

export function createSequenceGate(): SequenceGate {
  // `lastSeen` is the contiguous frontier. `highest` is the maximum accepted
  // sequence and may run ahead of the frontier when frames arrive out of
  // order; `pending` holds exactly those ahead-of-frontier sequences.
  let lastSeen = -1;
  let highest = -1;
  const pending = new Set<number>();

  function advance(): void {
    while (pending.delete(lastSeen + 1)) lastSeen += 1;
  }

  return {
    accept(seq: number): boolean {
      if (!Number.isInteger(seq) || seq < 0) return false;

      // Extends the contiguous run, including a frame that fills the first
      // hole of a gap (2 after 5 in `0, 1, 5, 2`).
      if (seq === lastSeen + 1) {
        lastSeen = seq;
        if (seq > highest) highest = seq;
        advance();
        return true;
      }

      // A new maximum is out of order but still useful: keep it pending until
      // the replay (or later frames) closes the hole.
      if (seq > highest) {
        highest = seq;
        pending.add(seq);
        return true;
      }

      // Duplicate, already applied, or an old hole that must come from replay.
      return false;
    },

    reset(headSeq: number): void {
      const head = Number.isInteger(headSeq) && headSeq > 0 ? headSeq : 0;
      lastSeen = head;
      highest = head;
      pending.clear();
    },

    get gap(): boolean {
      return pending.size > 0;
    },

    get resumeFrom(): number {
      return lastSeen + 1;
    },

    get lastSeen(): number {
      return lastSeen;
    },
  };
}

/** `http(s)://host` -> `ws(s)://host/ws/sim`, keeping any API subpath out. */
export function socketUrl(apiBase: string): string {
  const url = new URL("/ws/sim", apiBase);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
