import { describe, expect, it } from "vitest";

import { createSequenceGate, socketUrl } from "./ws";

describe("sequence gate", () => {
  it("accepts the first message and rejects duplicates", () => {
    const g = createSequenceGate();
    expect(g.accept(0)).toBe(true);
    expect(g.accept(0)).toBe(false);
    expect(g.accept(1)).toBe(true);
  });
  it("accepts out-of-order but records a gap", () => {
    const g = createSequenceGate();
    g.accept(0);
    expect(g.accept(3)).toBe(true);
    expect(g.gap).toBe(true);
    expect(g.resumeFrom).toBe(1);
  });
  it("resumeFrom is last contiguous seq + 1", () => {
    const g = createSequenceGate();
    g.accept(0);
    g.accept(1);
    g.accept(5);
    g.accept(2);
    expect(g.resumeFrom).toBe(3);
  });
  it("ignores messages older than the window", () => {
    const g = createSequenceGate();
    g.accept(10);
    expect(g.accept(4)).toBe(false);
  });
});

describe("sequence gate reset (session epochs)", () => {
  it("rewinds the cursor to a new session head", () => {
    const g = createSequenceGate();
    g.accept(0);
    g.accept(1);

    g.reset(0);

    expect(g.gap).toBe(false);
    expect(g.resumeFrom).toBe(1);
    expect(g.accept(0)).toBe(false);
    expect(g.accept(1)).toBe(true);
  });

  it("drops the pending window and accepts above a non-zero head", () => {
    const g = createSequenceGate();
    g.accept(0);
    g.accept(9);

    g.reset(4);

    expect(g.gap).toBe(false);
    expect(g.resumeFrom).toBe(5);
    expect(g.accept(4)).toBe(false);
    expect(g.accept(5)).toBe(true);
  });
});

describe("socketUrl", () => {
  it("derives ws from the http api base", () => {
    expect(socketUrl("http://127.0.0.1:8000")).toBe("ws://127.0.0.1:8000/ws/sim");
  });

  it("derives wss from an https api base", () => {
    expect(socketUrl("https://api.example.com/")).toBe("wss://api.example.com/ws/sim");
  });
});
