import { describe, expectTypeOf, it } from "vitest";

import type { SimMessage, StatusPayload, TradeOpenPayload, TradeSummary } from "./types";

declare const message: SimMessage;

describe("SimMessage", () => {
  it("narrows the payload on a switch over type", () => {
    switch (message.type) {
      case "status": {
        expectTypeOf(message.payload).toEqualTypeOf<StatusPayload>();
        expectTypeOf(message.payload.head_seq).toEqualTypeOf<number>();
        break;
      }
      case "bar": {
        expectTypeOf(message.payload.close).toEqualTypeOf<number>();
        expectTypeOf(message.payload.ts).toEqualTypeOf<string>();
        break;
      }
      case "decision": {
        expectTypeOf(message.payload.action).toEqualTypeOf<0 | 1 | 2 | 3>();
        expectTypeOf(message.payload.blocked).toEqualTypeOf<string | null>();
        break;
      }
      case "trade_open": {
        expectTypeOf(message.payload).toEqualTypeOf<TradeOpenPayload>();
        expectTypeOf(message.payload.entry).toEqualTypeOf<number>();
        break;
      }
      case "trade_close": {
        expectTypeOf(message.payload).toEqualTypeOf<TradeSummary>();
        expectTypeOf(message.payload.r_multiple).toEqualTypeOf<number>();
        break;
      }
      case "equity": {
        expectTypeOf(message.payload.equity).toEqualTypeOf<number>();
        break;
      }
      case "risk_blocked": {
        expectTypeOf(message.payload.reason).toEqualTypeOf<string>();
        break;
      }
      case "episode_end": {
        expectTypeOf(message.payload.reason).toEqualTypeOf<string>();
        break;
      }
    }
  });
});
