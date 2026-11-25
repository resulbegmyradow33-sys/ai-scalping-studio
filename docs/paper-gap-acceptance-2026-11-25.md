# Paper gap policy — real-store acceptance — 2026-11-25

**Status: plumbing only. A fixed test-only DSL spec ran through
`ContinuousPaperRunner`; there is no evidence champion and no profitability
claim. Not validated, not paper-traded as a champion.**

Machine-readable record:
[`docs/evidence/paper-gaps-2026-11-25.json`](evidence/paper-gaps-2026-11-25.json).
Policy contract: [`paper-trading.md`](paper-trading.md#gap-and-ordering-policy).

## Question

Issue #141 added a gap policy to the continuous runner after PR #140 found that
`paper_advance` refused both real stores before the first bar. This run proves
the merged policy on the two exact gaps from #140:

- M1: `2021-05-24T04:04Z` missing before `04:05` (`data-ticks5-store`);
- H1: `2009-05-25T17:00Z` missing before `22:00` (`data-real-store`).

It answers four things with real bars: `session_aware` advances and audits,
`strict` still refuses with nothing written, no bar is invented and no future
row changes an earlier signal, and a session resumes across the gap.

## Inputs

| Item | M1 | H1 |
|---|---|---|
| Source store (read-only) | `.worktrees/v1/data-ticks5-store` | `.worktrees/v1/data-real-store` |
| Bounded slice | 2021-05-23T00:00Z → 2021-05-26T00:00Z | 2009-05-01T00:00Z → 2009-06-05T00:00Z |
| Bars in slice | 2,760 | 592 |
| Slice checksum | `404374d2e3c51644d703b9265a2fbe4d` | `d91ed3e8dec16036360a02912fec2e96` |
| Source checksum | `f60e1260f64a9b68c77f874d5b0addf1` | `e48cd48d6aa1cdfb8071a511f0a2464a` |
| Provenance | `ticks:data-ticks5`, synthetic `false` | `csv:data-real-csv`, synthetic `false` |

Slices were copied into a writable, gitignored root
(`backend/runs/paper-gaps-2026-11-25/`) because the runner builds a feature
cache under the store root. The source stores were read only; their checksums
above still match their sidecars.

### Spec

No production or evidence champion exists for `XAUUSD:M1` or `XAUUSD:H1`
(acceptance-attempt-2026-11-25 stops at `NoChampionError`). The run therefore
uses one explicit test-only spec:

```
entries: long when 1.0 > 0.0
exits:   never by rule
holding: HoldingLimit(5)
max_lookback: 2
hash: 9fe13e03e88263bd3d6809eb390a3bb23de24ce917b49743956332c32a923e77
```

It reads no feature column, so it isolates the execution and gap path. It is
not a candidate, not a champion and says nothing about edge. Settings are the
defaults (`settings_digest 2226685b488a…`), including `daily_loss_pct=0.03`.

## Commands

Everything ran from `backend/` on the base commit `06c853c`.

```bash
# Build the bounded slices and run every case in a fresh process.
uv run python runs/paper-gaps-2026-11-25/measure.py --driver

# One case on its own (writes .../measurements.json only in --driver mode).
uv run python runs/paper-gaps-2026-11-25/measure.py --case m1-session-aware
```

The harness lives in the gitignored `backend/runs/` directory and is not part of
this change (docs/JSON only). The appendix carries the minimal reproduction.

## Results

### `session_aware` — M1, 2,760 stored bars

| Metric | Value |
|---|---|
| Bars processed (decisions) | **2,640** |
| Warm-up before first decision | 120 bars (60-bar feature warm-up + 60-bar window) |
| Provisional tail (never decided) | 1 bar (`2021-05-26T00:00Z`) |
| Gaps skipped / recorded | **3 / 3** |
| Missing in-session bars | **121** |
| Signals | 2,640 |
| Orders (filled / rejected) | 2,638 (22 / 2,616) |
| Trades | 22 |
| Runtime (cold / median of 3) | 0.70 s / 0.58 s |
| Peak RSS | 116 MB |
| Ledger rows | sessions 1, signals 2,640, orders 2,638, fills 22, positions 22, trades 22, equity 2,640, gaps 3 |

Signal reasons: 22 `entry_long`, 2,606 `entry_long; blocked=daily_loss_limit`,
10 `entry_long; blocked=max_positions`, 2 `holding_limit`. The daily-loss gate
is why most entries were rejected; that is the risk engine working, not a gap
or a skipped bar.

Recorded gaps:

| from_ts | to_ts | missing_bars | classification |
|---|---|---|---|
| 2021-05-24T04:03Z | 2021-05-24T04:05Z | 1 | the #140 feed hole at 04:04 |
| 2021-05-24T20:59Z | 2021-05-24T22:00Z | 60 | 21:00–21:59 hour absent from the 2021 feed |
| 2021-05-25T20:59Z | 2021-05-25T22:00Z | 60 | 21:00–21:59 hour absent from the 2021 feed |

The XAUUSD registry session is `sun22:00-fri21:00 UTC` with no daily break, so
the absent 21:00 hour is a real missing bar by that contract and is recorded.
The same M1 store prints 21:00 bars in 2022 and 2025, so this is a property of
the 2021 slice, not a blanket daily close.

### `session_aware` — H1, 592 stored bars

| Metric | Value |
|---|---|
| Bars processed (decisions) | **472** |
| Warm-up before first decision | 120 bars |
| Provisional tail | 1 bar (`2009-06-05T00:00Z`) |
| Gaps skipped / recorded | **1 / 1** |
| Missing in-session bars | **5** |
| Signals | 472 |
| Orders (filled / rejected) | 425 (136 / 289) |
| Trades | 136 |
| Runtime (cold / median of 3) | 0.13 s / 0.12 s |
| Peak RSS | 107 MB |
| Ledger rows | sessions 1, signals 472, orders 425, fills 136, positions 136, trades 136, equity 472, gaps 1 |

Recorded gap: `2009-05-25T16:00Z → 22:00Z`, 5 missing bars (17:00–21:59), the
exact #140 hole. This feed prints 21:00 bars, so the daily hour is not a gap
here. Signal reasons: 136 `entry_long`, 281
`entry_long; blocked=max_positions`, 8 `entry_long; blocked=daily_loss_limit`,
47 `holding_limit`.

### `strict` — same slices, fresh ledgers

| Store | Exact refusal | Rows written |
|---|---|---|
| M1 | `BarGapError: session 'auto-XAUUSD-M1-9fe13e03e88263bd': missing bar at 2021-05-24T04:04:00+00:00 before 2021-05-24T04:05:00+00:00` | 0 in every table |
| H1 | `BarGapError: session 'auto-XAUUSD-H1-9fe13e03e88263bd': missing bar at 2009-05-25T17:00:00+00:00 before 2009-05-25T22:00:00+00:00` | 0 in every table |

The refusal happens before the feature build and before the session row:
`sessions = signals = orders = fills = positions = trades = equity_curve =
paper_gaps = 0`. Runtime 0.04 s (M1) and 0.01 s (H1), peak RSS 93 MB for both.

## Verification

**No synthetic bar.** The ledger stores no OHLCV at all. Every signal and equity
timestamp is a stored bar timestamp: `all_signal_ts_in_store = true`,
`all_equity_ts_in_store = true`, distinct signal timestamps equal the signal
count. Signals equal the number of stored bars in
`[first_signal_ts, last_signal_ts]` exactly (2,640 and 472), so no decision slot
was invented. Every instant inside a recorded gap is absent from the store
(`gap_instants_absent_from_store = true`), and the bounded store checksum still
matches its sidecar after the runs.

**No look-ahead.** Appending later bars never changed an earlier decision:

| Check | Result |
|---|---|
| M1: 480 signals, then append 600 bars (10:00→20:00), re-advance | prefix of 480 identical, 600 appended |
| H1: 278 signals, then append 194 bars (across the gap), re-advance | prefix of 278 identical, 194 appended |
| Fresh full-slice M1 run vs incremental run | first 2,640 signals identical |
| Fresh full-slice H1 run vs incremental run | first 472 signals identical |

Row identity covers `(signal_id, ts, action, reason, model_version)`. The
full-run comparison also proves the feature matrix is causal: an incremental
run that first saw half the slice produced the same signals as a run that saw
the whole slice at once.

**Resume across a gap.** Each resume used a new runner and reopened the same
ledger, so this is the process-restart path, not an in-memory continuation.

- M1: store ended at `04:00Z`, first advance wrote 121 signals and 0 gaps. The
  store was then extended past the hole. Second advance wrote the gap row
  (`04:03 → 04:05`, 1 bar) and 2,519 more signals, ending at 2,640 with the 121
  prefix rows unchanged.
- H1: store ended at `2009-05-25T16:00Z`, first advance wrote 278 signals and
  0 gaps. Second advance recorded the gap (`16:00 → 22:00`, 5 bars) and 194 more
  signals; the 278 prefix rows were unchanged.

The gap is recorded on the call that first sees it, before that call's signals,
matching the documented crash order.

**Regression.** No source file changed, and the merged policy still passes its
suite:

```bash
uv run pytest tests/test_paper_continuous.py tests/test_paper_autonomous.py
# 30 passed in 3.77s
```

The source stores were re-read after the runs; both checksums still equal their
sidecars and neither store has a `.feature-cache` directory.

## Limitations

- The spec is arbitrary and loses money here (M1 −919.56, H1 −641.07 net over
  the slice). That is a plumbing artifact, not an edge measurement; no OOS, cost
  stress or holdout was involved.
- Default risk settings. The M1 run spent most bars blocked by
  `daily_loss_limit`; results describe gap handling, not strategy behaviour.
- Gap counts are slice-local. M1 shows 3 rows because the 2021 feed lacks the
  21:00 hour that the same store prints in 2022/2025; a whole-store gap census
  was not run.
- One symbol, two short slices, one M2 machine, one process at a time.
- Peak RSS is per-process `ru_maxrss` including imports; runtime includes
  interpreter startup. Warm medians reuse the on-disk feature cache.
- `skip_with_report`, out-of-order and duplicate refusal were not re-measured
  here; the merged tests cover them.
- Nothing here was promoted, paper-traded as a champion, or connected to a
  broker.

## What this shows and does not show

**Shows:** on the two exact real gaps from #140, `session_aware` advances,
audits each missing instant in `paper_gaps`, writes no bar for it, and resumes
across it without changing earlier signals; `strict` still fails closed with an
empty ledger; and a fresh full-slice run reproduces the incremental run's
signals row for row.

**Does not show:** any edge, any champion, any profitability, or the gap rate of
the full datasets.

## Appendix — minimal reproduction

```python
from datetime import UTC, datetime
from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.paper import PaperLedger
from scalping.paper.continuous import ContinuousPaperRunner
from scalping.strategy.dsl import Compare, Const, EntryRule, ExitRule, HoldingLimit, StrategySpec

# 1. Bounded slice from the real store (inclusive range).
src = ParquetStore("../../../data-ticks5-store")
bars = src.read("XAUUSD", "M1", datetime(2021, 5, 23, tzinfo=UTC), datetime(2021, 5, 26, tzinfo=UTC))
ParquetStore("runs/gap-store").write("XAUUSD", "M1", bars)  # copy the provenance sidecar too

# 2. Fixed test spec (plumbing only).
spec = StrategySpec(
    entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
    exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
    holding=HoldingLimit(5),
    max_lookback=2,
)

# 3. One advance; swap "session_aware" for "strict" to see the refusal.
with PaperLedger("runs/gap.db") as ledger:
    runner = ContinuousPaperRunner(
        ledger, spec, "XAUUSD", "M1", Settings(),
        store_root="runs/gap-store", gap_policy="session_aware",
    )
    print(runner.advance())
```
