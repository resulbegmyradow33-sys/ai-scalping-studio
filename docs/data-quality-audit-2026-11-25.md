# Data quality audit — 2026-11-25

Read-only audit of every `<symbol>/<timeframe>` slice present in the four local
data roots used by the research loop. No partition was modified, repaired or
backfilled; no raw rows are copied into this document or its JSON companion.

- Branch: `feat/multiqa`, base `main` `06c853c`
- Evidence: `docs/evidence/data-quality-2026-11-25.json` (per-slice aggregates)
- Scope: `data-ticks5-store`, `data-real-store`, `data-flow-store`,
  `data-forward-store` under `.worktrees/v1/`
- Method: streaming read of each parquet file, additive structural counts, then
  full-slice duplicate / gap / outlier aggregation from `ts`, `close`, `volume`.
  Validator semantics follow `scalping.data.validate.validate_bars` and
  `scalping.data.quality` (`OutlierConfig` defaults); gap threshold is
  `max(10, one bar width)` exactly as `/api/data/quality` computes it.
- Repro command (from `backend/`):

  ```bash
  uv run python <scanner> docs/evidence/data-quality-2026-11-25.json
  ```

  The scanner lives outside the repo (temp dir, not committed, per the
  no-code-change constraint) and calls only read APIs; nothing calls `write()`
  or `save_provenance()`. Newest file mtime in all four roots is 2026-11-22;
  the audit ran 2026-11-25, so the stores were not touched.
  Cost: ~12 s wall, ~760 MB peak RSS for 11.5 M rows / 1,309 parquet files.
- Status vocabulary maps the validator: `fail` = first error code, `warn` =
  first warning code, `ok` = no issue and a provenance sidecar exists.

## Headline

Nothing passes clean. Of 15 slices, **0 ok, 9 warn, 6 fail** — 11,502,309
physical rows across 1,309 partition files, 0 unreadable partitions.

1. Every `data-real-store` slice fails on `null_values`: `ask_close` and
   `spread` are null for 100 % of rows (and `volume` is all zero on all five
   timeframes). The documented import policy allows missing quotes, but the
   validator treats those nulls as errors, so 17 years of the only long history
   is permanently red at the quality gate.
2. `data-flow-store` is not a bar store and is invisible to `ParquetStore`:
   `/api/data/quality` sees 0 rows, while `flow_series()` sees 1,785,951 rows
   with 12,620 duplicate timestamps. Two readers of the same root disagree.
3. Intraday gaps are material and unclassified: ticks5 M1 has 1,037 in-session
   gaps (≈73.9 k missing minutes, 4.0 %), real M1 has 4,449 (≈287 k, 4.7 %).
   The shipped threshold also flags every D1 weekend, so D1 can never be `ok`.
4. Provenance is partly absent: 9 of 15 slices have no sidecar (all of
   `data-flow-store`, plus D1/M15 in ticks5 and D1/M15/M5 in real/forward), and
   the two M1 histories are not independent — a single-month spot check shows
   identical closes.
5. Coverage is single-symbol and stale in different places: only XAUUSD (1 of
   10 registered symbols); real ends 2026-01-09, ticks5 2026-05-22, forward
   2026-11-15 (9.2 days old on audit date). No L2/order-book data exists.

## Thresholds

| Check | Value |
|---|---|
| Gap tolerance | `max(10 min, one bar width)` → D1 1440 min, H1 60, M15 15, M5/M1 10 |
| Gap classification | `session_for("XAUUSD").is_open(gap midpoint)` separates session-closed (weekend) from in-session holes |
| Price outlier | robust z of log close returns ≥ 8, `min_rows` 30 |
| Volume outlier | robust z of `log1p(volume)` ≥ 8, `min_rows` 30 |
| Duplicate | duplicate `ts` across the whole slice |
| Structural | OHLC bracket, crossed quotes, negative spread/volume, null cells, non-finite prices, non-positive prices |

## data-ticks5-store

244 partitions, 1,922,744 rows. Provider `ticks5` on all rows. Status: 4 warn.

| TF | Rows | First → Last | Dup | Gaps in/out | Missing in-session bars | OHLC/crossed/neg spread/neg vol | Nulls | Outliers ret/vol | Sidecar | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 1,555 | 2021-05-24 → 2026-05-22 | 0 | 6/254 | 12 | 0/0/0/0 | 0 | 4/33 | no | warn/time_gaps |
| H1 | 29,586 | 2021-05-24 → 2026-05-22 | 0 | 1,031/260 | 1,213 | 0/0/0/0 | 0 | 159/0 | yes | warn/time_gaps |
| M1 | 1,773,331 | 2021-05-24 → 2026-05-22 | 0 | 1,037/260 | 73,937 | 0/0/0/0 | 0 | 9,786/0 | yes | warn/time_gaps |
| M15 | 118,272 | 2021-05-24 → 2026-05-22 | 0 | 1,033/260 | 4,919 | 0/0/0/0 | 0 | 630/0 | no | warn/time_gaps |

- Structure is clean: no duplicates, no OHLC violations, no crossed quotes, no
  negative spread/volume, no nulls, no non-positive prices, spacing matches the
  label (1/15/60/1440 min). Spread median ≈ 0.38–0.39 price units.
- Coverage: 99.36 % of session days; minute coverage 96.0 % on M1/H1/M15,
  99.2 % on D1. `last_ts` is 186–187 days old on the audit date.
- D1 warning is entirely weekends: threshold 1440 min, weekend gap 2880–4320 min
  → 260 gaps, 254 of them session-closed. D1 is warn-by-construction.
- Sidecars: H1/M1 exist and declared rows/first/last match exactly; D1 and M15
  have none (`provenance_missing`).
- M1 outliers: 9,786 returns (0.55 %) exceed 8 robust sigma, where the scale is
  `1.4826·MAD ≈ 0.000167`, i.e. a ≈0.13 % move. Only 126 sit on a gap boundary,
  so these are real large M1 moves, not gap artefacts — the check is just noisy
  at M1, not evidence of corruption.

## data-real-store

1,015 partitions, 7,578,294 rows. Providers: M1 `csv_eet`, every other TF
`resampled`. Status: 5 fail.

| TF | Rows | First → Last | Dup | Gaps in/out | Missing in-session bars | OHLC/crossed/neg spread/neg vol | Nulls | Outliers ret/vol | Sidecar | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 5,230 | 2009-03-15 → 2026-01-09 | 0 | 20/859 | 38 | 0/0/0/0 | 10,460 | 7/0 | no | fail/null_values |
| H1 | 100,083 | 2009-03-15 → 2026-01-09 | 0 | 2,725/877 | 3,578 | 0/0/0/0 | 200,166 | 483/0 | yes | fail/null_values |
| M1 | 5,888,800 | 2009-03-15 → 2026-01-09 | 0 | 4,449/887 | 287,137 | 0/0/0/0 | 11,777,600 | 33,991/0 | yes | fail/null_values |
| M15 | 396,855 | 2009-03-15 → 2026-01-09 | 0 | 4,111/882 | 17,698 | 0/0/0/0 | 793,710 | 1,885/0 | no | fail/null_values |
| M5 | 1,187,326 | 2009-03-15 → 2026-01-09 | 0 | 4,373/886 | 55,902 | 0/0/0/0 | 2,374,652 | 5,766/0 | no | fail/null_values |

- Every slice: `ask_close` and `spread` null on all rows (`nulls = 2 × rows`),
  `volume` zero on all rows (`zero_volume_pct = 1.0`), hence
  `capabilities_from_frame` reports `bar_volume=False, real_quotes=False` on a
  2024-01 M1 month, versus `True/True` on the same month from ticks5. No other
  structural failure: duplicates 0, OHLC 0, crossed 0, negative 0.
- The API validator also counts all rows as `non_finite_prices` because
  `is_finite(null)` yields null and the filter drops it; the null is the real
  defect. This audit records `non_finite_prices=0` and names nulls explicitly.
- Coverage: 99.28 % of session days; minute coverage 95.0–96.6 %, D1 99.3 %.
  `last_ts` 319–320 days old on the audit date.
- Sidecars exist for H1/M1 only (rows/first/last match); D1/M15/M5 have none.
- The `csv_eet` label does not mean the stored timestamps are shifted: at
  `ts + 0` the 2024-01 M1 closes match ticks5 exactly; +60/+120/+180 min and
  negative offsets all diverge (median |Δclose| 1.5–2.9 at ±60–180 min).

## data-flow-store

Flat layout `XAUUSD/M1/part-*.parquet`, 15 files, 1,785,951 physical rows,
1,773,331 unique timestamps. Schema `ts, sell_vol, buy_vol, flow` — a research
side table, not `BAR_SCHEMA`. Status: 1 fail.

| TF | Rows | First → Last | Dup | Gaps in/out | Missing in-session bars | Nulls/neg vol/|flow|>1 | Outliers ret | Sidecar | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 1,785,951 (1,773,331 unique) | 2021-05-24 → 2026-05-22 | 12,620 | 1,037/260 | 73,937 | 0/0/0 | 0 | no | fail/duplicate_timestamps |

- 10 file-span overlaps. Four are one-timestamp seams (`part-0024`, `-0060`,
  `-0096`, `-0108` start exactly on the previous end); `part-final.parquet`
  re-covers 2024-06-02 → 2026-05-22 after `part-0108…0168`, which is where the
  12,620 duplicates come from.
- Consumers: `flow_series()` concatenates and `.unique("ts")` keeps the **first**
  row, so the older part wins over the recomputed tail; the duplicate is hidden,
  not resolved. `ParquetStore.read("XAUUSD","M1")` returns 0 rows because
  `_partitions` only accepts `YYYY/MM.parquet`, so `/api/data/quality` and
  `/api/data/status` report this store as empty.
- Values are sane: `flow ∈ [-0.9757, 0.9768]`, median 0.0, no zero-total-minutes,
  no nulls or negatives. Coverage: 95.2 % of session minutes by the hourly
  estimate, 96.0 % by unique minutes vs missing bars — the same calendar holes
  as ticks5 M1 (identical gap counts), i.e. derived from the same tick archive.
- No `_meta` sidecar; source is only implicit in the script that wrote it.

## data-forward-store

35 partitions, 215,320 rows. Provider `dukascopy-2026` on all rows. Status: 5 warn.

| TF | Rows | First → Last | Dup | Gaps in/out | Missing in-session bars | OHLC/crossed/neg spread/neg vol | Nulls | Outliers ret/vol | Sidecar | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 147 | 2026-05-28 → 2026-11-15 | 0 | 0/25 | 0 | 0/0/0/0 | 0 | 0/9 | no | warn/time_gaps |
| H1 | 2,795 | 2026-05-28 → 2026-11-15 | 0 | 97/25 | 99 | 0/0/0/0 | 0 | 2/0 | yes | warn/time_gaps |
| M1 | 167,666 | 2026-05-28 → 2026-11-15 | 0 | 97/25 | 5,971 | 0/0/0/0 | 0 | 286/0 | yes | warn/time_gaps |
| M15 | 11,178 | 2026-05-28 → 2026-11-15 | 0 | 97/25 | 398 | 0/0/0/0 | 0 | 22/0 | no | warn/time_gaps |
| M5 | 33,534 | 2026-05-28 → 2026-11-15 | 0 | 97/25 | 1,194 | 0/0/0/0 | 0 | 59/0 | no | warn/time_gaps |

- Clean structurally (0 duplicates/OHLC/crossed/negative/null/non-positive),
  spacing matches the label, spread median ≈ 0.605–0.650.
- D1 warning is the 25 weekend gaps (0 in-session, 0 missing bars) — again
  warn-by-construction at the shipped threshold.
- 97 in-session gaps across ~6 months; largest 3,182–3,240 min (holidays).
  Minute coverage 96.6 % on intraday frames, 100 % on D1.
- Freshness: 9.2–10.2 days since `last_ts` on 2026-11-25. Sidecars only for
  H1/M1 (rows/first/last match); D1/M15/M5 have none.
- Provenance source is a third-party GitHub tick archive
  (`github:Touhid-HZ/xauusd-tick-data-2026`), not a broker feed.

## Cross-store observations

- Overlap is uneven, so multi-timeframe joins see different history depths:
  real 2009-03-15 → 2026-01-09, ticks5 2021-05-24 → 2026-05-22, forward
  2026-05-28 → 2026-11-15. Forward starts 6 days after ticks5 ends; real ends
  4.5 months before ticks5.
- Spot check (labelled, not a reconciliation): real M1 and ticks5 M1 for
  2024-01 join on 30,028 minutes with median and p90 |Δclose| = 0.0, and both
  report the same maximum |log return| 0.0268557. The two stores are not
  independent evidence; at least one is derived from the other or from a shared
  archive.
- Flow gap counts equal ticks5 M1 gap counts exactly (1,037/260, 73,937 missing
  minutes), confirming the flow table is derived from the same ticks.
- API parity spot check: `validate_bars` on real 2024-01 M1 returns
  `null_values` (60,056) + `non_finite_prices` (30,028) + `time_gaps` (22) +
  `price_outliers` (67); on ticks5 it returns only `time_gaps` (22) +
  `price_outliers` (66). This audit's slice-level numbers use the same counts,
  aggregated over the whole slice.

## Failed partitions

None. All 1,309 parquet files opened and parsed; `partitions_failed = 0` in all
15 slices. The scanner records per-file exceptions with `type: message` and
continues; no partition triggered that path, and no error was suppressed.

## Top 5 data risks for the autonomous loop

1. **The only 17-year history fails the quality gate and cannot support volume
   or real-quote features.** Every real-store slice is `fail/null_values`; volume
   is all zero and `ask_close`/`spread` are all null. Any manifest that selects
   `volume_*` or real quotes from this store is describing unsupported features.
   Follow-up: decide the null-quote contract (import fills a modeled spread and
   records it, or `validate_bars` treats null quote cells as a warning), expose
   `capabilities_from_frame` in `/api/data/quality`, and add a regression test
   with a bid-only frame.
2. **The flow channel is split between two readers and hides duplicates.**
   `ParquetStore` returns 0 rows for `data-flow-store`, while `flow_series()`
   reads 1,785,951 rows with 12,620 duplicate timestamps and keeps the stale
   first occurrence. A caller that checks `/api/data/quality` sees "store empty";
   a caller that builds features sees data.
   Follow-up: one reader per store (monthly partitions written through
   `ParquetStore`, or an explicit FlowStore), dedupe at write time, write a
   `_meta` sidecar, and make `/api/data/status` report "not a bar store" instead
   of `store_empty`.
3. **Gap and outlier policy is undefined and produces permanent warnings.**
   1,037 in-session M1 gaps in ticks5 and 4,449 in real accumulate to 4.0–4.7 %
   of session minutes, while D1 always warns on weekends (threshold = one bar
   width) and M1 always warns on ~0.55 % of returns (8σ ≈ 0.13 % move).
   No document says whether bars across a gap are dropped, imputed, or joined.
   Follow-up: separate session-closed gaps from in-session holes in the report,
   add a missing-minute metric and a gap policy per timeframe, and calibrate the
   outlier threshold to tick size or exclude gap boundaries.
4. **Provenance and independence are unproven for most slices.** 9/15 slices
   have no sidecar, `resampled` does not name its parent dataset, the 2026
   forward window comes from a third-party GitHub archive, and ticks5/real M1
   are identical on the 2024-01 overlap. The loop cannot currently tell
   independent evidence from the same data copied twice.
   Follow-up: require a sidecar (source, synthetic flag, checksum, parent
   dataset id) for every slice used in search or promotion, verify the checksum
   in the audit, and label derived stores as derived in `DatasetProvenance`.
5. **Coverage is single-symbol and freshness has no SLA.** Only XAUUSD exists
   in these four roots (1 of 10 registered symbols), there is no L2/depth data,
   and store tails differ by months (real 320 days stale, ticks5 186 days,
   forward 9 days). A loop that assumes multi-symbol or fresh data will
   silently train and paper-trade on the wrong window.
   Follow-up: add a per-store freshness SLA with alerting, a symbol-coverage
   gate before any multi-symbol claim, and an explicit "XAUUSD only, no L2"
   statement in the data contract.

## Honest limits

- **No second-source reconciliation.** Only the labelled 2024-01 M1 spot check
  was run; no row-by-row comparison of ticks5, real, forward or flow, no
  tolerance-based reconciliation report.
- **No L2 / order-book data.** These roots contain bars and a tick-volume
  imbalance side table; there is no depth, no full tick store (tick archives
  exist outside these roots), and "flow" is a proxy, not signed trade volume.
- **Checksums were not recomputed.** Sidecars were compared on declared rows and
  first/last timestamps only; the `checksum` field is unverified.
- **Outlier/gap/coverage numbers are retrospective descriptors**, not features
  and not promotion evidence. No model was trained or evaluated here.
- **`data-rich-store`** sits beside these roots but was outside the assigned
  store list; it was not scanned.
- **No code changes.** The scanner is an external read-only script; no validator,
  store or pipeline file was edited to make a partition pass. No task-board
  update and no push, per the assignment.
