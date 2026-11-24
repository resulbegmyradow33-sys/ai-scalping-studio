# M2 measurements on real stores — 2026-11-24

Local performance measurement of the merged bench harness on this Mac mini M2.
Raw aggregates are in `docs/evidence/bench-2026-11-24.json`.

## Claims block — read this first

- These are **local measurements** of wall time, CPU time, process high-water
  RSS and disk footprint on one machine at one time.
- They are **not validation** of any strategy or feature set.
- They are **not profitability evidence** and say nothing about returns.
- They are **not M2 acceptance**. No SLA, budget or gate is passed or failed
  here.
- The harness `claims` block is copied into the evidence JSON:
  `validation=false`, `profitability=false`, `m2_acceptance=false`.

## Environment

| Item | Value |
| --- | --- |
| Host | Mac14,3, Apple M2, 8 cores |
| RAM | 8192 MB |
| OS | macOS 27.2.1 (build 26B131) |
| Python | CPython 3.12.13 (via `uv`) |
| Repo base | `693f8a3` (`feat/perfreal`) |
| Free disk | 113 GiB |
| Load average at start | 3.69 3.03 2.79 |
| Concurrent load | interactive desktop session open; one benchmark process at a time |

The machine was not idle. Absolute numbers include that noise; compare runs on
the same idle machine, not across machines.

## Data identity

| Store | Timeframe | Parquet files | Size | Bars available | Range | Digest (sha256 of sorted file hashes) |
| --- | --- | --- | --- | --- | --- | --- |
| `data-ticks5-store` | M1 | 244 | 53.2 MiB | 1,773,331 | 2021-05-24 → 2026-05-22 | `a2c7bb44…e87084da` |
| `data-real-store` | H1 | 1,015 | 112.3 MiB | 100,083 | 2009-03-15 → 2026-01-09 | `fdb7c595…b90412c` |

Samples are the **first N chronological bars** (`frame.head(rows)`), not a
random sample. `--rows 200000` on H1 is longer than the store, so that run used
the whole 100,083-bar H1 store.

## Commands

All commands run from `backend/`:

```bash
uv run python scripts/bench_pipeline.py --real-store \
  --store-root ../../../data-ticks5-store --timeframe M1 --rows 20000 \
  --repeats 3 --warmup 1 --max-trials 5 --out ../../runs/bench/m1-ticks5-20k.json

uv run python scripts/bench_pipeline.py --real-store \
  --store-root ../../../data-ticks5-store --timeframe M1 --rows 200000 \
  --repeats 3 --warmup 1 --max-trials 5 --out ../../runs/bench/m1-ticks5-200k.json

uv run python scripts/bench_pipeline.py --real-store \
  --store-root ../../../data-real-store --timeframe H1 --rows 200000 \
  --repeats 3 --warmup 1 --max-trials 5 --out ../../runs/bench/h1-real-200k.json

uv run pytest tests/test_bench_pipeline.py -q
```

Harness tweak: `scripts/bench_pipeline.py` had no CLI control for the discovery
smoke trial count even though `discovery_smoke_stage` already accepted
`max_trials`. Added `--max-trials` (default 2, matching the old behaviour) and
pass it through. The existing CLI test now also exercises the flag. No `src`,
`paper` or `discovery` logic was changed.

Aggregation: 1 discarded warmup call per stage, 3 measured repeats, plain
median/min/max, CPU user+system per repeat, `ru_maxrss` high-water RSS.

## Suite results

Wall and CPU seconds per stage. `rows` is the stage's output row count (for the
discovery stage it is the validation split).

### `data-ticks5-store` M1, 20,000 bars

Suite peak RSS: **839.6 MiB** (whole process, includes imports).

| Stage | rows | median s | min s | max s | CPU s (median) | peak RSS MiB | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| features_build | 20,000 | 0.094 | 0.089 | 0.104 | 0.166 | 839.6 | yes |
| dataset_build | 19,939 | 0.105 | 0.095 | 0.110 | 0.171 | 839.6 | yes |
| discovery_smoke | 2,990 | 0.774 | 0.760 | 0.811 | 0.850 | 839.6 | yes |
| evidence_build | 19,939 | 1.539 | 1.523 | 3.462 | 1.626 | 839.6 | yes |
| paper_advance | — | — | — | — | — | — | **no** |

Median total for the four working stages: 2.51 s. Wall clock for the whole
suite: 18.6 s.

### `data-ticks5-store` M1, 200,000 bars

Suite peak RSS: **883.9 MiB**.

| Stage | rows | median s | min s | max s | CPU s (median) | peak RSS MiB | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| features_build | 200,000 | 0.833 | 0.825 | 0.834 | 1.099 | 884.0 | yes |
| dataset_build | 199,939 | 1.076 | 1.008 | 1.271 | 1.292 | 884.0 | yes |
| discovery_smoke | 29,990 | 8.283 | 8.273 | 10.183 | 8.360 | 884.0 | yes |
| evidence_build | 199,939 | 3.294 | 3.187 | 3.701 | 3.685 | 884.0 | yes |
| paper_advance | — | — | — | — | — | — | **no** |

Median total for the four working stages: 13.49 s. Wall clock for the whole
suite: 61.8 s.

### `data-real-store` H1, whole store (100,083 bars)

Suite peak RSS: **438.8 MiB**.

| Stage | rows | median s | min s | max s | CPU s (median) | peak RSS MiB | ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| features_build | 100,083 | 0.616 | 0.607 | 0.691 | 0.809 | 400.3 | yes |
| dataset_build | 100,022 | 0.650 | 0.626 | 0.654 | 0.840 | 422.3 | yes |
| discovery_smoke | 15,003 | 4.052 | 3.833 | 4.263 | 4.360 | 438.8 | yes |
| evidence_build | 100,022 | 0.906 | 0.896 | 0.924 | 1.192 | 438.8 | yes |
| paper_advance | — | — | — | — | — | — | **no** |

Median total for the four working stages: 6.22 s. Wall clock for the whole
suite: 27.9 s.

## Feature cache on real slices

Method: `ParquetStore.read(...).head(rows)` → `cached_build_features` once into
a fresh temp `FeatureCache` (miss, build, save), then three hit calls. Every
call recomputes `frame_meta`, so hit time includes hashing the raw bar columns.

| Store / TF | rows | cols | cold miss s | hit median s | hit min/max s | speedup | cache entry | counters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ticks5 M1 | 20,000 | 59 | 0.254 | 0.014 | 0.0137 / 0.0163 | 17.9× | 5.57 MiB | hits 3, misses 1, saves 1, corrupt 0, errors 0 |
| real H1 | 20,000 | 59 | 0.210 | 0.011 | 0.0095 / 0.0112 | 19.8× | 5.57 MiB | hits 3, misses 1, saves 1, corrupt 0, errors 0 |
| ticks5 M1 | 200,000 | 59 | 1.523 | 0.104 | 0.0975 / 0.1279 | 14.7× | 55.7 MiB | hits 3, misses 1, saves 1, corrupt 0, errors 0 |

Cache load and save time were 0.037 s / 0.014 s (20k M1), 0.026 s / 0.009 s
(20k H1) and 0.278 s / 0.213 s (200k M1). Hit cost grows with rows because the
input fingerprint is re-hashed on every call; it is not a pure file read.

## Disk

The harness records `disk_delta_mb` only when a stage sets `disk_path`; no
built-in stage does, so per-stage disk deltas are `null`. Measured separately:

| Item | Value |
| --- | --- |
| `data-ticks5-store` | 53.2 MiB, 244 parquet files |
| `data-real-store` | 112.3 MiB, 1,015 parquet files |
| Temp store write, M1 20k bars | 536,236 B in 0.086 s |
| Temp store write, M1 200k bars | 5,297,475 B in 0.094 s |
| Temp store write, H1 100,083 bars | 2,608,592 B in 0.692 s |
| Feature cache entry, 20k bars | 5,843,032 B |
| Feature cache entry, 200k bars | 58,403,038 B |

Discovery and paper stages copy their sampled frame into a temp store and delete
it at suite end, so the temp-store numbers above are their disk footprint per
repeat. The source stores are never written.

## Stage that could not run

`paper_advance` failed on both stores, in warmup, before any measurement:

- M1 ticks5: `BarGapError: session 'auto-XAUUSD-M1-…': missing bar at 2021-05-24T04:04:00+00:00 before 2021-05-24T04:05:00+00:00`
- H1 real: `BarGapError: session 'auto-XAUUSD-H1-…': missing bar at 2009-05-25T17:00:00+00:00 before 2009-05-25T22:00:00+00:00`

The paper runner requires contiguous session bars and fails closed on real
market gaps. Changing that behaviour is paper logic, which is out of scope for
this task, so it was left failing and is recorded as `ok=false` in the JSON.
The 20k M1 and H1 runs therefore measure 4 of 5 stages.

## Limitations

- First-N chronological samples, not random or regime-stratified.
- 200k M1 is 11.3% of the 1,773,331 stored M1 bars. There is no full-store M1
  measurement.
- H1 `--rows 200000` exceeds the store, so it is a whole-store run, not a
  200,000-bar run.
- `rss_delta_mb` is high-water growth. It reads 0.0 once an earlier stage set
  the peak, so `rss_peak_mb` is the number to use; the deltas are not additive.
- No `disk_path` instrumentation on the built-in stages.
- One measurement repeat in the 20k M1 evidence stage took 3.46 s against a
  1.54 s median. Treat min/max as noise bounds, not a distribution.
- `discovery_smoke` uses population 2, one generation, `max_trials=5`. It is a
  pipeline smoke, not a search experiment.
- `evidence_build` uses a fixed always-long spec, 3 folds, 100 bootstrap
  samples and cost stress (1.5, 2.0). It exercises fold geometry and cost
  arithmetic only.
- No confidence intervals, no thermal control, no CPU pinning, and the desktop
  session was active during the runs.
- No walk-forward stage was run; `--suite full` was not used.
- Requests, sockets and any network path are absent from these stages.

## Files

- `docs/evidence/bench-2026-11-24.json` — compact aggregates, commands,
  environment, cache and disk measurements, `claims` block.
- Raw per-run reports (git-ignored): `runs/bench/m1-ticks5-20k.json`,
  `runs/bench/m1-ticks5-200k.json`, `runs/bench/h1-real-200k.json`.
