# Mac mini M2 optimisation: pipeline measurement harness

P6 asks for measured RAM, CPU, disk and elapsed time on the target Mac mini M2
8 GB. This document describes the harness added for that measurement. It is a
measurement tool, **not an acceptance result**.

The audit's P6 order is in
[`docs/autonomous-system-audit-2026-11-24.md`](autonomous-system-audit-2026-11-24.md).
The harness covers item 1 (measure the stages) for the autonomous pipeline
slices that exist today. It does not enforce budgets, schedule jobs, prove
walk-forward validity or claim profitability.

## What the harness is

Two layers:

| Layer | File | Purpose |
|---|---|---|
| Measurement | `backend/src/scalping/bench/pipeline.py` | Run one callable `repeats` times after `warmup`; aggregate wall, CPU user/system, peak-RSS growth, optional temp-dir disk growth, and `rows`/`bytes` from a returned mapping. |
| Suite | `backend/src/scalping/bench/suite.py` | Measure named stages and write one JSON report with environment, per-stage samples, a summary and an explicit `claims` block. |
| CLI | `backend/scripts/bench_pipeline.py` | Build the built-in stages and print the report as JSON. |

The implementation is standard-library plus existing project modules. No new
dependency, no network, no subprocess and no real-money path. Measurements are
performed in the current process, so the numbers include this process's own
Python, Polars/NumPy and simulator work.

## Exact commands

Run from `backend/`:

```bash
cd backend
uv sync

# Synthetic smoke suite, the default. No data store and no network.
uv run python scripts/bench_pipeline.py

# Repeat each stage three times with one discarded warm-up call.
uv run python scripts/bench_pipeline.py --suite full --repeats 3 --warmup 1

# Real bars, bounded sample. The flag is required; synthetic remains the default.
uv run python scripts/bench_pipeline.py \
  --real-store --store-root /path/to/data --rows 50000

# Real bars, whole stored series. This can exceed 8 GB of RAM; run it alone.
uv run python scripts/bench_pipeline.py \
  --real-store --store-root /path/to/data --rows 0
```

The script writes `runs/bench/pipeline.json` by default and prints the same
object on stdout as one JSON line. Errors print one JSON line to stderr and exit
2 for an unusable input or a failed stage, 1 for an unexpected failure.

```bash
# Tests for the harness only.
uv run pytest tests/test_bench_pipeline.py -q

# Lint and types for the touched tree.
uv run ruff check src tests scripts
uv run mypy src
```

## Stages

| Stage | What is timed | Default data |
|---|---|---|
| `features_build` | `build_features` on one prepared frame | 481 synthetic M1 rows |
| `dataset_build` | `build_features` plus the chronological `build_dataset` cut | 481 synthetic M1 rows |
| `discovery_smoke` | A bounded discovery run with 2 trials, fresh archive per repeat | 1 201 synthetic M1 rows |
| `evidence_build` | `build_features`, `build_dataset`, then a 3-fold `build_evidence` | 1 200-row flat M1 tape |
| `paper_advance` | One `ContinuousPaperRunner.advance` with a fresh in-memory ledger | 401 synthetic M1 rows |
| `walk_forward` | `evaluate_walkforward` with a `ret_5` policy | 100 synthetic days, optional |

`--suite smoke` (the default) runs the first five. `--suite full` adds
`walk_forward`, which is the only stage that measures more than a second on the
synthetic defaults.

Synthetic mode never opens the store root you pass. `real_store=False` is
enforced by tests: `test_synthetic_default_never_reads_the_named_store`
monkeypatches `ParquetStore.read` and asserts the named root is never in the
call list. `real_store=True` is the only path that reads real bars, and it
reads them once when the stage is built, outside the timed region. Stages that
need a writable store (`discovery_smoke`, `paper_advance`) always use a
temporary directory. The source store is never written to.

`rows` is the same flag for every stage. `None` means a 20 000-bar sample for
most real stages and 200 000 for `walk_forward`; `0` means the whole real
store. The real defaults are small on purpose: a full 17-year M1 feature build
can exceed the 8 GB target, so using it must be deliberate and should be the
only heavy job on the machine.

## How to read the report

```jsonc
{
  "schema": "scalping.bench.suite/1",
  "generated_at": "2026-11-24T...",
  "environment": {
    "python": "3.12.x",
    "platform": "macOS-...-arm64",
    "machine": "arm64",
    "cpu_count": 8,
    "total_ram_mb": 8192.0
  },
  "claims": {
    "kind": "local_measurement",
    "validation": false,
    "profitability": false,
    "m2_acceptance": false
  },
  "stages": [
    {
      "name": "features_build",
      "repeats": 1,
      "measured_repeats": 1,
      "wall_seconds": { "median": 0.017, "min": 0.017, "max": 0.017 },
      "cpu_seconds": { "user_median": 0.016, "system_median": 0.001, "total_median": 0.017 },
      "rss_mb": { "start": 179.0, "peak": 187.0, "delta": 8.0 },
      "disk_delta_mb": null,
      "rows": 481,
      "bytes": null,
      "ok": true,
      "error": "",
      "probe_error": "",
      "runs": [ { "wall_seconds": 0.017, "...": "one raw repeat" } ]
    }
  ],
  "summary": {
    "stages": 5,
    "ok": 5,
    "failed": 0,
    "wall_seconds_median_total": 0.23,
    "max_rss_delta_mb": 8.3
  }
}
```

Read it in this order:

1. **`claims`** first. If a number is about to be repeated in a report, repeat
   this block with it. It says the run is a local measurement, not validation,
   not profitability and not M2 acceptance.
2. **`environment`** second. A timing without the machine, Python version and
   CPU count is not comparable to anything.
3. **`stages[].ok`** third. A failed stage still has a cost and an `error`; do
   not average a failed repeat into a success story. `measured_repeats` is the
   number of repeats that actually ran.
4. **`wall_seconds.median`** with `min`/`max`. The median is the headline;
   a wide min-to-max spread on an idle machine means the measurement is noisy
   or the stage has a cache effect. With one repeat, median, min and max are
   the same number by construction — that is a sample, not a distribution.
5. **`cpu_seconds`** separately. `user_median` plus `system_median` is not
   always `total_median`: each is a median of its own per-repeat series.
6. **`rss_mb.delta`** as growth of the high-water mark, not current residency.
   A stage that allocates 1 GiB and frees it still shows the 1 GiB. `start` is
   the reading before the first repeat, so it includes any frame the factory
   prepared outside the timed region.
7. **`disk_delta_mb`** only when a `disk_path` was supplied. It is file bytes
   under that directory, not free-space change. The built-in CLI stages do not
   set it; the pipeline API and tests do.
8. **`runs`** when a number looks wrong. It is the raw per-repeat wall, CPU,
   RSS and disk readings; every aggregate can be re-derived from it.

The `summary.wall_seconds_median_total` is a sum of stage medians, not a
measured end-to-end wall time: setup, report writing and gaps between stages are
excluded. Never quote it as pipeline latency.

## Measured smoke numbers (synthetic data)

One local run on 2026-11-24, Mac mini M2 8 GB, `--suite full --repeats 3
--warmup 1`, synthetic data, otherwise idle machine. These are illustrative, not
a baseline to compare against a different host or a loaded machine. The warm-up
call absorbs one-time import and allocator growth, so the RSS deltas are smaller
than a cold first run.

| Stage | Wall median (s) | Wall min–max (s) | Peak RSS delta (MB) | `rows` |
|---|---:|---:|---:|---:|
| `features_build` | 0.015 | 0.015–0.015 | 0.6 | 481 |
| `dataset_build` | 0.014 | 0.014–0.016 | 0.4 | 420 |
| `discovery_smoke` | 0.055 | 0.054–0.066 | 1.0 | 171 |
| `evidence_build` | 0.100 | 0.092–0.119 | 0.3 | 1 139 |
| `paper_advance` | 0.039 | 0.038–0.041 | 0.4 | 281 |
| `walk_forward` | 0.701 | 0.688–0.758 | 8.1 | 103 681 |

`summary.wall_seconds_median_total` was 0.92 s for the six stages. Process
start-up and report writing are outside that sum and add roughly a second.

What these numbers mean: the harness works on synthetic data and the stages
complete quickly enough to run in CI-scale tests. What they do **not** mean:
the pipeline is fast on real data, the 8 GB budget is met, a strategy is valid,
or the system is accepted. Synthetic bars are generated locally and the
feature/simulator work they exercise is not the 17-year store workload.

## Failure behaviour

- A stage that raises is recorded with `ok=false`, its error text and the cost
  of the failed repeat. Later stages still run.
- `--strict` aborts on the first failure and writes no report.
- A warm-up failure yields a zeroed measurement with no runs.
- A probe failure (RSS) is recorded in `probe_error`; the stage result remains
  usable for wall and CPU time.
- Duplicate stage names and an empty stage list are refused before measuring.

## Honest limits

- **One process, one machine, one moment.** Load, thermal throttling, other
  processes, filesystem cache and Python start-up all move the numbers. Compare
  repeated runs on the same idle machine, not absolute values across machines.
- **No enforcement.** This harness measures; it does not cap RAM, CPU, disk or
  concurrency. The `MeasuredResourceGuard` / orchestrator policy is a separate
  slice. `docs/orchestrator.md` describes that guard.
- **No ingest measurement.** Tick aggregation and validation are not built-in
  stages here. `backend/scripts/bench_fvg.py` measures the feature hot path;
  ingest needs its own measured run against the real archive.
- **Not a profiler.** There is no per-function attribution. Use a profiler for
  that; this harness answers "what did the stage cost", not "which line".
- **High-water RSS is cumulative.** `ru_maxrss` never decreases, so a later
  stage's delta can be zero after an earlier allocation. Run one stage in a
  fresh process when its own peak matters.
- **Disk delta is directory bytes.** It does not include free-space changes,
  filesystem overhead or files written outside the supplied directory.
- **Real-store sampling is not the full workload.** The 20 000-bar default is
  a bounded sample. `--rows 0` measures the full stored series, but a full M1
  feature build can exceed 8 GB; run it alone and watch system memory.
- **Synthetic results are not market evidence.** They prove the harness and the
  pipeline wiring execute; they say nothing about edge, costs or profitability.

The only honest summary for this slice is: the harness exists, it is tested
without the real store, and it produces comparable local numbers when given the
same machine and inputs. Acceptance still requires the P6 end-to-end run on real
XAUUSD data, measured on the target Mac mini, with code/test/training/
validation/paper evidence reported separately.


### PPO benchmark

```bash
uv run python -m scripts.benchmark_ppo \
  --timesteps 256 --n-steps 64 --predictions 50 --device cpu \
  --out ../runs/bench-ppo.json
```

The row reports `train_seconds`, `steps_per_second`, median `predict_ms` and the
device. It is a short training run on synthetic bars, not a policy-quality
claim; use the same shape/seed for comparisons. `--device mps` fails when MPS
is unavailable instead of silently benchmarking CPU.

