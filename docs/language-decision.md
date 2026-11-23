# Language and architecture decision

Date: 2026-11-23
Machine: Mac mini M2, 8 GB unified memory, 8 cores, macOS.
Inspected commit: `c3eac91` (main at the time of the review; later commits add
the transformer and API slices and do not change the conclusion).
Status: **decision record**, no native code added.

## Decision

**Keep Python 3.12 + TypeScript. Do not move any module to Rust, Go or C now.**

The heavy arithmetic already runs in native libraries: Polars/Arrow, NumPy,
PyArrow, PyTorch, scikit-learn, LightGBM, XGBoost and hmmlearn. Python is the
orchestration, contracts and research layer, which is where iteration speed and
scientific validity come from.

A native rewrite would add a second toolchain, a wheel matrix (macOS ARM64 +
Linux CI), a second test suite, a CodeQL coverage gap unless the workflow is
extended, and a divergence risk between research labels and paper fills. The
measured gain does not justify that.

## Evidence

Indicative single-run, warm-cache timings on the real stores; another team
benchmark was running during part of the session (load average ~4 on 8 cores),
so absolute numbers are slightly pessimistic.

| Path | Result |
|---|---|
| Tick aggregation (Polars `group_by_dynamic`) | 9.1M ticks → 4,633 M1 bars in **0.19 s** (47.6M ticks/s), 438 MB peak |
| Bar validation | 1.77M rows in **0.16 s** (11.2M rows/s) |
| H1 feature build | 29,586 rows in **1.17 s** |
| Bar backtest | 20,000 simulator steps in **0.40 s** (50k steps/s) |
| M1 feature build | 1.77M rows in **176 s** (10.1k rows/s), 1.07 GB peak |
| Label generation | 100k bars → 97,482 labels in **3.41 s** |
| PPO (existing run manifests) | 504k–508k steps in **7–22 min** end to end |

Extrapolations (linear, not measured): 5.89M M1 rows → ~9.8 min feature build,
~3.5 GB peak; 380M ticks → ~8 s aggregation CPU plus cold I/O.

## Module matrix

Native need is scored 1 (Python clearly sufficient) to 5 (Python infeasible);
it measures the need for native code, not whether native code would be faster in
isolation.

| Module | Current | Native need | Verdict |
|---|---|---|---|
| Tick ingestion/aggregation | Polars | 1 | Keep |
| Validation/quality | Polars + NumPy | 1 | Keep |
| Feature build | Polars, 59 columns | **2** | Keep; lazy plan + chunking + cache |
| Labels | Python loop + `PaperBroker` | 2 | Keep; vectorise joins |
| Backtest/simulator | Python bar loop | 1 | Keep |
| Classical ML | sklearn/LightGBM/XGBoost | 1 | Keep |
| DL training/inference | PyTorch | 1 | Keep |
| RL | SB3 + PyTorch + Gymnasium | 1 | Keep |
| API/WebSocket | FastAPI/uvicorn | 1 | Keep |
| Scheduler | not built | 1 | Keep Python |
| Frontend | Next.js/React/TS | 1 | Keep TS |

## Why not Rust, Go or C

- **Rust** is the only plausible native option. It would require reimplementing
  59 causal features and their exact window/EMA/session semantics, adding
  PyO3/maturin wheels for macOS ARM64 and Linux CI, type stubs, panic mapping
  and a dual implementation kept bit-identical in CI. Estimated 4–8 person-weeks
  for one of two people, plus permanent dual-stack maintenance, for a batch job
  that runs once per dataset build.
- **Go** has no identified role. It would either reimplement numerical code
  (worse than the existing native libraries) or sit behind cgo/IPC, adding a
  failure mode without removing Python.
- **C** is dominated by Rust if a native kernel is ever needed: the broker and
  risk logic are stateful, and memory safety matters more than the last few
  percent.
- **Memory is data-bound.** The M1 feature matrix is 419 MB because 59 float32
  columns × 1.77M rows is 419 MB. Rust would not shrink it; chunking, caching and
  one-job scheduling address the 8 GB limit.

## What would change the decision

Each needs a measurement, not an estimate:

1. **Tick-level simulation over the 380M-tick archive becomes a product
   requirement**, with per-tick broker/risk state. Today the simulator is
   bar-based and there is no tick store. A tick kernel that cannot finish in the
   overnight window would justify Rust called from Python, with `PaperBroker` as
   the oracle.
2. **The optimised Python feature build still misses an explicit wall-clock
   budget** for a multi-symbol daily rebuild. Suggested budget: < 10 minutes per
   symbol for the 17-year M1 store, < 60 s for the 5-year tick-derived store.
3. **A profiler shows a user-facing path dominated by Python bytecode** that
   cannot be cached or vectorised. Trigger: p95 `/api/sim/start` > 5 s after
   feature caching, or `label_events` > 5 min on the production store.

If none fires, keep improving the Python architecture.

## Higher-return Python work (the real bottlenecks)

These are the fixes the language question was masking:

1. **Feature cache.** Materialise `FeatureFrame` per `(symbol, timeframe,
   data_hash)` to Parquet/`.npy`. `/api/sim/start`, walk-forward folds and
   repeated training currently rebuild the same matrix.
2. **One lazy feature plan.** Convert `build_features` to a `LazyFrame` with one
   `collect`, chunk by month for the 17-year store, and keep the causality tests
   as the oracle.
3. **One-job enforcement for PPO.** Reuse the `JobSlot` pattern from
   `nn/train.py` in `rl/train.py`/`TrainRunner`; two distinct PPO runs can
   currently start at once on 8 GB.
4. **Metadata reads for status/coverage.** Use `scan_parquet` statistics instead
   of materialising M1 to answer row counts and first/last timestamps.
5. **Vectorise label joins.** Replace the per-label Python dictionary lookup in
   `bandit_training_data`/`_outcomes_from_table` with a Polars join; keep
   `PaperBroker` as the single source of fill semantics.

## Follow-up benchmark: one function, not the pipeline

A second independent benchmark pinpointed the M1 feature build: **96.6% of the
168–173 s is `features/liquidity.py::fvg`** (three calls: base, M5, M15), a
pure-Python row loop whose cost grows with the number of open gaps (1,567 open
up-gaps at 1.77M rows). A bit-identical C port runs the same loop in **1.69 s**
(87x), and the whole build would fall to ~7.5 s. `order_block_distance` is 2.1%
(3.6 s); everything else is ~1.3%.

That does **not** justify a native rewrite yet. The 87x is a language constant on
top of an O(n·k) algorithm. The correct next step is a Python algorithmic fix
(predecessor/successor query plus a next-smaller-element death pass), expected
to land in the 2–5 s range with no toolchain, wheel or FFI. The benchmarker's
naive heap variant reached 2.02 s but was not output-equivalent, so it is a lower
bound, not a drop-in. Extract only `fvg` to Rust/PyO3 if the corrected Python
implementation still misses the wall-clock budget on the 5.89M-row store.

## Honest limitations

- Single-run, warm-cache timings; no repetitions or confidence intervals.
- No `py-spy`/`cProfile` per-expression split; the feature-build cost is inferred
  from code structure (many materialising Polars passes).
- Linear extrapolations to 5.89M rows and 380M ticks; memory can scale
  superlinearly.
- PPO times come from run manifests, not a controlled benchmark.
- Only XAUUSD was exercised; multi-symbol memory is untested.
- The tick aggregation sample was 4 of 175 shards (~9.1M of ~380M ticks).
- PyO3 packaging cost is estimated, not measured; no wheel was built.

## Verification

No repository file was modified for this decision. The measurements used the
real stores under `data-ticks5-store` / `data-real-store` and the existing
dependencies. The next performance work is tracked as Python optimisations, not
a language migration.
