# Data pipeline

How market data enters the system, how it is validated, and where provenance is
recorded. Current state after Phase 2's first slice (registry, validation,
provenance, JSON import, M30/H4).

## Layers

```
raw source (CSV / JSON / Dukascopy / synthetic / tick archive)
        │
        ▼
provider or ingestion script
        │  conform_bars: schema + UTC
        ▼
ParquetStore  <symbol>/<timeframe>/<year>/<month>.parquet
        │  _meta/<symbol>/<timeframe>.json  (provenance)
        │
        ▼
resample → M5/M15/M30/H1/H4/D1   (M1 is the base)
        │
        ▼
feature builder / labels / simulator
```

The store is the only durable data layer. `runs/` is runtime state and is
gitignored; data stores are local and gitignored too.

### Feature availability

`scalping/features/availability.py` stops a dataset from claiming features its
source cannot support. `build_dataset(..., capabilities=Capabilities(...))`
checks the selected columns before the split is built; without `capabilities`
the historical behaviour is unchanged.

| Capability | Default | Satisfied by |
|---|---|---|
| `bar_volume` | true | a non-zero bar `volume` field |
| `real_quotes` | false | non-null `bid_close`/`ask_close`/`spread` rows |
| `modeled_spread` | true | the feature preparer's spread model |
| `tick_flow` | false | signed tick volumes; **never inferred from bars** |
| `m1_source` | true | an M1 frame the higher timeframes are derived from |

Rules: `volume_*` needs `bar_volume`; `spread_*` needs `spread`
(`real_quotes` **or** `modeled_spread`); `m5_*`/`m15_*`/`htf_*` need an M1
source; any `flow_*` column is reserved for the tick layer and needs
`tick_flow`. An unavailable selection raises `FeatureUnavailableError` and
names every missing input, so a manifest cannot quietly describe unsupported
features.

`capabilities_from_frame(frame)` infers volume/quote availability from a raw
provider frame. CSV exports without quotes are correctly reported as
`real_quotes=False`, while the allowed spread model keeps `spread_*` usable.

### Persisted dataset arrays

`scalping/datasets/artifact.py` saves a built dataset once and reloads it
without rebuilding from Parquet:

```python
from scalping.datasets.artifact import load_dataset_arrays, save_dataset_arrays

save_dataset_arrays("../runs/datasets/xauusd-m1-h1", bundle)
data = load_dataset_arrays("../runs/datasets/xauusd-m1-h1")   # X is a memmap
```

- Layout: `artifact.json` plus one `.npy` per array (`X`, `y_direction`,
  `y_return`, `ts`, `split`). `artifact.json` is written last and acts as the
  commit marker; arrays without it are not a dataset.
- `X` loads as a read-only `np.memmap`, so a large matrix is not copied into
  RAM. `mmap=False` returns ordinary arrays.
- Each file has a blake2b checksum in the manifest, verified on load by
  default. `verify=False` skips the read for a trusted artifact.
- The manifest keeps the dataset id and the builder manifest, so a model run
  can name the exact dataset it trained on. Shape/dtype mismatches and missing
  files are errors, not silent gaps.
- The manifest is untrusted input. Every `file` entry must be a plain file name,
  and the resolved path must stay inside the artifact directory: absolute paths,
  separators, `.`/`..` and symlinks that escape are rejected with
  `ArtifactError`. A manifest missing `file`, `dtype`, `shape` or `checksum`, or
  carrying invalid metadata, raises `ArtifactError` at parse time instead of
  `KeyError`.
- Saves are atomic and concurrency-safe. Each array and the manifest go to a
  unique temp file in the artifact directory (`tempfile.mkstemp`) and are moved
  into place with `os.replace`, so two saves cannot replace or unlink each
  other's scratch file.

This is the storage format; CLI/API wiring for "train from an artifact" lands
with the model registry (Phase 10).

## Schema

Every bar frame conforms to `scalping.data.base.BAR_SCHEMA`:

```text
ts        Datetime(us, UTC)
open      Float64
high      Float64
low       Float64
close     Float64
bid_close Float64
ask_close Float64
spread    Float64
volume    Int64
provider  String
```

Missing bid/ask/spread is allowed at import; the feature preparer applies its
spread model. A provider must not invent quotes silently.

## Symbols

`scalping/data/registry.py` holds research defaults for the ten required
instruments: XAUUSD, XAGUSD, EURUSD, GBPUSD, USDJPY, AUDUSD, NAS100, US30,
USOIL, BTCUSD. Each `SymbolSpec` carries contract size, point, precision,
min/max lot, lot step, commission, typical spread, sessions, leverage and an
optional Dukascopy symbol.

```python
from scalping.data.registry import get_symbol, contract_for

spec = get_symbol("XAUUSD")
contract = contract_for("XAUUSD")
```

Defaults are for research, not broker-verified production specifications. A
future MT5 adapter must replace them from the broker's contract details.

Current market-data coverage is still gold/silver for Dukascopy. The other
eight specs are registered but have **no real provider yet**; synthetic bars
can be generated for any symbol.

### Trading sessions

`session_for(symbol)` parses the registry's weekly UTC window
(`sun22:00-fri21:00 UTC` for metals/FX/indices, `24x7 UTC` for BTCUSD) into a
`SessionSpec`. `SessionSpec.is_open(ts)` treats naive timestamps as UTC and
handles the wrap from Sunday to Friday.

The simulator enforces it when a symbol is supplied: entries outside the window
are blocked with `info["blocked"] == "session_closed"`, while exits, stops and
targets always work. `run_backtest(..., symbol="XAUUSD")` resolves the session
and includes its label in the run id, so the same prices on a 24x7 and a
weekday symbol can never share an id. Without a symbol the simulator stays
session-free for custom frames.

```python
from datetime import UTC, datetime
from scalping.data.registry import session_for

session = session_for("XAUUSD")
session.is_open(datetime(2026, 1, 3, 12, tzinfo=UTC))  # Saturday -> False
```

## Timeframes

`TIMEFRAMES` maps M1, M5, M15, M30, H1, H4, D1 to Polars rules. `resample`
accepts any of them. Tick data is aggregated to M1 by the tick scripts; it is
not stored as a tick table in the main store.

### Alignment semantics

Buckets are **left-closed and left-labelled**: the bar stamped `10:00` holds
`[10:00, 10:30)` for M30 and `[08:00, 12:00)` for H4. Labels come from the UTC
epoch grid, not from the first source bar, so M30 starts at `:00`/`:30` and H4
at `00/04/08/12/16/20` even when a frame begins at `10:07`.

Within a bucket, `open`/`bid_close`/`ask_close` and `provider` are taken from
the first/last source bar (`bid_close` and `ask_close` from the last bar so the
quote matches `close`), `high`/`low` are the extrema, `volume` is the sum and
`spread_mean` is the bucket mean. Aggregation never reaches outside its window,
so truncating the source only changes a bucket that is still forming.

Chained derivation (`M1 → M30 → H4`) is associative for OHLC and volume. The
intermediate frame names the mean spread `spread_mean`; `data derive` renames it
back to `spread` before the next aggregation, which is what makes the second
`resample` call valid. A mean of means equals the overall mean only when every
sub-bucket is complete, so compare `spread_mean` across derive paths on complete
buckets only.

Higher-timeframe features (`features/context.py::htf_context`) join a bucket
only after it closes: the bucket timestamp is shifted forward by one bucket
width and applied with `join_asof(backward)`. A row at `t` therefore sees the
last bucket with `start + width <= t`; the bucket containing `t` is never used.
The rule is timeframe-agnostic (`_bucket_width` reads the frame's spacing), but
the feature matrix currently consumes M5 and M15 only: M30/H4 are data-layer
timeframes with no feature prefix yet.

Weekends are gaps, not buckets. `resample` emits no empty bucket and never
carries a Friday close into Sunday. Under the XAUUSD window
(`sun22:00-fri21:00 UTC`) the Sunday 22:00 bars fall into the H4 bucket labelled
`20:00`, because that grid window contains them.

Missing and duplicate source rows are not repaired or shifted. Duplicates are
aggregated into their bucket (volume summed) and reported by `validate_bars` as
`duplicate_timestamps`; the store also dedupes on write, so `data derive`
cannot consume them. A missing minute shortens its bucket and is reported as a
`time_gaps` warning above the threshold; labels stay on the grid.

### Feature naming and aliases

Column names are part of the cached registry version and of dataset checksums,
so a clearer name is added as a read-only alias instead of renaming a column.
`FEATURE_ALIASES` in `features/registry.py` maps aliases to canonical columns
and `canonical_name()` resolves them; `FEATURE_COLUMNS` keeps its exact order.

| Alias | Canonical | Reason |
|---|---|---|
| `log_ret_3` | `ret_3` | `ret_3` is a log return; `ret_1/5/15/60` are simple returns |

Consecutive-candle features (up/down streaks, consecutive closes above a level)
are **not implemented**. The matrix has candle anatomy (`body_atr`,
`upper_wick_atr`, `lower_wick_atr`, `range_atr`) and the BOS-derived `trend`
state; neither counts consecutive candles. Adding such a column needs a stated,
falsifiable research question rather than a rename.

## Validation

`scalping/data/validate.py` reports without repairing.

`validate_bars` errors:

- missing schema columns, non-UTC/non-datetime timestamps;
- unsorted timestamps, duplicates;
- nulls, non-finite prices;
- OHLC inconsistency (`low <= open/close <= high`, `low <= high`);
- crossed quotes (`ask_close < bid_close`);
- negative spread or volume.

Warnings: time gaps above a threshold (weekends are legitimate, so gaps do not
fail the report by itself) plus the statistical outliers below.

`validate_ticks` checks sorted stamps, crossed bid/ask, non-finite/negative
prices and volumes, and nulls.

### Statistical outliers

`scalping/data/quality.py` adds a retrospective warning layer to
`validate_bars` (disable with `include_outliers=False`):

- `price_outliers` — log returns beyond `8` robust sigma, where the scale is
  `1.4826 * MAD` (std fallback for a zero MAD);
- `volume_outliers` — `log1p(volume)` beyond `8` robust sigma.

Both are warnings: a news spike is real data. The check uses the whole slice's
median/MAD, so it is QA only — never a model feature (that would leak the
slice's distribution).

### Cross-source reconciliation

`reconcile_bars(primary, secondary, config=...)` compares two frames for the
same symbol/timeframe by timestamp:

| Code | Severity | Meaning |
|---|---|---|
| `only_in_primary` | warning | timestamps the secondary source lacks |
| `only_in_secondary` | warning | timestamps the primary source lacks |
| `close_mismatch` | error | closers differ beyond `price_tolerance` |
| `spread_mismatch` | error | spreads differ beyond `spread_tolerance` |
| `duplicate_timestamps` | error | a source repeats a timestamp |

`ok` is true when no error-severity issue exists. Nothing is merged or
repaired; the caller decides which source to trust.

API:

```bash
curl 'http://127.0.0.1:8000/api/data/validate?symbol=XAUUSD&timeframe=H1'
```

### Quality report API

`GET /api/data/quality?symbol=&timeframe=` is a read-only view over the same
validator. It adds no checks; it maps validator issue codes onto one stable
report for a dashboard or a data gate.

`GET /api/data/quality?symbol=XAUUSD&timeframe=M1&include_outliers=true`

| Field | Source |
|---|---|
| `rows`, `first_ts`, `last_ts` | the validated frame |
| `duplicates` | `duplicate_timestamps` issue count |
| `gaps` | `time_gaps` issue count |
| `ohlc_violations` | `ohlc_inconsistent` issue count |
| `crossed_quotes` | `crossed_quotes` issue count |
| `negative_spread` | `negative_spread` issue count |
| `negative_volume` | `negative_volume` issue count |
| `outliers` | `price_outliers` + `volume_outliers` counts |
| `provenance` | the `_meta/<symbol>/<timeframe>.json` sidecar, verbatim |
| `issues` | every validator issue (code, severity, count, detail, ts sample) |
| `thresholds` | gap and robust-z settings the run actually used |

Status thresholds:

| Status | Rule |
|---|---|
| `ok` | no error or warning issue, and a provenance sidecar exists |
| `warn` | no error, but a warning issue (`time_gaps`, `price_outliers`, `volume_outliers`) or no sidecar (`provenance_missing`) |
| `fail` | any error-severity issue; `status_rule` names the first one |

`status_rule` is the issue code that decided the status and `status_detail` is
its validator detail. The thresholds are deliberately the validator's:

- a gap is an adjacent diff wider than `max(10 min, one bar)`, so an H1 slice is
  not reported as gap-ridden for its normal 60-minute spacing; weekend and
  holiday closures still count as gaps. `/api/data/validate` keeps the
  validator's 10-minute default, so the two endpoints can report different gap
  counts on a higher timeframe; `thresholds` states what this report used;
- outliers are log-return / `log1p(volume)` values beyond `8` robust sigma
  (`1.4826 * MAD`, std fallback), computed only when the slice has at least
  `30` rows.

The report never invents a zero. An unconfigured slice returns `200` with
`configured: false`, `rows: 0`, null counts, `status: "fail"` and
`status_rule: "store_empty"`. On a configured slice, `outliers` is null (not
zero) when the slice is shorter than the robust-z floor or when
`include_outliers=false` skipped the pass; `outliers_checked` says which.
Likewise, a registered symbol with no bars is not a 422: only an unregistered
symbol or timeframe is, because the request itself is then invalid.

The endpoint reads the whole stored slice, like `/api/data/validate`, and never
writes, repairs or backfills. `include_outliers=false` is the escape hatch for
a large store; it skips only the O(n) robust-z pass.

## Provenance

`scalping/data/provenance.py` writes a sidecar:

```text
<store>/_meta/<symbol>/<timeframe>.json
```

Fields: `dataset_id`, `symbol`, `timeframe`, `source`, `synthetic`, `rows`,
`first_ts`, `last_ts`, `checksum`, `created_at`, `notes`.

The checksum covers timestamps, OHLC, spread and volume. A changed frame gets a
new checksum; the old sidecar is replaced only by an explicit write.

API status includes it:

```bash
curl 'http://127.0.0.1:8000/api/data/status?symbol=XAUUSD&timeframe=M1'
```

The Data page shows the sidecar verbatim (source, synthetic/real badge, dataset
id, checksum, created_at). A store written before provenance existed shows an
explicit "no provenance" note instead of a guessed value.

## Importing bars

Synthetic (deterministic):

```bash
cd backend
uv run python -m scalping.cli data synth \
  --root ../demo-data --symbol XAUUSD --start 2024-10-01 --end 2024-10-09
```

Broker CSV (M1, timestamp offset handled):

```bash
uv run python -m scalping.cli data csv \
  --path ../data-real-csv/XAUUSD_M1.csv --symbol XAUUSD --root ../data-real-store
```

The HTTP endpoint takes a **file name**, not a path. Copy the export into
`<data-root>/imports/` and name it in the body:

```bash
curl -X POST 'http://127.0.0.1:8000/api/data/download' \
  -H 'Content-Type: application/json' \
  -d '{"provider":"csv","symbol":"XAUUSD","path":"XAUUSD_M1.csv",
       "start":"2024-10-01T00:00:00Z","end":"2024-10-09T00:00:00Z","tz_offset":3}'
```

The server lists that directory and opens the matching entry, so a name with a
separator, an absolute path, `..`, or a symlink is a 422 and nothing outside
`imports/` can be read. The CLI keeps unrestricted paths.

The CSV provider parses the file in bounded chunks (100,000 rows) instead of
reading it into one string and one record list. A multi-GB export therefore
costs the frame plus one chunk, not several copies of the raw text. Source line
numbers stay exact and duplicate detection spans chunk boundaries, because the
"first seen" map is shared by every chunk. Pass `--tz-offset` explicitly; with
no offset the documented `+3` broker default is assumed (the old DST heuristic
only changed the log message, never the value).

The final canonical frame is still materialised, and the store write plus the
provenance checksum need it once; a fully streaming store import with a chunked
checksum is Phase 14 work. JSON import is parsed as one document (no streaming
JSON parser is a dependency).

JSON array (new):

```bash
uv run python -m scalping.cli data json \
  --path ../import/bars.json --symbol XAUUSD --root ../demo-data
```

Accepted JSON shapes: a list of objects, or `{"bars": [...]}`. Row keys:
`ts`/`timestamp`/`time` plus OHLC; `bid_close`, `ask_close`, `spread`,
`volume`, `provider` optional. Naive timestamps are UTC; epoch seconds/ms are
accepted.

Derive higher timeframes:

```bash
uv run python -m scalping.cli data derive \
  --root ../demo-data --symbol XAUUSD --timeframes M5,M15,M30,H1,H4,D1
```

Validate one stored slice (one JSON report line on stdout; exit 2 when a
structural error is found, so a script can gate on it):

```bash
uv run python -m scalping.cli data validate \
  --root ../demo-data --symbol XAUUSD --timeframe H1
```

Dukascopy (ticks → M1):

```bash
uv run python -m scalping.cli data download \
  --provider dukascopy --symbol XAUUSD \
  --start 2025-06-01T00:00:00Z --end 2025-06-02T00:00:00Z \
  --root ../data-rich-store
```

Tick archives with bid/ask volume:

```bash
# HuggingFace 5-year archive → M1 + flow
uv run python -m scripts.aggregate_ticks_v2 --dir ../data-ticks5 --root ../data-ticks5-store
uv run python -m scripts.aggregate_flow --dir ../data-ticks5 --root ../data-flow-store

# GitHub 2026 weekly archive → M1 + flow + M5..D1
uv run python -m scripts.ingest_2026_ticks --dir ../data-2026-ticks \
  --root ../data-forward-store --flow-root ../data-forward-flow --cache ../data-2026-cache
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/symbols` | ten symbol specs |
| `GET /api/timeframes` | supported bar sizes and their resample rules |
| `GET /api/data/status?symbol&timeframe` | rows, span, providers, provenance |
| `GET /api/data/validate?symbol&timeframe` | structural report |
| `GET /api/data/quality?symbol&timeframe&include_outliers` | read-only quality report with status and provenance |
| `POST /api/data/download` | synthetic/CSV/Dukascopy import; CSV names a file in `<data_root>/imports` |

## Current limits

- Real providers exist only for XAUUSD/XAGUSD (Dukascopy) and local CSV/JSON.
- Tick data is aggregated, not queried tick-by-tick from the store.
- No Level-2 order-book data, so DeepLOB/order-book models remain unavailable.
- The quality report is a view over the existing validator: structural checks
  plus robust-z outliers on log returns and `log1p(volume)`. There is no
  statistical anomaly model beyond that, no regime-conditioned baseline, and no
  cross-source reconciliation in the endpoint (the function exists in code but
  needs a second frame to compare).
- Quality checks read and sort the whole slice in memory, like
  `/api/data/validate`. A long M1 history can skip the outlier pass with
  `include_outliers=false`, but the frame is still loaded; streaming validation
  is not implemented.
- Gap detection is session-unaware: a weekend or holiday closure counts as a
  gap, so a multi-day slice normally reports `warn` even when every bar is well
  formed.
- Provenance is a JSON sidecar with a content checksum, not a signature: it
  detects a changed frame, not who produced it.
- Dataset versioning/checksums are recorded per dataset, but the Phase 3
  train/validation/test manifests are not built.
