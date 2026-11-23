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
fail the report by itself).

`validate_ticks` checks sorted stamps, crossed bid/ask, non-finite/negative
prices and volumes, and nulls.

API:

```bash
curl 'http://127.0.0.1:8000/api/data/validate?symbol=XAUUSD&timeframe=H1'
```

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
| `POST /api/data/download` | synthetic/CSV/Dukascopy import |

## Current limits

- Real providers exist only for XAUUSD/XAGUSD (Dukascopy) and local CSV/JSON.
- Tick data is aggregated, not queried tick-by-tick from the store.
- No Level-2 order-book data, so DeepLOB/order-book models remain unavailable.
- Validation is structural; statistical outliers and cross-source reconciliation
  are not implemented yet.
- Dataset versioning/checksums are recorded per dataset, but the Phase 3
  train/validation/test manifests are not built.
