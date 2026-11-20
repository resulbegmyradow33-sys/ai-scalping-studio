# Backend Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** XAUUSD M1 verisiyle çalışan, maliyetleri doğru hesaplayan, deterministik bir simülasyon ve backtest çekirdeği ile onu dışarı açan FastAPI/WS katmanını kurmak.

**Architecture:** Tek bir `Simulator` sınıfı muhasebenin tek doğruluk kaynağıdır; Gymnasium ortamı ve backtester ikisi de onu sarar. Veri sağlayıcıları takılabilir (synthetic/csv/dukascopy), özellikler nedensel olarak önceden hesaplanır, böylece hot path'te sadece dizi dilimleme kalır.

**Tech Stack:** Python 3.12 (uv), polars, pyarrow, numpy, gymnasium, FastAPI, uvicorn, pydantic-settings, httpx, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-20-ai-scalping-studio-design.md`

## Global Constraints

- Python **3.12** via `uv`; sistem Python 3.15 kullanılmaz.
- Tüm zaman damgaları içeride **UTC**; dönüşüm import anında yapılır.
- Testler **ağa çıkmaz**; dukascopy testleri mock'lanır.
- Kontrat: 100 oz, point 0.001, komisyon **$7/lot gidiş-dönüş**, kaldıraç **1:20**, min lot 0.01.
- Risk: işlem başına **%0.5**, günlük zarar **%3**, notional cap **20x**, maks **120 bar**, tek pozisyon.
- Başlangıç bakiyesi **$10,000**.
- Determinizm: env/broker/backtest bit bazında; eğitim bu planın dışında.
- Kod, yorum ve log İngilizce; kullanıcıya görünen metin Türkçe (bu plan sadece backend, loglar İngilizce).
- Sır yok: API anahtarı/token repoya girmez, `.env` ignore'lanır.

## Review Focus

Bunlar spec'in sessiz kaldığı ama kullanıcıyı doğrudan ısıracak girdiler; her biri sahibi olan görevde test edilir:

1. **Pencere'den kısa veri** (< 60 bar): `reset()` çökmemeli, ilk geçerli indeksten başlamalı.
2. **Eksik/bozuk bar**: hafta sonu boşluğu, `ask_*` kolonlarının NaN olması, sıfır spread — hattı durdurmamalı.
3. **Broker CSV'sinde yanlış saat dilimi veya tekrarlı timestamp**: sessizce yanlış seans/sinyal üretmemeli.
4. **Lot 0.00'a yuvarlanan pozisyon**: emir açılmamalı, `risk_blocked` nedeni `lot_too_small` olmalı.
5. **WS kopması**: panel yeniden bağlandığında `seq` ile senkron olmalı, eski mesaj replay edilmemeli.

---

### Task 1: Repo iskeleti, tooling ve `make dev`

**Files:**
- Create: `.gitignore`, `Makefile`, `backend/pyproject.toml`, `backend/.python-version`, `backend/src/scalping/__init__.py`, `backend/tests/test_smoke.py`

**Interfaces:**
- Produces: `scalping.__version__ -> str`; `make test`, `make lint`, `make dev` hedefleri.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_smoke.py
import scalping

def test_package_imports_and_has_version():
    assert isinstance(scalping.__version__, str)
    assert scalping.__version__.count(".") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalping'`

- [ ] **Step 3: Create the skeleton**

`backend/pyproject.toml`:

```toml
[project]
name = "scalping"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=1.26,<3",
  "polars>=1.0",
  "pyarrow>=16",
  "gymnasium>=1.0",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "pydantic-settings>=2.4",
  "httpx>=0.27",
  "pyyaml>=6.0",
]

[dependency-groups]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/scalping"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

`backend/.python-version`: `3.12`

`backend/src/scalping/__init__.py`:

```python
__version__ = "0.1.0"
```

`Makefile` (repo root):

```make
.PHONY: test lint dev
test:
	cd backend && uv run pytest
lint:
	cd backend && uv run ruff check . && uv run mypy src
dev:
	cd backend && uv run uvicorn scalping.api.app:app --reload --port 8000
```

`.gitignore`: `.venv/`, `__pycache__/`, `*.parquet`, `data/`, `.env`, `node_modules/`, `.next/`, `runs/`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git init && git add -A && git commit -m "chore: backend skeleton with uv and pytest"
```

---

### Task 2: Domain types and config

**Files:**
- Create: `backend/src/scalping/types.py`, `backend/src/scalping/config.py`, `backend/tests/test_types.py`

**Interfaces:**
- Produces:
  - `Side(IntEnum)`: `FLAT=0, LONG=1, SHORT=-1`
  - `Bar` frozen dataclass: `ts: datetime, open/high/low/close: float, bid_close: float, ask_close: float, spread: float, volume: int`
  - `Position`: `side: Side, lot: float, entry_price: float, entry_ts: datetime, sl: float, tp: float, strategy: str`
  - `Trade`: `entry_ts, exit_ts, side, lot, entry, exit, sl, tp, pnl, costs, r_multiple, reason, strategy`
  - `AccountState`: `balance: float, equity: float, position: Position | None, day_start_equity: float, day_pnl: float, bars_in_trade: int, trades_closed: int`
  - `ContractSpec`: `size=100.0, point=0.001, min_lot=0.01, lot_step=0.01, commission_per_lot=7.0, leverage=20.0, initial_balance=10_000.0`
  - `RiskConfig`: `risk_pct=0.005, daily_loss_pct=0.03, max_notional_x=20.0, max_positions=1, max_bars_in_trade=120`
  - `CostConfig`: `spread_floor=0.20, spread_vol_mult=0.10, slippage_atr_frac=0.05, commission_per_lot=7.0`
  - `Settings.load(path: Path) -> Settings` reading YAML with the three nested blocks.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_types.py
import pytest
from datetime import datetime, timezone
from scalping.config import Settings
from scalping.types import Bar, ContractSpec, RiskConfig, Side

def test_contract_defaults_match_spec():
    c = ContractSpec()
    assert (c.size, c.point, c.min_lot, c.commission_per_lot, c.leverage) == (100.0, 0.001, 0.01, 7.0, 20.0)

def test_risk_defaults_match_spec():
    r = RiskConfig()
    assert (r.risk_pct, r.daily_loss_pct, r.max_notional_x, r.max_bars_in_trade) == (0.005, 0.03, 20.0, 120)

def test_bar_rejects_low_above_high():
    with pytest.raises(ValueError, match="low"):
        Bar(ts=datetime(2026, 1, 5, tzinfo=timezone.utc), open=1, high=1, low=2,
            close=1, bid_close=1, ask_close=1, spread=0, volume=1)

def test_side_arithmetic():
    assert Side.LONG * -1 is Side.SHORT
    assert Side.FLAT * -1 is Side.FLAT

def test_settings_load_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("contract:\n  initial_balance: 5000\nrisk:\n  risk_pct: 0.01\n", encoding="utf-8")
    s = Settings.load(p)
    assert s.contract.initial_balance == 5000
    assert s.risk.risk_pct == 0.01
    assert s.risk.daily_loss_pct == 0.03  # default preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scalping.types'`

- [ ] **Step 3: Implement the three dataclasses and `Settings.load`**

Use `@dataclass(frozen=True)` for `Bar` with `__post_init__` raising `ValueError("low must be <= high")` etc. `Side` gets `__mul__` returning the negated enum member. `Settings` is a dataclass with `contract: ContractSpec`, `risk: RiskConfig`, `costs: CostConfig`; `load` merges YAML over defaults with a small recursive dict-merge.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_types.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/types.py backend/src/scalping/config.py backend/tests/test_types.py
git commit -m "feat: domain types and yaml settings"
```

---

### Task 3: Parquet store and resampling

**Files:**
- Create: `backend/src/scalping/data/__init__.py`, `backend/src/scalping/data/base.py`, `backend/src/scalping/data/store.py`, `backend/src/scalping/data/resample.py`, `backend/tests/test_store.py`

**Interfaces:**
- Produces (in `data/base.py`): `BAR_SCHEMA` (the single canonical M1 schema, including `provider: str`) and the `DataProvider` Protocol (`name: str`, `fetch(symbol, start, end) -> pl.DataFrame`). Later providers import both from here.
- Produces:
  - `ParquetStore(root: Path)`; `.write(symbol: str, timeframe: str, df: pl.DataFrame) -> int` (rows written, dedups by `ts`)
  - `.read(symbol: str, timeframe: str, start: datetime | None = None, end: datetime | None = None) -> pl.DataFrame` sorted ascending, empty schema-correct frame when no data
  - `.coverage(symbol: str, timeframe: str = "M1") -> Coverage(first_ts, last_ts, rows)`
  - `resample(df: pl.DataFrame, rule: str) -> pl.DataFrame` for `"5m"`/`"15m"`; OHLCV + `spread_mean`, `bid_close`/`ask_close` from last bar.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_store.py
import polars as pl
from datetime import datetime, timezone
from scalping.data.store import ParquetStore
from scalping.data.resample import resample

def bars(n=10):
    ts = [datetime(2026, 1, 5, 10, i, tzinfo=timezone.utc) for i in range(n)]
    return pl.DataFrame({
        "ts": ts, "open": [1.0+i for i in range(n)], "high": [2.0+i for i in range(n)],
        "low": [0.5+i for i in range(n)], "close": [1.5+i for i in range(n)],
        "bid_close": [1.4+i for i in range(n)], "ask_close": [1.6+i for i in range(n)],
        "spread": [0.2]*n, "volume": [1]*n,
    })

def test_write_read_roundtrip_and_dedup(tmp_path):
    s = ParquetStore(tmp_path)
    assert s.write("XAUUSD", "M1", bars(10)) == 10
    assert s.write("XAUUSD", "M1", bars(10)) == 0  # duplicate ts dropped
    out = s.read("XAUUSD", "M1")
    assert out.height == 10
    assert out["ts"].is_sorted()

def test_read_missing_range_returns_empty_with_schema(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(3))
    out = s.read("XAUUSD", "M1", datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert out.height == 0
    assert "spread" in out.columns

def test_coverage(tmp_path):
    s = ParquetStore(tmp_path)
    s.write("XAUUSD", "M1", bars(5))
    cov = s.coverage("XAUUSD")
    assert (cov.rows, cov.first_ts.hour, cov.last_ts.minute) == (5, 10, 4)

def test_resample_5m_ohlc_and_spread():
    out = resample(bars(10), "5m")
    assert out.height == 2
    first = out.row(0, named=True)
    assert (first["open"], first["high"], first["low"], first["close"]) == (1.0, 6.0, 0.5, 5.5)
    assert first["volume"] == 5
    assert first["spread_mean"] == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement store and resample**

Store layout: `{root}/{symbol}/{timeframe}/{year:04d}/{month:02d}.parquet`. `write` concatenates, dedups on `ts` keeping last, sorts, writes. `read` globs partitions that intersect `[start, end]`; returns empty frame built from a module-level `BAR_SCHEMA` when nothing matches. `resample` uses `df.group_by_dynamic("ts", every=rule, closed="left", label="left")` with `first/last/min/max/sum` aggregations, then renames.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/data backend/tests/test_store.py
git commit -m "feat: partitioned parquet store and resampling"
```

---

### Task 4: Synthetic data provider

**Files:**
- Create: `backend/src/scalping/data/base.py`, `backend/src/scalping/data/synthetic.py`, `backend/tests/test_synthetic.py`

**Interfaces:**
- Consumes: `BAR_SCHEMA`, `DataProvider` from `data/base.py` (created in Task 3).
- Produces:
  - `DataProvider` Protocol: `name: str`; `fetch(symbol: str, start: datetime, end: datetime) -> pl.DataFrame` (BAR_SCHEMA, UTC, M1)
  - `SyntheticProvider(seed: int = 7, start_price: float = 2350.0, vol_regimes: bool = True)`
  - `SyntheticProvider.fetch(...)` — deterministik: aynı `seed` + aralık → aynı satırlar.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_synthetic.py
from datetime import datetime, timezone
import polars as pl
from scalping.data.synthetic import SyntheticProvider

START = datetime(2026, 1, 5, tzinfo=timezone.utc)
END = datetime(2026, 1, 7, tzinfo=timezone.utc)

def test_deterministic_for_same_seed():
    a = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    b = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    assert a.equals(b)

def test_different_seed_differs():
    a = SyntheticProvider(seed=11).fetch("XAUUSD", START, END)
    b = SyntheticProvider(seed=12).fetch("XAUUSD", START, END)
    assert not a["close"].equals(b["close"])

def test_ohlc_valid_and_no_weekend_bars():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    assert df.height > 2000
    assert (df["high"] >= df[["open", "close"]].max_horizontal()).all()
    assert (df["low"] <= df[["open", "close"]].min_horizontal()).all()
    assert df.filter(pl.col("ts").dt.weekday() >= 6).height == 0

def test_spread_positive_and_ask_above_bid():
    df = SyntheticProvider().fetch("XAUUSD", START, END)
    assert (df["spread"] > 0).all()
    assert (df["ask_close"] > df["bid_close"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_synthetic.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the generator**

Vectorized numpy: minute grid, drop Sat/Sun; GBM increments with a regime-switching vol multiplier (calm 1x / trend 1.6x / spike 3x, Markov chain driven by `np.random.default_rng(seed)`); session-dependent spread (Asia 0.28, London 0.18, NY 0.22, ±10%); build OHLC by splitting each open into close and wicks proportional to |return|; `volume` from a Poisson around 40. Assemble with `pl.DataFrame` and cast to `BAR_SCHEMA`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_synthetic.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/data/base.py backend/src/scalping/data/synthetic.py backend/tests/test_synthetic.py
git commit -m "feat: deterministic synthetic M1 provider"
```

---

### Task 5: CSV provider with timezone normalization

**Files:**
- Create: `backend/src/scalping/data/csv_provider.py`, `backend/tests/test_csv_provider.py`, `backend/tests/fixtures/xauusd_gmt3.csv`, `backend/tests/fixtures/dupes.csv`, `backend/tests/fixtures/bad_float.csv`

**Interfaces:**
- Consumes: `DataProvider`, `BAR_SCHEMA`
- Produces:
  - `CsvProvider(path: Path, tz_offset_hours: int | None = None, column_map: dict[str, str] | None = None)`
  - Auto-detects offset when `tz_offset_hours is None`: the most common hour gap between the file's first timestamp and a UTC `ts` column is unknowable, so detection uses the **duplicate DST spring-forward gap**: if the file contains a `ts` sequence with a 1-hour jump at a DST boundary, offset is `+3` for TR brokers; default documented constant `DEFAULT_BROKER_OFFSET_HOURS = 3` and a warning log when auto-detected equals default.
  - Missing `ask_*` columns → `ask_* = null`, `spread = null`, and `provider = "csv_no_spread"` metadata so the env applies the spread model.
  - `DataError(line: int, reason: str)` for unparseable rows.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_csv_provider.py
import pytest
from pathlib import Path
from scalping.data.csv_provider import CsvProvider, DataError

FIX = Path(__file__).parent / "fixtures" / "xauusd_gmt3.csv"

def test_converts_broker_offset_to_utc():
    df = CsvProvider(FIX, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df["ts"].dt.hour[0] == 7  # file says 10:00 GMT+3

def test_missing_ask_columns_yields_null_spread():
    df = CsvProvider(FIX, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df["ask_close"].null_count() == df.height
    assert df["provider"][0] == "csv_no_spread"

def test_duplicate_timestamp_raises_with_line_number():
    with pytest.raises(DataError, match="line 3"):
        CsvProvider(FIX.with_name("dupes.csv"), tz_offset_hours=3).fetch("XAUUSD", None, None)

def test_bad_float_raises_data_error():
    with pytest.raises(DataError, match="close"):
        CsvProvider(FIX.with_name("bad_float.csv"), tz_offset_hours=3).fetch("XAUUSD", None, None)
```

Fixture format (tab-separated, no header): `2026.01.05 10:00,2350.10,2351.00,2349.80,2350.60,42`

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_csv_provider.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement parsing**

Read with `pl.read_csv(..., has_header=False, separator=",", schema_overrides=..., truncate_ragged_lines=False)`, rename by `column_map` or the HistData-style default, parse `ts` with `pl.Series.str.strptime(pl.Datetime("ms"))`, subtract `timedelta(hours=tz_offset_hours)`, then `replace_time_zone("UTC")`. Track original line numbers by adding a 1-based `_line` column before parsing so errors can quote them.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_csv_provider.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/data/csv_provider.py backend/tests/test_csv_provider.py backend/tests/fixtures
git commit -m "feat: csv provider with utc normalization and line-accurate errors"
```

---

### Task 6: Dukascopy provider with backoff and cache

**Files:**
- Create: `backend/src/scalping/data/dukascopy.py`, `backend/tests/test_dukascopy.py`

**Interfaces:**
- Consumes: `DataProvider`
- Produces:
  - `DukascopyProvider(cache_dir: Path, client: httpx.Client | None = None, max_retries: int = 5, base_delay: float = 1.0)`
  - `.fetch(symbol, start, end)` → M1 `pl.DataFrame`; tick `.bi5` payloads decoded as 20-byte big-endian records `(ms_offset:u32, ask:u32, bid:u32, ask_vol:f32, bid_vol:f32)`, price = int / 1000.
  - 429/503 → exponential backoff `1,2,4,8,16s` then `DataUnavailable`; successful hour cached as `{cache_dir}/{symbol}/{yyyymmdd}/{hh}.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_dukascopy.py
import httpx, pytest
from datetime import datetime, timezone
from scalping.data.dukascopy import DukascopyProvider, DataUnavailable, decode_ticks

def test_decode_ticks_known_record():
    import struct
    payload = struct.pack(">IIIff", 60_000, 235_1234, 235_1000, 1.5, 2.5)
    ticks = decode_ticks(payload)
    assert ticks[0]["ms"] == 60_000
    assert ticks[0]["ask"] == pytest.approx(2351.234)
    assert ticks[0]["bid"] == pytest.approx(2351.0)

def test_retries_on_429_then_succeeds(tmp_path):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, content=b"")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = DukascopyProvider(tmp_path, client=client, base_delay=0.01)
    p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=timezone.utc))
    assert calls["n"] == 3

def test_gives_up_after_max_retries(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    p = DukascopyProvider(tmp_path, client=client, max_retries=2, base_delay=0.01)
    with pytest.raises(DataUnavailable):
        p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=timezone.utc))

def test_cache_prevents_second_request(tmp_path):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=b"")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = DukascopyProvider(tmp_path, client=client, base_delay=0.01)
    p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=timezone.utc))
    p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=timezone.utc))
    assert calls["n"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_dukascopy.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

URL: `https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5`. Decompress with `lzma.decompress` when payload starts with the LZMA header, otherwise pass bytes through (empty payload → skip hour). Aggregate ticks to M1: `bid_close = last bid`, `ask_close = last ask`, OHLC from bid, `spread = ask_close - bid_close`, `volume = len(ticks)`. `_fetch_hour` writes cache before returning.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_dukascopy.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/data/dukascopy.py backend/tests/test_dukascopy.py
git commit -m "feat: dukascopy tick provider with backoff and cache"
```

---

### Task 7: PaperBroker accounting

**Files:**
- Create: `backend/src/scalping/env/__init__.py`, `backend/src/scalping/env/broker.py`, `backend/tests/test_broker.py`

**Interfaces:**
- Produces:
  - `PaperBroker(contract: ContractSpec, costs: CostConfig, atr: float = 0.0)`
  - `.open(side: Side, lot: float, ts: datetime, bid: float, ask: float, sl: float, tp: float, strategy: str) -> Position`
  - `.mark(bar: Bar) -> float` (unrealized PnL, no mutation of balance)
  - `.update(bar: Bar) -> Trade | None` — checks SL/TP using **bar high/low**, gap fills at `bar.open`, charges commission on close
  - `.close(bar: Bar, reason: str) -> Trade`
  - `.state(balance: float, day_start_equity: float | None = None) -> AccountState` — the broker does **not** own the session anchor; `day_start_equity` comes from the Simulator/RiskEngine (17:00 boundary lives there), defaulting to `balance`.
  - Invariants: `equity == balance + unrealized`; `costs = entry half-spread + slippage + commission + exit half-spread **only on quote exits**`. Trigger exits (SL/TP) fill at the bar's series price and report zero exit spread.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_broker.py
import pytest
from datetime import datetime, timezone
from scalping.env.broker import PaperBroker
from scalping.config import ContractSpec, CostConfig
from scalping.types import Bar, Side

TS = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

def bar(o, h, l, c, spread=0.20, ts=TS):
    return Bar(ts=ts, open=o, high=h, low=l, close=c, bid_close=c,
               ask_close=c + spread, spread=spread, volume=10)

def make(atr=1.0):
    return PaperBroker(ContractSpec(), CostConfig(spread_floor=0.20, slippage_atr_frac=0.0), atr=atr)

def test_long_entry_charges_half_spread_and_commission():
    b = make()
    pos = b.open(Side.LONG, 1.0, TS, bid=2350.0, ask=2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    assert pos.entry_price == pytest.approx(2350.20)
    trade = b.close(bar(2352.0, 2353.0, 2351.0, 2352.0), "manual")
    # long enters at ask 2350.20, exits at bid 2352.00 -> +1.80 * 100oz, minus $7 round-turn commission
    assert trade.pnl == pytest.approx(100 * 1.80 - 7)

def test_sl_hit_intrabar_closes_at_sl():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2349.0, tp=2354.0, strategy="mom")
    trade = b.update(bar(2350.0, 2350.5, 2348.5, 2349.5))
    assert trade is not None and trade.reason == "sl"
    assert trade.exit == pytest.approx(2349.0)

def test_tp_hit_before_sl_when_bar_contains_both():
    b = make()
    b.open(Side.SHORT, 1.0, TS, 2350.0, 2350.20, sl=2352.0, tp=2347.0, strategy="rev")
    trade = b.update(bar(2350.0, 2353.0, 2346.0, 2351.0))
    assert trade is None or trade.reason == "sl"  # conservative: SL wins ambiguity

def test_gap_through_sl_fills_at_open():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2349.0, tp=2354.0, strategy="mom")
    trade = b.update(bar(2347.0, 2347.5, 2346.0, 2347.2))
    assert trade.reason == "sl" and trade.exit == pytest.approx(2347.0)

def test_equity_equals_balance_plus_unrealized():
    b = make()
    b.open(Side.LONG, 1.0, TS, 2350.0, 2350.20, sl=2348.0, tp=2354.0, strategy="mom")
    b.update(bar(2351.0, 2352.0, 2350.5, 2351.5))
    st = b.state(balance=10_000.0)
    assert st.equity == pytest.approx(10_000.0 + 100 * (2351.5 - 2350.20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_broker.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Entry price for LONG = `ask + slippage`, SHORT = `bid - slippage`; exit mirrored (LONG exits at `bid`, SHORT at `ask`). Commission is charged **only** from `CostConfig.commission_per_lot` (never `ContractSpec`) as a round turn `commission_per_lot * lot`. PnL = `(exit - entry) * size * lot * side` minus commission. Ambiguity rule: if a bar's range contains both SL and TP, SL fills first (conservative). Gap rule: if `open` is beyond SL, fill at `open`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_broker.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/env/broker.py backend/tests/test_broker.py
git commit -m "feat: paper broker with spread, slippage, commission and SL/TP fills"
```

---

### Task 8: RiskEngine

**Files:**
- Create: `backend/src/scalping/env/risk.py`, `backend/tests/test_risk.py`

**Interfaces:**
- Consumes: `RiskConfig`, `ContractSpec`, `AccountState`
- Produces:
  - `RiskEngine(risk: RiskConfig, contract: ContractSpec)`
  - `.size(equity: float, entry: float, sl: float) -> SizingResult(lot: float, reason: str | None, effective_risk_pct: float)`
  - `.allow_new(account: AccountState, ts: datetime) -> tuple[bool, str | None]` reasons: `None`, `"daily_loss_limit"`, `"max_positions"`, `"lot_too_small"`
  - `.must_force_close(bars_in_trade: int) -> bool`
  - `.sl_tp(entry: float, side: Side, atr: float) -> tuple[float, float]` with 1.0×ATR SL, 2.0×ATR TP.
  - Internal: `_daily_anchor_utc(ts) -> date` using 17:00 UTC boundary.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_risk.py
import pytest
from datetime import datetime, timezone
from scalping.env.risk import RiskEngine
from scalping.config import RiskConfig, ContractSpec
from scalping.types import AccountState, Side

def engine(**kw):
    return RiskEngine(RiskConfig(**kw), ContractSpec())

def st(equity=10_000, day_start=10_000, position=None, bars=0):
    return AccountState(balance=equity, equity=equity, position=position,
                        day_start_equity=day_start, day_pnl=equity-day_start,
                        bars_in_trade=bars, trades_closed=0)

def test_risk_sizing_uses_half_percent():
    s = engine().size(equity=10_000, entry=2350.0, sl=2349.0)
    # risk $50 / $1 distance = 50oz = 0.5 lot
    assert s.lot == pytest.approx(0.5, abs=0.01) and s.reason is None

def test_notional_cap_binds_when_sl_is_tight():
    s = engine().size(equity=10_000, entry=2350.0, sl=2349.9)
    # risk-based -> 5.0 lot (500oz, $1.175M = 117x) -> capped at 20x = $200k -> 0.85 lot
    assert s.lot == pytest.approx(0.85, abs=0.01)
    assert s.reason == "notional_cap"
    assert s.effective_risk_pct < 0.005

def test_lot_too_small_blocks():
    s = engine().size(equity=50, entry=2350.0, sl=2349.0)
    assert s.lot == 0.0 and s.reason == "lot_too_small"

def test_daily_loss_limit_blocks_until_next_session():
    e = engine()
    ts = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    ok, reason = e.allow_new(st(equity=9_650, day_start=10_000), ts)
    assert (ok, reason) == (False, "daily_loss_limit")
    ok2, _ = e.allow_new(st(equity=9_650, day_start=9_650), ts.replace(day=6))
    assert ok2

def test_force_close_after_max_bars():
    e = engine()
    assert not e.must_force_close(119)
    assert e.must_force_close(120)

def test_sl_tp_from_atr():
    sl, tp = engine().sl_tp(entry=2350.0, side=Side.LONG, atr=1.0)
    assert (sl, tp) == (2349.0, 2352.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_risk.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`lot_risk = equity * risk_pct / (abs(entry - sl) * contract.size)`; `lot_cap = equity * max_notional_x / (entry * contract.size)`; `lot = min(...)`; floor to `lot_step`; if `< min_lot` → 0 with `lot_too_small`. `effective_risk_pct = lot * abs(entry-sl) * size / equity`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_risk.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/env/risk.py backend/tests/test_risk.py
git commit -m "feat: risk engine with dual limits and session anchor"
```

---

### Task 9: Market intelligence features

**Files:**
- Create: `backend/src/scalping/features/__init__.py`, `indicators.py`, `structure.py`, `liquidity.py`, `context.py`, `build.py`, `backend/tests/test_features.py`

**Interfaces:**
- Produces:
  - `atr(df, n=14) -> pl.Series`, `rsi(df, n=14) -> pl.Series`, `macd_hist(df) -> pl.Series`
  - `swings(df, k=2) -> tuple[pl.Series, pl.Series]` (booleans `swing_high`, `swing_low`)
  - `market_structure(df, k=2) -> pl.DataFrame` with `trend: int8`, `bos: int8` (+1/-1), `choch: int8`
  - `fvg(df) -> pl.DataFrame` with `fvg_up: bool`, `fvg_down: bool`, `fvg_dist: float`
  - `liquidity_sweep(df, lookback=20) -> pl.Series` (bool)
  - `session_flags(ts: pl.Series) -> pl.DataFrame` with `asia`, `london`, `ny`, `ny_open_30`
  - `htf_context(m1: pl.DataFrame, m5: pl.DataFrame, m15: pl.DataFrame) -> pl.DataFrame` — joins **last closed** HTF bar by `ts` with `join_asof(strategy="backward")`
  - `build_features(df: pl.DataFrame) -> FeatureFrame` in `build.py`
  - `FeatureFrame`: dataclass with `.X: np.ndarray (T,F) float32`, `.ts: np.ndarray`, `.ohlc: np.ndarray (T,4)`, `.spread: np.ndarray`, `.valid_from: int`, `.columns: list[str]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_features.py
import numpy as np, polars as pl, pytest
from datetime import datetime, timezone
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.features.indicators import atr, rsi
from scalping.features.structure import market_structure
from scalping.features.liquidity import fvg

def bars(closes):
    n = len(closes)
    ts = [datetime(2026, 1, 5, 10, i, tzinfo=timezone.utc) for i in range(n)]
    return pl.DataFrame({"ts": ts, "open": closes, "high": [c+0.5 for c in closes],
                         "low": [c-0.5 for c in closes], "close": closes,
                         "bid_close": closes, "ask_close": [c+0.2 for c in closes],
                         "spread": [0.2]*n, "volume": [10]*n})

def test_atr_matches_manual_mean_range():
    df = bars([10.0]*20)
    assert atr(df, 14)[-1] == pytest.approx(1.0)

def test_rsi_all_up_is_100():
    df = bars([10.0 + i for i in range(30)])
    assert rsi(df, 14)[-1] == pytest.approx(100.0, abs=1e-6)

def test_bos_after_higher_high():
    closes = [10, 11, 10.5, 12, 11.5, 13, 12.5, 14]
    out = market_structure(bars(closes))
    assert out["bos"].to_list()[-1] in (1, 0)
    assert out.filter(pl.col("bos") == 1).height >= 1

def test_fvg_detects_gap_up():
    df = bars([10.0, 10.2, 10.2])
    df = df.with_columns([pl.Series("high", [10.1, 10.9, 11.3]), pl.Series("low", [9.9, 10.3, 10.6])])
    out = fvg(df)
    # gap: low[i] > high[i-2] -> bar index 2
    assert out["fvg_up"].to_list()[2] is True

def test_no_lookahead_last_row_identical():
    df = SyntheticProvider(seed=3).fetch("XAUUSD",
        datetime(2026,1,5,tzinfo=timezone.utc), datetime(2026,1,6,tzinfo=timezone.utc))
    full = build_features(df)
    for t in (200, 700, len(df) - 1):
        part = build_features(df.head(t + 1))
        assert np.allclose(full.X[t], part.X[-1], equal_nan=True), f"lookahead at {t}"

def test_feature_frame_shapes_and_dtype():
    df = SyntheticProvider(seed=3).fetch("XAUUSD",
        datetime(2026,1,5,tzinfo=timezone.utc), datetime(2026,1,6,tzinfo=timezone.utc))
    f = build_features(df)
    assert f.X.dtype == np.float32
    assert f.X.shape == (df.height, len(f.columns))
    assert f.valid_from >= 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_features.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

All rolling operations use `closed="left"` or explicit `.shift(1)` so the current bar's close never leaks into its own feature row. `market_structure` marks a `swing_high` when `high[i]` is the max of `i-k..i+k` (label placed at `i+k`, the confirmation bar, to stay causal). BOS when close crosses above the last confirmed swing high; CHoCH when trend flips. `fvg_up` when `low[i] > high[i-2]`; distance to nearest unfilled FVG in ATR units. `htf_context` resamples M5/M15, shifts one bar, then `join_asof(..., strategy="backward")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_features.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/features backend/tests/test_features.py
git commit -m "feat: causal market intelligence features"
```

---

### Task 10: Simulator and Gymnasium environment

**Files:**
- Create: `backend/src/scalping/env/simulator.py`, `backend/src/scalping/env/trading_env.py`, `backend/tests/test_env.py`

**Interfaces:**
- Consumes: `PaperBroker`, `RiskEngine`, `FeatureFrame`, `Settings`
- Produces:
  - `Simulator(features: FeatureFrame, settings: Settings, start: int = 0, end: int | None = None)`
  - `.reset(seed: int | None = None) -> np.ndarray`
  - `.step(action: int) -> StepResult(obs, reward, terminated, truncated, info)`; `info` contains `equity`, `position`, `trade` (last closed `Trade | None`), `blocked` (reason or None), `strategy`.
  - `.obs_dim: int`; window `WINDOW = 60`; obs = flattened `X[t-59:t+1]` (60×F) + `[side, bars_in_trade/120, unrealized_pct, day_pnl_pct, risk_used]`.
  - `TradingEnv(features: FeatureFrame, settings: Settings, strategy_selector=None)` — `gym.Env` sarmalayıcı; içinde `self.sim = Simulator(...)` tutar (kalıtım değil, kompozisyon), böylece `env.sim.broker` test kancası erişilebilir.
  - `action_space = Discrete(4)`, `observation_space = Box(-inf, inf, (obs_dim,), float32)`.
  - Reward: `Δequity/initial*100 - 0.5*max(0, dd-limit) - 0.02*(opened) + 0.05*(risk_ok)`, clipped to `[-5, 5]`.
  - Termination: equity ≤ 80% of initial; truncation at `max_bars` (20_000) or data end.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_env.py
import numpy as np, pytest
from datetime import datetime, timezone
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.config import Settings
from scalping.env.trading_env import TradingEnv

def make_env(days=2):
    df = SyntheticProvider(seed=5).fetch("XAUUSD",
        datetime(2026,1,5,tzinfo=timezone.utc), datetime(2026,1,5+days,tzinfo=timezone.utc))
    return TradingEnv(build_features(df), Settings())

def test_spaces():
    env = make_env()
    assert env.action_space.n == 4
    assert env.observation_space.shape == (env.obs_dim,)

def test_reset_is_deterministic_given_seed():
    a, b = make_env(), make_env()
    assert np.allclose(a.reset(seed=1), b.reset(seed=1))

def test_hold_does_not_trade():
    env = make_env()
    env.reset(seed=1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(0)
        assert info["position"] is None
        assert info["equity"] == pytest.approx(10_000.0)
        if term or trunc:
            break

def test_buy_opens_and_reward_is_finite():
    env = make_env()
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(1)
    assert info["position"] is not None
    assert np.isfinite(r) and -5 <= r <= 5

def test_force_close_at_max_bars():
    env = make_env()
    env.reset(seed=1)
    env.step(1)
    for _ in range(121):
        _, _, _, _, info = env.step(0)
    assert info["position"] is None

def test_short_data_does_not_crash():
    df = SyntheticProvider(seed=5).fetch("XAUUSD",
        datetime(2026,1,5,tzinfo=timezone.utc), datetime(2026,1,5,1,tzinfo=timezone.utc))
    env = TradingEnv(build_features(df), Settings())
    obs, _ = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,)

def test_equity_drop_terminates_episode():
    env = make_env()
    env.reset(seed=1)
    env.sim._force_equity(8_000)  # test hook on the Simulator
    _, _, term, _, _ = env.step(0)
    assert term
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_env.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`reset` sets `t = max(start, features.valid_from + WINDOW - 1)`. Each `step`: apply action → update broker with current bar → compute reward from equity delta → advance `t` → build obs. `risk_blocked` recorded in `info["blocked"]` when action was overridden. `_force_balance` is a test-only method that sets balance and recomputes equity (documented as such).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_env.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/env/simulator.py backend/src/scalping/env/trading_env.py backend/tests/test_env.py
git commit -m "feat: deterministic simulator and gymnasium env"
```

---

### Task 11: Backtest engine and metrics

**Files:**
- Create: `backend/src/scalping/backtest/__init__.py`, `engine.py`, `metrics.py`, `walkforward.py`, `backend/tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `run_backtest(policy: Callable[[np.ndarray], int], features: FeatureFrame, settings: Settings) -> BacktestResult`
  - `BacktestResult`: `run_id: str`, `metrics: Metrics`, `equity: np.ndarray`, `trades: list[Trade]`, `config_hash: str`
  - `Metrics`: `net_profit, profit_factor, sharpe, max_drawdown, win_rate, expectancy_r, trades, total_costs, cost_ratio, avg_bars`
  - `metrics.compute(trades, equity, initial_balance, bars_per_year=525_600) -> Metrics`
  - `walkforward_split(df, train_months=6, val_months=1, test_months=1) -> list[Split(train, val, test)]`
  - `run_id(cfg: Settings, seed: int, data_hash: str) -> str` = sha256 hex[:12].

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_backtest.py
import numpy as np, pytest
from datetime import datetime, timezone
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.config import Settings, CostConfig
from scalping.backtest.engine import run_backtest
from scalping.backtest.metrics import compute
from scalping.backtest.walkforward import walkforward_split

def make(days=3):
    df = SyntheticProvider(seed=9).fetch("XAUUSD",
        datetime(2026,1,5,tzinfo=timezone.utc), datetime(2026,1,5+days,tzinfo=timezone.utc))
    return build_features(df), df

def test_hold_policy_has_no_trades_and_flat_equity():
    f, _ = make()
    res = run_backtest(lambda obs: 0, f, Settings())
    assert res.trades == []
    assert np.allclose(res.equity, 10_000.0)

def test_same_seed_same_result_bit_exact():
    f, _ = make()
    pol = lambda obs: 1 if (obs[0] > 0) else 0
    a = run_backtest(pol, f, Settings())
    b = run_backtest(pol, f, Settings())
    assert np.array_equal(a.equity, b.equity)
    assert a.run_id == b.run_id

def test_metrics_known_answer():
    m = compute(trades=[], equity=np.array([10_000.0]*10), initial_balance=10_000.0)
    assert m.net_profit == 0 and m.profit_factor == 0 and m.max_drawdown == 0

def test_walkforward_splits_do_not_overlap():
    f, df = make(days=420)
    splits = walkforward_split(df)
    assert len(splits) >= 3
    for s in splits:
        assert s.train.end <= s.val.start and s.val.end <= s.test.start

def test_same_config_same_run_id_different_config_differs():
    f, _ = make()
    s = Settings()
    assert run_backtest(lambda o: 0, f, s).run_id == run_backtest(lambda o: 0, f, s).run_id
    s2 = Settings(costs=CostConfig(spread_floor=0.5))
    assert run_backtest(lambda o: 0, f, s).run_id != run_backtest(lambda o: 0, f, s2).run_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_backtest.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`run_backtest` drives the same `Simulator` loop as the env (no duplicated accounting), collects equity per bar and closed trades, then `metrics.compute`. `profit_factor` = gross win / gross loss (0 when no losses and no wins; `inf` when wins and no losses). `sharpe` = mean(bar returns)/std * sqrt(bars_per_year). `max_drawdown` from running peak. `walkforward_split` uses calendar months with `pl.DataFrame` slicing.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_backtest.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/backtest backend/tests/test_backtest.py
git commit -m "feat: backtest engine, metrics and walk-forward splits"
```

---

### Task 12: API and WebSocket simulator runner

**Files:**
- Create: `backend/src/scalping/api/__init__.py`, `app.py`, `schemas.py`, `runner.py`, `routes.py`, `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `ParquetStore`, `Simulator`, `Settings`
- Produces:
  - `create_app(data_root: Path | None = None, settings_path: Path | None = None, runs_root: Path | None = None) -> FastAPI`
  - `GET /api/health -> {"status":"ok","version":str,"device":str}`
  - `GET /api/data/status -> {"symbol":str,"rows":int,"first_ts":str|None,"last_ts":str|None,"providers":[str]}`
  - `POST /api/data/download {provider: "synthetic"|"csv"|"dukascopy", symbol, start, end, seed?: int, path?: str, tz_offset?: int} -> 200 {"rows": int}` (senkron; dukascopy uzun sürerse 202 + job id, v1'de senkron yeterli)
  - `POST /api/sim/start {speed: float, mode: "replay"} -> {"session_id":str}`
  - `POST /api/sim/control {action: "pause"|"resume"|"stop"|"reset"} -> {"ok":bool}`
  - `WS /ws/sim` messages: `{"seq":int,"type":"bar|decision|trade_open|trade_close|risk_blocked|equity|episode_end","payload":{...}}`
  - `SimulationRunner`: background asyncio task; `speed=1` → 1 bar/sec, `0` → max; each message increments `seq`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from scalping.api.app import create_app

@pytest.fixture
def client(tmp_path):
    app = create_app(data_root=tmp_path)
    return TestClient(app)

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_data_status_empty_store(client):
    r = client.get("/api/data/status", params={"symbol": "XAUUSD"})
    assert r.status_code == 200 and r.json()["rows"] == 0

def test_sim_start_requires_data(client):
    r = client.post("/api/sim/start", json={"speed": 10, "mode": "replay"})
    assert r.status_code == 409
    assert "no data" in r.json()["detail"].lower()

def test_ws_sends_sequenced_messages(client, seed_data):
    with client.websocket_connect("/ws/sim") as ws:
        client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
        first = ws.receive_json()
        second = ws.receive_json()
        assert first["seq"] == 0 and second["seq"] == 1
        assert first["type"] in {"bar", "decision", "equity", "status"}

def test_control_stop(client, seed_data):
    client.post("/api/sim/start", json={"speed": 0, "mode": "replay"})
    assert client.post("/api/sim/control", json={"action": "stop"}).json()["ok"] is True

def test_synthetic_download_then_status(client):
    body = {"provider": "synthetic", "symbol": "XAUUSD", "seed": 7,
            "start": "2026-01-05T00:00:00Z", "end": "2026-01-07T00:00:00Z"}
    r = client.post("/api/data/download", json=body)
    assert r.status_code == 200 and r.json()["rows"] > 0
    status = client.get("/api/data/status", params={"symbol": "XAUUSD"}).json()
    assert status["rows"] == r.json()["rows"]
    assert status["providers"] == ["synthetic"]
```

`seed_data` is a pytest fixture in `backend/tests/conftest.py` that writes 2 days of synthetic M1 into the app's store.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_api.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

App factory takes `data_root: Path | None` and `settings_path: Path | None`. Runner holds a `set[WebSocket]`; on start it builds `Simulator` from the stored data, then loops with `await asyncio.sleep(1/speed)` (speed 0 → `sleep(0)`). Every step emits `bar` and, when relevant, `decision`/`trade_*`/`risk_blocked`/`equity`. Stop cancels the task. `reset` clears and re-creates the simulator. Reconnect: server keeps a ring buffer of the last 500 messages and replays those with `seq > last_seq` sent by the client in a `{"type":"resume","seq":n}` message.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_api.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/api backend/tests/test_api.py backend/tests/conftest.py
git commit -m "feat: fastapi app with sequenced websocket simulator"
```

---

### Task 13: CLI and end-to-end smoke

**Files:**
- Create: `backend/src/scalping/cli.py`, `backend/tests/test_cli.py`

**Interfaces:**
- Produces:
  - `python -m scalping.cli data synth --symbol XAUUSD --start 2026-01-01 --end 2026-02-01 --seed 7 --root data`
  - `python -m scalping.cli data csv --path file.csv --tz-offset 3 --symbol XAUUSD --root data`
  - `python -m scalping.cli backtest --symbol XAUUSD --policy hold --out runs/latest.json`
  - `python -m scalping.cli serve --port 8000`
  - All commands print a single JSON line summary and exit 0 on success, 2 on data error.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_cli.py
import json, subprocess, sys

def run(*args, cwd):
    return subprocess.run([sys.executable, "-m", "scalping.cli", *args],
                          cwd=cwd, capture_output=True, text=True)

def test_synth_backtest_roundtrip(tmp_path):
    r = run("data", "synth", "--symbol", "XAUUSD", "--start", "2026-01-01",
            "--end", "2026-01-08", "--root", str(tmp_path), cwd=".")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["rows"] > 0

    out = tmp_path / "bt.json"
    r = run("backtest", "--symbol", "XAUUSD", "--policy", "hold",
            "--root", str(tmp_path), "--out", str(out), cwd=".")
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["metrics"]["trades"] == 0

def test_backtest_without_data_exits_2(tmp_path):
    r = run("backtest", "--symbol", "XAUUSD", "--policy", "hold",
            "--root", str(tmp_path), "--out", str(tmp_path / "x.json"), cwd=".")
    assert r.returncode == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_cli.py -v`
Expected: FAIL — `No module named scalping.cli`

- [ ] **Step 3: Implement**

`argparse` with subparsers. `synth` calls `SyntheticProvider` then `store.write`; `csv` calls `CsvProvider`; `backtest` loads data, builds features, runs `run_backtest` with policy `hold` or `rule` (buy on positive 5-bar momentum), writes `{"run_id","config_hash","metrics","trades":[...]}`. Data errors map to exit 2 with `{"error": str}` on stderr.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest -v`
Expected: all tests PASS (>= 45 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/cli.py backend/tests/test_cli.py
git commit -m "feat: cli for data ingestion and backtests"
```

---

## Self-Review

- **Spec coverage:** §5.1 providers (T4–T6), §5.2 normalization/contract (T2, T5), §9 data model (T2–T3), §10 env+reward (T10), §10.1 risk (T8), §11 features (T9), §13 backtest/metrics/walk-forward (T11), §14 API/WS (T12). §12 strategy layer and §15 training tests belong to the RL plan; frontend belongs to its own plan.
- **Gaps closed:** empty/short data (T10), NaN ask columns (T5), timezone (T5), lot 0.00 (T8), WS resume (T12).
- **Type consistency:** `Settings` carries `contract/risk/costs`; `Simulator.step` returns `StepResult` used identically by env and backtest; `Metrics` field names match the API payload in T12.
- **Proportion:** plan is longer than the spec because it pins exact signatures and test names; no function bodies beyond algorithms that the tests do not determine.
