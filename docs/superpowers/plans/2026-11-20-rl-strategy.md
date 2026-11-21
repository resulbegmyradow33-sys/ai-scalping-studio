# RL and Strategy Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beş strateji ailesini, offline etiketli LinUCB seçiciyi ve PPO ajanını eğitip kaydeden, walk-forward ile değerlendiren katmanı kurmak.

**Architecture:** Stratejiler saf ve vektörel sinyal üretir. Etiketleme, aynı broker muhasebesini kullanarak her bar için beş stratejinin varsayımsal R'sini hesaplar. LinUCB bu etiketlerle kapalı formda ön-eğitilir ve PPO sırasında dondurulur. PPO, seçilen strateji bağlamını gözlemine alarak aksiyon öğrenir. Her koşu `runs/<run_id>/` altına model + config + metrik yazar.

**Tech Stack:** stable-baselines3 (PPO), torch, numpy, polars, gymnasium, optuna (sadece hiperparametre taraması için, opsiyonel).

**Spec:** `docs/superpowers/specs/2026-11-20-ai-scalping-studio-design.md`

## Global Constraints

- Bu plan **backend-core planı tamamlanmış** olmayı varsayar: `Simulator`, `FeatureFrame`, `PaperBroker`, `Settings`, `run_backtest` hazır.
- Bandit PPO eğitimi sırasında **dondurulur**; çevrimiçi güncelleme varsayılan kapalı.
- Eğitim determinizmi: aynı seed + aynı thread sayısı + aynı sürümler → aynı metrikler (tolerans raporlanır). Bit-exactlik iddia edilmez.
- Varsayılan eğitim bütçesi 2 saatlik checkpoint'lerle kesilebilir; üst sınır yok.
- Sentetik veride elde edilen sonuçlar **finansal kanıt değildir**; rapora bu cümle yazılır.
- `torch.set_num_threads(1)` ve `OMP_NUM_THREADS=1` eğitimde sabitlenir (8 GB makine).
- Hiçbir koşu, kendi test setinde raporlanmaz; walk-forward zorunlu.

## Review Focus

1. **Yetersiz veri**: < 5.000 bar ile eğitim başlatılırsa net hata vermeli, yarım model kaydetmemeli.
2. **Sıfır işlemli bandit**: hiçbir strateji sinyal üretmezse etiketleme boş dönmeli, LinUCB rastgele kollara düşmeli, çökmemeli.
3. **Yarıda kesilen eğitim**: Ctrl-C sonrası son checkpoint yüklenebilir olmalı, bozuk dosya bırakmamalı.
4. **Dejenere ödül**: tüm episode'lar aynı ödülü verirse eğitim "başarılı" sayılmamalı; entropi ve aksiyon dağılımı rapora girmeli.
5. **Aşırı uyum sinyali**: eğitim ödülü artarken test R'si düşüyorsa rapor bunu `overfit_warning` olarak işaretlemeli.

---

### Task 1: Strategy signal functions

**Files:**
- Create: `backend/src/scalping/strategy/__init__.py`, `signals.py`, `backend/tests/test_signals.py`

**Interfaces:**
- Produces:
  - `Strategy(StrEnum)`: `MOMENTUM="momentum"`, `REVERSAL="reversal"`, `SWEEP="sweep"`, `FVG="fvg"`, `TREND="trend"`
  - `ALL_STRATEGIES: tuple[Strategy, ...]` (sabit sıra — bandit kol indeksleri buna bağlı)
  - `strategy_signals(df: pl.DataFrame) -> pl.DataFrame` — `ts` + her strateji için `sig_<name>: int8` (-1/0/+1) ve `strength_<name>: float32` [0,1]
  - `active_strategy(row, bandit_choice: int) -> tuple[int, float]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_signals.py
import polars as pl, pytest
from datetime import datetime, timezone
from scalping.strategy.signals import ALL_STRATEGIES, Strategy, strategy_signals

def bars(closes, highs=None, lows=None):
    n = len(closes)
    ts = [datetime(2026, 1, 5, 10, i, tzinfo=timezone.utc) for i in range(n)]
    return pl.DataFrame({
        "ts": ts, "open": closes, "close": closes,
        "high": highs or [c + 0.5 for c in closes],
        "low": lows or [c - 0.5 for c in closes],
        "bid_close": closes, "ask_close": [c + 0.2 for c in closes],
        "spread": [0.2] * n, "volume": [10] * n,
    })

def test_all_strategies_are_stable_order():
    assert [s.value for s in ALL_STRATEGIES] == ["momentum", "reversal", "sweep", "fvg", "trend"]

def test_output_has_one_column_pair_per_strategy():
    out = strategy_signals(bars([10 + (i % 7) * 0.1 for i in range(300)]))
    for s in ALL_STRATEGIES:
        assert f"sig_{s.value}" in out.columns
        assert f"strength_{s.value}" in out.columns
    assert out.height == 300

def test_signals_only_take_valid_values():
    out = strategy_signals(bars([10 + (i % 11) * 0.2 for i in range(500)]))
    for s in ALL_STRATEGIES:
        assert set(out[f"sig_{s.value}"].unique()).issubset({-1, 0, 1})
        assert out[f"strength_{s.value}"].min() >= 0.0
        assert out[f"strength_{s.value}"].max() <= 1.0

def test_momentum_fires_on_sustained_uptrend():
    closes = [10 + i * 0.5 for i in range(60)]
    out = strategy_signals(bars(closes))
    assert (out["sig_momentum"] == 1).sum() > 0

def test_sweep_requires_wick_recovery():
    closes = [10.0] * 50
    lows = [9.8] * 50
    lows[45], lows[46] = 9.0, 9.9   # deep wick, recovered
    highs = [10.2] * 50
    out = strategy_signals(bars(closes, highs=highs, lows=lows))
    assert out["sig_sweep"][46] == 1

def test_signals_are_causal():
    closes = [10 + (i % 9) * 0.3 for i in range(200)]
    df = bars(closes)
    full = strategy_signals(df)
    part = strategy_signals(df.head(150))
    for s in ALL_STRATEGIES:
        assert full[f"sig_{s.value}"][:150].to_list() == part[f"sig_{s.value}"].to_list()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_signals.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the five rules**

- momentum: `close > ema(20)` and `close > close.shift(5)` and `max(atr_ratio, 5-bar displacement / ATR) > 1.1` → +1; mirror for -1. (Amended: a constant-range trend forces `atr_ratio == 1.0`, so the volatility-only gate could never fire on the brief's own trend test.)
- reversal: `rsi(14) > 75` and bearish candle → -1; `rsi < 25` and bullish candle → +1.
- sweep: `low < min(low, 20).shift(1)` and `close > prior_low` → +1; mirror → -1.
- fvg: last unfilled FVG within 1.5 ATR and price re-entering it → +1 (bull) / -1 (bear).
- trend: `bos` aligned with `trend`, `choch == 0` (continuation, not a reversal break) and pullback < 0.5 ATR → +1/-1. (Amended: `trend` is `bos` forward-filled, so the alignment conjunct alone is a tautology; the CHoCH exclusion is what makes it a continuation filter.)
`strength` = normalized |signal evidence| clipped to [0,1] (e.g. ATR ratio scaled, RSI distance / 25). All inputs already causal from `features`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_signals.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/strategy backend/tests/test_signals.py
git commit -m "feat(strategy): five causal signal families"
```

---

### Task 2: Offline counterfactual labeling

**Files:**
- Create: `backend/src/scalping/strategy/labeling.py`, `backend/tests/test_labeling.py`

**Interfaces:**
- Consumes: `Strategy`, `strategy_signals`, `PaperBroker`, `RiskEngine`, `Settings`
- Produces:
  - `label_events(df: pl.DataFrame, settings: Settings) -> pl.DataFrame` with `ts, strategy, side, entry, exit, bars, r_multiple, reason`
  - `context_matrix(features: FeatureFrame) -> np.ndarray (T, C)` — bandit bağlamı: rejim one-hot(3), seans one-hot(3), atr_pct, rsi/100, trend, spread/atr, son 20 bar strateji başarı ortalaması (5)
  - `CONTEXT_DIM` constant matching `context_matrix` width
  - Her strateji için sinyal barında **tek** varsayımsal işlem: giriş sonraki bar açılışı, çıkış SL/TP/120 bar.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_labeling.py
import numpy as np, polars as pl, pytest
from datetime import datetime, timezone
from scalping.config import Settings
from scalping.strategy.labeling import CONTEXT_DIM, context_matrix, label_events
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features

def data(days=3):
    df = SyntheticProvider(seed=21).fetch("XAUUSD",
        datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 5 + days, tzinfo=timezone.utc))
    return df

def test_labels_have_required_columns_and_values():
    out = label_events(data(), Settings())
    assert {"ts", "strategy", "side", "entry", "exit", "bars", "r_multiple", "reason"} <= set(out.columns)
    assert out.height > 0
    assert out["bars"].max() <= 120
    assert set(out["reason"].unique().to_list()) <= {"sl", "tp", "timeout"}

def test_r_multiple_sign_matches_outcome():
    out = label_events(data(), Settings())
    for row in out.head(50).iter_rows(named=True):
        gain = (row["exit"] - row["entry"]) * row["side"]
        assert (row["r_multiple"] > 0) == (gain > 0)

def test_no_signals_returns_empty_frame():
    flat = data(1).with_columns([pl.lit(2350.0).alias("open"), pl.lit(2350.0).alias("close"),
                                pl.lit(2350.01).alias("high"), pl.lit(2349.99).alias("low")])
    out = label_events(flat, Settings())
    assert out.height == 0
    assert "r_multiple" in out.columns

def test_context_matrix_shape_and_finiteness():
    f = build_features(data())
    X = context_matrix(f)
    assert X.shape == (f.X.shape[0], CONTEXT_DIM)
    assert np.isfinite(X).all()

def test_context_is_causal():
    df = data()
    f = build_features(df)
    full = context_matrix(f)
    part = context_matrix(build_features(df.head(400)))
    assert np.allclose(full[:400], part, equal_nan=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_labeling.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

For each strategy row where `sig != 0`, take the next bar's `ask` (long) / `bid` (short) as entry, size with `RiskEngine.size`, then run `PaperBroker.update` until SL/TP/timeout. `r_multiple = pnl / (initial_risk)` where `initial_risk = lot*size*|entry-sl|`. Drop labels whose entry is after data end. `context_matrix` reuses the same feature arrays; the "last 20 bars strategy success" column is a causal rolling mean of realized labels (0 before any).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_labeling.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/strategy/labeling.py backend/tests/test_labeling.py
git commit -m "feat(strategy): offline counterfactual R labeling"
```

---

### Task 3: LinUCB bandit

**Files:**
- Create: `backend/src/scalping/strategy/bandit.py`, `backend/tests/test_bandit.py`

**Interfaces:**
- Produces:
  - `LinUCB(n_arms: int = 5, dim: int = CONTEXT_DIM, alpha: float = 0.5, ridge: float = 1.0)`
  - `.select(x: np.ndarray) -> int`, `.update(arm: int, x: np.ndarray, reward: float) -> None`
  - `.fit_offline(X: np.ndarray, arms: np.ndarray, rewards: np.ndarray) -> LinUCB` (closed-form ridge per arm)
  - `.weights() -> np.ndarray (n_arms, dim)`, `.arm_stats() -> list[dict]` (pulls, mean reward)
  - `.save(path)` / `.load(path)` — joblib yok; `np.savez`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_bandit.py
import numpy as np, pytest
from scalping.strategy.bandit import LinUCB

def test_selects_in_range():
    b = LinUCB(dim=4)
    assert 0 <= b.select(np.ones(4)) < 5

def test_learns_dominant_arm_online():
    rng = np.random.default_rng(0)
    b = LinUCB(dim=4, alpha=0.1)
    for _ in range(200):
        x = rng.normal(size=4)
        a = b.select(x)
        b.update(a, x, 1.0 if a == 2 else 0.0)
    picks = [b.select(rng.normal(size=4)) for _ in range(50)]
    assert picks.count(2) >= 40

def test_offline_fit_recovers_best_arm():
    rng = np.random.default_rng(1)
    X, arms, rewards = [], [], []
    for _ in range(1000):
        arm = rng.integers(0, 5)
        x = rng.normal(size=4)
        X.append(x); arms.append(arm); rewards.append(1.0 if arm == 3 else 0.0)
    b = LinUCB(dim=4).fit_offline(np.array(X), np.array(arms), np.array(rewards))
    assert b.select(np.array([0.1, 0.2, 0.3, 0.4])) == 3

def test_save_load_roundtrip(tmp_path):
    b = LinUCB(dim=4)
    b.update(0, np.ones(4), 1.0)
    p = tmp_path / "bandit.npz"
    b.save(p)
    c = LinUCB.load(p)
    assert np.allclose(b.weights(), c.weights())

def test_empty_fit_does_not_crash():
    b = LinUCB(dim=4).fit_offline(np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))
    assert 0 <= b.select(np.ones(4)) < 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_bandit.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

Per arm: `A_i = ridge*I + Σ x xᵀ`, `b_i = Σ r x`, `theta_i = A_i⁻¹ b_i`, `p_i = theta_i·x + alpha*sqrt(xᵀ A_i⁻¹ x)`. `select` = argmax `p_i`. `fit_offline` accumulates in one pass then solves. Guard zero-pull arms with `theta=0` and a small exploration bonus so they still get picked once.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_bandit.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/strategy/bandit.py backend/tests/test_bandit.py
git commit -m "feat(strategy): linucb bandit with offline fit"
```

---

### Task 4: Observation with strategy context

**Files:**
- Modify: `backend/src/scalping/env/simulator.py` (obs assembly)
- Modify: `backend/src/scalping/env/trading_env.py` (`obs_dim`)
- Create: `backend/tests/test_env_strategy.py`

**Interfaces:**
- Produces: `Simulator(..., strategy_selector: Callable[[np.ndarray], int] | None = None)`; when present, obs gains `strategy_onehot(5) + strategy_strength(1)` at the end and `obs_dim` grows by 6.
- Produces: `info["strategy"]` is the `Strategy` value string of the active arm.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_env_strategy.py
import numpy as np, pytest
from datetime import datetime, timezone
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.config import Settings
from scalping.env.trading_env import TradingEnv
from scalping.strategy.bandit import LinUCB
from scalping.strategy.labeling import CONTEXT_DIM

def make():
    df = SyntheticProvider(seed=5).fetch("XAUUSD",
        datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 6, tzinfo=timezone.utc))
    return build_features(df)

def test_obs_dim_grows_by_six_with_selector():
    plain = TradingEnv(make(), Settings())
    with_sel = TradingEnv(make(), Settings(), strategy_selector=LinUCB(dim=CONTEXT_DIM))
    assert with_sel.obs_dim == plain.obs_dim + 6

def test_selected_strategy_is_reported():
    env = TradingEnv(make(), Settings(), strategy_selector=LinUCB(dim=CONTEXT_DIM))
    env.reset(seed=1)
    _, _, _, _, info = env.step(0)
    assert info["strategy"] in {"momentum", "reversal", "sweep", "fvg", "trend"}

def test_obs_remains_finite():
    env = TradingEnv(make(), Settings(), strategy_selector=LinUCB(dim=CONTEXT_DIM))
    obs, _ = env.reset(seed=1)
    assert np.isfinite(obs).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_env_strategy.py -v`
Expected: FAIL — `TradingEnv.__init__() got an unexpected keyword argument 'strategy_selector'`

- [ ] **Step 3: Implement**

Call the selector once per step with `context_matrix[row]`, convert the arm to one-hot of length 5, append the chosen strategy's `strength_<name>` from the signals frame (0 if absent). Keep the plain env path untouched when `strategy_selector is None` so existing tests keep passing.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_env_strategy.py tests/test_env.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/env backend/tests/test_env_strategy.py
git commit -m "feat(env): optional strategy context in observations"
```

---

### Task 5: PPO training pipeline and registry

**Files:**
- Create: `backend/src/scalping/rl/__init__.py`, `train.py`, `registry.py`, `policy.py`, `callbacks.py`, `backend/tests/test_registry.py`, `backend/tests/test_train_smoke.py`

**Interfaces:**
- Produces:
  - `TrainConfig(run_name: str, total_steps: int = 200_000, n_envs: int = 2, seed: int = 0, lr: float = 3e-4, n_steps: int = 1024, batch_size: int = 256, gamma: float = 0.99, ent_coef: float = 0.01, checkpoint_every: int = 10_000, data_hash: str = "")`
  - `train(config: TrainConfig, features: FeatureFrame, settings: Settings, selector=None, out_dir: Path) -> TrainResult(run_id, model_path, metrics_path, steps, reward_mean)`; `features.X.shape[0] < 4_000` ise `ValueError("not enough bars: need 4000")` (3 günlük sentetik ≈ 4.320 bar; eşik 5.000 olsaydı planın kendi smoke testi düşerdi).
  - `TrainResult` + `runs/<run_id>/{model.zip, config.json, metrics.json, bandit.npz}`
  - `load_policy(run_dir: Path) -> PolicyWrapper` with `.act(obs: np.ndarray) -> int`
  - `MetricsCallback` writes `metrics.json` every checkpoint: `step, reward_mean, ep_len_mean, entropy_loss, policy_loss, value_loss, action_dist[4]`.
  - `run_id` = `sha256(config_json + data_hash + seed)[:12]`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_registry.py
import json, numpy as np
from scalping.rl.registry import load_run, save_manifest, run_id_for

def test_run_id_stable_and_seeded():
    a = run_id_for({"lr": 3e-4, "total_steps": 100}, "abc", 7)
    b = run_id_for({"lr": 3e-4, "total_steps": 100}, "abc", 7)
    c = run_id_for({"lr": 3e-4, "total_steps": 100}, "abc", 8)
    assert a == b and a != c and len(a) == 12

def test_manifest_roundtrip(tmp_path):
    save_manifest(tmp_path, {"run_id": "x", "steps": 10, "metrics": {"reward_mean": 1.2}})
    assert load_run(tmp_path)["metrics"]["reward_mean"] == 1.2

def test_missing_manifest_returns_none(tmp_path):
    assert load_run(tmp_path) is None
```

```python
# backend/tests/test_train_smoke.py
import json, pytest, numpy as np
from datetime import datetime, timezone
import polars as pl
from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.rl.train import TrainConfig, train

@pytest.mark.slow
def test_training_produces_loadable_model(tmp_path):
    df = SyntheticProvider(seed=5).fetch("XAUUSD",
        datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 8, tzinfo=timezone.utc))
    f = build_features(df)
    cfg = TrainConfig(run_name="smoke", total_steps=1024, n_envs=1, n_steps=256,
                      batch_size=64, checkpoint_every=512, data_hash="smoke")
    res = train(cfg, f, Settings(), out_dir=tmp_path)
    assert (tmp_path / res.run_id / "model.zip").exists()
    metrics = json.loads((tmp_path / res.run_id / "metrics.json").read_text())
    assert np.isfinite(metrics["reward_mean"])
    assert sum(metrics["action_dist"]) == pytest.approx(1.0, abs=0.05)

@pytest.mark.slow
def test_training_refuses_tiny_dataset(tmp_path):
    df = SyntheticProvider(seed=5).fetch("XAUUSD",
        datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 5, 2, tzinfo=timezone.utc))
    f = build_features(df)
    with pytest.raises(ValueError, match="not enough bars"):
        train(TrainConfig(run_name="tiny", total_steps=256, data_hash="x"), f, Settings(), out_dir=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_registry.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`train` sets `torch.set_num_threads(1)`, builds `DummyVecEnv` of `n_envs` with fixed seeds, creates `PPO("MlpPolicy", env, learning_rate=..., n_steps=..., batch_size=..., ent_coef=..., seed=...)`, attaches `MetricsCallback` and `CheckpointCallback`, learns in chunks of `checkpoint_every` so Ctrl-C leaves a usable `model.zip`, then writes config + metrics + (if selector given) `bandit.npz`. Atomic writes: write `*.tmp` then `os.replace`. `load_policy` reads `model.zip` + `config.json` and returns a wrapper with `deterministic=True`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_registry.py -v && uv run pytest tests/test_train_smoke.py -v -m slow`
Expected: PASS (3 + 2)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/rl backend/tests/test_registry.py backend/tests/test_train_smoke.py
git commit -m "feat(rl): ppo training with checkpointed run registry"
```

---

### Task 6: Training API

**Files:**
- Modify: `backend/src/scalping/api/app.py`, `routes.py`, `schemas.py`
- Create: `backend/src/scalping/api/train_runner.py`, `backend/tests/test_api_train.py`

**Interfaces:**
- Produces:
  - `POST /api/train {total_steps, n_envs, seed, symbol} -> 202 {"run_id": str}`
  - `GET /api/runs -> [{run_id, status, created_at, steps, reward_mean}]`
  - `GET /api/runs/{run_id} -> {config, metrics, status}`
  - `POST /api/runs/{run_id}/control {action: "pause"|"resume"|"stop"} -> {"ok": bool}`
  - `POST /api/backtest {run_id, symbol} -> {"report_path": str, "metrics": {...}}`
  - Eğitim ayrı bir thread'de çalışır (`asyncio.to_thread`), API bloklanmaz.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_api_train.py
import pytest
from fastapi.testclient import TestClient
from scalping.api.app import create_app

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(data_root=tmp_path, runs_root=tmp_path / "runs"))

def test_train_rejects_without_data(client):
    r = client.post("/api/train", json={"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"})
    assert r.status_code == 409

def test_runs_empty(client):
    assert client.get("/api/runs").json() == []

def test_unknown_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404

def test_train_starts_and_is_listed(client, seed_data):
    r = client.post("/api/train", json={"total_steps": 256, "n_envs": 1, "seed": 0, "symbol": "XAUUSD"})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    listed = [x["run_id"] for x in client.get("/api/runs").json()]
    assert run_id in listed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_api_train.py -v`
Expected: FAIL — 404 on `/api/train`

- [ ] **Step 3: Implement**

`TrainRunner` keeps a `dict[run_id, RunState(status, thread, control_event)]`. `pause` sets an event the callback checks between chunks; `stop` sets a cancel flag and lets the current chunk finish so the model stays valid. `runs_root` defaults to `runs/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_api_train.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/api backend/tests/test_api_train.py
git commit -m "feat(api): training endpoints with pause and stop"
```

---

### Task 7: Walk-forward and cost stress evaluation

**Files:**
- Create: `backend/src/scalping/backtest/evaluate.py`, `backend/tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `evaluate_walkforward(policy, df: pl.DataFrame, settings: Settings, train_months=6, folds: int = 3) -> WalkForwardReport`
  - `WalkForwardReport`: `folds: list[FoldResult]`, `overall: Metrics`, `overfit_warning: bool`
  - `cost_stress(policy, features, settings, multipliers=(1, 2, 3)) -> list[StressResult]`
  - `FoldResult`: `train_range, test_range, metrics, trades`
  - `overfit_warning` = `train_reward_trend > 0 and test_expectancy_trend < 0`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_evaluate.py
import numpy as np, pytest
from datetime import datetime, timezone
from scalping.config import Settings
from scalping.data.synthetic import SyntheticProvider
from scalping.features.build import build_features
from scalping.backtest.evaluate import cost_stress, evaluate_walkforward

def make(days=200):
    df = SyntheticProvider(seed=33).fetch("XAUUSD",
        datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 5 + days, tzinfo=timezone.utc))
    return df, build_features(df)

def test_walkforward_produces_non_overlapping_folds():
    df, f = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    assert len(rep.folds) == 3
    for a, b in zip(rep.folds, rep.folds[1:]):
        assert a.test_range[1] <= b.test_range[0]

def test_zero_trade_policy_has_zero_expectancy_everywhere():
    df, f = make()
    rep = evaluate_walkforward(lambda o: 0, df, Settings(), folds=3)
    assert rep.overall.trades == 0
    assert not rep.overfit_warning

def test_cost_stress_monotonic_costs():
    df, f = make(30)
    out = cost_stress(lambda o: 1 if o[0] > 0 else 0, f, Settings())
    costs = [r.metrics.total_costs for r in out if r.metrics.trades > 0]
    assert costs == sorted(costs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

`evaluate_walkforward` slices calendar months, runs `run_backtest` on each test window with the same policy, aggregates trades into `overall`, and computes a simple slope (linear fit) of per-fold train reward and test expectancy for the warning. `cost_stress` temporarily multiplies `CostConfig.spread_floor`, `slippage_atr_frac` and commission by `m`, re-runs the backtest, restores settings after each run.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_evaluate.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/src/scalping/backtest/evaluate.py backend/tests/test_evaluate.py
git commit -m "feat(backtest): walk-forward and cost stress evaluation"
```

---

## Self-Review

- **Spec coverage:** §12 strategies (T1), offline labeling (T2), bandit (T3), frozen-during-PPO (T5 wiring), §10 obs extension (T4), §13 walk-forward + cost stress (T7), training registry/API (T5–T6), §15 determinism scope documented in Global Constraints.
- **Gaps closed:** tiny dataset (T5), zero-signal labels (T2), interrupted training (T5 atomic writes + chunked learning), degenerate reward (T5 action_dist metric), overfit (T7 warning).
- **Type consistency:** `CONTEXT_DIM` defined once in `labeling.py`, imported by env and tests; `Strategy` order is frozen and the bandit arm index is that order; `TrainConfig.data_hash` feeds `run_id_for`.
- **Proportion:** no HRL framework, no Optuna sweep in v1, no distributed training — the bandit + PPO pair is the whole hierarchy.
