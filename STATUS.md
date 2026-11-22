# STATUS — nerede kaldık?

Bu dosya, `docs/requirements/master-prompt.md` hedeflerine göre deponun
durumunu tek sayfada gösterir. Ayrıntılı kanıt: `docs/architecture-audit.md`.
Plan: `docs/implementation-roadmap.md`.

Son güncelleme: 2026-11-22 · Branch `main`

## Bugün gerçekten çalışan şey

- XAUUSD için **uçtan uca paper-trading araştırma hattı**: veri → özellik →
  simülatör → kural/PPO → walk-forward → panel.
- **`flow_momentum` kartı** canlı panelde H1 oynuyor: momentun + imzalı tick
  akış kapısı, 3/6 ATR, 48 bar.
- **Gerçek ileri test** (Haziran–Kasım 2026, kart değiştirilmeden):
  24 işlem, **+0.502 R**, PF 2.15, +$481, DD %2,4.
- Ablasyon: akış kapısı olmadan aynı dönem **−0.128 R**, PF 0.79.
- **PPO + taklit** yerelde eğitiliyor; iki bölünmede pozitif ama kuraldan zayıf.
- Testler: **556 backend**, **188 frontend**; ruff/mypy temiz; build geçiyor.

## Durum matrisi

| Alan | Durum | Not |
|---|---|---|
| Yerel, sıfır maliyet, LLM'siz | ✅ | Dış API yok; PyTorch/SB3 yerel |
| XAUUSD veri hattı | ✅ | 5 yıl tick, 17 yıl M1, 2026 ileri |
| Çoklu sembol (10 enstrüman) | 🟡 | 10 sembol registry + kontrat metadatası var; gerçek feed hâlâ XAU/XAG (Dukascopy) |
| Tick–D1 zaman dilimleri | ✅ | M1–D1 (M30/H4 dahil); tick ingestion scriptleri |
| CSV/Parquet/JSON ithal | ✅ | CSV, JSON, Parquet store, Dukascopy |
| Veri doğrulama + provenance | 🟡 | duplicate/OHLC/spread/gap raporu + checksum sidecar var; istatistiksel outlier/kaynak mutabakatı yok |
| Özellik mühendisliği | 🟡 | 59 nedensel özellik; seçilebilir set yok |
| Gerçek bid/ask/order-flow | 🟡 | İmzalı flow var; L2/deep book yok |
| Çoklu zaman dilimi analizi | 🟡 | M5/M15 bağlamı; bağımsız encoder/fusion yok |
| Klasik ML (RF, LGBM, XGB) | 🟡 | Logistic/RF/ExtraTrees/HistGB + metrikler var; LGBM/XGB, simülatör değerlendirmesi ve promosyon yok |
| Derin öğrenme (CNN/RNN/GRU/TCN) | ❌ | Sadece PPO'nun 64×64 MLP'si |
| Transformer araştırması | ❌ | Modül yok |
| RL (PPO + taklit) | 🟡 | PPO/BC var; SAC/TD3/DQN/A2C yok |
| Piyasa rejimi motoru | 🟡 | `vol_regime` özelliği var; HMM/GMM/KMeans motoru yok |
| Strateji araştırma/seçim | 🟡 | 5 aile + flow kuralı + LinUCB; rejim matrisi/NO TRADE motoru yok |
| Ensemble / karar motoru | 🟡 | LinUCB kol seçimi; stacking/oylama yok |
| Güven/kalibrasyon | ❌ | Brier/log loss/disagreement/OOD yok |
| Otonom öğrenme | 🟡 | Gece kalibrasyon + RL lab; SQLite/drift/rollback yok |
| Deney/HPO yönetimi | ❌ | Optuna/sweep yok |
| Bağımsız risk motoru | 🟡 | Günlük zarar, notional, pozisyon, SL/TP var; haftalık/DD kill switch yok |
| Gerçekçi backtest | 🟡 | Walk-forward + maliyet stresi; purged CV/Monte Carlo yok |
| Paper trading defteri | 🟡 | Simülatör var; kalıcı ledger/broker adapter yok |
| Model registry / SQLite | ❌ | JSON run klasörleri; champion/rollback yok |
| Panel | 🟡 | 4 sayfa; ML/NN/RL/rejim/paper/monitoring yok |
| Scheduler | 🟡 | İki arka plan döngüsü; konfigüre edilebilir scheduler yok |
| Testler | ✅ | Mevcut kapsam için güçlü; yeni modüller için test yok |
| Dokümantasyon | 🟡 | 4/15 zorunlu doküman hazır; 11 eksik |
| Kabul kriterleri | ❌ | ML/DL/ensemble/paper maddeleri henüz yok |

Lejant: ✅ tamam · 🟡 kısmi · ❌ yok.

## Canlı sistem hakkında dürüst notlar

- Panel bir **simülatördür**: gerçek broker bağlantısı, emir gönderimi veya
  gerçek para yoktur.
- Canlı oynayan strateji **kuraldır**, nöral ağ değil.
- PPO/BC **deneyseldir**; iki bölünmede pozitif ama kuralın altında.
- İleri test 24 işlemle pozitif; %95 güven aralığı hâlâ sıfırı içeriyor.
- Hiçbir model "kanıtlanmış kârlı" sayılmaz; sadece ölçülmüş sonuç vardır.

## Sıradaki işler (öncelik sırası)

1. **Faz 2 — Veri temeli:** registry, doğrulama, provenance, JSON, M30/H4
   tamamlandı; kalan: gerçek çoklu-sembol feed ve istatistiksel outlier raporu.
2. **Faz 3 — Dataset builder:** özellik registry + purged/embargo split +
   manifest/checksum tamamlandı; kalan: eğitim hattına bağlama, H4 hizalama
   testleri ve özellik kullanılabilirlik kuralları.
3. **Faz 5 — Klasik ML:** LightGBM/XGBoost/RF + baz çizgileri; simülatörle
   ticari değerlendirme.
4. **Faz 6 — GRU/TCN:** ortak PyTorch trainer, MPS/CPU, checkpoint.
5. **Faz 10 — SQLite registry:** aday → şampiyon kapısı, rollback, drift.

Faz 4 (rejim) ve Faz 11 (risk/backtest) paralel yürüyebilir; ağır eğitimler
aynı anda çalışmaz.

## Görev panosu

- [`docs/task-board.md`](docs/task-board.md): master prompt'un tek tek görev
  listesi; `[x]` yalnızca kod + test varsa işaretli.
- GitHub phase issue'ları aynı listeleri taşır.

## Ekip altyapısı

- `main` varsayılan branch; eski `feat/v1` uzak branch silindi.
- CI: her PR'da `backend` (ruff + mypy + 556 test) ve `frontend` (tsc + 188 test + build).
- PR/issue şablonları, CODEOWNERS, SECURITY, CODE_OF_CONDUCT, CHANGELOG.
- 27 etiket, 14 faz milestone'u, 5 açılış issue'su.
- `make hooks`: doğrudan `main` push'unu yerelde engeller.
- Repo **public**; sunucu tarafı branch protection, secret scanning, push
  protection ve CodeQL ücretsiz aktif.
- Yerel `pre-push` hook ek savunma olarak duruyor.

## Çalıştırma

```bash
cd backend
uv sync
uv run pytest
uv run python -m scalping.cli serve \
  --root ../data-ticks5-store --flow-root ../data-flow-store

cd ../frontend
npm install
npm run dev -- -p 3001
```

Tam kurulum: `README.md`. Kart: `backend/config/strategy_card.json`
(aktifleştirmek için `runs/strategy_card.json`'a kopyala).

## Katkı kuralı

- Push/PR/GitHub ayarı yalnızca açık izinle.
- Gerçek para yolu açılmaz; varsayılan paper.
- Bir mimariyi "yapıldı" saymak için kod + test gerekir.
- Bir modeli "iyi" saymak için iki kronolojik bölünme + maliyet stresi +
  ileri dilim gerekir.
