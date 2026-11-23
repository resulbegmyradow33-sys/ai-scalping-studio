# STATUS — nerede kaldık?

Bu dosya, `docs/requirements/master-prompt.md` hedeflerine göre deponun
durumunu tek sayfada gösterir. Ayrıntılı kanıt: `docs/architecture-audit.md`.
Plan: `docs/implementation-roadmap.md`. Görev listesi: `docs/task-board.md`.

Son güncelleme: 2026-11-23 · Branch `main` @ `7154d48`

## Bugün gerçekten çalışan şey

- XAUUSD için **uçtan uca paper-trading araştırma hattı**: veri → doğrulama →
  özellik → dataset → klasik ML / PyTorch / PPO → simülatör + maliyet stresi →
  panel.
- **`flow_momentum` kuralı** canlı panelde H1 oynuyor: momentum + imzalı tick
  akış kapısı, 3/6 ATR, 48 bar.
- **Gerçek ileri test** (Haziran–Kasım 2026, kart değiştirilmeden):
  24 işlem, **+0.502 R**, PF 2.15, +$481, DD %2,4. Örnek küçük; güven aralığı
  hâlâ sıfırı içeriyor.
- Ablasyon: akış kapısı olmadan aynı dönem **−0.128 R**, PF 0.79.
- **Purged expanding-window CV** (embargo dahil) ve kalibrasyon (Brier, log loss,
  ECE/MCE) var; hiçbir model promosyon kapısını geçmedi.
- **Testler: 843 backend, 194 frontend** (`main` @ `7154d48`); ruff/mypy/eslint
  temiz; Next build geçiyor.
- **Ölçülen performans:** M1 `build_features` 168–173 s → **8.65 s**; `fvg` taraması
  149.5 s → **1.93 s**, çıktı bit-identical. Dil kararı: Python + TypeScript kalıyor.

## Durum matrisi

| Alan | Durum | Not |
|---|---|---|
| Yerel, sıfır maliyet, LLM'siz | ✅ | Dış API yok; PyTorch/SB3 yerel |
| XAUUSD veri hattı | ✅ | 5 yıl tick, 17 yıl M1, 2026 ileri |
| Çoklu sembol (10 enstrüman) | 🟡 | 10 sembol registry + kontrat metadatası; gerçek feed hâlâ XAU/XAG |
| Tick–D1 zaman dilimleri | ✅ | M1–D1 (M30/H4 dahil) |
| CSV/Parquet/JSON ithal | ✅ | CSV chunked parse, JSON, Parquet, Dukascopy; HTTP import `imports/` allowlist |
| Veri doğrulama + provenance | ✅ | duplicate/OHLC/spread/gap + istatistiksel outlier/kaynak mutabakatı |
| Özellik mühendisliği | 🟡 | 59 nedensel özellik, yedi grup, kanonik sıra; ablation ve CVD/order-flow türevleri yok |
| Gerçek bid/ask/order-flow | 🟡 | İmzalı flow var; L2/deep book yok |
| Çoklu zaman dilimi analizi | 🟡 | M5/M15 bağlamı; bağımsız encoder/fusion yok |
| Klasik ML | 🟡 | Logistic/RF/ExtraTrees/HistGB/LightGBM/XGBoost + purged CV + kalibrasyon + simülatör değerlendirmesi; SGD/IF/SVM ve rejim/volatilite hedefleri yok |
| Derin öğrenme | 🟡 | MLP/RNN/LSTM/GRU/CNN/CNN-LSTM/causal TCN + ortak trainer; ML/NN/rule aynı dilimde karşılaştırma PR'da, NN için simulator/cost gate yok |
| Transformer araştırması | 🟡 | Encoder, causal encoder, PatchTST, iTransformer var; TFT ve ölçülü benchmark tablosu yok |
| RL (PPO + taklit) | 🟡 | PPO/BC + action mask (flow) var; SAC/TD3/DQN/A2C ve risk maskesi yok |
| Piyasa rejimi motoru | ✅ | Kural + KMeans/GMM/HMM/supervised + strateji×rejim tablosu + API/panel |
| Strateji araştırma/seçimi | 🟡 | 5 aile + flow kuralı + LinUCB; rejim bağımlı seçim yok |
| Ensemble / karar motoru | 🟡 | LinUCB kol seçimi; OOF stacking/oylama/NO TRADE dilimi geliştiriliyor |
| Güven/kalibrasyon | 🟡 | Brier/log loss/ECE/MCE var; temperature scaling ve OOD yok |
| Otonom öğrenme | 🟡 | SQLite registry dilimi PR'da; candidate gate/drift/scheduler yok |
| Deney/HPO yönetimi | ❌ | Optuna/sweep yok |
| Bağımsız risk motoru | 🟡 | Günlük + haftalık + DD + ardışık zarar + spread kill switch'leri var (varsayılan kapalı); slippage limiti yok |
| Gerçekçi backtest | 🟡 | Walk-forward + maliyet stresi + purged CV; Monte Carlo/parametre duyarlılığı yok |
| Paper trading defteri | 🟡 | Kalıcı SQLite ledger + restart-safe koşucu var; API dilimi geliştiriliyor; MT5 adapter iskelet |
| Model registry / SQLite | 🟡 | datasets/models/promotions/jobs + champion/rollback + scope PR'da; artefakt doğrulaması yok |
| Panel | 🟡 | Data, Intelligence, Performance, Training; ML/NN/RL/paper/monitoring sayfaları yok |
| Scheduler | 🟡 | İki arka plan döngüsü; konfigüre edilebilir scheduler yok |
| Testler | ✅ | 843 backend + 194 frontend; ruff/mypy/eslint/build temiz |
| Dokümantasyon | 🟡 | Zorunlu 15 dokümanın bir kısmı hazır; `docs/task-board.md` gerçek durumu izliyor |
| Kabul kriterleri | ❌ | Uçtan uca promote/reject → paper → monitor → rollback zinciri henüz yok |

Lejant: ✅ tamam · 🟡 kısmi · ❌ yok.

## Canlı sistem hakkında dürüst notlar

- Panel bir **simülatördür**: gerçek broker bağlantısı, emir gönderimi veya
  gerçek para yoktur. Gerçek para yolu bilerek kapalıdır.
- Canlı oynayan strateji **kuraldır**, nöral ağ veya ensemble değil.
- Hiçbir klasik/DL/RL modeli out-of-sample + maliyet stresinde kuralı geçmedi;
  hiçbiri "kanıtlanmış kârlı" değil. Sadece ölçülmüş sonuç var.
- Purged CV fold'ları erken geçmişi paylaşır; ortalama/std bağımsız tahmin değildir.
- Rejim kırılımı ve ensemble metrikleri tanımlayıcıdır, promosyon kapısı değildir.

## Sıradaki işler (öncelik sırası)

1. **Faz 9 — ensemble/karar:** OOF stacking, oylama, disagreement, NO TRADE karar
   kaydı; en iyi üye ile karşılaştırma.
2. **Faz 10 — registry:** candidate → champion kapısı, drift, scheduler,
   artefakt doğrulaması.
3. **Faz 11 — risk/backtest:** Monte Carlo, parametre duyarlılığı, execution
   delay, slippage limiti, data-quality trade block.
4. **Faz 12 — paper:** API uçları, panel sayfası, MT5 demo iskeleti (kapalı).
5. **Faz 13 — panel:** ML/NN/RL laboratuvarları, paper ve monitoring sayfaları;
   her sayı API/artefakttan gelir.
6. **Faz 14 — optimizasyon:** feature cache (#69), tek lazy plan (#70), global
   PPO job slot (#71), metadata coverage (#72), label join (#73).

Ağır eğitimler aynı anda çalışmaz; 8 GB RAM sınırı korunur.

## Görev panosu

Ayrıntılı tek tek liste: [`docs/task-board.md`](task-board.md).
