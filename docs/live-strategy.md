# Canlı strateji ve gece kalibrasyonu

Bu belge, panelin hangi kuralı oynattığını ve "sürekli öğrenme"nin bu projede
nasıl uygulandığını anlatır. Araştırma sonucu netti: beş yıllık gerçek spread
testinde ayakta kalan şey bir **kural**, PPO politikası değil. Bu yüzden canlı
sistem kuralı çalıştırır; öğrenme, kuralın serbest parametrelerinin gece
yeniden seçilip bir holdout kapısından geçirilmesiyle olur.

## Kural

`flow_momentum`, H1:

- `momentum` ailesi sinyal verir (EMA20 + 5 bar yer değiştirme + ATR genişlemesi),
- yalnızca `|flow|` son 480 barın %85'lik dilimini aşarsa işlem açar,
- çıkış: 3×ATR stop, 6×ATR hedef, 48 bar zaman aşımı.

`flow`, tick arşivindeki `bid_volume` / `ask_volume` farkının dakikalık
normalize hâlidir: `(buy − sell) / (buy + sell)`. Fiyatın türevi değildir.

Kart, bu parametrelerin yanında terfiyi hak eden eğitim ve holdout metriklerini
taşır: `runs/strategy_card.json` (sunucunun çalışma dizini: `backend/`). Panel her `start` sırasında kartı yeniden
okur; yeni kart için süreç yeniden başlatılmaz.

## Kalibrasyon yöntemi

1. Veri, zaman sırasına göre %80 eğitim / %20 holdout bölünür.
2. 27 kombinasyon (3 risk profili × 3 akış dilimi × 3 pencere) **yalnızca
   eğitim diliminde** skorlanır; en yüksek ortalama R kazanır.
3. Kazanan **bir kez** holdout'ta değerlendirilir.
4. Terfi koşulu: holdout'ta en az 30 işlem, pozitif ortalama R ve mevcut
   karttan daha iyi holdout R'si.
5. Terfi etmezse hiçbir şey yazılmaz; rapor `runs/strategy_history.jsonl`'e
   eklenir.

Bu kapı dürüst bir sonsuz out-of-sample testi değildir: her gece veri uzasa da
aynı tarihsel holdout tekrar görülür. Kartın `note` alanı bunu kaydeder.

## Çalıştırma

Tek sefer (cron/launchd için):

```bash
cd backend
uv run python -m scripts.nightly_calibrate \
  --store ../data-ticks5-store \
  --flow-root ../data-flow-store \
  --card runs/strategy_card.json \
  --history runs/strategy_history.jsonl
```

Sürekli döngü (24 saatte bir):

```bash
cd backend
nohup uv run python -m scripts.nightly_calibrate --loop --interval-hours 24 \
  > runs/nightly.log 2>&1 &
```

API ile (panel düğmesi bunu çağırır):

```bash
curl -X POST http://127.0.0.1:8000/api/strategy/calibrate \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"XAUUSD","timeframe":"H1"}'
```

`"dry_run": true` adayı değerlendirir ama kartı yazmaz.

## Sunucu

Kartın uygulanması için akış yan deposu verilmelidir:

```bash
cd backend
uv run python -m scalping.cli serve \
  --root ../data-ticks5-store \
  --flow-root ../data-flow-store \
  --port 8000
```

`--flow-root` yoksa veya kartın zaman diliminde bar yoksa panel sessizce eski
M1 momentumuna döner; bunu `GET /api/strategy/card` (`active: false`) ve
sunucu logu gösterir.

## Doğrulanmış sonuç (22 Kasım 2026)

Kartın ilk terfisi: `flow_quantile=0.85`, `flow_window=480`, 3/6 ATR, 48 bar.
Kart ve kalibrasyon kaydı: `docs/cards/strategy_card_2026-11-22.json`,
`docs/cards/strategy_history_2026-11-22.jsonl`.

| Dilim | İşlem | R/işlem | PF | Max DD | Net |
|---|---|---|---|---|---|
| Eğitim (2021–2025-05) | 1.718 | +0.042 | 1.08 | %40 | — |
| Holdout (2025-05→2026-05) | 215 | **+0.159** | **1.32** | **%8,1** | — |
| Holdout, sıralı simülatör | 73 | **+0.144** | 1.21 | %4,2 | +$291 |
| 5 yıl, sıralı simülatör | 604 | +0.046 | 1.09 | %11,4 | +$1.201 |

Sıralı simülatör, aynı anda tek pozisyon kuralını uyguladığı için etiket
sayısından daha az işlem yapar; bu yüzden asıl kanıt ikinci satırdır.

Başlangıç yılına göre (her satır o yıldan 2026-05'e kadar, pencereler örtüşür):

| Başlangıç | İşlem | R/işlem | PF | Max DD | Net |
|---|---|---|---|---|---|
| 2021-05 | 604 | +0.046 | 1.09 | %11,4 | +$1.201 |
| 2022-01 | 531 | +0.063 | 1.13 | %7,5 | +$1.494 |
| 2023-01 | 410 | +0.114 | 1.23 | %5,4 | +$2.166 |
| 2024-01 | 266 | +0.068 | 1.11 | %4,9 | +$616 |
| 2025-01 | 123 | +0.103 | 1.12 | %4,9 | +$308 |
| 2026-01 | 6 | +0.998 | 3.96 | %0,5 | +$264 |

Her başlangıç yılı pozitif; bu tek başına kanıt değil (pencereler örtüşüyor),
ama rejim bağımlılığının 2021-2022 ile sınırlı olmadığını gösteriyor.

## Akış gerçekten imzalı olmalı

Kolay bir kısayol var mı diye test ettim: toplam işlem hacmi veya mum yönü ×
hacim, imzalı bid/ask hacim farkının yerini tutar mı?

| Kapı | Eğitim (≤2023) | Holdout (2024+) |
|---|---|---|
| Toplam hacim (üst %15, 480 bar) | −0.035 R | −0.002 R |
| Mum yönü × hacim (5 bar, üst %15) | −0.044 R | −0.004 R |
| **Bid/ask hacim farkı** | **+0.003 R** | **+0.133 R** |

Yani kenar "hareketlilik" veya "hacim" değil, **alıcı/satıcı tarafının
dengesizliği**. Bu yüzden ileri veri crawl'ı toplam hacimli mumlarla değil,
Dukascopy'nin bid/ask hacimli tick'leriyle yapılıyor.

## İleri test (Haziran–Kasım 2026)

Kart, Dukascopy'nin 2026 tick verisinde hiç yeniden kalibre edilmeden
oynatıldı: **24 işlem, +0.502 R, PF 2.15, +$481, DD %2,4**. Maliyet ×2'de
+0.496 R. Örnek küçük ve %95 güven aralığı sıfırı içeriyor; ayrıntı ve
istatistik `docs/forward-test.md` içinde.

## Sınırlar

- Veri 2021-05 → 2026-05; tek sembol (XAUUSD), tek broker feed'i.
- Spread arşivi ortalama 0.446; canlı broker spread'i farklıysa sonuç değişir.
- Kart, PPO ajanını terfi ettirmez. RL panelde deneysel kalır; taklit deneyi
  ve sonuçları `docs/rl-imitation.md`, otomatik tekrarı `scripts/nightly_rl.py`.
- Gece kalibrasyonu bir mühendislik döngüsüdür, garanti değil: her terfi aynı
  holdout'u tekrar gördüğü için kanıt gücü zamanla azalır.
