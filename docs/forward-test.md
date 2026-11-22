# İleri test: Haziran–Kasım 2026

Kartın gerçekten görülmemiş veride çalışıp çalışmadığını ölçen test. Kart
**hiç yeniden kalibre edilmedi**; Haziran 2026'dan itibaren aynı parametrelerle
oynatıldı.

## Veri

- Kaynak: `Touhid-HZ/xauusd-tick-data-2026` (GitHub) — Dukascopy `bi5`
  feed'inin haftalık CSV.gz paketlemesi.
- Kapsam: **2026-05-28 → 2026-11-15**, 25 hafta, ~310 MB sıkıştırılmış,
  ~45 milyon tick.
- Alanlar: `timestamp` (UTC ms), `ask`, `bid`, `ask_volume`, `bid_volume`.
- Spread ortalaması **0.646** (Tickmill geçmişinde 0.446).
- Akış büyüklüğü Tickmill'den ~10 kat küçük (Dukascopy birimleri), ama dağılım
  şekli benzer; kapı yüzdelik dilimle çalıştığı için ölçek farkı önemsiz.

Dukascopy'yi doğrudan saatlik crawl etmek denendi; feed saatte bir bağlantıda
30 sn stall ve 429 banı verdiği için bırakıldı. Aynı kaynağın haftalık aynası
sorunsuz indi.

## Üretim

```bash
cd backend
uv run python -m scripts.ingest_2026_ticks \
  --dir ../data-2026-ticks \
  --root ../data-forward-store \
  --flow-root ../data-forward-flow \
  --cache ../data-2026-cache
```

Her hafta M1'e iner; bid OHLC, gerçek spread, tick sayısı ve dakikalık
`(ask_volume - bid_volume) / (ask_volume + bid_volume)` üretilir. M5/M15/H1/D1
aynı store'a türetilir.

## Sonuç

Kart: H1, akış eşiği üst %15 / 480 bar, 3×ATR stop, 6×ATR hedef, 48 bar.
Sıralı simülatör, gerçek spread, maliyetler dahil:

| Dilim | İşlem | R/işlem | PF | Max DD | Net | Win |
|---|---|---|---|---|---|---|
| Önceki 6 ay (2025-12→2026-05) | 12 | +0.377 | 1.76 | %1,6 | +$202 | %50 |
| **İleri (2026-06→2026-11)** | **24** | **+0.502** | **2.15** | **%2,4** | **+$481** | **%58,3** |
| İleri, maliyet ×2 | 24 | +0.496 | 2.14 | %2,4 | +$474 | %58,3 |
| **Temmuz–Kasım (tam ısınma)** | **20** | **+0.653** | **3.03** | **%2,4** | **+$549** | **%65** |

Aylara göre işlem: Haziran 3, Temmuz 2, Ağustos 4, Eylül 3, Ekim 7, Kasım 5.
Tek bir ayda toplanmıyor.

Haziran başındaki iki hafta akış ölçeği geçişinde: 480 barlık pencere hâlâ
Tickmill akışını taşıyor, Dukascopy akışı ise ~10 kat küçük. Bu yüzden ayrıca
**1 Temmuz** başlangıcıyla (pencere tamamen Dukascopy) koştum: +0.653 R,
PF 3.03, 20 işlem. İki sonucu da raporluyorum; Temmuz'u seçip Haziran'ı
gizlemek olmaz. Asıl ileri test Haziran başlangıcıdır.

## İstatistiksel dürüstlük

24 işlem az. R serisinin ortalaması +0.502, standart sapması 1.34:

- t-istatistiği **1.83**
- %95 güven aralığı **−0.04 … +1.04** (sıfırı içeriyor)
- bootstrap ile P(gerçek ortalama ≤ 0) ≈ **%3,0**

Yani "kenar var" demek için hâlâ erken; "ilk gerçek ileri testte pozitif ve
maliyet stresine dayandı" demek doğru olan. 24 işlemin 14'ü kârlı, 10'u zararlı;
9 hedef, 8 stop, 7 kısmi/timeout.

## Tekrarlama

```bash
cd backend
uv run python -m scripts.forward_test \
  --start 2026-06-01 --end 2026-11-21
```

`forward_test` geçmiş ve ileri store'u birleştirir (ısınma için), kartı
dosyadan okur, Haziran başında simülatörü başlatır. Hiçbir parametre ileri
veride seçilmez.

Haftalık güncelleme (GitHub aynası Pazartesi yenilenir):

```bash
make forward-update
```

Yeni haftaları indirir, cache'i atlar, store'u yeniden kurar, testi koşar ve
sonucu `backend/runs/forward_history.jsonl`'e ekler.

## Sonraki adım

Aynı haftalık ayna Pazarteleri güncelleniyor. Kasım–Aralık 2026 verisi
geldikçe aynı komut yeniden koşulur; işlem sayısı 50'yi geçtiğinde sonuç
istatistiksel olarak anlamlı hale gelir. O zamana kadar canlı kart değişmez.
