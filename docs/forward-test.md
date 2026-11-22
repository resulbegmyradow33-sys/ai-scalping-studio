# İleri test: Haziran–Kasım 2026

Kartın gerçekten görülmemiş veride çalışıp çalışmadığını ölçen test. Kart
**hiç yeniden kalibre edilmedi**; Haziran 2026'dan itibaren aynı parametrelerle
oynatıldı.

## Veri

- Kaynak: `Touhid-HZ/xauusd-tick-data-2026` (GitHub) — Dukascopy `bi5`
  feed'inin haftalık CSV.gz paketlemesi.
- Kapsam: **2026-05-28 → 2026-11-11**, 24 hafta, 301 MB sıkıştırılmış,
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
| **İleri (2026-06→2026-11)** | **23** | **+0.451** | **2.03** | **%2,4** | **+$422** | **%60,9** |
| İleri, maliyet ×2 | 23 | +0.445 | 2.01 | %2,4 | +$416 | %60,9 |

Aylara göre işlem: Haziran 3, Temmuz 2, Ağustos 4, Eylül 3, Ekim 7, Kasım 4.
Tek bir ayda toplanmıyor.

## İstatistiksel dürüstlük

23 işlem az. R serisinin ortalaması +0.452, standart sapması 1.33:

- t-istatistiği **1.63**
- %95 güven aralığı **−0.09 … +1.00** (sıfırı içeriyor)
- bootstrap ile P(gerçek ortalama ≤ 0) ≈ **%4,8**

Yani "kenar var" demek için hâlâ erken; "ilk gerçek ileri testte pozitif ve
maliyet stresine dayandı" demek doğru olan. 23 işlemin 14'ü kârlı, 9'u zararlı;
7 hedef, 8 stop, 8 kısmi/timeout.

## Tekrarlama

```bash
cd backend
uv run python -m scripts.forward_test \
  --start 2026-06-01 --end 2026-11-21
```

`forward_test` geçmiş ve ileri store'u birleştirir (ısınma için), kartı
dosyadan okur, Haziran başında simülatörü başlatır. Hiçbir parametre ileri
veride seçilmez.

## Sonraki adım

Aynı haftalık ayna Pazarteleri güncelleniyor. Kasım–Aralık 2026 verisi
geldikçe aynı komut yeniden koşulur; işlem sayısı 50'yi geçtiğinde sonuç
istatistiksel olarak anlamlı hale gelir. O zamana kadar canlı kart değişmez.
