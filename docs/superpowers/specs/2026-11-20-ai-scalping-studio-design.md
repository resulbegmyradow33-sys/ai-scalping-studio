# AI Scalping Studio — Tasarım Dokümanı

Tarih: 2026-11-20
Sahip: resul2004
Durum: Onay bekliyor

## 1. Amaç

Mac mini M2 (8 GB) üzerinde tamamen yerel çalışan, XAUUSD odaklı bir scalping
sistemi kurmak. Sistem piyasayı M1/M5/M15 üzerinde izler, sanal broker üzerinde
işlem yapar, sonuçlardan öğrenir ve tüm süreç tarayıcıda canlı izlenebilir olur.

Bunu neden istiyoruz: hazır bir "sinyal botu" değil, kendi kararlarını deneyerek
geliştiren bir ajan ve onu şeffaf biçimde izleyebileceğimiz bir stüdyo. Ücretli
AI API'si yok; öğrenme tamamen yerelde, pekiştirmeli öğrenme ile.

## 2. Hedefler

- XAUUSD M1 verisiyle uçtan uca çalışan bir hat: veri → özellik → ortam → ajan → backtest → panel.
- Gerçekçi işlem maliyeti: bid/ask spread, komisyon, slippage, gecikme.
- Ajanın değiştiremediği bağımsız risk katmanı (pozisyon boyutu, günlük zarar limiti, maruziyet).
- Strateji seçimi: hangi koşulda hangi strateji ailesinin kullanılacağını öğrenen ikinci katman.
- Walk-forward doğrulama: eğitim / validasyon / görülmemiş test ayrımı.
- Canlı simülasyon paneli: mum grafiği, işlem işaretleri, equity, açık pozisyon, strateji, eğitim metrikleri.
- Aynı motorun ileride XAGUSD, NAS100, USOIL, BTCUSD ve Forex'e genişlemesi.

## 3. Hedef olmayanlar (v1)

- Gerçek para ile canlı işlem. v1'de kesinlikle yok.
- MT5 canlı bağlantısı. macOS'ta MT5 Python API'si çalışmıyor; v2 konusu.
- Tick seviyesinde order book / footprint analizi. v1 M1 bar + bid/ask spread ile çalışır.
- LLM tabanlı haber/sentiment analizi.
- Bulut eğitim, dağıtık RL, çoklu ajan paralel eğitimi.
- Mobil uygulama.

## 4. Anti-hedefler (teknik olarak çalışsa bile başarısız sayılır)

- Geçmiş veriyi ezberleyip test setinde çöken model.
- İşlem maliyetlerini simülasyonda atlayıp "kârlı" görünen sonuç.
- Ajanın risk limitlerini aşabildiği bir mimari.
- Yeniden üretilemeyen (seed'siz) eğitim/backtest.
- Sadece grafik gösteren, karar üretmeyen bir dashboard.

## 5. Kısıtlar ve ortam gerçekleri

| Kısıt | Etki |
|---|---|
| macOS (arm64), 8 GB RAM | Küçük MLP politikalar, 2–4 paralel ortam, veri bellekte parça parça |
| Python 3.15 sistemde kurulu | PyTorch/SB3 uyumu için `uv` ile Python 3.12 izole ortam |
| MT5 Python API yok | Veri ve emir katmanı brokerdan bağımsız soyutlanır |
| Ücretli API yok | Tüm eğitim yerel, veri ücretsiz kaynaklardan |
| Ağ rate limitleri | Dukascopy 429, Yahoo 429, Stooq JS challenge (2026-11-20 ölçümü) |
| Tek geliştirici | Tek repo, tek komutla çalışan akış |

### 5.1 Veri kaynağı gerçeği

2026-11-20 tarihinde bu ağdan yapılan ölçüm:

- `datafeed.dukascopy.com` → HTTP 429 ("Too Many Requests"), EURUSD dahil.
- `query1.finance.yahoo.com` → HTTP 429.
- `stooq.com` → JS proof-of-work challenge.
- `histdata.com` → sayfa açılıyor, indirme token'ı boş dönüyor; güvenilir değil.

Sonuç: veri katmanı **takılabilir sağlayıcı** olacak. Üç sağlayıcı:

1. `dukascopy` — gerçek tick → M1 toplama, üstel backoff + disk cache. 429 geçince çalışır.
2. `csv` — kullanıcının brokerından (Exness/MT5) dışa aktardığı M1 CSV/Parquet. En gerçekçi yol.
3. `synthetic` — tohumlanmış, rejim değişimli fiyat üreteci. Ağ yokken tüm hattı test eder.

Gerçek veri gelene kadar eğitilen model **finansal olarak anlamsızdır**; sadece
hattın doğruluğunu kanıtlar. Bu dokümanda ve panelde bu açıkça yazılır.

### 5.2 Veri normalizasyonu ve kontrat varsayılanları

Tüm zaman damgaları içeride **UTC**'dir. Broker export'u GMT+2/+3 ise import
sırasında offset tespit edilir ve UTC'ye çevrilir; ham dosya değiştirilmez,
dönüşüm manifestte kaydedilir. Seans bayrakları ve günlük zarar limitinin
sıfırlanması UTC üzerinden hesaplanır (NY 17:00 = gün sınırı yapılandırılabilir).

| Parametre | Varsayılan | Not |
|---|---|---|
| Kontrat büyüklüğü | 100 oz | XAUUSD standart |
| Fiyat adımı (point) | 0.001 | tick 0.01 |
| Komisyon | $7 / lot gidiş-dönüş | config ile değiştirilir |
| Kaldıraç | 1:20 | notional cap ile tutarlı |
| Minimum lot | 0.01 | |

CSV'de bid/ask kolonları varsa spread gerçek, yoksa §9'daki spread modeli uygulanır.

## 6. Kullanıcının verdiği kararlar

1. Veri: Dukascopy'dan ücretsiz indirme (gerçek veri hedefi).
2. Panel: React/Next modern panel (Streamlit değil).
3. Kapsam: Tam v1 + strateji seçici ajan.

## 7. Benim verdiğim kararlar (gözden geçirilebilir)

- **Monorepo**: `backend/` (Python) + `frontend/` (Next.js), tek `Makefile`.
- **Backend**: FastAPI + WebSocket. Simülasyon backend'de döner, panel sadece render eder.
- **Python**: `uv` ile 3.12; `pyarrow` + `polars` (pandas'tan hızlı, 8 GB'da rahat).
- **Frontend**: Next.js 15 App Router, TypeScript, Tailwind, TradingView `lightweight-charts`, SWR.
- **RL**: Stable-Baselines3 PPO, MlpPolicy [128,128]. SAC sonra karşılaştırma için.
- **Hiyerarşi**: Contextual Bandit (LinUCB) strateji seçer → PPO seçilen stratejinin sinyal bağlamında aksiyon seçer. Tam HRL (options/frameworks) v1'de yok; bandit + PPO gerçek bir iki katman ve ölçülebilir.
- **Aksiyon uzayı**: `Discrete(4)` → Hold / Buy / Sell / Close. SL ve TP ATR çarpanıyla risk modülü tarafından atanır; ajan bunları değiştiremez (v2'de ayrı aksiyonlar).
- **Ödül**: equity değişimi (maliyetler düşülmüş) + drawdown ihlali cezası + aşırı işlem cezası. Detay §10.
- **Depolama**: Parquet, `symbol/timeframe/yıl/ay` bölümlenmiş. M1 taban, M5/M15 türetilir.
- **Yeniden üretilebilirlik**: her eğitim/backtest `run_id` + seed + config hash ile kaydedilir.

## 8. Mimari

```
frontend (Next.js)
   │  REST + WS
backend (FastAPI)
   ├── api/          REST + WS kontratları, oturum yönetimi
   ├── data/         sağlayıcılar (dukascopy | csv | synthetic), parquet store, resampling
   ├── features/     market intelligence (yapı, likidite, FVG, rejim, seans)
   ├── env/          Gymnasium ortamı + sanal broker (spread, komisyon, slippage, SL/TP)
   ├── rl/           PPO eğitimi, model kayıt defteri, değerlendirme
   ├── strategy/     LinUCB bandit, strateji aileleri, sinyal üretimi
   ├── risk/         bağımsız limit motoru (ajanın dışında)
   ├── backtest/     olay güdümlü backtester, metrikler, walk-forward
   └── execution/    PaperBroker (v1), broker arayüzü (v2: OANDA/MT5 köprüsü)
```

Bağımlılık yönü tek yönlü: `api → (rl, backtest, strategy) → env → (features, data) → store`.
`risk` herkesin altında, kimse onu atlayamaz.

### 8.1 Veri akışı

1. Sağlayıcı M1 barlarını parquet'e yazar (bid/ask OHLC + tick volume).
2. `Store` sembol+zaman aralığı ile dilim okur; M5/M15 talep üzerine yeniden örneklenir.
3. `features.build(df)` nedensel (look-ahead yok) özellik tablosu üretir.
4. `env` her adımda bir M1 barı ilerletir, ajanın aksiyonunu sanal brokerda uygular.
5. `backtest` aynı env'i deterministik modda koşar; `rl` ise eğitim modunda.
6. `api` canlı simülasyonda her N adımda WS'e `state` yayını yapar.

## 9. Veri modeli

M1 bar kaydı (parquet kolonları):

| Kolon | Tip | Not |
|---|---|---|
| ts | datetime64[ms, UTC] | bar açılış zamanı |
| open/high/low/close | float64 | bid tarafı |
| ask_open/ask_high/ask_low/ask_close | float64 | ask tarafı (yoksa NaN) |
| tick_volume | int64 | işlem sayısı |
| spread_mean | float32 | ortalama spread (fiyat birimi) |
| provider | string | kaynak etiketi |

Spread yoksa (CSV kaynağı) spread modeli devreye girer: seans bazlı taban + volatilite çarpanı, panelde ayarlanabilir.

## 10. Ortam (Gymnasium) sözleşmesi

- **Gözlem** (`Box`, float32): son 60 M1 barın özellikleri (≈40 boyut) + pozisyon durumu
  (side one-hot, girişten beri geçen bar, açık kâr/zarar, kullanılan risk%, günlük PnL%).
- **Aksiyon** (`Discrete(4)`): 0 Hold, 1 Buy, 2 Sell, 3 Close.
- **Adım**: bar kapanışında karar, sonraki barın açılışında spread+slippage ile fill.
- **Maliyetler**: yarım spread giriş + yarım spread çıkış, komisyon (lot başına sabit),
  slippage (ATR'nin yüzdesi, rejim bazlı).
- **SL/TP**: girişte ATR(14) × {1.0, 2.0}; bar içinde high/low ile tetiklenir; gap'te açılış fiyatından dolar.
- **Episode**: 20.000 bar veya equity %20 düşüşü (erken bitiş).
- **Ödül**:
  `r = Δequity / initial_balance × 100`
  `− 50 × max(0, drawdown − limit)`
  `− 0.05 × (yeni pozisyon açıldı mı)`
  `+ 0.02 × (giriş anında risk limitine uyum bonusu, tek sefer)`
  Ödül ölçeği PPO için kırpılır `[-5, +5]`.
  Not: drawdown terimi equity terimiyle aynı birimde (×100) olmalı; aksi halde
  şekillendirme etkisiz kalır. Giriş cezası, risk bonusundan büyük olmalı; yoksa
  ajan sadece pozisyon açarak pozitif ödül toplar.

### 10.1 Risk motoru (ajanın dışında)

İki limit aynı anda uygulanır; **hangisi önce bağlarsa o geçerlidir**:

- İşlem başına risk: equity'nin %0.5'i (SL mesafesinden lot hesaplanır).
- Notional maruziyet: equity'nin 20 katı (kaldıraç 1:20). Bu, risk bazlı lotu
  aşağı çekebilir; gerçekleşen işlem riski %0.5'ten küçük olabilir ve bu normaldir.

Altın fiyatı ~$2.350 ve M1 ATR ~$0.80 iken: %0.5 risk = $50 → SL 1×ATR ile
~62 oz → notional ~$147k ≈ 14.7x. 20x cap bağlamaz. ATR $0.25'e düşerse lot
büyür ve cap devreye girer. Bu davranış test edilir, panelde "efektif risk"
olarak gösterilir.

Diğer limitler:

- Günlük zarar limiti: %3 → limit aşılırsa gün sonuna kadar yeni pozisyon yok.
- Eşzamanlı pozisyon: 1 (scalping için).
- Maksimum bar sayısı: 120 bar sonra zorunlu kapanış.

Bu limitler ajan aksiyonundan önce uygulanır; ajan "Buy" dese bile limit doluysa
aksiyon `Hold`'a düşürülür ve bu olay loglanır (`risk_blocked` olayı panelde görünür).

## 11. Market Intelligence (özellikler)

Hepsi nedensel, sadece `t` ve öncesini kullanır. Testte bunu doğrulayan bir
"look-ahead" testi var (§15).

- Getiri ve momentum: 1/3/5/15/60 bar log getiri, RSI(14), MACD histogram.
- Volatilite: ATR(14), gerçekleşen vol (30 bar), volatilite rejimi (düşük/orta/yüksek).
- Yapı: swing high/low (fractal 2), BOS, CHoCH, trend yönü.
- Likidite: son N barda eşit yüksek/dip seviyeleri, sweep bayrağı (fitil geri dönüşü).
- FVG: 3 mumluk boşluk tespiti, doldurulma durumu, mesafe.
- Order block: son karşıt mumun gövdesi, fiyatın içinde olup olmadığı.
- Seans: Asya/Londra/New York bayrakları, açılış sonrası ilk 30 dk.
- Konum: günlük VWAP'a uzaklık, gün içi high/low'a göre konum.

Çoklu zaman dilimi (M5/M15) ajana **tam pencere** olarak verilmez. Build sırasında
nedensel olarak M1 satırlarına şu bağlam kolonları eklenir: son **kapanmış** M5 ve
M15 barının yönü, ATR'si, yapı durumu (BOS/CHoCH) ve FVG yakınlığı. Böylece hot
path'te resampling maliyeti olmaz, gözlem boyutu sabit kalır (~40). M5/M15 tam
grafikleri panelde gösterilir.

## 12. Strateji katmanı

Beş strateji ailesi, her biri bar başına `{-1, 0, +1}` sinyali ve güven skoru üretir:

| Strateji | Tetik |
|---|---|
| Momentum / Breakout | yapı kırılımı + hacim artışı |
| Reversal | aşırı uzama + dönüş mumu |
| Liquidity Sweep | seviye süpürme sonrası geri dönüş |
| FVG Fill | FVG'ye geri çekilme |
| Trend Follow | BOS sonrası geri çekilme |

`LinUCB` bandit bağlamı: rejim, seans, volatilite, son 20 barın strateji bazlı
başarı ortalaması. Ödül: işlem kapandığında net R katsayısı.

Eğitim sırası (ve neden):

1. **Offline etiketleme**: her bar için beş stratejinin sinyali varsayımsal işleme
   çevrilir (giriş, SL/TP/120-bar çıkış) ve net R etiketi üretilir. Bu, gerçek
   işlem beklemeye göre ~100 kat daha fazla örnek verir.
2. **Bandit ön-eğitimi**: LinUCB bu etiketler üzerinde kapalı formda çözülür.
3. **Bandit dondurulur**, PPO eğitilir. Aksi halde PPO değişirken bandit'in ödülü
   non-stationary olur ve iki katman birbirini bozar.

Canlı simülasyonda bandit yalnızca gerçekleşen kapanışlarla çevrimiçi güncellenir
(opsiyonel, varsayılan kapalı).

PPO gözlemine bandit'in seçtiği strateji one-hot + güven skoru eklenir.

## 13. Backtest ve metrikler

- Olay güdümlü, env ile **aynı** muhasebe kodu (tek doğruluk kaynağı).
- Metrikler: net kâr, profit factor, Sharpe (bar getirisinden yıllıklandırılmış),
  max drawdown, win rate, expectancy (R), ortalama işlem süresi, işlem sayısı,
  toplam maliyet, maliyet/kâr oranı, strateji bazlı kırılım.
- Walk-forward: eğitim 6 ay → validasyon 1 ay → test 1 ay, 1 ay kaydırmalı.
  **v1'de her fold'da yeniden eğitim yok**: tek model eğitilir, kayan pencerelerde
  yalnızca dışarıdan değerlendirilir (out-of-sample). Fold başına yeniden eğitim
  `--retrain-per-fold` bayrağıyla opsiyonel; 8 GB M2'de 3 fold ile sınırlıdır.
- Maliyet stresi: spread ×1, ×2, ×3 ve slippage ×2 senaryoları.
- Kıyas tabanı: "her zaman hold" ve "rastgele aksiyon" — ajan bunları geçemiyorsa başarısız.

## 14. API ve panel

### 14.1 REST

| Endpoint | İş |
|---|---|
| `GET /api/health` | sürüm, cihaz, provider durumu |
| `GET /api/data/status` | sembol/timeframe kapsamı, bar sayısı, son güncelleme |
| `POST /api/data/download` | sağlayıcı + aralık ile indirme işi başlatır |
| `POST /api/train` | config ile eğitim başlatır, `run_id` döner |
| `GET /api/runs` | eğitim/backtest koşuları listesi |
| `GET /api/runs/{id}` | metrikler, eğri, config |
| `POST /api/backtest` | kayıtlı modelle backtest |
| `POST /api/sim/start` | canlı simülasyon (hız, mod) |
| `POST /api/sim/control` | durdur / hızlandır / sıfırla |

### 14.2 WebSocket `/ws/sim`

Mesaj tipleri: `bar`, `decision`, `trade_open`, `trade_close`, `risk_blocked`,
`equity`, `episode_end`, `status`. Her mesaj `seq` taşır; panel kayıp durumunda
`seq` ile yeniden senkron ister.

### 14.3 Ekranlar

1. **Simulator** — mum grafiği (M1/M5/M15 geçişi), alım/satım işaretleri, SL/TP çizgileri,
   equity eğrisi, açık pozisyon kartı, seçili strateji, hız kontrolü (1x/10x/100x/max).
2. **Training** — ödül eğrisi, episode uzunluğu, entropi, kayıp, bandit ağırlıkları,
   eğitim ilerlemesi ve durdur/devam.
3. **Performance** — metrik kartları, drawdown grafiği, strateji bazlı kırılım,
   maliyet analizi, walk-forward sonuç tablosu.
4. **Data** — provider durumu, indirme işleri, kapsam takvimi, sentetik veri üretme.

**Simülasyon semantiği**: v1 "canlı" modu geçmiş bar replay'idir, gerçek zamanlı
feed değil. `1x` = saniyede 1 bar (duvar saati), `10x`/`100x` doğrusal çarpan,
`max` = mümkün olduğunca hızlı. Veri bitince simülasyon `episode_end` yayınlar ve
durur; panelde "başa sar" ve "sıfırla" ayrı eylemlerdir. Gerçek zamanlı feed v2.

Panelde her yerde "bu sonuç gerçek para değil / sentetik veri" etiketi görünür.

## 15. Test stratejisi

Zorunlu, yeşil olmadan iş bitmiş sayılmaz:

- **Muhasebe invariantları**: equity = bakiye + açık PnL; işlem PnL'i maliyetlerle tutarlı;
  SL/TP fill fiyatları bar aralığında.
- **Look-ahead testi**: `features(df.iloc[:t])` ile `features(df).iloc[t]` aynı olmalı
  (rastgele 20 `t` için).
- **Determinizm**: aynı seed + config → aynı equity eğrisi. Kapsam: env, broker
  muhasebesi, backtest ve simülasyon **bit bazında** (tek süreç, tek thread).
  Eğitimde bit-exactlik aranmaz; aynı seed + aynı thread sayısı + aynı sürümlerle
  metrikler tekrar üretilebilir düzeyde olmalı, tolerans raporlanır.
- **Risk motoru**: limitler aşılamıyor; `risk_blocked` doğru üretiliyor.
- **Backtest = env**: aynı politika ile ikisi aynı metrikleri vermeli.
- **API kontrat testleri**: şemalar, hata durumları.
- **Frontend**: `vitest` bileşen testleri + Playwright ile "simülasyon başlat → bar akışı → trade marker" smoke testi.
- **Veri sağlayıcı**: `synthetic` deterministik; `csv` parser bozuk satır/eksik kolon testleri; `dukascopy` 429 → backoff testi (mock).

## 16. Kilometre taşları

| # | Çıktı | Kabul |
|---|---|---|
| M0 | Repo iskeleti, uv + Next kurulumu, CI'sız yerel `make dev` | `make dev` panel + API ayağa kalkar |
| M1 | Sentetik veri + M1 store + sanal broker + env | pytest muhasebe testleri yeşil |
| M2 | Simulator ekranı canlı mum + rastgele politika akışı | Panelde 100x simülasyon akar |
| M3 | Özellikler + backtest + metrikler + baseline kıyası | Look-ahead testi yeşil, rapor üretilir |
| M4 | PPO eğitimi + Training ekranı | 200k adım eğitim koşar, eğri görünür |
| M5 | Bandit strateji katmanı + Performance ekranı | Strateji kırılımı raporlanır |
| M6 | Walk-forward + maliyet stresi | Kayan pencere raporu (sentetik) üretilir |
| M7 | Gerçek veri geldiğinde aynı hattı yeniden koşma | Gerçek XAUUSD M1 ile rapor; veri yoksa atlanır |

M2'den sonra görsel bir ürün var; M3'ten sonra ölçülebilir sonuç. M3–M6 sonuçları
sentetik veride olduğu için **finansal kanıt sayılmaz**, sadece hat doğrulamasıdır.

## 17. Riskler ve dürüst sınırlar

- **Sentetik veri finansal olarak anlamsız.** M4'teki "iyi" sonuç gerçek değil.
- **Scalping'de maliyet belirleyici.** Spread/komisyon gerçekçi değilse backtest yalan söyler.
- **8 GB RAM** büyük deneyleri sınırlar; 1–3M adım üstü sabır ister.
- **Dukascopy 429** sürebilir; gerçek veri için CSV export veya OANDA demo hesabı gerekebilir.
- **Kâr garantisi yok.** Başarı ölçütü "kârlı olmak" değil, "maliyet sonrası kıyas tabanlarını geçmek ve test setinde çökmemek".

## 18. Kalan kararlar (senin cevabın gerekiyor)

1. **Gerçek veri**: şimdilik yok. Sistem sentetik + CSV hazır adaptörüyle kurulur;
   gerçek veri geldiğinde M7 yeniden koşulur. Bu yüzden M3–M6 kabul kapıları
   "hat doğru çalışıyor" düzeyinde tutulur, "kârlı" iddiası yapılmaz.
2. **Eğitim bütçesi**: sınır yok. Varsayılan yine 2 saatlik checkpoint'lerle
   ilerler (kesilebilir olması için), ama üst sınır yok.
3. **OANDA demo**: şimdilik yok. `execution/` arayüzü hazır bırakılır.

Varsayılanlar: panel Türkçe, eğitim kesilebilir, gerçek veri sonra.

## 19. Reviewer sonrası verdiğim kararlar (kontrol et)

- Risk: %0.5 risk **ve** 20x notional cap birlikte, hangisi önce bağlarsa. 5x cap
  altın M1'de neredeyse her emri bloklardı.
- Walk-forward: v1'de tek model + kayan out-of-sample değerlendirme; fold başına
  yeniden eğitim opsiyonel bayrak.
- Simülasyon: geçmiş bar replay'i; 1x = saniyede 1 bar. Gerçek zamanlı feed v2.
- Bandit: offline counterfactual R etiketleriyle ön-eğitim, PPO sırasında dondurulur.
- M5/M15: ajana tam pencere değil, son kapanmış barın bağlam kolonları.
- Determinizm: muhasebe/backtest bit-exact, eğitim istatistiksel tekrarlanabilir.
- Kontrat: 100 oz, point 0.001, komisyon $7/lot gidiş-dönüş, kaldıraç 1:20.
- Zaman: içeride UTC, broker offset'i import'ta dönüştürülür ve manifeste yazılır.
