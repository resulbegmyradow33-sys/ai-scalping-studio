# Taklit ile pekiştirmeli öğrenme denemesi

Bu belge, "robot yürümeyi nasıl öğreniyorsa ajan da öyle öğrensin" fikrinin
ölçülebilir hâlini kaydeder. Sonuç dürüst: taklit, PPO'yu kararlı hale getirdi
ama doğrulanmış kuralı geçemedi.

## Neden taklit

Sıfırdan PPO iki bölünmede tutarsızdı: 2024+ testinde +0.099 R, 2025+ testinde
−0.049 R. Az işlem (5 yılda ~600 sıralı işlem), seyrek ödül ve değişen rejim
keşfi zorlaştırıyor. Standart çözüm, ajanı önce kanıtlanmış kuralı taklit
edecek şekilde eğitip sonra PPO ile ince ayar yapmak.

## Yöntem

1. Öğretmen: canlı kartın `flow_momentum` kuralı (H1, akış kapısı, 3/6 ATR, 48 bar).
2. 8 paralel ortamda rastgele başlangıçlardan 100.000 `(gözlem, aksiyon)` çifti
   toplanır; gözlemler `VecNormalize`'dan geldiği için PPO'nun göreceği ölçekte.
3. Sınıf dengesizliği ters frekans ağırlıklı çapraz entropi ile giderilir;
   girişler barların ~%5'i.
4. Gösterilerin %15'i doğrulama için ayrılır; doğrulama kaybı iki tur
   iyileşmezse eğitim durur (erken durdurma).
5. Klon kaydedilir, ardından 300.000 adım PPO ince ayarı yapılır.
6. Değerlendirme, kartın kendi kuralıyla aynı dilimde ve sıralı simülatörde.

İki öğretmen tanımı denendi:

- **İstediği aksiyon** (stateless): kural her aşırı akış barında giriş ister;
  pozisyon açıkken ortam reddeder. Klon bu reddedilen girişleri de öğrenir.
- **Uygulanan aksiyon** (stateful): ortamın gerçekten uyguladığı aksiyon
  kaydedilir; klon pozisyon açıkken tutmayı öğrenir.

## Sonuçlar

H1, gerçek spread, sıralı simülatör:

| Deney | 2024+ | 2025+ |
|---|---|---|
| **Kural (canlı kart)** | **+0.068 R**, PF 1.11, 266 işlem | **+0.103 R**, PF 1.12, 123 işlem |
| PPO sıfırdan | +0.099 R, PF 1.25 | −0.049 R, PF 0.69 |
| BC + PPO (stateless) | +0.012 R, PF 1.07, 452 işlem | +0.020 R, PF 1.05, 242 işlem |
| Sadece BC (stateless) | −0.117 R, PF 0.74, 41 işlem | — |
| BC + PPO (uygulanan aksiyon) | −0.018 R, PF 0.96, 403 işlem | — |

Klonun doğrulama başarımı yüksekti:

| Öğretmen | Doğrulama doğruluğu | Giriş recall | Tahmin edilen giriş oranı |
|---|---|---|---|
| stateless | %95,1 | — | — |
| uygulanan | %99,96 | %100 | %0,82 |

Yüksek doğruluk, eğitim rejiminde kuralı neredeyse birebir kopyaladığını
gösteriyor; 2024+ diliminde kaybetmesi bu kopyalamanın **rejim ezberi**
olduğunu gösteriyor. Uygulanan-aksiyon öğretmeni daha sadık ama daha az giriş
öğrettiği için testte neredeyse hiç işlem açmadı.

Run kimlikleri: stateless BC+PPO `07da94f85e51` (2024+) ve `e7969734070a`
(2025+); sadece BC `15c31a9492dc`; uygulanan aksiyon `9cb0b06c9371`.

## Sürekli deney: `nightly_rl`

`scripts/nightly_rl.py` bu deneyi veri değiştiğinde otomatik tekrarlar:

- veri özeti (digest) son kayıtla aynıysa hiçbir şey yapmaz;
- değiştiyse her bölünme için BC+PPO eğitir, aynı dilimde kuralı da koşturur;
- `beats_rule` yalnızca ajan **her** bölünmede pozitif ve kuraldan iyi ise true olur;
- sonuç `runs/rl_history.jsonl`'e yazılır; canlı kart **asla** değişmez.

```bash
cd backend
uv run python -m scripts.nightly_rl --splits 2024-01-01 2025-01-01
nohup uv run python -m scripts.nightly_rl --loop --interval-hours 24 \
  > runs/rl_lab.log 2>&1 &
```

## Verdict

RL bu problemde artık "kararsız" değil: taklit ile iki bölünmede de pozitif,
ama kuralın yarısından az kenar üretiyor ve maliyet stresine dayanma payı yok.
Canlı strateji kural olarak kalır. RL, yeni veri geldiğinde otomatik yeniden
denenir; kuralı geçerse `beats_rule: true` kaydı çıkar ve terfi insan kararına
bırakılır.
