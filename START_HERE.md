# START HERE — yeni geliştirici

Bu sayfa, repoya ilk kez gelen birinin **hiçbir şey sormadan** başlaması için
gereken her adımı tek yerde toplar. Daha ayrıntılı kurallar alttaki dosyalarda.

## 0. Erişim

Repo **public**: okumak ve klonlamak için izin gerekmez.

- **Yazma izni** (collaborator): aynı repoda branch açıp PR gönderir.
- **Fork**: kendi kopyanda çalışıp upstream'e PR gönderir. Issue atanamaz;
  issue yorumunda "bende" yaz.

Davet komutu (repo sahibi):

```bash
gh api -X PUT repos/resulbegmyradow33-sys/ai-scalping-studio/collaborators/KULLANICI \
  -f permission=push
```

## 1. Kurulum (yaklaşık 10 dakika)

Gerekenler: Git, GitHub hesabı, Python 3.12 + `uv`, Node 20+.

```bash
gh auth login
gh auth setup-git
git clone https://github.com/resulbegmyradow33-sys/ai-scalping-studio.git
cd ai-scalping-studio
make hooks

cd backend
uv sync
uv run pytest          # 539 test, ~2 dakika

cd ../frontend
npm ci
npm test               # 184 test
```

Fork kullanıyorsan:

```bash
gh repo fork resulbegmyradow33-sys/ai-scalping-studio --clone
cd ai-scalping-studio
git remote add upstream https://github.com/resulbegmyradow33-sys/ai-scalping-studio.git
git fetch upstream
```

## 2. Sistemi çalıştır

Veri repoda yok. Sentetik veriyle hemen çalışır:

```bash
cd backend
uv run python -m scalping.cli data synth \
  --root ../demo-data --symbol XAUUSD --start 2024-10-01 --end 2024-10-09
uv run python -m scalping.cli serve --root ../demo-data --port 8000
```

Başka terminal:

```bash
cd frontend
npm run dev -- -p 3001
```

Aç: http://127.0.0.1:3001

Gerçek veri kaynakları ve import komutları: `README.md`,
`docs/data-pipeline.md`, `docs/forward-test.md`.

## 3. Bir iş seç

Sıra:

1. `docs/task-board.md` — her madde tek tek, `[x]` / `[ ]`.
2. GitHub faz issue'ları (#12–#28) — hangi iş hangi fazda.
3. `STATUS.md` — ne yapıldı, ne kısmi, ne yok.
4. `docs/implementation-roadmap.md` — sıra, dosyalar, RAM, testler.

İşi seçince issue'ya yorum yaz: **"bende"**. Fork kullanıyorsan issue atanamaz,
aynı yorum yeterli.

İlk katkı için `good first issue` etiketli issue'ları seç. Şu an açık olanlar:

- `API: /api/timeframes endpoint`
- `CLI: data validate command`
- `Frontend: show provenance on the Data page`

## 4. Çalışma döngüsü

```bash
git switch main
git pull --rebase origin main
git switch -c feat/kisa-isim

# kodla + test yaz

make check             # CI'ın çalıştırdığı her şey
git add <dosyalar>
git commit -m "feat(scope): kısa açıklama"
git push -u origin feat/kisa-isim
gh pr create --base main
```

PR açıklamasında:
- ne değişti,
- hangi komutları çalıştırdın ve sonucu,
- veri/model kullandıysan dataset id + run id,
- risk ve geri alma planı.

`main` korumalı: CI yeşil + 1 review + güncel branch olmadan merge edilmez.
Merge squash ile yapılır, branch silinir.

## 5. PR kabul kriteri

- [ ] Kod + test.
- [ ] `make check` yeşil (ruff, mypy, backend test, frontend test, build).
- [ ] Veri store'u, `runs/`, model ağırlığı, `.env` veya sır yok.
- [ ] İlgili `docs/*` ve `docs/task-board.md` kutuları güncel.
- [ ] Gerçek para yolu açılmadı.
- [ ] İddia edilen sonuç, yazılan commit ve dataset ile tekrar üretilebilir.

## 6. Kurallar

- Veri ve sır commit edilmez.
- "Yapıldı" = kod + test. "İyi model" = iki kronolojik bölünme + maliyet
  stresi + ileri test.
- Model sonuçları uydurulmaz; başarısız deney silinmez, kaydedilir.
- Gerçek para yok; varsayılan paper.
- `main`'e doğrudan push yok (sunucu + yerel hook).
- Büyük değişiklikten önce issue aç; tek PR'da tek mantıksal değişiklik.

## 7. Nerede ne var

| Dosya | Ne için |
|---|---|
| `README.md` | Genel bakış, kurulum, veri kaynakları |
| `CONTRIBUTING.md` | Clone, branch, commit, push, PR |
| `docs/team-workflow.md` | Roller, review, veri/model kuralları, board |
| `docs/task-board.md` | Tek tek görev listesi ve durum |
| `STATUS.md` | Yönetici özeti: yapıldı/kısmi/yok |
| `docs/implementation-roadmap.md` | Faz sırası, dosyalar, RAM, testler |
| `docs/architecture-audit.md` | Mevcut mimari ve eksikler |
| `docs/model-validation.md` | Split, purge/embargo, promotion kapısı |
| `docs/data-pipeline.md` | Veri katmanları, doğrulama, provenance |
| `.github/ISSUE_TEMPLATE/` | Bug/feature/data/experiment formları |

## 8. Takılırsan

- Hata: `bug_report.yml` şablonuyla issue aç; commit + komut + hata metni ekle.
- Tasarım sorusu: Discussions.
- Güvenlik: `SECURITY.md` (public issue açma).
- Emin değilsen issue'ya yorum yaz; körlemesine büyük değişiklik yapma.
