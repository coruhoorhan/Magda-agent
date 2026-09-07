# Repo Slim-Cleanup Planı — airbnb-app (Pazartesi)

Tarih: 2026-09-04 (plan), uygulama: Pazartesi. Hedef repo: `coruhoorhan/airbnb-app`.
Kural: temizlik **PR ile** yapılır (CI + denetçi kapısından geçer), doğrudan `main`'e push yok.

## Envanter (04.09.2026, `main`)

| Dizin/Dosya | Adet | Hüküm |
|---|---|---|
| `magda_agent/` | 745 | KORU — canlı beyin |
| `src/`, `server.js`, `tests/`, `package.json` | ~60 | KORU — ürün + kapı testleri |
| `agent_tasks.json`, `AGENTS.md`, `docs/PRD.md` | — | KORU — kuyruk + bağlam |
| `.github/`, `scripts/` | 19 | KORU — loop makinesi |
| `__pycache__/`, `*.pyc` | 41 | SİL — derleme artığı, `.gitignore`'a ekle |
| `.scaffolding/` | 292 | DOĞRULA — veyyon/Agent-Stack harness; Jules kullanmıyor. Veyyon kullanımı varsa KORU, yoksa SİL |
| `.agents/`, `.harness/`, `.archcore/` | 17 | DOĞRULA — aynı harness ailesi |
| `orchestrator-src/` | 5 | DOĞRULA — ne olduğu belirsiz, sahibine sor |
| `dist/` | 0 tracked | TAMAM — zaten repoda yok, sunucuda build alınıyor |
| `.db` dosyaları | 0 tracked | TAMAM — repoda yok |
| Session HTML'leri | 0 tracked | TAMAM — repoya girmemiş |
| Ölü dallar | 0 | TAMAM — temizlendi |

## `.gitignore` eklenecekler (Pazartesi adım 1)

```
__pycache__/
*.py[cod]
*.db
*.sqlite3
veyyon-session-*.html
.DS_Store
```

## Pazartesi adımları (sırayla, kapılı)

1. **Doğrulama (insan):** `.scaffolding`, `.agents`, `.harness`, `.archcore`, `orchestrator-src` — veyyon kullanıyor mu? Liste parça parça silinecek, toplu değil.
2. **PR-1 (güvenli):** 41 `.pyc` sil + `.gitignore` sıkılaştır. CI yeşili beklenir (etki: sıfır).
3. **PR-2 (harness):** Adım-1 cevabına göre harness klasörlerini kaldır. CI + denetçi onaylar.
4. **PR-3 (isteğe bağlı):** `orchestrator-src` kararı.
5. **Doğrulama:** kopya sayısı, klon süresi, Jules oturumunda dosya listesi. Sentinel yeşil kalır.

## Güvenlik notları

- Silinen tracked dosya, sunucuda `git pull` ile de silinir. `dist/` zaten tracked değil — sorun yok. Başka servis edilen tracked dosya varsa deploy planına `npm run build` şartı eklenir.
- `.env`, `data/`, `node_modules` repoda yok — dokunulmaz, sunucuda kalır.
- Rollback: her adım ayrı PR + ayrı commit; geri alma = revert tek komut.
