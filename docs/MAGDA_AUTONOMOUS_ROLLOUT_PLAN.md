# Magda Otonom Döngü — Rollout Planı (magda-agent → ürün repoları → taşınabilir kit)

Tarih: 2026-09-04. Sahibi: @coruhoorhan.
Hedef sırası: **Faz 1** magda-agent self-loop → **Faz 2** ürün repolarına dağıtım → **Faz 3** taşınabilir kit.

## Tespit edilen eksikler (04.09.2026 incelemesi)

1. **57 modül `main`'de değil:** `feature/magda-cognitive-v2-architecture` dalında duruyor; Jules `startingBranch: main` ile çalıştığı için bu kodu göremez.
2. **Local'de commitlenmemiş iş var:** `feature/...` dalında M + ?? dosyalar (`llm_client.py` değişikliği, `secret_redaction`, `airbnb_bridge`, testler, session HTML'leri). GitHub'da yok = Jules için yok.
3. **Denetçi + responder magda-agent'ta yok:** `scripts/` içinde `pr_code_auditor.py` ve `jules_responder.py` yok; `jules_responder.yml` workflow'u yok. `jules_automerge.yml` eski sürüm (YAML kolon hatası + 422 bug'ı + hardcoded key düzeltmeleri sadece airbnb-app'e gitti).
4. **Secret eksik:** Magda-agent Secrets'ta `JULES_API_KEY` + `PAT` var; `GH_PAT` ve `OPENAI_API_KEY` yok → auditor çalışamaz.
5. **Prompt ↔ politika çelişkisi:** `jules_next_task.yml` prompt 6. maddesi her PR'da 2-3 YENİ görev ekletiyor (görev şişmesinin motoru: 1857 görev). `agent_tasks.json` içindeki `replenishment_policy` ise `always_add_tasks: false` diyor. İkisi barışmalı.
6. **Seçim kuralı riskli:** 989 `todo` var, ilk sıralar trend-üretilmiş modüller (Letta sync v3, Claude planner export v2...). Politika `stabilization_first` diyor ama seçim "FIRST todo" yapıyor. Tam otonomi bu havuzla başlarsa kaos + maliyet çıkar.
7. **airbnb-app AGENTS.md uyumsuzluğu:** airbnb-app'teki AGENTS.md, veyyon/Agent-Stack kılavuzu (`.archcore/.scaffolding/.harness`); Jules prompt'u "AGENTS.md'yi oku" diyor ama dosya Jules'a göre yazılmamış. (Faz 2'de düzelir.)
8. **Python uyarlaması yok:** automerge workflow'u `npm ci/build` varsayıyor; magda-agent Python → `pytest` hedefi gerekli. AST smoke check `.py` parse ediyor (uygun).

## Faz 1 — magda-agent self-loop (önce burası)

0. **Hazırlık (manuel + tek seferlik):**
   - Local değişiklikleri toparla → `feature/...` → PR → `main` merge (57 modül + secret_redaction + bridge + testler).
   - Magda-agent Secrets'a `GH_PAT` ve `OPENAI_API_KEY` ekle (airbnb-app'teki değerler).
   - `agent_tasks.json` triyajı: pilot şeridi seç (öneri: `stabilization` + `testing` area'larındaki 5 todo + `blocked` 60 kaydın ayıklanması). İlk 5 trend modülü pilota ALINMAZ.
1. **Loop dosyalarını taşı (airbnb-app'ten kanıtlı sürümler):**
   - `scripts/pr_code_auditor.py` (COMMENT + fallback sürümü), `scripts/jules_responder.py`, `.github/workflows/jules_responder.yml` (cron 15dk).
   - `jules_automerge.yml`: tırnaklı step isimleri, auditor ilk adım + fail-closed, **npm yerine pytest** (`pip install -r requirements.txt` + `pytest tests/<hedef> -q`), merge adımı aynı.
   - `jules_next_task.yml`: done-kapısı guard (airbnb'deki), prompt magda-agent'a özel kalır + şu 3 satır eklenir: `in_progress` kuralı, "todo OR in_progress" seçim kuralı, replenishment çelişkisinin çözümü (6. madde → "sadece havuz minimumun altına düşerse, politikaya uygun ekle").
2. **Pilot (3-5 görev, gözetimli):** workflow_dispatch ile tek oturum başlat → PR → auditor → merge zincirini izle. Maliyet ve kalite onayı.
3. **Tam otonom:** pilot yeşilse cron/responder devrede, merge → next-task → responder döngüsü kendi kendine döner. `high/critical` PR'lar insana kalır (merge_policy).

Kabul: pilot 3 PR üst üste auditor+test yeşiliyle merge olur; spurious (görevsiz) Jules oturumu açılmaz; responder bekleyen oturumu 15dk içinde cevaplar.

## Faz 2 — Ürün repolarına dağıtım

- Merkez (magda-agent) karar verir, hedef repo (airbnb-app, spacebot, TEYİT: hangi repo?) uygular.
- Hedef repoda olması gereken set: `agent_tasks.json` (o ürüne özel görevler), `scripts/{validate,pr_code_auditor}.py`, 3 workflow (automerge dil-uyarlamalı, next_task ürün prompt'lu, responder ortak), 3 secret.
- airbnb-app AGENTS.md: Jules'a göre sadeleştirilmiş sürümle değiştirilir (veyyon kılavuzu `.archcore/`'da kalır, prompt doğru dosyayı gösterir).
- Merkez→hedef görev akışı: magda-agent daemon'u hedef repo AST'sini tarar, bulguyu o reponun `agent_tasks.json`'ına yazar (kanıtlı: Mercury-2'nin ürettiği 2 güvenlik görevi gibi).

## Faz 3 — Taşınabilir kit (tek komut)

- `python -m magda_agent.kit init --target <repo> --stack python|node|rust --prompt "<ürün>"`: yukarıdaki seti üretir, secret listesini yazdırır, ilk smoke PR'ını doğrular.
- Kit, Faz 1'de kanıtlanmış dosya sürümlerinden üretilir; sürüm etiketi taşır.

## Maliyet / güvenlik korkulukları (tüm fazlar)

- Responder cron 15dk, oturum başına max 3 cevap, 2 saatten eski oturuma dokunmaz, fazlasında `needs-human-review` issue'su.
- Next-task guard: `done`'a dönen görev yoksa oturum açılmaz.
- `high/critical` asla auto-merge olmaz. Hardcoded secret yasak (auditor secret taraması yapar).
- Jules oturum maliyeti izlenir; pilot öncesi tam-gaz açılmaz.
