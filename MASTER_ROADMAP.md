# 🌌 Otonom Yapay Zekâ Ekosistemi — Ana Yol Haritası & Canlı Durum Raporu
**Proje Sahibi:** @coruhoorhan  
**Tarih:** 2026-09-04  
**Durum:** Faz 1 & 2 Canlı Üretim (Production-Grade Autonomous Loop)

---

## 🏛️ 1. Mimari Roller & Görev Dağılımı

Bu ekosistem; insan müdahalesine ihtiyaç duymadan kendi kodunu yazan, güvenlik açıklarını denetleyen ve sistemlerini otonom yaşatan bir Yapay Zekâ Organizasyonudur.

```text
                                 ┌─────────────────────────────────────────────────────────┐
                                 │       1. MERKEZİ BEYİN & GÖZCÜ: MAGDA-AGENT             │
                                 │   • 57 Bilişsel Modül (ACS Guard, MemGPT, DAG Planner)  │
                                 │   • 7/24 Fullstack LLM Watchdog (Inception Mercury-2)   │
                                 │   • Bağımsız Güvenlik & Kod Denetçisi (Quality Gate)    │
                                 └────────────────────────────┬────────────────────────────┘
                                                              │ (A2A / GitHub Actions / REST API)
                                                              ▼
                                 ┌─────────────────────────────────────────────────────────┐
                                 │       2. OTONOM İŞÇİ & KODLAYICI: JULES / CODEX         │
                                 │   • agent_tasks.json'dan sıradaki 'todo' görevini alır  │
                                 │   • Kodu ve testleri yazar, Pull Request (PR) açar      │
                                 └────────────────────────────┬────────────────────────────┘
                                                              │ (PR Denetimi & Auto-Merge)
                                                              ▼
                                 ┌─────────────────────────────────────────────────────────┐
                                 │       3. HEDEF ÜRÜNLER (Çalışma Alanları)               │
                                 │   • Airbnb Fatsa Clone (coruhoorhan/airbnb-app)         │
                                 │   • Spacebot Rust İletişim Botu (coruhoorhan/spacebot)  │
                                 │   • Gelecek Projeler (E-Ticaret, CRM, E-Belediye)       │
                                 └─────────────────────────────────────────────────────────┘
```

---

## 🛡️ 2. 5 Aşamalı Katı Kalite Kapısı (Audit Quality Gate)

```text
       ┌─────────────────────────────────────────────────────────┐
       │                   JULES KODU YAZAR & PR AÇAR            │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 🛡️ 1. ADIM: MAGDA AI BAĞIMSIZ GÜVENLİK VE KOD DENETÇİSİ │
       │    (Mercury-2 LLM Diff Analizi & Güvenlik Taraması)     │
       └──────────────┬───────────────────────────┬──────────────┘
                      │                           │
          [CHANGES REQUESTED]                 [APPROVED]
                      │                           │
                      ▼                           ▼
       ┌──────────────────────────────┐   ┌──────────────────────────────┐
       │ 🛑 MERGE KESİNLİKLE KİLİTLENİR│   │ 📦 2. ADIM: Derleme & AST Test│
       │                              │   │                              │
       │ • GitHub Actions FAIL eder   │   │ • npm run build              │
       │ • Auto-Merge kilitlenir      │   │ • AST Smoke Check            │
       │ • PR'a detaylı rapor basılır │   │ • manifest doğrulaması       │
       │ • Düzeltme commit'i beklenir │   └──────────────┬───────────────┘
       └──────────────────────────────┘                  │ (Hepsi Yeşil)
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │ ✅ 3. ADIM: OTOMATİK MERGE   │
                                          │    ve SONRAKİ GÖREVE GEÇİŞ   │
                                          └──────────────────────────────┘
```

---

## 🚀 3. Airbnb Projesi (`coruhoorhan/airbnb-app`) Canlı İlerleme Durumu

### A. Tamamlanan Görevler (`Done / Merged`):
1. **`task-01-database-schema`:** SQLite WAL Database Schema & iyzico Payments *(Arşivlendi / Tamamlandı)*.
2. **`auto-fix-missing_rate_limiter` (PR #7):** `POST /api/listings` endpoint'ine `listingsRateLimiter` middleware'i eklendi ve `main` dalına merge edildi.
3. **`feat-01-dynamic-pricing` (PR #8):** `pricingEngine.js` dinamik fiyatlandırma motoru yazıldı, Magda AI Denetçisi tarafından satır satır incelendi ve `main` dalına merge edildi.

### B. Sıradaki Aktif Görev Kuyruğu (`agent_tasks.json`):
1. **`feat-02-secure-auth` (Sıradaki):** OAuth2 ve JWT Tabanlı Güvenli Kimlik Doğrulama (`src/lib/auth.js`).
2. **`feat-03-graphql-api`:** GraphQL Endpoint ile Tek Sorgu Çok Veri Çekimi (`src/lib/graphqlSchema.js`).
3. **`feat-04-responsive-ui`:** Mobil ve Tablet İçin Tamamen Duyarlı Tasarım (Tailwind CSS).
4. **`feat-05-accessibility-audit`:** WCAG 2.1 AA Uyumlu Erişilebilirlik Kontrolleri (`src/lib/accessibility.js`).
5. **`security-rate-limit-input-sanitization`:** Mercury-2 LLM'in keşfettiği genel API girdi temizleme ve XSS koruma katmanı.
6. **`backend-db-connection-pooling-caching`:** Mercury-2 LLM'in keşfettiği SQLite bağlantı havuzu ve in-memory önbellekleme (cache).

---

## 🧠 4. Magda-Agent 7/24 Bilişsel Daemon & Gözcü Sistemi

* **Canlı Motor:** Inception Labs `mercury-2` reasoning LLM (`magda_airbnb_daemon.py`).
* **Otomatik Kod İnceleme:** Her periyotta kod tabanını tarar, yeni güvenlik veya mimari ihtiyaçlar bulduğunda doğrudan `agent_tasks.json` içine yeni `todo` görevleri yazar.
* **Veritabanı Auto-Healer:** 0 TL ilanları ve süresi geçmiş kuponları canlıda otomatik onarır.
* **Bağımsız Denetçi:** `scripts/pr_code_auditor.py` ile Jules'un açtığı her PR'ı bağımsız bir müfettiş gibi inceler ve `CHANGES REQUESTED` durumunda merge'i kilitler.

---

## 🗺️ 5. Sonraki Adımlar (Gelecek Planı)

1. **Airbnb Otonom Görev Döngüsünün Tamamlanması:** `feat-02`'den `feat-05`'e kadar tüm modüllerin Jules tarafından kodlanıp Quality Gate onayından geçerek `main` dalına birleştirilmesi.
2. **Merkezi Magda Dashboard (Gelecek Aşama):** Tüm repoların üstünde, bağımsız bir siber güvenlik ve otonom komuta paneli (`coruhoorhan/magda-dashboard`) kurulması.
3. **Spacebot Rust Entegrasyonu:** Rust iletişim botu (`coruhoorhan/spacebot`) PR #10 merge ve PR #11 inceleme aşamasına geçilmesi.
