# 🏛️ Otonom Yapay Zekâ Yazılım Ekosistemi — Ana Plan & Mimari Standartlar
**Tarih:** 2026-09-04  
**Proje Sahibi:** @coruhoorhan  
**Durum:** Faz 1 & 2 Aktif Geliştirme (Production-Grade Autonomous Loop)

---

## 🌟 1. Mimari Vizyon ve Görev Dağılımı

Sistem, insan müdahalesine ihtiyaç duymadan yazılım projelerini denetleyen, açıklarını bulan, görev üreten, kod yazan ve güvenli bir şekilde ana dala birleştiren **tam otonom bir Yapay Zekâ Organizasyonudur**.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   1. MERKEZİ BEYİN & KOMUTA MERKEZİ: MAGDA-AGENT                       │
│  • 57 Bilişsel Modül (DAG Planlayıcı, ACS Guard, MemGPT, OpenClaw RL)                  │
│  • 7/24 Fullstack LLM Watchdog Daemon (Inception Labs Mercury-2)                       │
│  • Bağımsız Güvenlik & Kod Denetçisi (Quality Gate Auditor)                            │
│  • Merkezi Bilişsel Komuta & Siber Güvenlik Paneli (Magda Command Dashboard)           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (A2A / GitHub REST API / Daemon)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   2. HEDEF ÜRÜNLER & ÇALIŞMA ALANLARI (Repolar)                         │
│  • Airbnb Fatsa Clone (coruhoorhan/airbnb-app)                                         │
│  • Spacebot Rust İletişim Botu (coruhoorhan/spacebot)                                  │
│  • Yeni Gelecek Projeler (E-Ticaret, CRM, E-Belediye)                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ 2. Kalite Kapısı ve 5 Aşamalı Otonom Yazılım Döngüsü

```text
       ┌─────────────────────────────────────────────────────────┐
       │                   JULES KODU YAZAR & PR AÇAR            │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 1. KAPI: STATİK TESTLER & DERLEME (Build & AST Smoke)   │
       └────────────────────────────┬────────────────────────────┘
                                    │ (Geçerse)
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 2. KAPI: MAGDA AI BAĞIMSIZ GÜVENLİK VE KOD MÜFETTİŞİ    │
       │    (Mercury-2 / Claude Diff İncelemesi)                 │
       └──────────────┬───────────────────────────┬──────────────┘
                      │                           │
         [CHANGES REQUESTED]                  [APPROVED]
                      │                           │
                      ▼                           ▼
       ┌──────────────────────────────┐   ┌──────────────────────────────┐
       │ 🛑 MERGE KESİNLİKLE KİLİTLENİR│   │ 5. KAPI: OTOMATİK MERGE      │
       │                              │   │          & SONRAKİ GÖREV     │
       │ • GitHub Actions FAIL eder   │   │ • PR main'e merge edilir     │
       │ • Auto-Merge kilitlenir      │   │ • Geçici dal silinir         │
       │ • Revizyon görevi Jules'a    │   │ • Sıradaki todo tetiklenir   │
       │   otomatik geri döner        │   └──────────────────────────────┘
       └──────────────┬───────────────┘
                      │ (Düzeltildi)
                      └─────────► (Tekrar 2. Kapıya Gider)
```

---

## 🚀 3. İnşa Edilecek 4 Kritik Sistem

### 1. Otomatik Düzeltme Tetikleyicisi (Self-Correction Revision Loop)
- **Amaç:** Denetçi `CHANGES REQUESTED` dediğinde, PR'ı kapatmadan aynı PR dalına Jules için revizyon talimatı iletilir.
- **İşleyiş:** Jules eksikleri (input validation, auth, error handling) tamamlayıp aynı dala commit atar, denetçi onaylayana kadar döngü devam eder.

### 2. Bağımlılık & Paket Koruyucusu (Dependency Sentinel)
- **Amaç:** Kodda `import` / `require` edilen yeni kütüphaneler (`express-validator`, `joi` vb.) `package.json` dosyasında yoksa otomatik tespit edilir ve PR'a eklenir.

### 3. Merkezi Magda-Agent Komuta & Güvenlik Paneli (Magda Dashboard)
- **Amaç:** Tüm repoların üstünde, Magda-Agent'ın kendi bağımsız kontrol merkezidir.
- **İçerik:**
  - Canlı Siber Saldırı & Tehdit İzleme (ACS 5-Checkpoint, Taint Tracking, SQL/XSS girişimleri).
  - Canlı Denetim Defteri (Audit Ledger - PR onay/ret geçmişi, güvenlik skorları).
  - Hedef Repoların AST Kod Grafı ve Sağlık Durumu.
  - Jules / Codex Otonom İşçi Kuyrukları.

### 4. Canlı Sunucu Geri Alma Kalkanı (Runtime Auto-Rollback)
- **Amaç:** Merge edilen kod canlı sunucuda 500 hatası veya bellek sızıntısı yaratırsa, Daemon otomatik olarak `git revert` yaparak sistemi kurtarır.

---

## 🖥️ 4. Magda-Agent Dashboard Mimari Yapısı

Magda Dashboard; hedef ürünlerin (Airbnb, Spacebot) içindeki arayüzlerden **tamamen bağımsız, ayrı bir Bilişsel Komuta Merkezi** olarak konumlanır:

1. **Bağımsız Web Arayüzü:** `coruhoorhan/magda-dashboard` (React, Tailwind CSS, Recharts, Lucide Icons).
2. **REST & WebSocket API:** `magda_agent/api.py` (Port 8000 / 5173).
3. **Merkezi Veritabanı:** `operations.sqlite3` / `data/guardian_audit.db`.
