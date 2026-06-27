# Btk Datathon 2026 — Career Success Score Tahmini Yaklaşımımız

Öğrencilerin kariyer başarı skorunu (`career_success_score`, 0–100) tahmin eden makine öğrenimi pipeline'ı.
Yarışma değerlendirme metriği MSE'dir (düşük = iyi).

## Yaklaşım

### NLP — Triple Ridge Meta-Özellikleri

Mentor geri bildirim metni (`mentor_feedback_text`) üç ayrı TF-IDF vektörleştirici ile işlenir:

- **Word TF-IDF** (unigram–trigram, 5.000 özellik, α=5.0) — kelime düzeyinde anlam
- **Char TF-IDF** (3-gram–5-gram, 10.000 özellik, α=3.0) — Türkçe morfoloji ve yazım varyantları
- **Combined** (Word + Char hstack, α=4.0) — genel metin sinyali

Her Ridge modeli 5-fold OOF ile eğitilir; tahminler ağaç modellerine meta-özellik olarak beslenir.

### BERTurk Fine-Tuning (GPU)

`dbmdz/bert-base-turkish-cased` modeli `career_success_score` tahminine yönelik fine-tune edilir.
Hedef değişken fold bazında z-score normalleştirilir, 3 epoch eğitim yapılır.
OOF tahminleri ağaç modellerine ek meta-özellik olarak aktarılır.

GPU yoksa bu aşama atlanır; notebook hata vermeden çalışmaya devam eder.

### Tabular Özellik Mühendisliği

- Eksik veri gösterge bayrakları (7 sütun)
- Teknik beceri istatistikleri: ortalama, max, min, std, en iyi 3 ortalaması
- Bileşik indeksler: `combined_master`, `weighted_exp`, `portfolio_strength`, `github_strength`, `academic_str`, `career_readiness`
- Etkileşim terimleri: yüksek korelasyonlu özellik çiftleri çarpımı
- OOF Bayesian (Smooth) Target Encoding — kategorik sütunlar için data leakage korumalı

### Ensemble

LightGBM, XGBoost ve CatBoost ağırlıklı ortalama ile birleştirilir.

| Mod | LightGBM | XGBoost | CatBoost |
|-----|----------|---------|----------|
| GPU (BERTurk aktif) | 0.20 | 0.05 | 0.75 |
| CPU Fallback | 0.30 | 0.07 | 0.63 |

## Çalıştırma

Notebook Kaggle GPU ortamında çalıştırılması önerilir. BERTurk fine-tuning aşaması CUDA gerektirmektedir.

1. Kaggle'da yeni bir notebook açın.
2. `datathon-v11-clean.ipynb` dosyasını import edin.
3. Yarışma veri setini input olarak ekleyin.
4. GPU açık olarak "Run All" yapın.

Yerel makinede çalıştırmak isterseniz veri yollarını güncelleyin ve GPU olmadığı için BERTurk aşaması otomatik olarak atlanır.

## Ekip 
- Mehmet Doğukan Özer
- Sema Nur Çetin
- Sena Demirbaş
