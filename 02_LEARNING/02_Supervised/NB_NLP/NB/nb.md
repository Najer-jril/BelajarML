# Learning Brief — Naive Bayes

Dokumen ini menjelaskan **tujuan, prasyarat, dataset, alur belajar, dan output** untuk materi **Naive Bayes** pada data tabular sesuai dengan RPS dan Sub-CPMK.

---

## Tujuan Pembelajaran

Setelah menyelesaikan fase ini, mahasiswa mampu:

* Menjelaskan konsep probabilistik Naive Bayes dan Teorema Bayes
* Memahami asumsi conditional independence pada Naive Bayes
* Mengimplementasi Gaussian Naive Bayes pada data tabular
* Melakukan preprocessing dan feature engineering untuk Naive Bayes
* Mengevaluasi performa model menggunakan confusion matrix, precision, recall, F1-score, dan ROC-AUC
* Membandingkan performa Naive Bayes dengan algoritma supervised learning lainnya
* Memvisualisasikan hasil evaluasi model

**Sub-CPMK terkait:**

> Mahasiswa mampu mengimplementasi dan mengevaluasi model naive bayes. (C5, P4, A4)

---

## Prasyarat (SUDAH TERPENUHI)

✔ Feature engineering & preprocessing
✔ Train-test split
✔ Classification metrics (precision, recall, F1, ROC)
✔ Pipeline & GridSearch
✔ Standardization dan encoding

---

## Konsep Inti yang Dipelajari

### 1. Naive Bayes

* **Bayes Theorem** - Probabilitas posterior berdasarkan prior dan likelihood
* **Conditional Probability** - P(A|B) = P(B|A) * P(A) / P(B)
* **Asumsi Conditional Independence** - Setiap fitur dianggap independen terhadap fitur lainnya
* **Gaussian Naive Bayes** - Untuk data kontinu dengan asumsi distribusi normal
* **Prior Probability** - Probabilitas awal sebelum melihat data
* **Likelihood** - Probabilitas data given class
* **Posterior Probability** - Probabilitas class given data

### 2. Karakteristik Naive Bayes

* Model probabilistik yang cepat dan efisien
* Cocok untuk dataset dengan banyak fitur
* Bekerja baik meski dengan data training yang sedikit
* Asumsi independensi yang kuat (naive)
* Tidak sensitif terhadap irrelevant features
* Hasil probabilitas dapat digunakan untuk confidence scoring

---

## Alur Belajar (STEP-BY-STEP)

---

## STEP 1 — Naive Bayes Learning (Adult Income Dataset)

### Tujuan

Memahami konsep dasar Naive Bayes dan implementasinya pada dataset klasifikasi biner dengan preprocessing yang kompleks.

### Dataset

* **Adult Income Dataset** (UCI ML Repository)
* Target: income (>50K atau <=50K)
* Fitur campuran: numerik dan kategorikal
* Memerlukan feature engineering

### Aktivitas

1. **Data Loading & Exploration**
   * Load dataset Adult Income
   * Eksplorasi struktur data
   * Identifikasi missing values
   * Analisis distribusi fitur

2. **Feature Engineering**
   * Grouping native-country menjadi US vs Non-US
   * Grouping workclass menjadi kategori yang lebih sederhana
   * Grouping occupation menjadi kategori makro
   * Handling missing values
   * Encoding kategorikal

3. **Exploratory Data Analysis**
   * Visualisasi distribusi target
   * Analisis korelasi fitur numerik
   * Distribusi fitur kategorikal
   * Identifikasi pola data

4. **Data Preprocessing**
   * Split train-test
   * Standardization untuk fitur numerik
   * One-hot encoding untuk fitur kategorikal
   * Pipeline preprocessing

5. **Model Training**
   * Implementasi Gaussian Naive Bayes
   * Training model
   * Prediksi pada test set

6. **Model Evaluation**
   * Confusion Matrix
   * Classification Report (Precision, Recall, F1)
   * ROC Curve dan AUC Score
   * Precision-Recall Curve
   * Feature importance analysis

### Output

* Notebook: `NB_Learn.ipynb`
* Pemahaman preprocessing untuk Naive Bayes
* Insight tentang performa model pada data real-world

---

## STEP 2 — Naive Bayes pada Credit Card Default

### Tujuan

Mengaplikasikan Naive Bayes pada dataset imbalanced dan memahami penanganan fitur numerik kontinyu.

### Dataset

* **Default of Credit Card Clients** (UCI ML Repository)
* Target: default payment next month (binary)
* Fitur: payment history, bill amounts, payment amounts
* Karakteristik: imbalanced dataset

### Aktivitas

1. **Data Loading & Cleaning**
   * Load dataset credit card
   * Drop unnecessary columns (ID)
   * Handle invalid categorical values
   * Data type validation

2. **Feature Engineering**
   * Replace invalid MARRIAGE values dengan 'Unknown'
   * Replace invalid EDUCATION values dengan 'Unknown'
   * Validasi kategori fitur

3. **Exploratory Data Analysis**
   * Distribusi target (class imbalance)
   * Visualisasi payment history patterns
   * Korelasi antar fitur
   * Distribusi bill amounts dan payment amounts

4. **Data Preprocessing**
   * Train-test split dengan stratification
   * Standardization fitur numerik
   * Handling class imbalance

5. **Model Training & Evaluation**
   * Gaussian Naive Bayes implementation
   * Confusion Matrix analysis
   * Precision-Recall trade-off pada imbalanced data
   * ROC-AUC evaluation
   * Threshold tuning untuk optimasi

### Fokus Analisis

* Pengaruh class imbalance pada Naive Bayes
* Precision vs Recall trade-off
* Interpretasi probabilitas prediksi
* Feature contribution analysis

### Output

* Notebook: `CC_NB.ipynb`
* Strategi handling imbalanced data dengan NB
* Pemahaman threshold tuning

---

## STEP 3 — Naive Bayes pada Diabetes Prediction

### Tujuan

Implementasi Naive Bayes pada health indicators dataset dengan fokus pada interpretasi medis.

### Dataset

* **Diabetes Health Indicators** (BRFSS 2015)
* Target: Diabetes_binary (0 atau 1)
* Fitur: health indicators (binary, ordinal, continuous)
* Dataset balanced (50-50 split)

### Aktivitas

1. **Data Understanding**
   * Load balanced diabetes dataset
   * Identifikasi tipe fitur:
     - Binary features (HighBP, Smoker, dll)
     - Ordinal features (Age, Income, Education)
     - Continuous features (BMI, GenHlth, dll)

2. **Exploratory Data Analysis**
   * Distribusi diabetes (balanced)
   * Korelasi health indicators dengan diabetes
   * Visualisasi distribusi fitur
   * Identifikasi risk factors

3. **Feature Analysis**
   * Analisis importance dari health indicators
   * Korelasi antar health features
   * Distribusi BMI, blood pressure, cholesterol

4. **Model Development**
   * Preprocessing pipeline
   * Gaussian Naive Bayes training
   * Probability calibration

5. **Model Evaluation**
   * Confusion Matrix interpretation
   * Clinical metrics (Sensitivity, Specificity)
   * ROC-AUC analysis
   * Precision-Recall curves
   * False Positive vs False Negative analysis

### Fokus Analisis

* Interpretasi model dalam konteks medis
* Trade-off antara False Negative (miss diabetes) vs False Positive (false alarm)
* Feature importance untuk diabetes prediction
* Confidence scoring untuk medical decision support

### Output

* Notebook: `Diabetes__NB.ipynb`
* Medical interpretation dari model
* Insight tentang diabetes risk factors

---

## Evaluasi Kritis & Insight

### Kelebihan Naive Bayes

* **Cepat dan efisien** - Training dan prediksi sangat cepat
* **Bekerja baik dengan data terbatas** - Tidak memerlukan banyak training data
* **Probabilistic output** - Memberikan confidence score
* **Handling multi-class** - Mudah diperluas untuk klasifikasi multi-class
* **Robust terhadap irrelevant features** - Asumsi independensi membantu

### Keterbatasan Naive Bayes

* **Asumsi independensi yang kuat** - Jarang terpenuhi di dunia nyata
* **Zero frequency problem** - Jika kategori tidak muncul di training
* **Estimasi probabilitas bisa bias** - Jika asumsi dilanggar
* **Tidak menangkap interaksi fitur** - Feature interactions diabaikan

### Kapan Menggunakan Naive Bayes

✔ Dataset dengan banyak fitur independen
✔ Baseline model yang cepat
✔ Real-time prediction dengan computational constraint
✔ Text classification (NLP)
✔ Spam detection
✔ Medical diagnosis dengan independent symptoms

### Kapan TIDAK Menggunakan Naive Bayes

✖ Fitur sangat berkorelasi satu sama lain
✖ Butuh capture complex feature interactions
✖ Data sangat non-linear
✖ Perlu model interpretability yang detail

---

## Struktur Direktori

```
02_LEARNING/02_Supervised/NB_NLP/NB/
│
├── NB_Learn.ipynb          # Adult Income - comprehensive NB learning
├── CC_NB.ipynb             # Credit Card Default - imbalanced data
├── Diabetes__NB.ipynb      # Diabetes Prediction - medical context
└── nb.md                   # Dokumentasi pembelajaran (file ini)
```

---

## Kriteria "SUDAH PAHAM"

Kamu boleh lanjut ke **NLP** kalau:

* Bisa menjelaskan Bayes Theorem tanpa melihat rumus
* Paham kenapa disebut "Naive" dan implikasinya
* Bisa membaca dan menginterpretasi confusion matrix
* Paham trade-off Precision vs Recall dalam konteks aplikasi
* Bisa menjelaskan kapan NB cocok dan tidak cocok digunakan
* Paham perbedaan Gaussian, Multinomial, dan Bernoulli NB
* Bisa mengimplementasi preprocessing untuk Naive Bayes

Kalau belum → **ulangi dan pelajari lebih dalam**

---

## Setelah Naive Bayes Tabular

Next logical step:

1. **Natural Language Processing (NLP)** - Naive Bayes untuk teks
2. Multinomial dan Bernoulli Naive Bayes
3. Text vectorization (BoW, TF-IDF)
4. Sentiment analysis

---

## Catatan

> Naive Bayes = **baseline probabilistic classifier**
> Paham ini dengan baik → fondasi untuk probabilistic models & NLP
> Asumsi "naive" = trade-off antara simplicity & accuracy

---