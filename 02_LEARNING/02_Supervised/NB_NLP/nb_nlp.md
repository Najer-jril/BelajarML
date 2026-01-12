# 📘 Learning Brief — Naive Bayes & NLP

Dokumen ini menjelaskan **tujuan, prasyarat, dataset, alur belajar, dan output** untuk materi **Naive Bayes dan Natural Language Processing (NLP)** sesuai dengan RPS dan Sub-CPMK.

---

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan fase ini, mahasiswa mampu:

* Menjelaskan konsep probabilistik Naive Bayes
* Mengimplementasi dan mengevaluasi Naive Bayes pada data tabular
* Memahami dasar Natural Language Processing (NLP)
* Mengimplementasi klasifikasi teks menggunakan Naive Bayes
* Mengevaluasi performa model NLP secara kuantitatif dan kritis

**Sub-CPMK terkait:**

> Mahasiswa mampu mengimplementasi dan mengevaluasi model naive bayes dan teknik natural language processing. (C5, P4, A4)

---

## 📌 Prasyarat (SUDAH TERPENUHI)

✔ Feature engineering & preprocessing
✔ Train-test split
✔ Classification metrics (precision, recall, F1, ROC)
✔ Pipeline & GridSearch
✔ Threshold tuning

> ⚠️ Tidak perlu regresi, SVM, atau unsupervised untuk masuk ke NB & NLP

---

## 🧠 Konsep Inti yang Akan Dipelajari

### 1️⃣ Naive Bayes

* Bayes Theorem
* Conditional Probability
* Asumsi **conditional independence**
* Perbedaan:

  * Gaussian Naive Bayes
  * Multinomial Naive Bayes
  * Bernoulli Naive Bayes

---

### 2️⃣ NLP (Natural Language Processing)

* Representasi teks ke numerik
* Tokenization
* Stopwords removal
* Bag of Words
* TF-IDF
* Sparse vector & curse of dimensionality

---

## 🗂️ Alur Belajar (STEP-BY-STEP)

---

## 🔹 STEP 1 — Naive Bayes (Tabular Data)

### Tujuan

Mengenal Naive Bayes sebagai **baseline classifier probabilistik** dan membandingkannya dengan Logistic Regression & SVM.

### Dataset

* **Boleh pakai dataset yang sama (Adult Income)**
* Tujuannya **comparative study**, bukan performa maksimal

### Aktivitas

1. Import Gaussian Naive Bayes
2. Training model
3. Evaluasi:

   * Confusion matrix
   * Precision, Recall, F1
   * ROC-AUC
4. Bandingkan dengan:

   * Logistic Regression
   * SVM

### Fokus Analisis

* Kenapa NB bisa kalah/menang?
* Pengaruh asumsi independensi
* Overfitting vs underfitting

### Output

* Notebook: `01_naive_bayes_tabular.ipynb`
* Kesimpulan komparatif model

---

## 🔹 STEP 2 — Transisi ke NLP (Konsep & Preprocessing)

### Tujuan

Memahami kenapa Naive Bayes **sangat cocok** untuk teks.

### Materi

* Karakteristik data teks
* Perbedaan numerik vs teks
* Masalah dimensionalitas tinggi
* Kenapa model linear & NB efektif di NLP

### Aktivitas

* Eksplorasi dataset teks
* Analisis panjang dokumen
* Distribusi kata

### Output

* Notebook: `02_nlp_text_preprocessing.ipynb`

---

## 🔹 STEP 3 — Feature Extraction untuk Teks

### Tujuan

Mengubah teks → vektor numerik

### Teknik

1. **CountVectorizer**
2. **TF-IDF Vectorizer**

### Aktivitas

* Tokenization
* Stopwords
* N-grams (opsional)
* Bandingkan BoW vs TF-IDF

### Output

* Visualisasi sparse matrix
* Notebook: `03_text_vectorization.ipynb`

---

## 🔹 STEP 4 — Naive Bayes untuk NLP

### Tujuan

Membangun **text classifier berbasis probabilistik**

### Model

* Multinomial Naive Bayes

### Aktivitas

1. Pipeline:

   * Vectorizer → Classifier
2. Training
3. Evaluasi:

   * Confusion matrix
   * Precision, Recall, F1
   * Precision-Recall Curve

### Dataset Contoh

* SMS Spam Detection
* Movie Review Sentiment
* Tweet Sentiment

### Output

* Notebook: `04_naive_bayes_nlp.ipynb`

---

## 🔹 STEP 5 — Evaluasi Kritis & Insight

### Fokus Analisis

* Precision vs Recall pada NLP
* False Positive vs False Negative
* Kapan NB lebih baik dari Logistic Regression?
* Kapan NB tidak cocok?

### Output

* Kesimpulan tertulis (Markdown / Notebook)

---

## 📁 Struktur Direktori contoh

```
06_naive_bayes_nlp/
│
├── naive_bayes_tabular.ipynb
├── nlp_text_preprocessing.ipynb
├── text_vectorization.ipynb
├── naive_bayes_nlp.ipynb
└── README.md
```

---

## 🚦 Kriteria “SUDAH PAHAM”

Kamu **boleh lanjut ke Unsupervised / DL** kalau:

* Bisa menjelaskan NB tanpa lihat rumus
* Bisa jelaskan kenapa NB cocok untuk teks
* Bisa membaca confusion matrix NLP
* Bisa menjelaskan trade-off precision vs recall

Kalau belum → **belum DL**

---

## 🔜 Setelah NB & NLP

Next logical step:

1. Unsupervised Learning (KMeans & PCA)
2. Baru masuk:

   * Neural Network
   * Backpropagation
   * Deep Learning

---

## 🧠 Catatan

> NB + NLP = **fondasi DL untuk teks**
> Kalau ini lemah → DL NLP bakal kerasa “magic tanpa logika”.

---