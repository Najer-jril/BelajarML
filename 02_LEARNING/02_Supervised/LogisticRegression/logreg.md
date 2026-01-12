# 📘 Learning Brief — Logistic Regression

Dokumen ini menjelaskan **tujuan, prasyarat, dataset, alur belajar, dan output** untuk materi **Logistic Regression** sesuai dengan RPS dan Sub-CPMK.

---

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan fase ini, mahasiswa mampu:

* Memahami perbedaan regresi dan klasifikasi
* Mengimplementasi Logistic Regression untuk binary classification
* Melakukan preprocessing data (imputation, scaling, encoding)
* Mengevaluasi model klasifikasi dengan confusion matrix dan classification metrics
* Menginterpretasi precision, recall, F1-score, dan accuracy
* Memahami trade-off antara precision dan recall
* Menganalisis class imbalance

**Sub-CPMK terkait:**

> Mahasiswa mampu mengimplementasi dan mengevaluasi model klasifikasi logistic regression untuk prediksi kategori binary. (C3, P3, A3)

---

## 📌 Prasyarat (SUDAH TERPENUHI)

✔ Linear Regression (konsep regresi dasar)
✔ Python basics (pandas, numpy, matplotlib, seaborn)
✔ Exploratory Data Analysis (EDA)
✔ Data preprocessing (missing values, encoding, scaling)
✔ Train-test split
✔ Statistical distributions

---

## 🧠 Konsep Inti yang Akan Dipelajari

### 1️⃣ Logistic Regression

* Sigmoid Function: σ(z) = 1 / (1 + e⁻ᶻ)
* Log-Odds & Probability
* Decision Boundary
* Perbedaan dengan Linear Regression:
  * Linear → continuous output
  * Logistic → probability (0-1) → class

---

### 2️⃣ Classification Metrics

* **Confusion Matrix**
  * True Positive (TP)
  * True Negative (TN)
  * False Positive (FP) - Type I Error
  * False Negative (FN) - Type II Error

* **Metrics**
  * **Accuracy** = (TP + TN) / Total
  * **Precision** = TP / (TP + FP) - "dari yang diprediksi positif, berapa yang benar?"
  * **Recall** = TP / (TP + FN) - "dari yang benar positif, berapa yang berhasil diprediksi?"
  * **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) - harmonic mean

---

### 3️⃣ Data Preprocessing untuk Classification

* **Missing Values Handling**
  * SimpleImputer (mean, median, mode)
  
* **Scaling**
  * StandardScaler - untuk features dengan skala berbeda
  * Penting untuk Logistic Regression karena sensitif terhadap skala

* **Class Imbalance**
  * Deteksi imbalance
  * Interpretasi metrik dengan class imbalance

---

## 🗂️ Alur Belajar (STEP-BY-STEP)

---

## 🔹 STEP 1 — Data Loading & Exploration

### Tujuan

Memahami karakteristik data untuk klasifikasi.

### Dataset

* **Heart Disease Prediction** (`framingham.csv`)
* Target: `TenYearCHD` (0 = no disease, 1 = has disease)
* Features: age, education, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose, dll

### Aktivitas

1. **Load Data**
   * Import dataset dari Kaggle
   * Explore struktur data

2. **Check Missing Values**
   * Identify columns dengan missing values
   * Decide strategy: imputation

3. **Class Distribution**
   * Countplot untuk target variable
   * Check class imbalance (berapa ratio 0:1?)

### Output

* Understanding dataset structure
* Missing values report
* Class distribution visualization

---

## 🔹 STEP 2 — Exploratory Data Analysis (EDA)

### Tujuan

Memahami distribusi features dan relationship dengan target.

### Aktivitas

1. **Univariate Analysis**
   * Histogram + KDE untuk continuous features
   * Boxplot untuk detect outliers

2. **Correlation Analysis**
   * Heatmap korelasi antar continuous features
   * Korelasi features dengan target

### Fokus Analisis

* Feature mana yang paling berkorelasi dengan target?
* Apakah ada outliers ekstrim?
* Apakah ada multicollinearity?

### Output

* Distribution plots
* Correlation heatmap
* Insight untuk feature selection

---

## 🔹 STEP 3 — Data Preprocessing

### Tujuan

Menyiapkan data agar siap untuk model training.

### Aktivitas

1. **Missing Values Imputation**
   * SimpleImputer dengan strategy='median'
   * Transform dan replace missing values

2. **Train-Test Split**
   * Split 80:20
   * Stratify jika class imbalance

3. **Feature Scaling**
   * StandardScaler untuk continuous features
   * Fit pada training set, transform pada test set
   * **Penting:** Jangan fit scaler pada test set!

### Output

* Clean dataset tanpa missing values
* Scaled features
* Training & test sets siap pakai

---

## 🔹 STEP 4 — Model Training & Evaluation

### Tujuan

Training Logistic Regression dan evaluasi performa.

### Aktivitas

1. **Model Training**
   * Import LogisticRegression
   * Fit model pada X_train, y_train

2. **Prediction**
   * Predict pada test set
   * Get probability predictions (opsional)

3. **Evaluasi Metrics**
   * Accuracy score
   * Precision score
   * Recall score
   * F1-score

4. **Confusion Matrix**
   * Heatmap visualization
   * Interpretasi TP, TN, FP, FN

5. **Classification Report**
   * Per-class metrics
   * Support (jumlah data per class)

### Fokus Analisis

* Apakah accuracy tinggi sudah cukup? (ingat class imbalance!)
* Precision vs Recall: mana yang lebih penting untuk kasus ini?
  * Heart disease → **Recall** lebih penting (jangan sampai miss positive case)
* Berapa banyak False Negative? (bahaya!)

### Output

* Notebook: `Logistics_Regression_learning.ipynb`
* Model evaluation metrics
* Confusion matrix
* Insight tentang performa model

---

## 🔹 STEP 5 — Exercise: Wine Classification

### Tujuan

Mengaplikasikan Logistic Regression pada dataset baru secara mandiri.

### Dataset

* **Wine Quality Dataset**
* Binary classification task

### Aktivitas

1. Complete end-to-end pipeline:
   * EDA
   * Missing values handling
   * Scaling
   * Model training
   * Evaluation
   * Interpretation

2. Eksperimen:
   * Coba threshold tuning untuk trade-off precision-recall
   * Feature selection
   * Cross-validation

### Output

* Notebook: `Wine_LogReg_exercise.ipynb`
* Full analysis dan kesimpulan

---

## 📁 Struktur Direktori

```
LogisticRegression/
│
├── Logistics_Regression_learning.ipynb
├── Wine_LogReg_exercise.ipynb
└── logreg.md
```

---

## 🚦 Kriteria "SUDAH PAHAM"

Kamu **boleh lanjut ke SVM / Advanced Classification** kalau:

* Bisa menjelaskan perbedaan precision, recall, dan F1
* Bisa membaca dan interpret confusion matrix
* Paham kenapa accuracy bisa misleading (class imbalance)
* Bisa menjelaskan kapan prioritas precision vs recall
* Bisa explain kenapa scaling penting untuk Logistic Regression
* Paham sigmoid function dan decision boundary (konseptual)

Kalau belum → **ulangi dengan dataset imbalanced yang berbeda**

---

## 🔜 Setelah Logistic Regression

Next logical step:

1. **Support Vector Machines (SVM)** - classification dengan kernel tricks
2. **Decision Trees** - non-linear classification
3. **Naive Bayes & NLP** - probabilistic classification
4. **Ensemble Methods** (Random Forest, XGBoost)

---

## 🧠 Catatan

> Logistic Regression = **fondasi classification & base model untuk benchmark**
> Kalau precision-recall trade-off tidak paham → evaluation model lain akan salah interpretasi.

> **Class Imbalance** = masalah real-world yang paling sering diabaikan!

---
