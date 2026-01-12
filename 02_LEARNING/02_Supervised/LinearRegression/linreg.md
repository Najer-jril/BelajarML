# 📘 Learning Brief — Linear Regression

Dokumen ini menjelaskan **tujuan, prasyarat, dataset, alur belajar, dan output** untuk materi **Linear Regression** sesuai dengan RPS dan Sub-CPMK.

---

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan fase ini, mahasiswa mampu:

* Memahami konsep dasar regresi linear dan fungsi cost
* Mengimplementasi Linear Regression menggunakan scikit-learn
* Melakukan preprocessing data (encoding, scaling, train-test split)
* Mengevaluasi model regresi dengan metrik MAE, RMSE, dan R²
* Memahami regularisasi (Ridge Regression)
* Menginterpretasi koefisien model dan visualisasi hasil prediksi

**Sub-CPMK terkait:**

> Mahasiswa mampu mengimplementasi dan mengevaluasi model regresi linear untuk prediksi nilai kontinu. (C3, P3, A3)

---

## 📌 Prasyarat (SUDAH TERPENUHI)

✔ Python basics (pandas, numpy, matplotlib)
✔ Exploratory Data Analysis (EDA)
✔ Data preprocessing (missing values, encoding)
✔ Train-test split
✔ Basic statistics

---

## 🧠 Konsep Inti yang Akan Dipelajari

### 1️⃣ Linear Regression

* Fungsi linear: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
* Least Squares Method
* Gradient Descent (konseptual)
* Asumsi Linear Regression:
  * Linearitas
  * Independence
  * Homoscedasticity
  * Normalitas residual

---

### 2️⃣ Evaluation Metrics

* **MAE** (Mean Absolute Error) - rata-rata error absolut
* **RMSE** (Root Mean Squared Error) - penalti untuk error besar
* **R²** (R-squared) - proporsi variance yang dijelaskan model
* Actual vs Predicted Plot

---

### 3️⃣ Regularization

* **Ridge Regression** (L2 Regularization)
* Parameter alpha (λ)
* Trade-off antara bias dan variance
* Kapan menggunakan regularisasi

---

## 🗂️ Alur Belajar (STEP-BY-STEP)

---

## 🔹 STEP 1 — Linear Regression Dasar

### Tujuan

Memahami dan mengimplementasi Linear Regression pada data tabular.

### Dataset

* **Insurance Dataset** (`insurance_kaggle.csv`)
* Target: `charges` (biaya asuransi)
* Features: age, sex, bmi, children, smoker, region

### Aktivitas

1. **Load & Explore Data**
   * Import dataset
   * Check missing values
   * Descriptive statistics

2. **Preprocessing**
   * Log transformation pada target (untuk stabilisasi variance)
   * OneHotEncoder untuk categorical features
   * ColumnTransformer untuk pipeline preprocessing

3. **Model Training**
   * Train-test split (80:20)
   * Pipeline: Preprocessor → LinearRegression
   * Fit model pada training data

4. **Evaluasi**
   * Prediksi pada test set
   * Hitung MAE, RMSE, R²
   * Analisis koefisien model
   * Visualisasi: Actual vs Predicted

### Fokus Analisis

* Feature mana yang paling berpengaruh?
* Apakah R² sudah cukup baik?
* Apakah ada pola pada residual plot?

### Output

* Notebook: `Learn_Linear_Regression.ipynb`
* Metrik evaluasi model
* Interpretasi koefisien

---

## 🔹 STEP 2 — Ridge Regression (Regularization)

### Tujuan

Memahami konsep regularisasi untuk mencegah overfitting.

### Aktivitas

1. **Implementasi Ridge**
   * Import Ridge dari sklearn
   * Set parameter alpha (e.g., 10.0)
   * Training dengan pipeline yang sama

2. **Evaluasi & Perbandingan**
   * Bandingkan MAE, RMSE, R² dengan Linear Regression biasa
   * Analisis perbedaan koefisien
   * Visualisasi Actual vs Predicted

3. **Hyperparameter Tuning** (opsional)
   * GridSearchCV untuk cari alpha terbaik
   * Cross-validation

### Fokus Analisis

* Apakah Ridge lebih baik dari Linear Regression biasa?
* Bagaimana regularisasi mempengaruhi koefisien?
* Kapan sebaiknya menggunakan Ridge?

### Output

* Perbandingan Linear vs Ridge
* Insight tentang regularisasi

---

## 🔹 STEP 3 — Exercise: Student Performance Prediction

### Tujuan

Mengaplikasikan Linear Regression pada dataset baru secara mandiri.

### Dataset

* **Student Performance Dataset**
* Target: student performance score
* Features: study hours, previous scores, activities, dll

### Aktivitas

1. Complete end-to-end pipeline:
   * EDA
   * Preprocessing
   * Model training
   * Evaluation
   * Interpretation

2. Eksperimen:
   * Coba feature engineering
   * Bandingkan Linear vs Ridge
   * Tuning hyperparameters

### Output

* Notebook: `student_Performance_LinReg_exercise.ipynb`
* Analisis dan kesimpulan

---

## 📁 Struktur Direktori

```
LinearRegression/
│
├── Learn_Linear_Regression.ipynb
├── student_Performance_LinReg_exercise.ipynb
└── linreg.md
```

---

## 🚦 Kriteria "SUDAH PAHAM"

Kamu **boleh lanjut ke Logistic Regression / Classification** kalau:

* Bisa menjelaskan perbedaan MAE, RMSE, dan R²
* Bisa menginterpretasi koefisien model
* Paham kapan pakai Linear vs Ridge
* Bisa membaca Actual vs Predicted plot
* Bisa menjelaskan kenapa log transformation diperlukan

Kalau belum → **ulangi latihan dengan dataset lain**

---

## 🔜 Setelah Linear Regression

Next logical step:

1. **Logistic Regression** (untuk klasifikasi)
2. **Support Vector Machines** (SVM)
3. **Decision Trees & Random Forest**

---

## 🧠 Catatan

> Linear Regression = **fondasi semua model supervised learning**
> Kalau ini lemah → model lain akan susah dipahami.

---
