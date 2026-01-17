# Learning Brief — K-Means Clustering & PCA

Dokumen ini menjelaskan **tujuan, prasyarat, dataset, alur belajar, dan output** untuk materi **K-Means Clustering** dan **Principal Component Analysis (PCA)** pada unsupervised learning.

---

## Tujuan Pembelajaran

Setelah menyelesaikan fase ini, mahasiswa mampu:

* Memahami konsep clustering sebagai unsupervised learning
* Mengimplementasi algoritma K-Means Clustering
* Menentukan jumlah cluster optimal dengan Elbow Method dan Silhouette Score
* Memahami konsep dan implementasi PCA untuk dimensionality reduction
* Melakukan preprocessing data untuk clustering (scaling, transformation)
* Memvisualisasikan hasil clustering dalam 2D dan 3D
* Menginterpretasi hasil cluster untuk business insight
* Memahami kapan menggunakan PCA dan kapan tidak

**Sub-CPMK terkait:**

> Mahasiswa mampu mengimplementasi dan mengevaluasi model clustering untuk segmentasi data. (C5, P4, A4)

---

## Prasyarat (SUDAH TERPENUHI)

- Statistik deskriptif dan distribusi data
- Data preprocessing (cleaning, handling missing values)
- Feature engineering
- Standardization dan scaling
- Visualisasi data (matplotlib, seaborn, plotly)

---

## Konsep Inti yang Dipelajari

### 1. K-Means Clustering

* **Centroid-based Clustering** - Membagi data ke dalam K cluster berdasarkan jarak ke centroid
* **Inertia** - Sum of squared distances dari setiap point ke centroid terdekat
* **K-Means++ Initialization** - Metode inisialisasi centroid yang lebih baik
* **Convergence** - Iterasi berhenti ketika centroid tidak berubah atau mencapai max_iter
* **Algorithm Types** - Lloyd's algorithm vs Elkan's algorithm

### 2. Menentukan Nilai K Optimal

* **Elbow Method** - Plot inertia vs K, cari "siku" di mana penurunan mulai landai
* **Silhouette Score** - Mengukur seberapa mirip objek dengan clusternya dibanding cluster lain
* **Silhouette Range** - Score antara -1 sampai 1, semakin tinggi semakin baik
* **Trade-off** - K terlalu kecil = underfitting, K terlalu besar = overfitting

### 3. Principal Component Analysis (PCA)

* **Dimensionality Reduction** - Mengurangi jumlah fitur sambil mempertahankan variance
* **Principal Components** - Linear combination dari fitur asli
* **Explained Variance Ratio** - Persentase informasi yang dipertahankan per komponen
* **Projection** - Memproyeksikan data ke ruang berdimensi lebih rendah
* **Orthogonality** - Komponen utama saling tegak lurus (tidak berkorelasi)

### 4. Preprocessing untuk Clustering

* **StandardScaler** - Mean = 0, Std = 1 (sensitif terhadap outlier)
* **RobustScaler** - Menggunakan median dan IQR (robust terhadap outlier)
* **Log Transformation** - Mengurangi skewness pada data
* **Feature Selection** - Memilih fitur yang relevan untuk clustering

---

## Alur Belajar (STEP-BY-STEP)

---

## STEP 1 — K-Means Learning (Mall Customers Dataset)

### Tujuan

Memahami konsep dasar K-Means Clustering dengan dataset sederhana dan eksplorasi berbagai kombinasi fitur 2D dan 3D.

### Dataset

* **Mall Customer Segmentation Data**
* Source: Kaggle (vjchoudhary7/customer-segmentation-tutorial-in-python)
* File: Mall_Customers.csv
* Total: 200 customers
* Fitur asli: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

### Data Information

* Columns: 5 kolom
* CustomerID: identifier (tidak digunakan)
* Gender: Male/Female (tidak digunakan untuk clustering numerik)
* Age: 18-70 tahun
* Annual Income: 15-137 k$
* Spending Score: 1-100 (skor belanja dari mall)
* Missing Values: Tidak ada
* Data Type: Numerik (untuk Age, Income, Spending Score)

### Feature Engineering

* **Rename Columns**
  - Annual Income (k$) → Annual Income
  - Spending Score (1-100) → Spending Score
  - Mempermudah akses kolom

* **Feature Selection untuk Clustering**
  - Tidak ada transformasi kompleks
  - Menggunakan fitur asli tanpa scaling
  - Eksplorasi berbagai kombinasi fitur

### Model

* **Algorithm**: KMeans dari sklearn.cluster
* **Parameters**:
  - init='k-means++'
  - n_init=10
  - max_iter=300
  - tol=0.0001
  - random_state=42
  - algorithm='elkan'

### Pipeline

1. Load data dari Kaggle
2. Rename columns
3. EDA (distribusi, pairplot)
4. Pilih kombinasi fitur untuk clustering
5. Elbow Method untuk cari K
6. Silhouette Analysis untuk validasi K
7. Training K-Means dengan K terpilih
8. Visualisasi decision boundaries dan centroids
9. Analisis cluster

### Eksperimen Clustering

#### 2D Clustering: Age vs Spending Score

* **K yang diuji**: 4 dan 5
* **Elbow Method**: Siku terlihat di K=4-5
* **Silhouette Score**:
  - K=4: ~0.4
  - K=5: ~0.38
* **K Terpilih**: 4 (lebih sederhana, skor tidak jauh berbeda)
* **Insight**: Segmentasi berdasarkan usia dan pola belanja

#### 2D Clustering: Age vs Annual Income

* **K yang diuji**: 4 dan 17
* **Elbow Method**: Tidak ada siku yang jelas
* **Silhouette Score**:
  - K=4: ~0.35
  - K=17: lebih rendah
* **K Terpilih**: 4 (trade-off simplicity vs granularity)
* **Insight**: Data tidak memiliki cluster natural yang jelas

#### 2D Clustering: Spending Score vs Annual Income

* **K yang diuji**: 5 dan 6
* **Elbow Method**: Siku jelas di K=5
* **Silhouette Score**:
  - K=5: ~0.55 (tertinggi)
  - K=6: ~0.45
* **K Terpilih**: 5 (cluster natural paling jelas)
* **Insight**: Kombinasi fitur terbaik untuk segmentasi pelanggan
* **Cluster Pattern**: 5 segmen customer yang distinct

#### 3D Clustering: Age, Annual Income, Spending Score

* **K yang diuji**: 5 dan 6
* **Elbow Method**: Siku di K=5-6
* **K Terpilih**: 5-6
* **Visualisasi**: 3D scatter plot dengan Plotly
* **Insight**: Menambah dimensi memberikan perspektif tambahan

### Hasil Clustering (Spending Score vs Annual Income, K=5)

* **Cluster 0**: High Income, High Spending (Premium)
* **Cluster 1**: High Income, Low Spending (Careful)
* **Cluster 2**: Low Income, High Spending (Spenders)
* **Cluster 3**: Low Income, Low Spending (Sensible)
* **Cluster 4**: Medium Income, Medium Spending (Standard)

### Mengapa TIDAK Menggunakan PCA

* **Dimensi Rendah**: Hanya 3 fitur numerik
* **Interpretabilitas**: Fitur asli mudah diinterpretasi
* **Visualisasi Langsung**: Bisa plot 2D dan 3D tanpa reduksi
* **Tidak Ada Multikolinearitas Tinggi**: Fitur cukup independen
* **Tujuan Pembelajaran**: Fokus pada K-Means dasar

### Output

* Notebook: `K_means_learn.ipynb`
* Pemahaman dasar K-Means
* Elbow Method dan Silhouette Analysis
* Visualisasi decision boundaries
* Tidak menggunakan PCA

---

## STEP 2 — Customer Personality Analysis (dengan PCA)

### Tujuan

Mengaplikasikan K-Means pada dataset dengan banyak fitur dan menggunakan PCA untuk dimensionality reduction dan visualisasi.

### Dataset

* **Customer Personality Analysis**
* Source: Kaggle (imakash3011/customer-personality-analysis)
* File: marketing_campaign.csv (tab-separated)
* Total: 2240 customers (setelah cleaning: 2216)
* Fitur asli: 29 kolom

### Data Information

* **Demographic Features**:
  - Year_Birth → diubah jadi Age
  - Education: Graduation, PhD, Master, Basic, 2n Cycle
  - Marital_Status: Single, Together, Married, Divorced, Widow, Alone, Absurd, YOLO
  - Income: pendapatan tahunan (ada 24 missing values)
  - Kidhome, Teenhome: jumlah anak

* **Product Spending Features**:
  - MntWines, MntFruits, MntMeatProducts
  - MntFishProducts, MntSweetProducts, MntGoldProds

* **Campaign Features** (di-drop):
  - AcceptedCmp1-5, Response

* **Purchase Channel Features**:
  - NumWebPurchases, NumCatalogPurchases, NumStorePurchases
  - NumDealsPurchases, NumWebVisitsMonth

* **Other Features**:
  - Dt_Customer (tanggal jadi customer)
  - Recency (hari sejak pembelian terakhir)
  - Complain (binary)
  - Z_CostContact, Z_Revenue (constant, di-drop)

### Feature Engineering

* **Drop Unnecessary Columns**
  - ID, Dt_Customer
  - AcceptedCmp1-5, Response (campaign results)
  - Z_CostContact, Z_Revenue (constant values)

* **Marital Status Mapping**
  - Single, Divorced, Widow, Alone, Absurd, YOLO → 'Alone'
  - Together, Married → 'Partner'

* **Age Calculation**
  - Age = 2014 - Year_Birth
  - Filter: Age < 100 (remove outliers)

* **Education Aggregation**
  - Master, PhD → HighEducation = 1
  - Others → HighEducation = 0

* **Family Status**
  - Family = 1 jika Partner dan ada Children
  - Children = Kidhome + Teenhome

* **Total Spending Aggregation**
  - TotalSpending = sum(MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds)

* **Handle Missing Values**
  - Drop rows dengan Income null

### Final Features untuk Clustering

```
Income, TotalSpending, Recency, Age, Children,
NumDealsPurchases, NumWebVisitsMonth, NumWebPurchases,
NumCatalogPurchases, NumStorePurchases
```
Total: 10 fitur

### Model

* **Preprocessing**: StandardScaler
* **Dimensionality Reduction**: PCA (n_components=2)
* **Clustering**: KMeans
* **Parameters**:
  - n_clusters=3
  - init='k-means++'
  - n_init=10
  - random_state=42
  - algorithm='elkan'

### Pipeline

1. Load data dari Kaggle
2. Feature engineering ekstensif
3. Drop missing values
4. EDA (scatter plots, boxplots, pairplot, heatmap)
5. Select features
6. StandardScaler
7. Elbow Method dan Silhouette Analysis
8. PCA transformation (2 komponen)
9. K-Means pada data PCA
10. Visualisasi decision boundaries

### Hasil Clustering

* **K Optimal**: 3 (berdasarkan Silhouette dan Elbow)
* **Silhouette Score**: ~0.25-0.3
* **Visualisasi**: 2D PCA projection dengan decision boundaries

### Mengapa Menggunakan PCA

* **Dimensi Tinggi**: 10 fitur numerik
* **Visualisasi**: Tidak bisa plot 10D, perlu reduksi ke 2D
* **Multikolinearitas**: Income dan TotalSpending berkorelasi tinggi
* **Noise Reduction**: PCA membantu menghilangkan noise
* **Clustering Performance**: Clustering pada ruang PCA seringkali lebih baik
* **Decision Boundary**: Bisa visualisasi decision boundary setelah PCA

### Catatan Penting

* PCA dilakukan setelah scaling
* Clustering dilakukan pada hasil PCA, bukan data asli
* Trade-off: kehilangan interpretabilitas fitur asli
* Explained variance ratio perlu dicek

### Output

* Notebook: `customerpersonal_kmeans.ipynb`
* Feature engineering kompleks
* PCA untuk dimensionality reduction
* Visualisasi decision boundaries di ruang PCA

---

## STEP 3 — Online Retail RFM Analysis (dengan PCA)

### Tujuan

Implementasi K-Means untuk customer segmentation berbasis RFM (Recency, Frequency, Monetary) dengan preprocessing yang robust dan PCA untuk visualisasi.

### Dataset

* **Online Retail II (UCI)**
* Source: Kaggle (mashlyn/online-retail-ii-uci)
* File: online_retail_II.csv
* Total: 1,067,371 transaksi (raw)
* Periode: 2009-2011
* Customer base: UK dan Non-UK

### Data Information

* **Transaction Columns**:
  - Invoice: nomor invoice
  - StockCode: kode produk
  - Description: nama produk
  - Quantity: jumlah item
  - InvoiceDate: tanggal transaksi
  - Price: harga per unit
  - Customer ID: identifier customer
  - Country: negara customer

* **Data Quality Issues**:
  - Customer ID null (perlu di-drop)
  - Quantity <= 0 (returns, perlu di-filter)
  - Price <= 0 (perlu di-filter)

### Feature Engineering

* **Country Mapping**
  - United Kingdom → 'UK'
  - Others → 'Non-UK'

* **Data Cleaning**
  - Drop rows dengan Customer ID null
  - Filter Quantity > 0
  - Filter Price > 0
  - Convert InvoiceDate ke datetime

* **RFM Calculation**
  - Reference Date = max(InvoiceDate) + 1 day
  - **Recency**: Hari sejak transaksi terakhir per customer
  - **Frequency**: Jumlah unique invoice per customer
  - **Monetary**: Total (Quantity * Price) per customer

### RFM Aggregation

```python
recency = df.groupby('Customer ID')['InvoiceDate'].max()
         .apply(lambda x: (reference_date - x).days)

frequency = df.groupby('Customer ID')['Invoice'].nunique()

monetary = df.groupby('Customer ID')['TotalPrice'].sum()
```

### Model

* **Preprocessing**:
  - Log1p transformation (mengurangi skewness)
  - RobustScaler (robust terhadap outlier)

* **Dimensionality Reduction**: PCA (2 dan 3 komponen)

* **Clustering**: KMeans
* **Parameters**:
  - n_clusters=4
  - init='k-means++'
  - n_init=10
  - random_state=42
  - algorithm='elkan'

### Pipeline

1. Load data dari Kaggle
2. Data cleaning (null, negative values)
3. Feature engineering (Country mapping)
4. RFM calculation
5. EDA ekstensif:
   - Time series analysis (revenue per bulan)
   - Heatmap transaksi (hari vs jam)
   - UK vs Non-UK comparison
   - Top products
   - RFM distributions
   - Outlier detection
   - Lorenz curve (inequality)
   - Correlation matrix
6. Log transformation untuk reduce skewness
7. RobustScaler
8. Elbow Method dan Silhouette Analysis
9. PCA transformation (2D dan 3D)
10. K-Means pada data PCA
11. Visualisasi 2D dan 3D clusters
12. Cluster profiling

### Hasil Clustering

* **K Optimal**: 4
* **Elbow Method**: Siku di K=4
* **Silhouette Score**: Tertinggi di K=4

* **Cluster Profiling** (mean values):
  - Cluster 0: Low Recency, Low Frequency, Low Monetary (New/Occasional)
  - Cluster 1: High Recency, Low Frequency, Low Monetary (Churned)
  - Cluster 2: Low Recency, High Frequency, High Monetary (Champions)
  - Cluster 3: Medium all (Average)

### PCA Results

* **2D PCA**:
  - PC1 dan PC2
  - Decision boundaries terlihat jelas
  - Cluster separation cukup baik

* **3D PCA**:
  - Explained Variance: ~95%+ dengan 3 komponen
  - Interactive visualization dengan Plotly
  - Perspektif tambahan untuk cluster separation

### Mengapa Menggunakan PCA

* **Skewed Data**: RFM sangat skewed, PCA membantu setelah transformation
* **Visualisasi**: Perlu project ke 2D/3D untuk visualisasi
* **Correlation**: F dan M berkorelasi tinggi
* **Noise Reduction**: Menghilangkan noise dari data
* **Better Separation**: Cluster lebih terpisah di ruang PCA

### Mengapa Menggunakan RobustScaler

* **Outlier Heavy**: RFM data memiliki banyak outlier
* **Tidak Sensitif Outlier**: RobustScaler menggunakan median dan IQR
* **Better Performance**: Lebih baik dari StandardScaler untuk data ini

### EDA Highlights

* **Lorenz Curve**: Menunjukkan 1% customer berkontribusi signifikan terhadap revenue
* **Time Series**: Peak revenue di bulan tertentu
* **UK Dominance**: 90%+ transaksi dari UK
* **Skewness**: Semua RFM metric highly skewed
* **Business Sanity Check**: Scatter plot R vs M untuk identifikasi customer segments

### Output

* Notebook: `onlineretailRFM.ipynb`
* RFM analysis
* Log transformation + RobustScaler
* PCA untuk visualisasi 2D dan 3D
* Interactive 3D plot dengan Plotly

---

## Evaluasi Kritis & Insight

### Perbandingan Ketiga Notebook

| Aspek | K_means_learn | customerpersonal | onlineretailRFM |
|-------|---------------|------------------|-----------------|
| Dataset Size | 200 | 2,216 | ~5,000 customers |
| Fitur Asli | 3 | 29 | 8 → 3 (RFM) |
| Fitur Clustering | 2-3 | 10 | 3 |
| PCA | Tidak | Ya (2D) | Ya (2D, 3D) |
| Scaler | Tidak | StandardScaler | RobustScaler |
| Transformation | Tidak | Tidak | Log1p |
| K Optimal | 4-5 | 3 | 4 |
| Silhouette | ~0.55 | ~0.25-0.3 | ~0.4 |

### Kapan Menggunakan PCA

**GUNAKAN PCA jika:**
- Fitur > 3 dimensi (tidak bisa visualisasi langsung)
- Banyak fitur yang berkorelasi tinggi
- Perlu menghilangkan noise
- Ingin visualisasi decision boundaries
- Data memiliki redundansi informasi

**TIDAK PERLU PCA jika:**
- Fitur <= 3 dimensi
- Interpretabilitas fitur penting
- Fitur sudah independen
- Dataset sederhana untuk pembelajaran dasar

### Kapan Menggunakan Scaler Tertentu

**StandardScaler:**
- Data mendekati distribusi normal
- Tidak banyak outlier
- Default choice untuk kebanyakan kasus

**RobustScaler:**
- Data memiliki banyak outlier
- Distribusi sangat skewed
- RFM data, transaction data

**MinMaxScaler:**
- Butuh nilai dalam range tertentu
- Neural network input
- Data sudah bounded

### Pemilihan K Optimal

**Elbow Method:**
- Visualisasi inertia vs K
- Cari titik "siku" (diminishing returns)
- Subjektif, bisa ambigu

**Silhouette Score:**
- Metrik objektif (-1 to 1)
- Lebih tinggi lebih baik
- Pertimbangkan dengan interpretabilitas

**Domain Knowledge:**
- Berapa segmen yang masuk akal secara bisnis?
- Apakah cluster dapat diinterpretasi?
- Trade-off granularity vs simplicity

### Best Practices

1. **Selalu scaling sebelum K-Means** - K-Means sensitif terhadap skala
2. **Cek distribusi data** - Transform jika perlu (log, sqrt)
3. **Handle outlier** - RobustScaler atau remove
4. **Gunakan multiple methods** untuk pilih K - Elbow + Silhouette
5. **Validasi dengan domain knowledge** - Apakah cluster masuk akal?
6. **PCA setelah scaling** - Urutan penting
7. **Cek explained variance** - Pastikan informasi cukup retained

---

## Kelebihan K-Means

* Sederhana dan mudah diimplementasi
* Scalable untuk dataset besar
* Konvergen dengan cepat
* Hasil mudah diinterpretasi
* Works well untuk cluster spherical

## Keterbatasan K-Means

* Harus tentukan K di awal
* Sensitif terhadap inisialisasi (mitigasi: k-means++)
* Sensitif terhadap outlier
* Asumsi cluster spherical (sama ukuran)
* Tidak handle cluster non-convex
* Tidak optimal untuk cluster dengan density berbeda

---

## Struktur Direktori

```
02_LEARNING/03_Unsupervised/Kmeans/
|
├── K_means_learn.ipynb        # Basic K-Means, no PCA
├── customerpersonal_kmeans.ipynb  # Feature engineering + PCA
├── onlineretailRFM.ipynb      # RFM analysis + PCA + 3D
└── kmeans_pca.md              # Dokumentasi pembelajaran (file ini)
```

---

## Kriteria "SUDAH PAHAM"

Kamu boleh lanjut ke materi berikutnya kalau:

* Bisa menjelaskan cara kerja K-Means tanpa melihat dokumentasi
* Paham perbedaan inertia dan silhouette score
* Bisa menentukan K optimal dengan Elbow dan Silhouette
* Paham kapan perlu scaling dan jenis scaler yang tepat
* Bisa menjelaskan konsep PCA dan kapan menggunakannya
* Paham trade-off PCA (information loss vs dimensionality reduction)
* Bisa menginterpretasi hasil cluster untuk business insight
* Paham preprocessing yang tepat untuk data skewed/outlier

Kalau belum → **ulangi dan pelajari lebih dalam**

---

## Progression Path

### Sudah Dikuasai

- K-Means Clustering fundamentals
- Elbow Method dan Silhouette Score
- PCA untuk dimensionality reduction
- Preprocessing untuk clustering
- RFM Analysis
- 2D dan 3D visualization

### Next Steps

1. **Algoritma Clustering Lain**:
   - DBSCAN (density-based)
   - Hierarchical Clustering
   - Gaussian Mixture Models
   - Mean Shift

2. **Evaluasi Clustering Lanjutan**:
   - Davies-Bouldin Index
   - Calinski-Harabasz Index
   - Gap Statistic

3. **Dimensionality Reduction Lain**:
   - t-SNE
   - UMAP
   - LDA (Linear Discriminant Analysis)

4. **Aplikasi**:
   - Image segmentation
   - Anomaly detection
   - Customer lifetime value prediction

---

## Catatan

> K-Means = **baseline clustering algorithm**
> PCA = **dimensionality reduction untuk visualisasi dan noise reduction**
> Preprocessing sangat penting untuk clustering
> Pilih K berdasarkan kombinasi metrik dan domain knowledge
> Interpretabilitas cluster sama pentingnya dengan metrik

---

## Key Takeaways

### Dataset Characteristics

* **Mall Customers**: Small, clean, 3 features, no PCA needed
* **Customer Personality**: Medium, complex features, needs PCA
* **Online Retail RFM**: Large, transactional, needs transformation + PCA

### Preprocessing Techniques

* **K_means_learn**: Minimal preprocessing (rename columns only)
* **customerpersonal**: Extensive feature engineering + StandardScaler
* **onlineretailRFM**: RFM aggregation + Log transformation + RobustScaler

### PCA Usage

* **K_means_learn**: Tidak perlu (fitur <= 3)
* **customerpersonal**: Ya, untuk visualisasi 10 fitur
* **onlineretailRFM**: Ya, untuk visualisasi dan noise reduction

### K Selection

* **K_means_learn**: 4-5 (cluster natural terlihat jelas)
* **customerpersonal**: 3 (silhouette optimal)
* **onlineretailRFM**: 4 (elbow + silhouette + business sense)

### Business Application

* **Mall Customers**: Customer segmentation untuk targeted marketing
* **Customer Personality**: Persona development untuk campaign
* **Online Retail RFM**: Customer value segmentation untuk retention

---
