# Dataset Directory

Folder ini menyimpan semua dataset yang digunakan untuk pembelajaran dan eksperimen machine learning.

## Available Datasets

Dataset properti rental dari Pararius untuk berbagai kota di Belanda:

- amsterdam_Pararius.csv - Data properti rental Amsterdam
- delft_Pararius.csv - Data properti rental Delft
- denhaag_Pararius.csv - Data properti rental Den Haag
- eindhoven_Pararius.csv - Data properti rental Eindhoven
- groningen_Pararius.csv - Data properti rental Groningen
- rotterdam_Pararius.csv - Data properti rental Rotterdam
- utrecht_Pararius.csv - Data properti rental Utrecht

## Dataset Usage

Dataset ini dapat digunakan untuk berbagai task machine learning:

### Regression Tasks
- Prediksi harga rental berdasarkan features properti
- Analisis faktor yang mempengaruhi harga
- Time series forecasting jika ada data temporal

### Classification Tasks
- Klasifikasi tipe properti
- Kategorisasi range harga
- Klasifikasi lokasi berdasarkan karakteristik

### Clustering Tasks
- Segmentasi area berdasarkan harga dan features
- Grouping properti dengan karakteristik serupa
- Market analysis

## Data Management Best Practices

### Loading Data

```python
import pandas as pd

# Load single city
df = pd.read_csv('../01_DATASETS/amsterdam_Pararius.csv')

# Load dan merge multiple cities
import glob
all_files = glob.glob('../01_DATASETS/*_Pararius.csv')
df_list = [pd.read_csv(file) for file in all_files]
df_combined = pd.concat(df_list, ignore_index=True)
```

### Data Versioning

1. Raw data tidak boleh dimodifikasi langsung
2. Simpan hasil preprocessing di folder terpisah atau dengan suffix
3. Dokumentasikan setiap perubahan pada data
4. Gunakan version control untuk tracking changes

### Data Preprocessing

1. Buat copy dari raw data sebelum preprocessing
2. Dokumentasikan semua transformasi yang dilakukan
3. Save preprocessing pipeline untuk reproducibility
4. Validasi data setelah preprocessing

## File Organization

```
01_DATASETS/
├── raw/              # Original datasets (read-only)
├── processed/        # Cleaned dan preprocessed data
├── interim/          # Intermediate processing results
└── external/         # Data dari sumber eksternal
```

## Adding New Datasets

Ketika menambahkan dataset baru:

1. Simpan dalam format yang sesuai (CSV, Parquet, Excel)
2. Tambahkan dokumentasi di file ini
3. Include data dictionary jika tersedia
4. Dokumentasikan source dan acquisition date
5. Check data quality dan missing values
6. Update .gitignore jika file terlalu besar

## Data Dictionary

Untuk setiap dataset, sebaiknya ada dokumentasi:
- Column names dan descriptions
- Data types
- Valid value ranges
- Missing value handling
- Units of measurement
- Categorical value meanings

## Dataset Size Considerations

Untuk dataset besar:
- Pertimbangkan menggunakan Parquet untuk efficiency
- Gunakan chunking untuk processing
- Store di external storage dan download on-demand
- Add ke .gitignore dan gunakan DVC atau Git LFS
- Dokumentasikan cara download dataset

## Data Privacy dan Ethics

1. Pastikan data acquisition legal dan ethical
2. Anonymize personal information jika diperlukan
3. Follow data protection regulations
4. Dokumentasikan data usage rights
5. Remove sensitive information sebelum sharing

## Common Data Issues

### Missing Values
- Identify pattern missing values
- Decide strategy: imputation, deletion, atau flagging
- Document approach yang digunakan

### Outliers
- Detect using statistical methods
- Decide apakah legitimate atau errors
- Handle appropriately based on context

### Data Quality
- Check untuk duplicates
- Validate data types
- Check value ranges
- Verify categorical values

## Backup Strategy

1. Keep original raw data untouched
2. Backup ke cloud storage atau external drive
3. Version important datasets
4. Document backup locations dan procedures

