# Models Directory

Folder ini digunakan untuk menyimpan trained models yang telah dibuat selama proses pembelajaran.

## Naming Convention

Gunakan naming convention yang jelas untuk memudahkan tracking:

```
<task>_<algorithm>_<date>_<version>.pkl
<task>_<algorithm>_<date>_<version>.h5
<task>_<algorithm>_<date>_<version>.joblib
```

Contoh:
- rental_price_linear_regression_20260103_v1.pkl
- property_classification_random_forest_20260103_v2.joblib
- clustering_kmeans_20260103_v1.pkl

## Format Penyimpanan

### Scikit-learn Models
- Gunakan `joblib` untuk efisiensi (lebih baik untuk numpy arrays)
- Alternatif: `pickle` untuk compatibility

### Deep Learning Models
- TensorFlow/Keras: format .h5 atau SavedModel
- PyTorch: format .pt atau .pth

### Model Metadata

Setiap model sebaiknya disertai dengan file metadata berisi:
- Tanggal training
- Hyperparameters yang digunakan
- Performance metrics
- Dataset yang digunakan
- Feature columns
- Preprocessing steps

Contoh: rental_price_linear_regression_20260103_v1_metadata.json

## Best Practices

1. Simpan preprocessing pipeline bersama dengan model
2. Version control untuk tracking perubahan
3. Dokumentasikan performa model
4. Backup model terbaik secara terpisah
5. Hapus model lama yang tidak digunakan untuk menghemat storage

## File Structure

```
03_MODELS/
├── regression/
│   ├── rental_price_model_v1.pkl
│   └── rental_price_model_v1_metadata.json
├── classification/
│   ├── property_type_model_v1.pkl
│   └── property_type_model_v1_metadata.json
└── clustering/
    ├── area_clustering_v1.pkl
    └── area_clustering_v1_metadata.json
```
