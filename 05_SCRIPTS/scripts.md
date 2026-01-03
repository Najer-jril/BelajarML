# Scripts Directory

Folder ini berisi Python scripts untuk automasi dan utility functions yang sering digunakan dalam workflow machine learning.

## Kategori Scripts

### Data Processing
- data_loader.py: Fungsi untuk load dan merge dataset dari 01_DATASETS
- data_cleaner.py: Cleaning dan preprocessing routines
- feature_engineer.py: Feature engineering functions
- data_validator.py: Validasi kualitas data

### Model Training
- train_model.py: Script untuk training model dengan parameter configuration
- evaluate_model.py: Evaluasi model dengan berbagai metrics
- hyperparameter_tuning.py: Grid search dan random search automation
- cross_validation.py: K-fold cross validation utilities

### Utilities
- config.py: Konfigurasi global (paths, constants, settings)
- logger.py: Custom logging setup
- visualization.py: Plotting functions untuk EDA dan model evaluation
- metrics.py: Custom evaluation metrics

### Deployment
- model_loader.py: Load saved models dengan preprocessing pipeline
- predict.py: Inference script untuk production
- api_server.py: Simple API untuk model serving (Flask/FastAPI)

## Best Practices

1. Setiap script harus memiliki docstring yang jelas
2. Gunakan argparse untuk command-line arguments
3. Implement error handling dan logging
4. Pisahkan configuration dari logic (gunakan config files)
5. Write modular dan reusable functions
6. Include usage examples di docstring

## Example Usage

```python
# Train model dengan script
python scripts/train_model.py --data ../01_DATASETS/amsterdam_Pararius.csv --model linear_regression --output ../03_MODELS/

# Evaluate model
python scripts/evaluate_model.py --model ../03_MODELS/rental_price_model_v1.pkl --test_data ../01_DATASETS/utrecht_Pararius.csv

# Hyperparameter tuning
python scripts/hyperparameter_tuning.py --algorithm random_forest --search grid --cv 5
```

## Dependency Management

Pastikan semua dependencies tercatat di environment specification:
- requirements.txt untuk pip
- environment.yml untuk conda

Update dependencies ketika menambahkan library baru.
