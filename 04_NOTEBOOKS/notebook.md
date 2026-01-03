# Notebooks Directory

Folder ini digunakan untuk menyimpan Jupyter notebooks yang berisi eksperimen, analisis ad-hoc, dan exploratory work yang tidak masuk ke dalam struktur pembelajaran formal di folder 02_LEARNING.

## Tujuan Folder Ini

Berbeda dengan folder 02_LEARNING yang berisi materi pembelajaran terstruktur, folder 04_NOTEBOOKS digunakan untuk:

- Eksperimen cepat dengan algoritma baru
- Analisis data ad-hoc untuk menjawab pertanyaan spesifik
- Prototyping sebelum dipindahkan ke scripts
- Testing dan debugging code
- Exploratory data analysis (EDA) untuk dataset baru
- Proof of concept untuk ide-ide baru
- Scratch work dan iterasi cepat

## Naming Convention

Gunakan naming yang deskriptif dengan prefix untuk kategori:

```
<kategori>_<deskripsi>_<tanggal>.ipynb
```

Kategori:
- exp: Experiment
- eda: Exploratory Data Analysis
- test: Testing
- poc: Proof of Concept
- debug: Debugging
- analysis: Analysis

Contoh:
- exp_random_forest_tuning_20260103.ipynb
- eda_amsterdam_rental_prices_20260103.ipynb
- poc_neural_network_price_prediction_20260103.ipynb
- test_custom_preprocessing_pipeline_20260103.ipynb

## Best Practices

1. Tambahkan markdown cells untuk dokumentasi
2. Restart kernel dan run all cells sebelum save untuk memastikan reproducibility
3. Clear output untuk large results sebelum commit ke git
4. Gunakan relative paths untuk akses data dan models
5. Import libraries di cell pertama
6. Set random seed untuk reproducibility
7. Tambahkan summary di cell terakhir untuk key findings

## Template Structure

Setiap notebook sebaiknya mengikuti struktur:

```
1. Title dan Description
2. Imports
3. Configuration (paths, parameters, seeds)
4. Data Loading
5. Exploratory Analysis / Experiment
6. Results
7. Conclusions dan Next Steps
```

## Transition ke Production

Ketika notebook sudah mature dan siap untuk production:

1. Extract reusable functions ke 05_SCRIPTS
2. Move ke 02_LEARNING jika menjadi materi pembelajaran
3. Save trained model ke 03_MODELS
4. Export visualizations ke 07_EXPORTS
5. Archive atau delete notebook jika sudah tidak diperlukan

## Jupyter Configuration

Untuk menjalankan Jupyter:

```bash
# Dari root directory
jupyter lab

# Atau jupyter notebook
jupyter notebook
```

Pastikan conda environment ml-core sudah aktif sebelum menjalankan jupyter.

## Common Workflows

### Quick EDA
1. Load dataset dari 01_DATASETS
2. Check data info, missing values, distributions
3. Create visualizations
4. Document findings

### Model Experimentation
1. Load preprocessed data
2. Try different algorithms atau hyperparameters
3. Compare performance metrics
4. Select best approach
5. Transition to script untuk final implementation

### Debugging
1. Reproduce error dalam isolated environment
2. Test solutions incrementally
3. Validate fix
4. Update production code

## Gitignore Considerations

Notebook outputs dapat membuat file size besar:
- Clear outputs sebelum commit untuk notebooks dengan large results
- Gunakan nbstripout untuk automated output clearing
- Commit hanya code dan markdown cells, bukan results

Tambahkan ke .gitignore jika diperlukan:
```
# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*
```
