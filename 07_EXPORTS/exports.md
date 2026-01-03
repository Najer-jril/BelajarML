# Exports Directory

Folder ini digunakan untuk menyimpan hasil export dari analysis, visualizations, reports, dan deliverables lainnya.

## Jenis Exports

### Visualizations
- Plots dan charts dari EDA
- Model performance visualizations
- Feature importance plots
- Confusion matrices
- ROC curves
- Learning curves

Format: PNG, SVG, PDF

### Reports
- Model performance reports (PDF, HTML)
- Data analysis reports
- Experiment results summary
- Comparative analysis

Format: PDF, HTML, Markdown

### Data Exports
- Preprocessed datasets
- Predictions hasil inference
- Feature engineered datasets
- Aggregated statistics

Format: CSV, Excel, Parquet

### Presentations
- Slide deck untuk presentasi hasil
- Summary untuk stakeholders

Format: PDF, PPTX

## Naming Convention

```
<type>_<description>_<date>.<extension>
```

Contoh:
- plot_feature_correlation_20260103.png
- report_model_performance_20260103.html
- data_predictions_amsterdam_20260103.csv
- presentation_project_summary_20260103.pdf

## Struktur Folder

```
07_EXPORTS/
├── plots/
│   ├── eda/
│   ├── model_performance/
│   └── feature_analysis/
├── reports/
│   ├── weekly/
│   └── final/
├── data/
│   ├── predictions/
│   └── processed/
└── presentations/
```

## Best Practices

1. Gunakan high resolution untuk plots (300 DPI minimum)
2. Save plots dalam multiple formats (PNG untuk preview, SVG untuk editing)
3. Include timestamp dalam filename untuk versioning
4. Compress large files sebelum sharing
5. Backup exports penting
6. Dokumentasikan metadata untuk setiap export

## Automated Export

Gunakan scripts untuk automated export:

```python
# Example export script
import matplotlib.pyplot as plt
from datetime import datetime

def save_plot(fig, name, export_dir='../07_EXPORTS/plots/'):
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{name}_{timestamp}"
    fig.savefig(f"{export_dir}{filename}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{export_dir}{filename}.svg", bbox_inches='tight')
```

## Gitignore Considerations

Beberapa exports mungkin perlu diabaikan dari git:
- File berukuran besar
- Intermediate results
- Temporary visualizations

Tambahkan ke .gitignore jika diperlukan.
