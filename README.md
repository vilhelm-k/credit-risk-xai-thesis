# Explainable AI for Credit Risk Prediction

A Bachelor's thesis on applying explainable AI (SHAP, ALE) to corporate credit risk prediction for Swedish SMEs using LightGBM.

## Requirements

- Python 3.12
- Raw data files (not included in repository):
  - 10 Serrano dataset files: `serrano1.dta` through `serrano10.dta` (place in `data/raw/`)
  - 4 macroeconomic CSV files from SCB (place in `data/external/`)

## Setup

1. **Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Build the processed dataset:**

If you have the raw data files in place, run the pipeline scripts in order:

```bash
python3 credit_risk_xai/data/make_dataset.py   # Raw Serrano .dta -> interim parquet
python3 credit_risk_xai/data/make_macro.py     # External CSVs -> macro parquet
python3 credit_risk_xai/features/engineer.py   # Interim + macro -> processed features
```

## Analysis Notebooks

The following notebooks contain the analysis presented in the thesis:

| Notebook | Description |
|----------|-------------|
| `notebooks/00_data_exploration.ipynb` | Descriptive statistics, filter analysis, temporal distribution |
| `notebooks/05a_xai_global.ipynb` | Global XAI analysis (SHAP importance, beeswarm plots, feature interactions) |
| `notebooks/05b_xai_temporal.ipynb` | Temporal XAI analysis (ALE plots, time-varying effects) |

### Exploratory Notebooks

The following notebooks were used during development and exploration but are not part of the final analysis. Their outputs have been cleared to reduce file size:

- `notebooks/00a_hyperparameter_tuning.ipynb` - Optuna hyperparameter optimization
- `notebooks/00b_ml_comparison.ipynb` - Model comparison experiments
- `notebooks/00_data_quality_audit.ipynb` - Data quality checks
- `notebooks/01_exploration.ipynb` - Initial data exploration
- `notebooks/02_modelling_test.ipynb` - Model prototyping
- `notebooks/03_feature_selection.ipynb` - Feature selection experiments
- `notebooks/04_feature_pruning.ipynb` - Feature pruning analysis
- `notebooks/05_xai_exploration.ipynb` - XAI method exploration
- `notebooks/05bb_xai_temporal_oneyear.ipynb` - Single-year temporal analysis
- `notebooks/05c_xai_case_studies.ipynb` - Individual case studies
- `notebooks/05d_xai_clustering.ipynb` - SHAP-based clustering

## Project Structure

```
├── credit_risk_xai/        <- Source code
│   ├── config.py           <- Configuration and feature definitions
│   ├── data/               <- Data loading and preprocessing
│   ├── features/           <- Feature engineering
│   ├── modeling/           <- Model training, evaluation, XAI
│   └── plotting.py         <- Thesis-quality plotting utilities
│
├── data/
│   ├── raw/                <- Original Serrano .dta files
│   ├── external/           <- Macroeconomic CSV files from SCB
│   ├── interim/            <- Intermediate parquet caches
│   └── processed/          <- Final feature matrix
│
├── notebooks/              <- Jupyter notebooks for analysis
├── figures/                <- Generated figures for thesis
├── tables/                 <- Generated LaTeX tables
└── requirements.txt        <- Python dependencies
```
