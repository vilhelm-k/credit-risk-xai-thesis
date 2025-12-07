# Explainable AI for Credit Risk Prediction

**Thesis Title**: Explainable AI for Credit Risk Prediction: A Study of Swedish SMEs Using LightGBM and SHAP

**Authors**: Vilhelm Karlin, Minna Olsson

**Contact**: 26012@student.hhs.se, 26027@student.hhs.se

**Course**: BE451 - Degree Project in Finance, Stockholm School of Economics, Fall 2025

---

## Data Sources

### Serrano Database

The Serrano database is provided by the Swedish House of Finance Research Data Center at the Stockholm School of Economics. It contains comprehensive financial statement data for Swedish companies from 1998-2023.

- **Access**: Restricted to SSE researchers via the Research Data Center
- **URL**: https://www.hhs.se/en/houseoffinance/data-center/
- **Files required**: `serrano1.dta` through `serrano10.dta` (place in `data/raw/`)

### Macroeconomic Data

Interest rate data for computing term spread is obtained from Statistics Sweden (SCB):

```bibtex
@misc{scb2025lending,
  author       = {{Statistics Sweden}},
  title        = {Lending Rates to Households and Non-Financial Corporations,
                  Breakdown by Fixation Periods},
  year         = {2025},
  howpublished = {Statistical Database},
  url          = {https://www.statistikdatabasen.scb.se/pxweb/en/ssd/START__FM__FM5001__FM5001C/RantaT01N/},
  note         = {Accessed: September 2025}
}
```

---

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`

---

## Setup and Replication

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place data files

- Place Serrano `.dta` files in `data/raw/`
- Place SCB macro CSV files in `data/external/`

### 4. Build the processed dataset

Run the data pipeline scripts in order:

```bash
# Step 1: Load raw Serrano .dta files → interim parquet
python3 credit_risk_xai/data/make_dataset.py

# Step 2: Process external macro CSVs → macro parquet
python3 credit_risk_xai/data/make_macro.py

# Step 3: Engineer features → final processed parquet
python3 credit_risk_xai/features/engineer.py
```

### 5. Run analysis notebooks

Execute the notebooks in order to reproduce all figures and tables:

| Notebook | Description | Outputs | Runtime |
|----------|-------------|---------|---------|
| `notebooks/00_data_exploration.ipynb` | Descriptive statistics, filter analysis | `figures/default_rate_by_year.pdf`, `tables/descriptive_stats.tex` | ~1 min |
| `notebooks/05a_xai_global.ipynb` | Global XAI analysis (SHAP, ALE) | `figures/roc_curve.pdf`, `figures/calibration_curve.pdf`, `figures/shap_*.pdf`, `figures/ale_*.pdf`, `figures/interaction_*.pdf` | ~15 min |
| `notebooks/05b_xai_temporal.ipynb` | Temporal XAI analysis | `figures/temporal_*.pdf` | ~20 min |

---

## Project Structure

```
credit-risk-xai-thesis/
├── README.md                      <- This file
├── FEATURES.md                    <- Feature definitions with formulas
├── requirements.txt               <- Python dependencies
├── pyproject.toml                 <- Project metadata
│
├── credit_risk_xai/               <- Source code package
│   ├── config.py                  <- Configuration and feature definitions
│   ├── plotting.py                <- Thesis-quality plotting utilities
│   ├── data/
│   │   ├── make_dataset.py        <- Raw Serrano .dta → interim parquet
│   │   └── make_macro.py          <- External CSVs → macro parquet
│   ├── features/
│   │   └── engineer.py            <- Feature engineering pipeline
│   └── modeling/
│       ├── train.py               <- LightGBM training
│       ├── evaluate.py            <- Evaluation metrics (AUC, ECE, etc.)
│       ├── explain.py             <- SHAP explanations
│       ├── ale.py                 <- ALE plot computation
│       └── logit.py               <- Logistic regression baseline
│
├── data/
│   ├── raw/                       <- Serrano .dta files (not in repo)
│   ├── external/                  <- SCB macro CSV files
│   ├── interim/                   <- Intermediate parquet caches
│   └── processed/                 <- Final feature matrix
│
├── notebooks/
│   ├── 00_data_exploration.ipynb  <- Descriptive statistics
│   ├── 05a_xai_global.ipynb       <- Global XAI analysis
│   ├── 05b_xai_temporal.ipynb     <- Temporal XAI analysis
│   └── archive/                   <- Exploratory notebooks (not part of final analysis)
│
├── figures/                       <- Generated thesis figures
└── tables/                        <- Generated LaTeX tables
```

---

## Output Mapping

The following table maps each thesis output to its source code location:

### Tables

| Table | Source | Notebook Cell |
|-------|--------|---------------|
| Descriptive Statistics | `tables/descriptive_stats.tex` | `00_data_exploration.ipynb`, cell 18 |

### Figures

| Figure | Source | Notebook Cell |
|--------|--------|---------------|
| Default Rate by Year | `figures/default_rate_by_year.pdf` | `00_data_exploration.ipynb`, cell 21 |
| ROC Curve | `figures/roc_curve.pdf` | `05a_xai_global.ipynb` |
| Calibration Curve | `figures/calibration_curve.pdf` | `05a_xai_global.ipynb` |
| SHAP Importance | `figures/shap_importance_comparison.pdf` | `05a_xai_global.ipynb` |
| SHAP Beeswarm (LightGBM) | `figures/shap_beeswarm_lgbm.pdf` | `05a_xai_global.ipynb` |
| SHAP Beeswarm (Logit) | `figures/shap_beeswarm_logit.pdf` | `05a_xai_global.ipynb` |
| ALE Plots (Full) | `figures/ale_full.pdf` | `05a_xai_global.ipynb` |
| ALE Plots (Individual) | `figures/ale_*.pdf` | `05a_xai_global.ipynb` |
| Interaction Analysis | `figures/interaction_*.pdf` | `05a_xai_global.ipynb` |
| SHAP Interaction Heatmap | `figures/shap_interaction_heatmap.pdf` | `05a_xai_global.ipynb` |
| Temporal Importance Ranks | `figures/temporal_importance_ranks.pdf` | `05b_xai_temporal.ipynb` |
| ALE Temporal Evolution | `figures/ale_temporal_evolution.pdf` | `05b_xai_temporal.ipynb` |
| Group Importance Evolution | `figures/temporal_group_importance.pdf` | `05b_xai_temporal.ipynb` |
| Group Correlation | `figures/temporal_group_correlation.pdf` | `05b_xai_temporal.ipynb` |

---

## Feature Documentation

See [FEATURES.md](FEATURES.md) for complete documentation of all 22 model features, including:
- Variable definitions and formulas
- Source column mappings to Serrano database
- Data filters applied
- Macroeconomic data sources

---

## Model Configuration

The model uses LightGBM with the following configuration (defined in `credit_risk_xai/modeling/train.py`):

- **Objective**: Binary classification (default prediction)
- **Metric**: Log-loss with early stopping
- **Features**: 22 features (V2 Altman/Ohlson aligned)
- **Categorical handling**: Native LightGBM categorical support for `sni_group_3digit`

---

## Exploratory Notebooks

The `notebooks/archive/` directory contains notebooks used during development but not part of the final analysis:

- `00a_hyperparameter_tuning.ipynb` - Optuna hyperparameter optimization
- `00b_ml_comparison.ipynb` - Model comparison experiments
- `00_data_quality_audit.ipynb` - Data quality checks
- `01_exploration.ipynb` - Initial data exploration
- `02_modelling_test.ipynb` - Model prototyping
- `03_feature_selection.ipynb` - Feature selection experiments
- `04_feature_pruning.ipynb` - Feature pruning analysis
- `05_xai_exploration.ipynb` - XAI method exploration
- `05bb_xai_temporal_oneyear.ipynb` - Single-year temporal analysis
- `05c_xai_case_studies.ipynb` - Individual case studies
- `05d_xai_clustering.ipynb` - SHAP-based clustering

These notebooks have cleared outputs to reduce file size.
