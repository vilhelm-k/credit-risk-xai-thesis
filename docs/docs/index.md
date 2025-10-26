# Explainable AI for Credit Risk Prediction documentation!

## Description

A Bachelor's thesis on applying explainable AI to corporate credit risk prediction for SMEs.

## Command reference

Use the Makefile targets for repeatable data processing:

| Command | Description |
| --- | --- |
| `make data-raw [FORCE=true]` | Build the interim Serrano base (`data/interim/serrano_base.parquet`). Set `FORCE=true` to overwrite existing cache. |
| `make data-macro [FORCE=true]` | Aggregate macro CSVs into `data/interim/macro_annual.parquet`. |
| `make features [FORCE=true]` | Create the full engineered feature matrix (`data/processed/serrano_features.parquet`). |
| `make build RAW=true MACRO=true FEATURES=true FORCE=true` | Convenience wrapper to run selected stages in one go. |

Model training utilities live under `credit_risk_xai.modeling` and can be invoked directly, for example:

```
python -m credit_risk_xai.modeling.train lightgbm --feature-path data/processed/serrano_features.parquet
python -m credit_risk_xai.modeling.train optuna --feature-path data/processed/serrano_features.parquet --mlflow-experiment credit-risk
```

These commands share the same functions exposed to notebooks for interactive experimentation.
