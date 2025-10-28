from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_risk_xai.config import FEATURE_CACHE_PATH
from credit_risk_xai.features.engineer import prepare_modeling_data
from credit_risk_xai.modeling.utils import positive_class_weight, split_train_validation

app = typer.Typer(help="Model training utilities (LightGBM + Optuna + MLflow).")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    num_boost_round: int = 10_000,
    early_stopping_rounds: int = 50,
    eval_metric: str = "auc",
    verbose_eval: int = 100,
) -> tuple[lgb.LGBMClassifier, Dict[str, float], float]:
    """
    Train a LightGBM classifier with early stopping and return model + metrics.
    """
    scale_pos_weight = positive_class_weight(y_train)
    default_params = {
        "objective": "binary",
        "n_estimators": num_boost_round,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "is_unbalance": False,
        "scale_pos_weight": scale_pos_weight,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "metric": eval_metric,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval),
        ],
    )
    training_time = time.time() - start

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "auc": roc_auc_score(y_val, y_pred_proba),
        "average_precision": average_precision_score(y_val, y_pred_proba),
    }
    return model, metrics, training_time


def run_optuna_study(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: Optional[str] = None,
    direction: str = "maximize",
    mlflow_experiment: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> optuna.Study:
    """
    Optimise LightGBM hyperparameters with Optuna and (optionally) MLflow logging.
    """

    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }

        X_train, X_val, y_train, y_val = split_train_validation(
            X,
            y,
            test_size=test_size,
            random_state=random_state + trial.number,
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) if mlflow_experiment else _nullcontext():
            model, metrics, training_time = train_lightgbm(
                X_train,
                y_train,
                X_val,
                y_val,
                params=params,
                verbose_eval=0,
            )

            if mlflow_experiment:
                mlflow.log_params(params)
                mlflow.log_metric("auc", metrics["auc"])
                mlflow.log_metric("average_precision", metrics["average_precision"])
                mlflow.log_metric("training_time", training_time)

        return metrics["auc"]

    study = optuna.create_study(direction=direction, study_name=study_name)
    logger.info("Starting Optuna study (%d trials, direction=%s)", n_trials, direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.success("Optuna completed. Best trial: %.4f (params=%s)", study.best_value, study.best_params)
    return study


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _nullcontext():
    return _NullContext()


@app.command("lightgbm")
def cli_train_lightgbm(
    feature_path: Path = typer.Option(FEATURE_CACHE_PATH, help="Path to features parquet."),
    min_revenue: int = typer.Option(MIN_REVENUE_KSEK, help="Revenue threshold (kSEK)."),
    test_size: float = typer.Option(0.2, help="Validation split fraction."),
    random_state: int = typer.Option(42, help="Random seed."),
    output_model: Optional[Path] = typer.Option(None, help="Optional path to save trained model."),
) -> None:
    """Train a simple LightGBM model on the processed feature dataset."""
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}. Run feature pipeline first.")

    df = pd.read_parquet(feature_path)
    # Apply filtering: active companies with minimum revenue
    filtered = df[(df["ser_aktiv"] == 1) & (df["rr01_ntoms"] >= min_revenue)]
    X, y = prepare_modeling_data(filtered)
    X_train, X_val, y_train, y_val = split_train_validation(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Training LightGBM on %d features (%d train / %d val)",
        X.shape[1],
        len(X_train),
        len(X_val),
    )

    model, metrics, training_time = train_lightgbm(X_train, y_train, X_val, y_val)
    logger.success(
        "Validation AUC=%.4f | PR-AUC=%.4f | training_time=%.1fs",
        metrics["auc"],
        metrics["average_precision"],
        training_time,
    )

    if output_model:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(output_model)
        logger.info("Saved LightGBM model to %s", output_model)


@app.command("optuna")
def cli_run_optuna(
    feature_path: Path = typer.Option(FEATURE_CACHE_PATH, help="Path to features parquet."),
    min_revenue: int = typer.Option(MIN_REVENUE_KSEK, help="Revenue threshold (kSEK)."),
    n_trials: int = typer.Option(25, help="Number of Optuna trials."),
    mlflow_experiment: Optional[str] = typer.Option(
        None, help="Optional MLflow experiment name for logging."
    ),
) -> None:
    """Run hyperparameter optimisation using Optuna (and optionally MLflow)."""
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}.")

    df = pd.read_parquet(feature_path)
    # Apply filtering: active companies with minimum revenue
    filtered = df[(df["ser_aktiv"] == 1) & (df["rr01_ntoms"] >= min_revenue)]
    X, y = prepare_modeling_data(filtered)

    run_optuna_study(
        X,
        y,
        n_trials=n_trials,
        mlflow_experiment=mlflow_experiment,
    )


if __name__ == "__main__":
    app()
