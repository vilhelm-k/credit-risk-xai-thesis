from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    classification_report,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from credit_risk_xai.config import FEATURE_CACHE_PATH, FEATURES_FOR_MODEL, PROJ_ROOT
from credit_risk_xai.features.engineer import prepare_modeling_data
from credit_risk_xai.modeling.utils import split_train_validation
from wandb.sdk.wandb_run import Run

DEFAULT_PARAMS: Dict[str, Any] = {
    # Core settings
    "objective": "binary",
    "n_estimators": 10_000,
    "metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "is_unbalance": False,
    # Tuned hyperparameters (Optuna, 2024-12-01, Log Loss: 0.0644, ROC-AUC: 0.8990)
    "learning_rate": 0.0567,
    "num_leaves": 214,
    "max_depth": 6,
    "min_child_samples": 97,
    "min_child_weight": 0.308,
    "reg_alpha": 4.764,
    "reg_lambda": 9.83e-05,
    "min_split_gain": 0.846,
    "subsample": 0.826,
    "subsample_freq": 3,
    "colsample_bytree": 0.505,
}


def apply_default_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    mask = (df["ser_aktiv"] == 1) & (df["sme_category"].isin(["Small", "Medium"]))
    filtered = df.loc[mask].copy()
    description = "ser_aktiv == 1 AND sme_category in ['Small', 'Medium']"
    return filtered, description


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = binids == i
        count = np.sum(mask)
        if count == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (count / total) * abs(avg_confidence - avg_accuracy)
    return float(ece)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict[str, Any]],
    eval_metric: str,
    early_stopping_rounds: int,
    log_frequency: int,
) -> Tuple[lgb.LGBMClassifier, Dict[str, float], float]:
    merged_params = dict(DEFAULT_PARAMS)
    merged_params["metric"] = eval_metric
    if params:
        merged_params.update(params)

    model = lgb.LGBMClassifier(**merged_params)
    
    # Use official wandb callback - it handles iteration logging automatically
    booster_callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=log_frequency),
    ]
    if wandb.run is not None:
        booster_callbacks.append(wandb_callback())
    
    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=eval_metric,
        categorical_feature='auto',  # Auto-detect category dtype for native categorical support
        callbacks=booster_callbacks,
    )
    duration = time.perf_counter() - start

    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    ece = expected_calibration_error(y_val.to_numpy(), proba)
    metrics = {
        "roc_auc": roc_auc_score(y_val, proba),
        "pr_auc": average_precision_score(y_val, proba),
        "logloss": log_loss(y_val, proba),
        "brier": brier_score_loss(y_val, proba),
        "ece": ece,
        "precision@0.5": precision_score(y_val, preds, zero_division=0),
        "recall@0.5": recall_score(y_val, preds, zero_division=0),
        "f1@0.5": f1_score(y_val, preds, zero_division=0),
    }
    
    # No need to return evals_result anymore
    return model, metrics, duration


def feature_importance(model: lgb.LGBMClassifier, feature_names: Sequence[str]) -> pd.DataFrame:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total_gain = np.sum(gain) or 1.0
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "gain": gain,
                "gain_pct": gain / total_gain,
                "split": split,
            }
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

def log_wandb(
    run: Run,
    model: lgb.LGBMClassifier,
    metrics: Dict[str, float],
    dataset_description: str,
    feature_names: Sequence[str],
    X_train_shape: Tuple[int, int],
    X_val_shape: Tuple[int, int],
    y_val: pd.Series,
    proba: np.ndarray,
    training_time: float,
) -> None:
    preds = (proba >= 0.5).astype(int)

    # Log config
    run.config.update(
        {
            "model_params": model.get_params(),
            "features": list(feature_names),
            "dataset_description": dataset_description,
        },
        allow_val_change=True,
    )

    # Log final metrics (standard + custom)
    wandb.log({f"metrics/{k}": v for k, v in metrics.items()})
    wandb.log(
        {
            "dataset/train_samples": X_train_shape[0],
            "dataset/train_features": X_train_shape[1],
            "dataset/val_samples": X_val_shape[0],
            "dataset/val_features": X_val_shape[1],
            "dataset/val_positive_rate": float(np.mean(y_val)),
            "metrics/best_iteration": model.best_iteration_,
            "metrics/training_time": training_time,
        }
    )

    # Confusion matrix - you're already doing this correctly
    wandb.log(
        {
            "plots/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_val.tolist(),
                preds=preds.tolist(),
                class_names=["no_default", "default"],
            )
        }
    )

    # PR curve - use wandb's built-in
    proba_2d = np.column_stack([1 - proba, proba])
    wandb.log(
        {
            "plots/pr_curve": wandb.plot.pr_curve(
                y_true=y_val.tolist(),
                y_probas=proba_2d.tolist(),
                labels=["no_default", "default"],
            )
        }
    )

    # ROC curve - use wandb's built-in
    wandb.log(
        {
            "plots/roc_curve": wandb.plot.roc_curve(
                y_true=y_val.tolist(),
                y_probas=proba_2d.tolist(),
                labels=["no_default", "default"],
            )
        }
    )

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_val, proba, n_bins=20, strategy='quantile'
    )
    
    wandb.log(
        {
            "plots/calibration_curve": wandb.plot.line_series(
                xs=mean_predicted_value.tolist(),
                ys=[
                    fraction_of_positives.tolist(),
                    mean_predicted_value.tolist(),  # Perfect calibration line
                ],
                keys=["Model", "Perfect Calibration"],
                title="Calibration Curve",
                xname="Mean Predicted Probability",
            )
        }
    )

    # Use official log_summary - handles feature importance chart + model artifact
    log_summary(model.booster_, save_model_checkpoint=True)

def run_lightgbm_training(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    dataset_description: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    eval_metric: str = "logloss",
    test_size: float = 0.2,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
    log_frequency: int = 50,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if dataset_description is None:
        dataset_description = "User supplied dataset (no description provided)."
    
    X_train, X_val, y_train, y_val = split_train_validation(
        X, y, test_size=test_size, random_state=random_state
    )

    # Start wandb run BEFORE training (so callback works)
    run = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            tags=list(wandb_tags) if wandb_tags else None,
            dir=PROJ_ROOT,
            config={
                "random_state": random_state,
                "test_size": test_size,
                "eval_metric": eval_metric,
                "early_stopping_rounds": early_stopping_rounds,
                "log_frequency": log_frequency,
            },
        )

    model, metrics, elapsed = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        params=params,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        log_frequency=log_frequency,
    )

    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    report_data = classification_report(y_val, preds, output_dict=True, zero_division=0)
    cm_array = confusion_matrix(y_val, preds, labels=[0, 1])
    importance_df = feature_importance(model, X_train.columns)

    results = {
        "model": model,
        "metrics": metrics,
        "dataset_description": dataset_description,
        "training_time": elapsed,
        "feature_importance": importance_df,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_val_proba": proba,
        "classification_report": report_data,
        "confusion_matrix": cm_array,
    }

    if use_wandb and run:
        log_wandb(
            run,
            model=model,
            metrics=metrics,
            dataset_description=dataset_description,
            feature_names=X_train.columns,
            X_train_shape=X_train.shape,
            X_val_shape=X_val.shape,
            y_val=y_val,
            proba=proba,
            training_time=elapsed,
        )
        run.finish()

    return results


app = typer.Typer()


@app.command()
def train(
    feature_path: Path = typer.Option(FEATURE_CACHE_PATH, help="Path to features parquet."),
    dataset_description: Optional[str] = typer.Option(
        None, help="Optional free-text description of dataset filters."
    ),
    eval_metric: str = typer.Option("logloss"),
    test_size: float = typer.Option(0.2),
    random_state: int = typer.Option(42),
    early_stopping_rounds: int = typer.Option(50),
    log_frequency: int = typer.Option(50),
    use_wandb: bool = typer.Option(False, help="Enable Weights & Biases logging."),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_entity: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
    wandb_tags: Optional[str] = typer.Option(None, help="Comma separated tags."),
    params: Optional[str] = typer.Option(None, help="JSON dict of LightGBM params."),
) -> None:
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}")

    df = pd.read_parquet(feature_path)
    parsed_params = json.loads(params) if params else None
    tag_list = [tag.strip() for tag in wandb_tags.split(",")] if wandb_tags else None

    filtered_df, default_desc = apply_default_filters(df)
    if dataset_description is None:
        dataset_description = default_desc
    X, y = prepare_modeling_data(filtered_df)

    results = run_lightgbm_training(
        X=X,
        y=y,
        dataset_description=dataset_description,
        params=parsed_params,
        eval_metric=eval_metric,
        test_size=test_size,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        log_frequency=log_frequency,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=tag_list,
    )

    metrics = results["metrics"]
    logger.info(
        "Validation metrics | AUC %.4f | PR-AUC %.4f | LogLoss %.4f | Precision@0.5 %.3f | Recall@0.5 %.3f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["logloss"],
        metrics["precision@0.5"],
        metrics["recall@0.5"],
    )


if __name__ == "__main__":
    app()
