from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from mfp.core.config import settings
from mfp.core.logging import get_logger
from mfp.core.seeds import set_global_seeds
from mfp.features.scaling import FeatureScaler
from mfp.features.sequences import (
    create_sequences,
)
from mfp.models.forecast import SensorForecaster
from mfp.models.risk import RiskClassifier

logger = get_logger(__name__)


def suggest_forecaster_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for forecaster."""
    return {
        "lstm_units": trial.suggest_categorical(
            "lstm_units",
            [[32, 32], [64, 64], [128, 64], [128, 128], [64, 32]],
        ),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 2.0),
    }


def suggest_classifier_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for classifier."""
    return {
        "hidden_units": trial.suggest_categorical(
            "hidden_units",
            [[32, 16], [64, 32], [128, 64], [128, 64, 32], [64, 32, 16]],
        ),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }


def objective_forecaster(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective for forecaster tuning."""
    params = suggest_forecaster_params(trial)
    set_global_seeds(settings.random_seed)

    forecaster = SensorForecaster(config=params)
    forecaster.build_model()

    history = forecaster.train(
        X_train, y_train, X_val, y_val,
        epochs=settings.forecaster.epochs,
        batch_size=settings.forecaster.batch_size,
    )

    val_loss = min(history["val_loss"])
    trial.report(val_loss, step=len(history["val_loss"]))

    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss


def objective_classifier(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective for classifier tuning."""
    params = suggest_classifier_params(trial)
    set_global_seeds(settings.random_seed)

    classifier = RiskClassifier(config=params)
    classifier.build_model()

    history = classifier.train(
        X_train, y_train, X_val, y_val,
        epochs=settings.classifier.epochs,
        batch_size=settings.classifier.batch_size,
        class_weight=settings.classifier.class_weight,
    )

    metric_name = settings.optuna.metric
    val_metric = min(history[f"val_{metric_name}"]) if metric_name != "loss" else min(history["val_loss"])
    trial.report(val_metric, step=len(history["val_loss"]))

    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_metric


def run_forecaster_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Run Optuna hyperparameter tuning for forecaster."""
    sampler = TPESampler(seed=settings.random_seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction=settings.optuna.direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"{settings.optuna.study_name}_forecaster",
        storage=settings.optuna.storage,
        load_if_exists=True,
    )

    def objective(trial):
        return objective_forecaster(trial, X_train, y_train, X_val, y_val)

    study.optimize(objective, n_trials=settings.optuna.n_trials, timeout=settings.optuna.timeout)

    logger.info("forecaster_tuning_complete", best_value=study.best_value, best_params=study.best_params)
    return study.best_params


def run_classifier_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Run Optuna hyperparameter tuning for classifier."""
    sampler = TPESampler(seed=settings.random_seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction=settings.optuna.direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"{settings.optuna.study_name}_classifier",
        storage=settings.optuna.storage,
        load_if_exists=True,
    )

    def objective(trial):
        return objective_classifier(trial, X_train, y_train, X_val, y_val)

    study.optimize(objective, n_trials=settings.optuna.n_trials, timeout=settings.optuna.timeout)

    logger.info("classifier_tuning_complete", best_value=study.best_value, best_params=study.best_params)
    return study.best_params


def setup_mlflow() -> None:
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("mfp_training")


def log_forecaster_mlflow(forecaster: SensorForecaster, metrics: dict, params: dict, artifact_path: str) -> None:
    """Log forecaster to MLflow."""
    with mlflow.start_run(run_name="forecaster") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(artifact_path)
        logger.info("forecaster_logged_mlflow", run_id=run.info.run_id)


def log_classifier_mlflow(classifier: RiskClassifier, metrics: dict, params: dict, artifact_path: str) -> None:
    """Log classifier to MLflow."""
    with mlflow.start_run(run_name="classifier") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(artifact_path)
        logger.info("classifier_logged_mlflow", run_id=run.info.run_id)


def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_gap: int = 40,
    seq_len: int = 10,
) -> list:
    """
    Walk-forward cross-validation with purge gaps.

    Returns list of (train_idx, val_idx, test_idx) for each fold.
    """
    n = len(df)
    fold_size = n // n_splits
    folds = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)

        val_start = max(0, test_start - int(n * val_ratio) - purge_gap)
        val_end = max(0, test_start - purge_gap)

        train_end = max(0, val_start - purge_gap)
        train_start = max(0, train_end - int(n * train_ratio))

        train_idx = list(range(train_start, train_end))
        val_idx = list(range(val_start, val_end))
        test_idx = list(range(test_start, test_end))

        if len(train_idx) > seq_len and len(val_idx) > seq_len and len(test_idx) > seq_len:
            folds.append((train_idx, val_idx, test_idx))

    logger.info("walk_forward_cv_created", n_folds=len(folds))
    return folds


def run_walk_forward_training(
    df: pd.DataFrame,
    n_splits: int = 5,
    artifact_dir: str | None = None,
) -> list:
    """Run walk-forward CV training and return metrics per fold."""
    artifact_path = Path(artifact_dir or settings.artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    folds = walk_forward_cv(
        df,
        n_splits=n_splits,
        train_ratio=settings.split.train_ratio,
        val_ratio=settings.split.val_ratio,
        test_ratio=settings.split.test_ratio,
        purge_gap=settings.split.purge_gap,
        seq_len=settings.split.seq_len,
    )

    all_results = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds):
        logger.info("walk_forward_fold_start", fold=fold_idx)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        scaler = FeatureScaler()
        train_scaled = scaler.fit_transform(train_df)
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)

        Xf_train, yf_train = create_sequences(train_scaled, settings.split.seq_len, horizon=1)
        Xf_val, yf_val = create_sequences(val_scaled, settings.split.seq_len, horizon=1)
        Xf_test, yf_test = create_sequences(test_scaled, settings.split.seq_len, horizon=1)

        forecaster = SensorForecaster()
        forecaster.build_model()
        forecaster.train(Xf_train, yf_train, Xf_val, yf_val)

        forecaster_metrics = forecaster.evaluate(Xf_test, yf_test)

        Xc_train = train_scaled[settings.split.seq_len:]
        yc_train = train_df["risk"].values[settings.split.seq_len:]
        Xc_val = val_scaled[settings.split.seq_len:]
        yc_val = val_df["risk"].values[settings.split.seq_len:]
        Xc_test = test_scaled[settings.split.seq_len:]
        yc_test = test_df["risk"].values[settings.split.seq_len:]

        classifier = RiskClassifier()
        classifier.build_model()
        classifier.train(Xc_train, yc_train, Xc_val, yc_val)
        classifier.tune_threshold(Xc_val, yc_val)

        classifier_metrics = classifier.evaluate(Xc_test, yc_test)

        fold_results = {
            "fold": fold_idx,
            "forecaster_test": forecaster_metrics,
            "classifier_test": classifier_metrics,
        }
        all_results.append(fold_results)

        if settings.optuna.enabled:
            best_forecaster_params = run_forecaster_tuning(Xf_train, yf_train, Xf_val, yf_val)
            best_classifier_params = run_classifier_tuning(Xc_train, yc_train, Xc_val, yc_val)
            fold_results["best_forecaster_params"] = best_forecaster_params
            fold_results["best_classifier_params"] = best_classifier_params

    logger.info("walk_forward_training_complete", n_folds=len(all_results))
    return all_results
