import os
from pathlib import Path

import numpy as np
import pandas as pd

from mfp.core.config import settings
from mfp.core.logging import get_logger
from mfp.core.seeds import set_global_seeds
from mfp.data.ingest import preprocess_pipeline
from mfp.features.scaling import FeatureScaler
from mfp.features.sequences import (
    create_sequences,
    temporal_split_with_purge,
    check_split_leakage,
)
from mfp.models.forecast import SensorForecaster
from mfp.models.risk import RiskClassifier
from mfp.models.baselines import compare_with_baselines

logger = get_logger(__name__)


def prepare_forecaster_data(df: pd.DataFrame) -> tuple:
    """Prepare data for forecaster: scale, sequence, split."""
    scaler = FeatureScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(
        scaled_data,
        seq_len=settings.split.seq_len,
        horizon=1,
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split_with_purge(
        X,
        y,
        train_ratio=settings.split.train_ratio,
        val_ratio=settings.split.val_ratio,
        test_ratio=settings.split.test_ratio,
        purge_gap=settings.split.purge_gap,
    )

    # Verify no leakage
    assert check_split_leakage(
        X_train, X_val, X_test, settings.split.seq_len, settings.split.purge_gap
    ), "Leakage detected in splits!"

    logger.info(
        "forecaster_data_prepared",
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        test_shape=X_test.shape,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def prepare_classifier_data(
    df: pd.DataFrame,
    scaler: FeatureScaler | None = None,
) -> tuple:
    """Prepare data for risk classifier: use actual sensor readings (not forecasts)."""
    if scaler is None:
        scaler = FeatureScaler()
        scaled_features = scaler.fit_transform(df[settings.data.sensor_columns])
    else:
        scaled_features = scaler.transform(df[settings.data.sensor_columns])

    X = scaled_features
    y = df["risk"].values

    n = len(X)
    train_end = int(n * settings.split.train_ratio)
    val_end = train_end + int(n * settings.split.val_ratio)

    # Apply same purge logic (no sequences, just row indices)
    purge = settings.split.purge_gap
    X_train, y_train = X[: max(0, train_end - purge)], y[: max(0, train_end - purge)]
    X_val, y_val = X[train_end + purge : val_end - purge], y[train_end + purge : val_end - purge]
    X_test, y_test = X[val_end + purge :], y[val_end + purge :]

    logger.info(
        "classifier_data_prepared",
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        test_shape=X_test.shape,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def train_forecaster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> SensorForecaster:
    """Train the sensor forecaster."""
    forecaster = SensorForecaster()
    forecaster.build_model()
    forecaster.train(X_train, y_train, X_val, y_val)
    return forecaster


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RiskClassifier:
    """Train the risk classifier."""
    classifier = RiskClassifier()
    classifier.build_model()
    classifier.train(X_train, y_train, X_val, y_val)
    # Tune threshold on validation
    classifier.tune_threshold(X_val, y_val)
    return classifier


def run_full_training_pipeline(
    data_path: str | None = None,
    artifact_dir: str | None = None,
) -> dict:
    """Run the complete training pipeline: forecaster -> classifier -> baselines."""
    set_global_seeds(settings.random_seed)

    artifact_path = Path(artifact_dir or settings.artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess data
    logger.info("starting_training_pipeline")
    df = preprocess_pipeline(data_path)

    # 2. Prepare forecaster data
    (Xf_train, yf_train), (Xf_val, yf_val), (Xf_test, yf_test), scaler = prepare_forecaster_data(df)

    # 3. Train forecaster
    forecaster = train_forecaster(Xf_train, yf_train, Xf_val, yf_val)

    # 4. Evaluate forecaster on test
    forecaster_metrics = forecaster.evaluate(Xf_test, yf_test)
    persist_metrics = forecaster.persistence_baseline(Xf_test, yf_test)

    logger.info("forecaster_test_metrics", **forecaster_metrics)
    logger.info("forecaster_persistence_baseline", **persist_metrics)

    # 5. Prepare classifier data (use actual sensor readings, not forecasts)
    (Xc_train, yc_train), (Xc_val, yc_val), (Xc_test, yc_test), _ = prepare_classifier_data(df, scaler)

    # 6. Train classifier
    classifier = train_classifier(Xc_train, yc_train, Xc_val, yc_val)

    # 7. Evaluate classifier on test
    classifier_metrics = classifier.evaluate(Xc_test, yc_test)

    # 8. Compare with baselines
    baseline_results = compare_with_baselines(
        forecaster, Xf_test, yf_test,
        classifier, Xc_test, yc_test,
    )

    # 9. Save artifacts
    forecaster.save(str(artifact_path / "forecaster"))
    classifier.save(str(artifact_path / "classifier"))
    scaler.save(str(artifact_path / "scaler.joblib"))

    results = {
        "forecaster_test": forecaster_metrics,
        "forecaster_persistence": persist_metrics,
        "classifier_test": classifier_metrics,
        "baselines": baseline_results,
        "scaler_params": scaler.get_params(),
    }

    logger.info("training_pipeline_complete", results=results)
    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_training_pipeline(path)