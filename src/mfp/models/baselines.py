import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mfp.core.logging import get_logger

logger = get_logger(__name__)


class PersistenceBaseline:
    """Persistence (naive) baseline: predict last observed value."""

    def __init__(self):
        self.name = "persistence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        """No fitting needed for persistence."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return last timestep of each sequence."""
        return X[:, -1, :]

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute MSE and MAE."""
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        mae = np.mean(np.abs(y_pred - y))
        return {"mse": float(mse), "mae": float(mae)}


class RidgeBaseline:
    """Ridge regression baseline for sensor forecasting."""

    def __init__(self, alpha: float = 1.0):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifier(alpha=alpha)),
        ])
        self.name = "ridge"
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        """Fit ridge on flattened sequences."""
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(n_samples, seq_len * n_features)
        self.pipeline.fit(X_flat, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ridge."""
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(n_samples, seq_len * n_features)
        return self.pipeline.predict(X_flat)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute accuracy."""
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        return {"accuracy": float(acc)}


class RiskPersistenceBaseline:
    """Persistence baseline for risk: if any sensor in window exceeds threshold, risk=1."""

    def __init__(self, temp_threshold: float = 85.0, press_threshold: float = 110.0):
        self.temp_threshold = temp_threshold
        self.press_threshold = press_threshold
        self.name = "risk_persistence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiskPersistenceBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Risk = 1 if any temp/pressure in window exceeds threshold."""
        # X shape: (n_samples, seq_len, n_features)
        # Temperature1 is index 0, Pressure1 is index 2
        temp1_max = X[:, :, 0].max(axis=1)
        press1_max = X[:, :, 2].max(axis=1)
        return ((temp1_max > self.temp_threshold) | (press1_max > self.press_threshold)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-probabilities."""
        preds = self.predict(X)
        return preds.astype(float)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "pr_auc": float(average_precision_score(y, y_proba)),
        }


def compare_with_baselines(
    forecaster,  # SensorForecaster
    X_test: np.ndarray,
    y_test: np.ndarray,
    risk_classifier=None,
    X_risk_test: np.ndarray | None = None,
    y_risk_test: np.ndarray | None = None,
) -> dict:
    """Compare models with baselines."""
    results = {}

    # Forecaster baselines
    persist = PersistenceBaseline()
    persist.fit(X_test, y_test)
    results["forecaster_persistence"] = persist.score(X_test, y_test)

    ridge = RidgeBaseline()
    ridge.fit(X_test, y_test)
    results["forecaster_ridge"] = ridge.score(X_test, y_test)

    # Forecaster model
    if forecaster and forecaster.model:
        y_pred = forecaster.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        results["forecaster_lstm"] = {"mse": float(mse), "mae": float(mae)}

    # Risk classifier baselines
    if risk_classifier and X_risk_test is not None and y_risk_test is not None:
        risk_persist = RiskPersistenceBaseline()
        risk_persist.fit(X_risk_test, y_risk_test)
        results["risk_persistence"] = risk_persist.score(X_risk_test, y_risk_test)

        # Risk classifier model
        if risk_classifier.model:
            results["risk_classifier"] = risk_classifier.evaluate(X_risk_test, y_risk_test)

    logger.info("baseline_comparison", results=results)
    return results
