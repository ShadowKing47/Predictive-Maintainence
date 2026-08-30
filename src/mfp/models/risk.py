import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import joblib

from mfp.core.config import settings
from mfp.core.exceptions import ModelError
from mfp.core.logging import get_logger
from mfp.features.scaling import FeatureScaler

logger = get_logger(__name__)


class RiskClassifier:
    """Binary classifier for failure risk (sigmoid + BCE)."""

    def __init__(self, config: dict | None = None):
        self.config = config or {
            "hidden_units": settings.classifier.hidden_units,
            "dropout": settings.classifier.dropout,
            "learning_rate": settings.classifier.learning_rate,
        }
        self.model: models.Model | None = None
        self.scaler = FeatureScaler()
        self.n_features = len(settings.data.sensor_columns)
        self.decision_threshold = 0.5
        self.history = None

    def build_model(self) -> models.Model:
        """Build MLP classifier with sigmoid output."""
        inputs = layers.Input(shape=(self.n_features,))

        x = inputs
        for units in self.config["hidden_units"]:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.config["dropout"])(x)

        outputs = layers.Dense(1, activation="sigmoid", name="risk_prob")(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        optimizer = optimizers.Adam(learning_rate=self.config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                "binary_accuracy",
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model
        logger.info("classifier_model_built", params=model.count_params())
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
        class_weight: dict | None = None,
    ) -> dict:
        """Train the risk classifier."""
        if self.model is None:
            self.build_model()

        epochs = epochs or settings.classifier.epochs
        batch_size = batch_size or settings.classifier.batch_size
        class_weight = class_weight or settings.classifier.class_weight

        cb = [
            callbacks.EarlyStopping(
                monitor="val_pr_auc",
                patience=settings.classifier.patience,
                restore_best_weights=True,
                mode="max",
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode="max",
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            class_weight=class_weight,
            verbose=1,
        )

        logger.info("classifier_trained", epochs=len(self.history.history["loss"]))
        return self.history.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict risk probabilities."""
        if self.model is None:
            raise ModelError("Model not trained or loaded")
        return self.model.predict(X, verbose=0).flatten()

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predict risk class."""
        proba = self.predict_proba(X)
        thresh = threshold or self.decision_threshold
        return (proba >= thresh).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate classifier on test set."""
        if self.model is None:
            raise ModelError("Model not trained or loaded")
        results = self.model.evaluate(X, y, verbose=0, return_dict=True)
        logger.info("classifier_evaluated", **results)
        return results

    def tune_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Tune decision threshold on validation set using PR curve."""
        from sklearn.metrics import precision_recall_curve

        proba = self.predict_proba(X_val)
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        self.decision_threshold = float(best_threshold)
        logger.info("threshold_tuned", threshold=self.decision_threshold, f1=f1_scores[best_idx])
        return self.decision_threshold

    def save(self, path: str) -> None:
        """Save model, scaler, and threshold."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "classifier.keras"))
        self.scaler.save(os.path.join(path, "scaler.joblib"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({**self.config, "decision_threshold": self.decision_threshold}, f)
        logger.info("classifier_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "RiskClassifier":
        """Load model, scaler, and threshold."""
        obj = cls()
        obj.model = models.load_model(os.path.join(path, "classifier.keras"))
        obj.scaler = FeatureScaler.load(os.path.join(path, "scaler.joblib"))
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
            obj.config = {k: v for k, v in config.items() if k != "decision_threshold"}
            obj.decision_threshold = config.get("decision_threshold", 0.5)
        logger.info("classifier_loaded", path=path)
        return obj