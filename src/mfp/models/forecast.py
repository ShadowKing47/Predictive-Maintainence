import json
from pathlib import Path

import numpy as np
from tensorflow.keras import callbacks, layers, models, optimizers

from mfp.core.config import settings
from mfp.core.exceptions import ModelError
from mfp.core.logging import get_logger
from mfp.features.scaling import FeatureScaler

logger = get_logger(__name__)


class SensorForecaster:
    """LSTM forecaster for sensor values only (no Status/risk)."""

    def __init__(self, config: dict | None = None):
        self.config = config or {
            "lstm_units": settings.forecaster.lstm_units,
            "dropout": settings.forecaster.dropout,
            "learning_rate": settings.forecaster.learning_rate,
            "gradient_clip": settings.forecaster.gradient_clip,
        }
        self.model: models.Model | None = None
        self.scaler = FeatureScaler()
        self.seq_len = settings.split.seq_len
        self.n_features = len(settings.data.sensor_columns)
        self.history = None

    def build_model(self) -> models.Model:
        """Build LSTM model for multi-sensor forecasting."""
        inputs = layers.Input(shape=(self.seq_len, self.n_features))

        x = inputs
        for i, units in enumerate(self.config["lstm_units"]):
            return_sequences = i < len(self.config["lstm_units"]) - 1
            x = layers.LSTM(units, return_sequences=return_sequences)(x)
            x = layers.Dropout(self.config["dropout"])(x)

        outputs = layers.Dense(self.n_features, name="sensor_forecast")(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        optimizer = optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=self.config["gradient_clip"],
        )
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"],
        )

        self.model = model
        logger.info("forecaster_model_built", params=model.count_params())
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> dict:
        """Train the forecaster."""
        if self.model is None:
            self.build_model()

        epochs = epochs or settings.forecaster.epochs
        batch_size = batch_size or settings.forecaster.batch_size

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=settings.forecaster.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
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
            verbose=1,
        )

        logger.info("forecaster_trained", epochs=len(self.history.history["loss"]))
        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast sensor values."""
        if self.model is None:
            raise ModelError("Model not trained or loaded")
        return self.model.predict(X, verbose=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate forecaster on test set."""
        if self.model is None:
            raise ModelError("Model not trained or loaded")
        results = self.model.evaluate(X, y, verbose=0, return_dict=True)
        logger.info("forecaster_evaluated", **results)
        return results

    def persistence_baseline(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute persistence (naive) baseline: last known value."""
        # Last timestep of input sequence
        last_vals = X[:, -1, :]
        mse = np.mean((last_vals - y) ** 2)
        mae = np.mean(np.abs(last_vals - y))
        return {"mse": float(mse), "mae": float(mae)}

    def save(self, path: str) -> None:
        """Save model and scaler."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        self.model.save(path_obj / "forecaster.keras")
        self.scaler.save(path_obj / "scaler.joblib")
        with (path_obj / "config.json").open("w") as f:
            json.dump(self.config, f)
        logger.info("forecaster_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "SensorForecaster":
        """Load model and scaler."""
        path_obj = Path(path)
        obj = cls()
        obj.model = models.load_model(path_obj / "forecaster.keras")
        obj.scaler = FeatureScaler.load(path_obj / "scaler.joblib")
        with (path_obj / "config.json").open() as f:
            obj.config = json.load(f)
        logger.info("forecaster_loaded", path=path)
        return obj
