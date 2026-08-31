import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from mfp.core.config import settings
from mfp.core.exceptions import DataValidationError
from mfp.core.logging import get_logger

logger = get_logger(__name__)

FEATURE_COLS = settings.data.sensor_columns


class FeatureScaler:
    """Wrapper around MinMaxScaler with train-only fit and persistence."""

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.feature_names = FEATURE_COLS

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler on training data only."""
        self.scaler.fit(df[self.feature_names].values)
        self.is_fitted = True
        logger.info("scaler_fitted", n_samples=len(df), features=self.feature_names)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise DataValidationError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(df[self.feature_names].values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data."""
        if not self.is_fitted:
            raise DataValidationError("Scaler not fitted.")
        return self.scaler.inverse_transform(arr)

    def save(self, path: str) -> None:
        """Save scaler to disk."""
        joblib.dump({"scaler": self.scaler, "features": self.feature_names}, path)
        logger.info("scaler_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load scaler from disk."""
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.feature_names = data["features"]
        obj.is_fitted = True
        logger.info("scaler_loaded", path=path)
        return obj

    def get_params(self) -> dict:
        """Get scaler parameters for verification."""
        return {
            "min": self.scaler.data_min_.tolist() if hasattr(self.scaler, "data_min_") else None,
            "max": self.scaler.data_max_.tolist() if hasattr(self.scaler, "data_max_") else None,
            "scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            "features": self.feature_names,
        }
