import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from mfp.core.config import settings
from mfp.core.exceptions import DataValidationError
from mfp.core.logging import get_logger

logger = get_logger(__name__)


def get_sensor_schema() -> DataFrameSchema:
    """Get Pandera schema for sensor data validation."""
    checks = []
    for col in settings.data.sensor_columns:
        checks.append(Column(col, float, Check.greater_than(-1000), Check.less_than(10000)))

    return DataFrameSchema(
        {
            **{col: Column(float, Check.greater_than(-1000), Check.less_than(10000)) for col in settings.data.sensor_columns},
        },
        strict=True,
        coerce=True,
    )


def validate_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate sensor data against schema."""
    schema = get_sensor_schema()
    try:
        validated = schema.validate(df, lazy=True)
        logger.info("data_validation_passed", rows=len(validated), columns=list(validated.columns))
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error("data_validation_failed", errors=exc.failure_cases.to_dict(orient="records"))
        raise DataValidationError(f"Data validation failed: {exc}") from exc


def check_temporal_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure data is temporally ordered (no timestamp column, so assume row order)."""
    return df.reset_index(drop=True)


def check_missing_values(df: pd.DataFrame, max_missing_ratio: float = 0.05) -> pd.DataFrame:
    """Check for missing values."""
    missing = df.isnull().sum()
    if (missing / len(df) > max_missing_ratio).any():
        bad_cols = missing[missing / len(df) > max_missing_ratio].index.tolist()
        raise DataValidationError(f"Columns exceed missing ratio threshold: {bad_cols}")

    return df.ffill().bfill()


def check_sensor_stuck(df: pd.DataFrame, window: int = 100, min_std: float = 0.01) -> pd.DataFrame:
    """Detect stuck sensors (constant values)."""
    for col in settings.data.sensor_columns:
        rolling_std = df[col].rolling(window=window, min_periods=window).std()
        stuck_mask = rolling_std < min_std
        if stuck_mask.any():
            logger.warning("sensor_stuck_detected", column=col, count=int(stuck_mask.sum()))
    return df
