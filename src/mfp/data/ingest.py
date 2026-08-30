import pandas as pd

from mfp.core.config import settings
from mfp.core.exceptions import DataValidationError
from mfp.core.logging import get_logger
from mfp.data.validate import (
    validate_sensor_data,
    check_temporal_order,
    check_missing_values,
    check_sensor_stuck,
)
from mfp.data.label import add_risk_labels, label_sanity_report

logger = get_logger(__name__)


def ingest_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load raw sensor data from CSV."""
    data_path = path or settings.data.raw_data_path
    df = pd.read_csv(data_path)
    logger.info("raw_data_loaded", path=str(data_path), shape=df.shape, columns=list(df.columns))
    return df


def preprocess_pipeline(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full preprocessing pipeline: validate -> clean -> label."""
    if df is None:
        df = ingest_raw_data()

    df = validate_sensor_data(df)
    df = check_temporal_order(df)
    df = check_missing_values(df)
    df = check_sensor_stuck(df)
    df = add_risk_labels(df)

    report = label_sanity_report(df)
    logger.info("label_sanity_report", **report)

    return df