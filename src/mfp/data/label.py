import pandas as pd
import numpy as np

from mfp.core.config import settings
from mfp.core.logging import get_logger

logger = get_logger(__name__)


def add_risk_labels(
    df: pd.DataFrame,
    temp_threshold: float | None = None,
    press_threshold: float | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """
    Add horizon-based risk labels.
    
    risk[t] = 1 if a threshold breach (failure) occurs in (t, t+horizon].
    
    This is the standard predictive maintenance framing from §2.3.
    """
    temp_thresh = temp_threshold or settings.data.temp_threshold
    press_thresh = press_threshold or settings.data.press_threshold
    h = horizon or settings.data.horizon

    df = df.sort_index().reset_index(drop=True).copy()

    failure = (
        (df["Temperature1"] > temp_thresh) | (df["Pressure1"] > press_thresh)
    ).astype(int)

    df["risk"] = (
        failure.rolling(window=h, min_periods=1).max().shift(-h).fillna(0).astype(int)
    )

    risk_rate = df["risk"].mean()
    logger.info(
        "risk_labels_added",
        horizon=h,
        temp_threshold=temp_thresh,
        press_threshold=press_thresh,
        risk_rate=float(risk_rate),
        positive_samples=int(df["risk"].sum()),
    )

    return df


def label_sanity_report(df: pd.DataFrame) -> dict:
    """Generate label sanity report: distribution, temporal autocorrelation."""
    risk = df["risk"].values
    n = len(risk)
    pos_rate = risk.mean()

    autocorr = []
    for lag in [1, 5, 10, 30, 60]:
        if n > lag:
            corr = np.corrcoef(risk[:-lag], risk[lag:])[0, 1]
            autocorr.append({"lag": lag, "autocorr": float(corr) if not np.isnan(corr) else 0.0})

    clusters = []
    in_cluster = False
    cluster_len = 0
    for r in risk:
        if r == 1 and not in_cluster:
            in_cluster = True
            cluster_len = 1
        elif r == 1 and in_cluster:
            cluster_len += 1
        elif r == 0 and in_cluster:
            clusters.append(cluster_len)
            in_cluster = False
            cluster_len = 0
    if in_cluster:
        clusters.append(cluster_len)

    return {
        "positive_rate": float(pos_rate),
        "total_samples": n,
        "positive_samples": int(risk.sum()),
        "autocorrelation": autocorr,
        "cluster_sizes": clusters,
        "num_clusters": len(clusters),
        "avg_cluster_size": float(np.mean(clusters)) if clusters else 0.0,
    }