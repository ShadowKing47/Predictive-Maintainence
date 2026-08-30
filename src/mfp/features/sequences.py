import numpy as np
import pandas as pd

from mfp.core.config import settings
from mfp.core.exceptions import SplitError
from mfp.core.logging import get_logger

logger = get_logger(__name__)


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.
    
    X[i] = data[i : i + seq_len]
    y[i] = data[i + seq_len + horizon - 1]  # forecast horizon steps ahead
    """
    n_samples = len(data) - seq_len - horizon + 1
    if n_samples <= 0:
        raise SplitError(f"Not enough data for seq_len={seq_len}, horizon={horizon}")

    X = np.lib.stride_tricks.sliding_window_view(data, window_shape=(seq_len, data.shape[1]))
    X = X[:n_samples]

    y = data[seq_len + horizon - 1 : seq_len + horizon - 1 + n_samples]

    return X, y


def temporal_split_with_purge(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_gap: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Split sequences temporally with purge gap between splits.
    
    This prevents leakage by ensuring train/val/test sequences don't
    share timestamps within the purge gap (seq_len + horizon).
    
    Returns: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    total = len(X)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Apply purge gaps
    train_end = max(0, train_end - purge_gap)
    val_start = train_end + purge_gap
    val_end = max(val_start, val_end - purge_gap)
    test_start = val_end + purge_gap

    if test_start >= total:
        raise SplitError(
            f"Purge gap too large: test_start={test_start} >= total={total}. "
            f"Reduce purge_gap or increase data size."
        )

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[val_start:val_end], y[val_start:val_end]
    X_test, y_test = X[test_start:], y[test_start:]

    logger.info(
        "temporal_split_done",
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
        purge_gap=purge_gap,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def check_split_leakage(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    seq_len: int,
    purge_gap: int,
) -> bool:
    """
    Verify no temporal leakage between splits.
    
    Checks that the last timestamp in train + purge_gap <= first timestamp in val,
    and similarly for val -> test.
    """
    train_last_idx = len(X_train) + seq_len - 1
    val_first_idx = len(X_train) + purge_gap
    val_last_idx = val_first_idx + len(X_val) + seq_len - 1
    test_first_idx = val_first_idx + len(X_val) + purge_gap

    train_val_ok = train_last_idx < val_first_idx
    val_test_ok = val_last_idx < test_first_idx

    logger.info(
        "leakage_check",
        train_last_idx=train_last_idx,
        val_first_idx=val_first_idx,
        val_last_idx=val_last_idx,
        test_first_idx=test_first_idx,
        train_val_ok=train_val_ok,
        val_test_ok=val_test_ok,
    )

    return train_val_ok and val_test_ok