import os
import random

import numpy as np
import tensorflow as tf


def set_global_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.config.experimental.enable_op_determinism()


def get_tf_config() -> tf.compat.v1.ConfigProto:
    """Get TF config for deterministic ops."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return config