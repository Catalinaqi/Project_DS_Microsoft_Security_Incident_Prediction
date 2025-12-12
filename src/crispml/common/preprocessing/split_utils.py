from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from src.crispml.config.enums.enums_config import ProblemType
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def train_val_test_split(
        X: np.ndarray,
        y: Optional[np.ndarray],
        problem_type: ProblemType,
        test_size: float = 0.2,
        val_size: float = 0.0,
        random_state: int = 42,
        time_order=None,
) -> Tuple:
    """
    Performs train/val/test splitting for supervised and unsupervised tasks.
    """
    n = X.shape[0]
    if n == 0:
        raise ValueError("X is empty.")

    logger.info("[preprocessing] Train/val/test split: n=%d", n)

    # Time-series split
    if problem_type == ProblemType.TIME_SERIES and time_order is not None:
        logger.info("[preprocessing] Time-series split applied.")
        order = np.argsort(np.asarray(time_order))
    else:
        rng = np.random.RandomState(random_state)
        order = np.arange(n)
        rng.shuffle(order)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val] if n_val > 0 else np.array([])
    test_idx = order[n_train + n_val :]

    X_train = X[train_idx]
    X_val = X[val_idx] if len(val_idx) > 0 else None
    X_test = X[test_idx]

    if y is None:
        logger.info("[preprocessing] Unsupervised split (y=None).")
        return X_train, X_val, X_test, None, None, None

    y = np.asarray(y)
    y_train = y[train_idx]
    y_val = y[val_idx] if len(val_idx) > 0 else None
    y_test = y[test_idx]


    return X_train, X_val, X_test, y_train, y_val, y_test
