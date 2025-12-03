from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.crispml.common.logging.logging_utils import get_logger
logger = get_logger(__name__)


def scale_features(df: pd.DataFrame, scaling: str = "standard") -> np.ndarray:
    """
    Applies numerical feature scaling.
    """
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")

    X = df.select_dtypes(include=["number"]).values

    logger.info("[preprocessing] Scaling '%s' applied to shape %s", scaling, X.shape)
    return scaler.fit_transform(X)
