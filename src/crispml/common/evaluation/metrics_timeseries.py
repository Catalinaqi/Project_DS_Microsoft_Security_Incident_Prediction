# src/crispml/common/evaluation/metrics_timeseries.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)

def compute_ts_metrics(y_true, y_pred):
    """
    Computes TS metrics: MAE, RMSE, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

    logger.info(
        "[eval] TS metrics: MAE=%.4f RMSE=%.4f MAPE=%.2f%%",
        mae, rmse, mape
    )

    return pd.DataFrame({
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE": [mape],
    })
