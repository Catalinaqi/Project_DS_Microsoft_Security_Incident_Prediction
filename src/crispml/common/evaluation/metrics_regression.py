# src/crispml/common/evaluation/metrics_regression.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)

def compute_regression_metrics(models, X_test, y_test):
    """
    Computes regression metrics: MSE, RMSE, MAE, R2.
    """
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rows.append({
            "model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        })

        logger.info(
            "[eval] Regressor %s: RMSE=%.4f MAE=%.4f R2=%.4f",
            name, rmse, mae, r2
        )

    return pd.DataFrame(rows)
