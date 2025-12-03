from __future__ import annotations
import numpy as np
import pandas as pd

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def treat_outliers(df: pd.DataFrame, method: str = "winsorize") -> pd.DataFrame:
    """
    Treats outliers using IQR-based winsorization or clipping.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = s.clip(lower=lower, upper=upper)

    logger.info("[preprocessing] Outlier treatment (%s) applied to %d columns",
                method, len(num_cols))
    return df
