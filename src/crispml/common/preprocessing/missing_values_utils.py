from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def drop_high_nan_columns(df: pd.DataFrame, max_nan_ratio: float = 0.8) -> pd.DataFrame:
    """
    Removes columns with too many missing values.
    """
    nan_ratio = df.isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > max_nan_ratio].index.tolist()

    if cols_to_drop:
        logger.info("[preprocessing] Dropping high-NaN columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def simple_imputation(
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """
    Simple missing-value imputation using median/mode strategies.
    """
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if len(num_cols):
        imputer = SimpleImputer(strategy=numeric_strategy)
        df[num_cols] = imputer.fit_transform(df[num_cols])
        logger.info("[preprocessing] Numeric imputation on %d columns", len(num_cols))

    if len(cat_cols):
        imputer = SimpleImputer(strategy=categorical_strategy)
        df[cat_cols] = imputer.fit_transform(df[cat_cols])
        logger.info("[preprocessing] Categorical imputation on %d columns", len(cat_cols))

    return df
