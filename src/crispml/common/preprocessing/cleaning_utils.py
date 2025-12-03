"""
Cleaning utilities for Phase 3 – Data Preparation

Contains:
- remove_duplicates()
- apply_log_transform()

These are REAL transformations applied to the dataset.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows and returns a cleaned DataFrame.
    """
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)

    logger.info("[PREP][DUPLICATES] Removed %d duplicate rows.", before - after)
    return df_clean


# ---------------------------------------------------------
# APPLY LOG TRANSFORM
# ---------------------------------------------------------
def apply_log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Applies log1p transform to selected columns.
    Uses np.log1p to handle zeros safely.
    """
    if not columns:
        return df

    df_new = df.copy()

    for col in columns:
        if col not in df_new.columns:
            continue

        if (df_new[col] < 0).any():
            logger.warning("[PREP][LOG] Column '%s' contains negative values, skipping.", col)
            continue

        df_new[col] = np.log1p(df_new[col])
        logger.info("[PREP][LOG] Applied log-transform to column '%s'.", col)

    return df_new
