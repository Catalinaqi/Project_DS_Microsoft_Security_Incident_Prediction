# src/crispml/common/preprocessing/preprocessing_utils.py

"""
High-level preprocessing orchestrator for CRISP-ML.

This module provides unified preprocessing pipelines required for
Phase 2 (Data Understanding) and Phase 3 (Data Preparation).

It integrates specialized utilities from:
    - missing_values_utils
    - outlier_utils
    - categorical_utils
    - scaling_utils
    - split_utils

The goal is to centralize the coordination logic while keeping the
individual transformations in their own dedicated modules.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Tuple

from src.crispml.common.preprocessing.missing_values_utils import (
    drop_high_nan_columns,
    simple_imputation,
)
from src.crispml.common.preprocessing.outlier_utils import treat_outliers
from src.crispml.common.preprocessing.categorical_utils import encode_category_utils
from src.crispml.common.preprocessing.scaling_utils import scale_features
from src.crispml.common.preprocessing.split_utils import train_val_test_split

from src.crispml.config.enums.enums_config import ProblemType
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# PHASE 2 — LIGHT PREPROCESSING (EDA)
# ---------------------------------------------------------------------
def phase2_quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight preprocessing used during Phase 2 EDA:
        - remove extreme NaN columns
        - simple imputation
        - mild outlier treatment

    Heavy transformations (encoding, scaling) are *not* done here
    to preserve the original structure for human interpretation.
    """
    logger.info("[preprocessing] Phase 2 quick cleaning started.")

    df = drop_high_nan_columns(df, max_nan_ratio=0.8)
    df = simple_imputation(df)
    df = treat_outliers(df)

    logger.info("[preprocessing] Phase 2 quick cleaning completed with shape %s", df.shape)
    return df


# ---------------------------------------------------------------------
# PHASE 3 — FULL PREPROCESSING PIPELINE
# ---------------------------------------------------------------------
def phase3_full_pipeline(
        df: pd.DataFrame,
        target_col: Optional[str],
        problem_type: ProblemType,
        scaling: str = "standard",
) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
    """
    Complete preprocessing pipeline for Phase 3:

        1. Drop columns with too many NaNs
        2. Simple imputation (can be replaced by advanced imputation)
        3. Outlier treatment
        4. Encode categorical variables
        5. Extract target vector (if supervised)
        6. Scale numerical features
        7. Return (X, y, feature_names)

    This pipeline is intentionally kept generic — actual steps
    are activated/deactivated using Phase3Techniques.
    """

    logger.info("[preprocessing] Phase 3 full pipeline started.")

    df = drop_high_nan_columns(df)
    df = simple_imputation(df)
    df = treat_outliers(df)
    df = encode_category_utils(df, target_col=target_col)

    # Extract target if supervised
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].values
        df = df.drop(columns=[target_col])
        logger.info("[preprocessing] Target column '%s' extracted.", target_col)
    else:
        logger.info("[preprocessing] No target column to extract.")

    # Numeric scaling
    X = scale_features(df, scaling=scaling)
    feature_names = df.columns.tolist()

    logger.info("[preprocessing] Phase 3 preprocessing completed. X shape: %s", X.shape)
    return X, y, feature_names


# ---------------------------------------------------------------------
# SPLITTING PIPELINE
# ---------------------------------------------------------------------
def split_dataset(
        X: np.ndarray,
        y: Optional[np.ndarray],
        problem_type: ProblemType,
        test_size: float = 0.2,
        val_size: float = 0.0,
        random_state: int = 42,
        time_order=None,
):
    """
    Wrapper for train/val/test splitting.
    """
    logger.info("[preprocessing] Performing train/val/test split.")

    result = train_val_test_split(
        X=X,
        y=y,
        problem_type=problem_type,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        time_order=time_order,
    )

    logger.info("[preprocessing] Dataset split done.")
    return result
