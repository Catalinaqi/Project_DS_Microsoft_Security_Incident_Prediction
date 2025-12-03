# src/crispml/common/feature_selection/feature_selection_utils.py

"""
Feature selection utilities for CRISP-ML.
These functions are used during Phase 2 (EDA) and Phase 3 (Data Preparation).

Responsibilities:
-----------------
- Removing ID / target / time columns
- Applying AUTO / INCLUDE / EXCLUDE feature strategies
- Identifying numeric vs categorical columns
- Enforcing max_features limit
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import pandas as pd

# package imports CONFIG DATASET
from src.crispml.config.dataset.dataset_config import DatasetConfig
# package imports CONFIG DATASET
from src.crispml.config.dataset.feature_config import FeatureConfig
# package imports COMMON LOGGING
from src.crispml.config.enums.enums import FeatureSelectionMode

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------
def _infer_feature_columns(
        df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
) -> List[str]:
    """
    Determines which columns should be used as features based on:
        - DatasetConfig (target, time, ID columns)
        - FeatureConfig (AUTO, INCLUDE, EXCLUDE)

    Only columns present in the DataFrame are returned.
    """
    all_cols = list(df.columns)

    target = dataset_cfg.target_col
    time_col = dataset_cfg.time_col
    id_cols = dataset_cfg.id_cols

    excluded = set()

    # Always remove target, time, and ID columns
    if target and target in all_cols:
        excluded.add(target)
    if time_col and time_col in all_cols:
        excluded.add(time_col)
    for c in id_cols:
        if c in all_cols:
            excluded.add(c)

    # -------------------------------------------------------------
    # INCLUDE MODE
    # -------------------------------------------------------------
    if feature_cfg.mode == FeatureSelectionMode.INCLUDE and feature_cfg.include:
        feat_cols = [
            c for c in feature_cfg.include
            if c in all_cols and c not in excluded
        ]

    # -------------------------------------------------------------
    # AUTO / EXCLUDE MODE
    # -------------------------------------------------------------
    else:
        feat_cols = [c for c in all_cols if c not in excluded]

        # Apply exclude list explicitly
        if feature_cfg.mode == FeatureSelectionMode.EXCLUDE and feature_cfg.exclude:
            feat_cols = [c for c in feat_cols if c not in feature_cfg.exclude]

    # Max feature limit
    if len(feat_cols) > feature_cfg.max_features:
        logger.info(
            "[feature_selection] Feature count (%d) > max_features (%d). Keeping only first %d.",
            len(feat_cols), feature_cfg.max_features, feature_cfg.max_features,
        )
        feat_cols = feat_cols[: feature_cfg.max_features]

    return feat_cols


def _split_numeric_categorical(
        df: pd.DataFrame,
        feature_cols: List[str],
        max_unique_for_cat: int,
) -> Tuple[List[str], List[str]]:
    """
    Splits feature columns into:
        - numeric (int, float)
        - categorical (object, bool, category)
    """

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        dtype = df[col].dtype

        if np.issubdtype(dtype, np.number):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


# ---------------------------------------------------------------------
# PUBLIC API — THREE MAIN ENTRY POINTS
# ---------------------------------------------------------------------
def select_features_auto(
        df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    AUTO MODE:
    Uses all columns (except target/time/IDs) as features.
    """
    feature_cfg.mode = FeatureSelectionMode.AUTO
    return _select_features_core(df, dataset_cfg, feature_cfg)


def select_features_include(
        df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    INCLUDE MODE:
    Uses only columns listed in FeatureConfig.include.
    """
    feature_cfg.mode = FeatureSelectionMode.INCLUDE
    return _select_features_core(df, dataset_cfg, feature_cfg)


def select_features_exclude(
        df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    EXCLUDE MODE:
    Uses all columns except those in FeatureConfig.exclude.
    """
    feature_cfg.mode = FeatureSelectionMode.EXCLUDE
    return _select_features_core(df, dataset_cfg, feature_cfg)


# ---------------------------------------------------------------------
# INTERNAL CORE LOGIC (shared by AUTO / INCLUDE / EXCLUDE)
# ---------------------------------------------------------------------
def _select_features_core(
        df: pd.DataFrame,
        dataset_cfg: DatasetConfig,
        feature_cfg: FeatureConfig,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Core implementation of feature selection logic.
    Shared by AUTO, INCLUDE, EXCLUDE wrappers.
    """

    logger.info("[feature_selection] Starting feature selection...")

    feat_cols = _infer_feature_columns(df, dataset_cfg, feature_cfg)
    logger.info("[feature_selection] Candidate features: %s", feat_cols)

    numeric_cols, categorical_cols = _split_numeric_categorical(
        df, feat_cols, feature_cfg.max_unique_for_cat
    )

    df_features = df[feat_cols].copy()

    logger.info(
        "[feature_selection] Selected %d features (%d numeric, %d categorical).",
        len(feat_cols),
        len(numeric_cols),
        len(categorical_cols),
    )

    return df_features, numeric_cols, categorical_cols
