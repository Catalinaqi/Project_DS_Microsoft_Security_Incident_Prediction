# src/crispml/common/quality/missing_analysis_quality_utils.py

"""
Missing values diagnostic (Phase 2).
This module computes missing-value statistics WITHOUT modifying the dataset.
"""

from __future__ import annotations
import pandas as pd

from src.crispml.common.logging.logging_utils import get_logger


logger = get_logger(__name__)


def analyze_missing_values(df: pd.DataFrame, phase_name: str = "phase2_data_understanding"):
    """
    Computes percentage of missing values per column and exports it as an image.
    """
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100

    result = missing_pct.to_frame(name="Missing (%)")
    # Convert index → real column
    result = result.reset_index().rename(columns={"index": "Column"})

    logger.info("[QUALITY][MISSING] Missing-value analysis completed.")

    return result
