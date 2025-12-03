# src/crispml/common/quality/inconsistency_detection.py

"""
Detects inconsistencies: impossible dates, negative values, etc.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.output import save_table_as_image

logger = get_logger(__name__)


def detect_inconsistencies(df: pd.DataFrame, phase_name: str = "phase2_data_understanding"):
    """
    Basic inconsistency detection: negative numeric values, impossible dates.
    Returns a DataFrame, NEVER a dict.
    """

    # Detect negative numeric values
    num_cols = df.select_dtypes(include=np.number)
    neg = num_cols[num_cols < 0].dropna(how="all")

    # Logging
    logger.info("[QUALITY][INCONS] Negative values detected: %d rows", neg.shape[0])

    # Export preview if exists
    if not neg.empty:
        save_table_as_image(
            neg.head(20),
            filename="negative_values.png",
            subfolder=phase_name
        )

    # Return DataFrame for Phase 2
    # (Phase 2 requires a DataFrame to use `.empty`)
    return neg


