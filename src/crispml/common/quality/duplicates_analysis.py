# src/crispml/common/quality/duplicates_analysis.py

"""
Duplicate detection (Phase 2).
"""

from __future__ import annotations
import pandas as pd

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.output import save_table_as_image

logger = get_logger(__name__)


def analyze_duplicates(df: pd.DataFrame, phase_name: str = "phase2_data_understanding"):
    """
    Detects duplicated rows and shows a small sample.
    """
    dup = df[df.duplicated()]

    logger.info("[QUALITY][DUPLICATES] Found %d duplicated rows.", len(dup))

    if dup.empty:
        return pd.DataFrame({"info": ["No duplicates found"]})

    sample = dup.head(10)
    save_table_as_image(sample, "duplicates_found.png", phase_name)

    return dup
