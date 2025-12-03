# src/crispml/common/quality/range_validation.py

"""
Range validation (example: age 0–120).
"""

from __future__ import annotations
import pandas as pd

# package imports COMMON LOGGING -LOGGING UTILS
from src.crispml.common.logging.logging_utils import get_logger
# package imports COMMON OUTPUT - OUTPUT UTILS
from src.crispml.common.output.table_utils import save_table_as_image

logger = get_logger(__name__)


def validate_range(
        df: pd.DataFrame,
        column: str,
        min_val: float,
        max_val: float,
        phase_name: str = "phase2_data_understanding"
):
    """
    Detects values outside a valid range.
    """

    if column not in df.columns:
        logger.warning("[QUALITY][RANGE] Column '%s' not found.", column)
        return pd.DataFrame()

    bad = df[(df[column] < min_val) | (df[column] > max_val)]

    logger.info("[QUALITY][RANGE] Column '%s' out-of-range values: %d", column, len(bad))

    if not bad.empty:
        save_table_as_image(bad.head(20), f"{column}_range_violation.png", phase_name)

    return bad
