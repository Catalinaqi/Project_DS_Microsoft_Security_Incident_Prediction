# src/crispml/common/quality/range_validation_quality_utils.py

"""
Range validation (example: age 0–120).
"""

from __future__ import annotations
import pandas as pd

# package imports COMMON LOGGING -LOGGING UTILS
from src.crispml.common.logging.logging_utils import get_logger
# package imports COMMON OUTPUT - OUTPUT UTILS
from src.crispml.common.output.table_utils import save_table_as_image
from src.crispml.config.enums.enums_config import PhaseName
#PHASE_NAME = PhaseName.PHASE2_DATA_UNDERSTANDING
#phase_name_str: str = PhaseName.PHASE2_DATA_UNDERSTANDING.name.lower()

logger = get_logger(__name__)


def validate_range(
        df: pd.DataFrame,
        column: str,
        min_val: float,
        max_val: float,
        phase_name: PhaseName
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
