# src/crispml/common/quality/duplicates_analysis_quality_utils.py

"""
Duplicate detection (Phase 2).
"""

from __future__ import annotations
import pandas as pd

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.output import save_table_as_image

from src.crispml.config.enums.enums_config import ProblemType
from src.crispml.config.enums.enums_config import PhaseName
PHASE_NAME = PhaseName.PHASE2_DATA_UNDERSTANDING

logger = get_logger(__name__)


def analyze_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects duplicated rows and shows a small sample.
    """
    dup = df[df.duplicated(keep=False)]
    logger.info("[QUALITY][DUPLICATES] Found %d duplicated rows.", len(dup))

    if dup.empty:
        return pd.DataFrame({"info": ["No duplicates found"]})

    return dup


#from src.crispml.common.visualization.table_export import save_table_as_image


#
#
# def analyze_duplicates(
#         df: pd.DataFrame,
#         phase_name: PhaseName = PHASE_NAME,
#         problem_type: ProblemType | str = None,
# ) -> pd.DataFrame:
#     """
#     Analyze duplicate rows in the dataframe.
#     Saves a sample of duplicates as an image table.
#     """
#     dup = df[df.duplicated(keep=False)]
#
#     if dup.empty:
#         return pd.DataFrame({"info": ["No duplicates found"]})
#
#     sample = dup.head(10)
#
#     # --- aquí convertimos el enum a string correctamente ---
#     if isinstance(problem_type, ProblemType):
#         problem_type_str = problem_type.name.lower()
#     else:
#         problem_type_str = str(problem_type).lower()
#
#     save_table_as_image(
#         sample,
#         f"{problem_type_str}_duplicates_found.png",
#         phase_name,
#     )
#
#     return dup
#
