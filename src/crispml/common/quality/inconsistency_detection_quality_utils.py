# # src/crispml/common/quality/inconsistency_detection_quality_utils.py
#
# """
# Detects inconsistencies: impossible dates, negative values, etc.
# """
#
# from __future__ import annotations
# import pandas as pd
# import numpy as np
#
# from src.crispml.common.logging.logging_utils import get_logger
# from src.crispml.common.output import save_table_as_image
# from src.crispml.config.enums import PhaseName
#
# phase_name_num = PhaseName.PHASE2_DATA_UNDERSTANDING
# logger = get_logger(__name__)
#
#
# def detect_inconsistencies(df: pd.DataFrame,
#                            phase_name: PhaseName = phase_name_num
#                            #phase_name: str = "phase2_data_understanding"
#                            ) -> pd.DataFrame:
#     """
#     Basic inconsistency detection: negative numeric values, impossible dates.
#     Returns a DataFrame, NEVER a dict.
#     """
#
#     # Detect negative numeric values
#     num_cols = df.select_dtypes(include=np.number)
#     neg = num_cols[num_cols < 0].dropna(how="all")
#
#     # Logging
#     logger.info("[QUALITY][INCONS] Negative values detected: %d rows", neg.shape[0])
#
#     # Export preview if exists
#     if not neg.empty:
#         save_table_as_image(
#             neg.head(20),
#             filename="negative_values.png",
#             subfolder=phase_name
#         )
#
#     # Return DataFrame for Phase 2
#     # (Phase 2 requires a DataFrame to use `.empty`)
#     return neg
#
#

from __future__ import annotations

import pandas as pd
import numpy as np

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.config.enums import PhaseName

logger = get_logger(__name__)

DEFAULT_PHASE = PhaseName.PHASE2_DATA_UNDERSTANDING


def detect_inconsistencies(
        df: pd.DataFrame,
        rules: dict | None = None,
        phase_name: PhaseName = DEFAULT_PHASE
) -> pd.DataFrame:
    """
    Generic inconsistency detection engine.

    This function is DOMAIN-AGNOSTIC and TEMPLATE-LEVEL.
    It executes inconsistency rules provided by the project configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (never modified).

    rules : dict
        Dictionary defining inconsistency rules.
        Structure is project-specific.

    phase_name : PhaseName
        Current CRISP-ML phase (used only for logging/output).

    Returns
    -------
    pd.DataFrame
        Rows violating at least one inconsistency rule.
        Empty DataFrame if no inconsistencies are detected.
    """

    if not rules:
        logger.info("[QUALITY][INCONS] No inconsistency rules provided. Skipping.")
        return pd.DataFrame()

    issues: list[pd.DataFrame] = []

    for rule_name, rule in rules.items():
        rule_type = rule.get("type")
        column = rule.get("column")

        if column not in df.columns:
            logger.warning(
                "[QUALITY][INCONS] Rule '%s' skipped: column '%s' not found",
                rule_name, column
            )
            continue

        # --------------------------------------------------
        # Numeric rules
        # --------------------------------------------------
        if rule_type == "numeric_non_negative":
            invalid = df[df[column] < 0]

        # --------------------------------------------------
        # Datetime rules
        # --------------------------------------------------
        elif rule_type == "datetime_not_future":
            ts = pd.to_datetime(df[column], errors="coerce", utc=True)
            invalid = df[ts > pd.Timestamp.utcnow()]

        elif rule_type == "datetime_parseable":
            ts = pd.to_datetime(df[column], errors="coerce")
            invalid = df[ts.isna()]

        # --------------------------------------------------
        # Categorical rules
        # --------------------------------------------------
        elif rule_type == "not_null":
            invalid = df[df[column].isna()]

        else:
            logger.warning(
                "[QUALITY][INCONS] Unknown rule type '%s' for rule '%s'",
                rule_type, rule_name
            )
            continue

        if not invalid.empty:
            invalid = invalid.copy()
            invalid["__issue_rule"] = rule_name
            invalid["__issue_type"] = rule_type
            issues.append(invalid)

        logger.info(
            "[QUALITY][INCONS] Rule '%s' detected %d inconsistent rows",
            rule_name, invalid.shape[0]
        )

    if issues:
        return pd.concat(issues, axis=0).drop_duplicates()

    return pd.DataFrame()
