import pandas as pd
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def filter_columns_for_boxplot(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    """
    Filters numeric columns to avoid ID-like fields, hashed values, high-cardinality
    variables and degenerate columns that break or slow down EDA visualizations.

    This is used *only for Phase 2 Data Understanding* (EDA) to improve
    plot readability and performance. The original dataframe remains unchanged.

    Filtering logic:
        1. Exclude ID-like fields (based on column name patterns).
        2. Exclude columns with no variability (unique <= 1).
        3. Exclude extremely high-cardinality fields (likely disguised IDs).
        4. Keep only meaningful numeric features for EDA boxplots.

    Parameters
    ----------
    df : pd.DataFrame
        Full original dataset.
    numeric_cols : list[str]
        List of numeric column names detected by dtype.

    Returns
    -------
    list[str]
        Filtered list of numeric columns suitable for boxplot visualization.
    """

    logger.info("[VIS][BOX] Applying filtering rules to numeric columns for boxplot generation...")

    filtered = []
    total_rows = len(df)

    for col in numeric_cols:
        col_lower = col.lower()

        # ----------------------------------------------------
        # 1. Exclude ID-like columns based on name patterns
        # ----------------------------------------------------
        if any(key in col_lower for key in ["id", "hash", "key", "path", "sha"]):
            logger.debug(f"[VIS][BOX] Excluding column '{col}' (ID-like pattern detected).")
            logger.info(f"[VIS][BOX] Excluding column '{col}' (ID-like pattern detected).")
            continue

        # Count unique values once
        nunique = df[col].nunique()

        # ----------------------------------------------------
        # 2. Exclude columns with no variability
        # ----------------------------------------------------
        if nunique <= 1:
            logger.debug(f"[VIS][BOX] Excluding column '{col}' (no variance: nunique={nunique}).")
            logger.info(f"[VIS][BOX] Excluding column '{col}' (no variance: nunique={nunique}).")
            continue

        # ----------------------------------------------------
        # 3. Exclude extremely high-cardinality columns
        #    These are often IDs disguised as numeric fields.
        # ----------------------------------------------------
        if nunique > 0.5 * total_rows:
            logger.debug(f"[VIS][BOX] Excluding column '{col}' (high cardinality: nunique={nunique}).")
            logger.info(f"[VIS][BOX] Excluding column '{col}' (high cardinality: nunique={nunique}).")
            continue

        # ----------------------------------------------------
        # 4. Column is valid for visualization
        # ----------------------------------------------------
        filtered.append(col)

    logger.info(
        f"[VIS][BOX] Filtering completed. "
        f"Original numeric columns: {len(numeric_cols)}, "
        f"Filtered columns: {len(filtered)}."
    )

    return filtered
