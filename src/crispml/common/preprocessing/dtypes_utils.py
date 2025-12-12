# src/crispml/common/preprocessing/dtypes_utils.py

import numpy as np
import pandas as pd

ID_KEYWORDS = [
    "id", "hash", "key", "path", "token", "sha", "guid"
]


def is_id_column(col_name: str) -> bool:
    """
    Detects ID-like columns based on common naming patterns.
    """
    col_lower = col_name.lower()
    return any(keyword in col_lower for keyword in ID_KEYWORDS)


def get_real_numeric_columns(df: pd.DataFrame, max_unique_ratio: float = 0.5):
    """
    Returns only the truly meaningful numeric columns:
        - excludes ID-like columns
        - excludes columns with 1 unique value
        - excludes columns with extremely high cardinality (ID disguised as numeric)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    real_numeric = []

    for col in numeric_cols:

        # Remove IDs
        if is_id_column(col):
            continue

        nunique = df[col].nunique()

        # Remove degenerate columns
        if nunique <= 1:
            continue

        # Remove numeric columns that are actually IDs (too many unique values)
        if nunique > max_unique_ratio * len(df):
            continue

        real_numeric.append(col)

    return real_numeric


def get_categorical_columns(df: pd.DataFrame):
    """
    Detects columns that should be treated as categorical, even if dtype is numeric.
    """
    cat_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            cat_cols.append(col)
        elif df[col].dtype in ["int64", "float64"]:
            # numeric but low cardinality → treat as category
            if df[col].nunique() < 50:
                cat_cols.append(col)

    return cat_cols
