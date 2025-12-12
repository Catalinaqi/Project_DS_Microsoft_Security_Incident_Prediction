from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def encode_category_utils(
        df: pd.DataFrame,
        encoding: str = "onehot",
        target_col: Optional[str] = None,
        max_onehot_cardinality: int = 20,
) -> pd.DataFrame:
    """
    Encodes categorical variables in a memory-safe way.

    Strategy (for encoding in {"onehot", "hybrid"}):
    - Categorical columns with <= max_onehot_cardinality distinct values:
        → One-Hot (dense, drop="first")
    - Categorical columns with  > max_onehot_cardinality distinct values:
        → Ordinal encoding (integer index per category: col__idx)
    """
    if encoding not in {"onehot", "hybrid"}:
        raise NotImplementedError(
            "Only 'onehot' (hybrid) encoding is currently supported."
        )

    df = df.copy()

    # Separate target (if provided) to avoid encoding it
    y = None
    if target_col is not None and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df

    # Categorical columns (non-numerical)
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    if not cat_cols:
        logger.info("[preprocessing] No categorical columns to encode.")
        # Re-attach target if we had separated it
        if y is not None:
            X[target_col] = y
        return X

    # Cardinality for each categorical column
    cardinalities = {col: X[col].nunique(dropna=True) for col in cat_cols}
    low_card_cols = [
        c for c in cat_cols if cardinalities[c] <= max_onehot_cardinality
    ]
    high_card_cols = [
        c for c in cat_cols if cardinalities[c] > max_onehot_cardinality
    ]

    logger.info("[preprocessing] Categorical columns: %s", cat_cols)
    logger.info(
        "[preprocessing] Low-cardinality (one-hot) columns (<= %d): %s",
        max_onehot_cardinality,
        low_card_cols,
    )
    logger.info(
        "[preprocessing] High-cardinality (ordinal) columns (> %d): %s",
        max_onehot_cardinality,
        high_card_cols,
    )

    # Numerical / non-categorical part
    df_num = X.drop(columns=cat_cols)

    # ============================
    # 1) ONE-HOT for low_card_cols
    # ============================
    if low_card_cols:
        df_cat_low = X[low_card_cols]

        encoder = OneHotEncoder(
            drop="first",
            sparse_output=False,  # safe: number of columns is limited
            handle_unknown="ignore",
        )
        encoded = encoder.fit_transform(df_cat_low)

        df_ohe = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(low_card_cols),
            index=df.index,
        )

        df_encoded = pd.concat([df_num, df_ohe], axis=1)
    else:
        df_encoded = df_num

    # =======================================
    # 2) ORDINAL encoding for high_card_cols
    # =======================================
    for col in high_card_cols:
        # .cat.codes → integer code per category, -1 for NaN
        codes = X[col].astype("category").cat.codes.astype(np.int32)
        new_col = f"{col}__idx"
        df_encoded[new_col] = codes

    # Reattach target if it exists
    if y is not None:
        df_encoded[target_col] = y

    return df_encoded
