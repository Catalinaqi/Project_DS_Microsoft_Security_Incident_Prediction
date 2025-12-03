from __future__ import annotations
import pandas as pd
from typing import Optional
from sklearn.preprocessing import OneHotEncoder

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


def encode_categoricals(
        df: pd.DataFrame,
        encoding: str = "onehot",
        target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Encodes categorical variables (default: One-Hot Encoding).
    """
    if encoding != "onehot":
        raise NotImplementedError("Only onehot encoding is currently supported.")

    df = df.copy()

    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    if not cat_cols:
        logger.info("[preprocessing] No categorical columns to encode.")
        return df

    df_cat = df[cat_cols]
    df_num = df.drop(columns=cat_cols)

    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df_cat)

    df_encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index,
    )

    logger.info("[preprocessing] One-Hot Encoding applied to: %s", cat_cols)

    return pd.concat([df_num, df_encoded], axis=1)
