"""
Public API for CRISP-ML preprocessing utilities.
"""

# Missing Values (Phase 3 Cleaning)
from .missing_values_utils import (
    drop_high_nan_columns,
    simple_imputation,
)

# Outliers (Phase 3 Cleaning)
from .outlier_utils import treat_outliers

# Cleaning (duplicates + log-transform)
from .cleaning_utils import remove_duplicates, apply_log_transform

# Categorical Encoding
from .categorical_utils import encode_category_utils

# Scaling
from .scaling_utils import scale_features

# Splitting
from .split_utils import train_val_test_split

# High-level pipelines
from .preprocessing_utils import (
    phase2_quick_clean,
    phase3_full_pipeline,
    split_dataset,
)

# Data Types Utilities
from .dtypes_utils import (
    is_id_column,
    get_real_numeric_columns,
    get_categorical_columns,)

__all__ = [
    # Missing values
    "drop_high_nan_columns",
    "simple_imputation",

    # Outliers
    "treat_outliers",

    # Cleaning
    "remove_duplicates",
    "apply_log_transform",

    # Categorical
    "encode_category_utils",

    # Scaling
    "scale_features",

    # Splitting
    "train_val_test_split",

    # Pipelines
    "phase2_quick_clean",
    "phase3_full_pipeline",
    "split_dataset",

    # Dtypes utils
    "is_id_column",
    "get_real_numeric_columns",
    "get_categorical_columns",
]
