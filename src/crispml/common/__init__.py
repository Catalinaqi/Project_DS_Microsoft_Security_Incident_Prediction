"""
Public API for CRISP-ML common utilities.

This package contains modular utility functions used across all phases
of the CRISP-ML workflow, including:

- Logging utilities
- I/O utilities
- Preprocessing utilities
- Feature selection utilities
- Modeling utilities
- Evaluation utilities
- Output (image/table saving) utilities

Only high-level, stable functions are exported here.
"""

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
from .logging.logging_utils import get_logger

# ---------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------
from .io import load_dataset

# ---------------------------------------------------------
# Preprocessing utilities (high-level API only)
# ---------------------------------------------------------
from .preprocessing import (
    phase2_quick_clean,
    phase3_full_pipeline,
    split_dataset,
)

# ---------------------------------------------------------
# Feature selection utilities
# ---------------------------------------------------------
from .feature_selection import (
    select_features_auto,
    select_features_include,
    select_features_exclude,
)

# ---------------------------------------------------------
# Modeling utilities
# ---------------------------------------------------------
from .modeling import (
    run_clustering_algos,
    run_classification_algos,
    run_regression_algos,
    run_time_series_algos,
    run_models,
)

# ---------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------
from .evaluation import (
    compute_clustering_metrics,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_ts_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_residuals,
)

# ---------------------------------------------------------
# Output utilities
# ---------------------------------------------------------
from .output import (
    save_figure,
    save_table_as_image,
    get_output_dir,
)

# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------
__all__ = [
    # Logging
    "get_logger",

    # IO
    "load_dataset",

    # Preprocessing
    "phase2_quick_clean",
    "phase3_full_pipeline",
    "split_dataset",

    # Feature Selection
    "select_features_auto",
    "select_features_include",
    "select_features_exclude",

    # Modeling
    "run_clustering_algos",
    "run_classification_algos",
    "run_regression_algos",
    "run_time_series_algos",
    "run_models",

    # Evaluation
    "compute_clustering_metrics",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_ts_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_residuals",

    # Output
    "save_figure",
    "save_table_as_image",
    "get_output_dir",
]
