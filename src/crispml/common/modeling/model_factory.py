"""
Unified wrapper to call the correct modeling pipeline automatically.
"""

from __future__ import annotations
from typing import Dict, Any

from src.crispml.config.enums.enums_config import ProblemType

from .clustering_models import run_clustering_algos
from .classification_models import run_classification_algos
from .regression_models import run_regression_algos
from .timeseries_models import run_time_series_algos

def run_models(problem_type: ProblemType, **kwargs) -> Dict[str, Any]:
    """
    Dispatch to the correct modeling family based on ProblemType.
    """

    if problem_type == ProblemType.CLUSTERING:
        return run_clustering_algos(
            kwargs["X"],
            kwargs["algos"],
            kwargs["hyperparams"],
        )

    if problem_type == ProblemType.CLASSIFICATION:
        return run_classification_algos(
            kwargs["X_train"], kwargs["X_test"],
            kwargs["y_train"], kwargs["y_test"],
            kwargs["algos"], kwargs["hyperparams"],
        )

    if problem_type == ProblemType.REGRESSION:
        return run_regression_algos(
            kwargs["X_train"], kwargs["X_test"],
            kwargs["y_train"], kwargs["y_test"],
            kwargs["algos"], kwargs["hyperparams"],
        )

    if problem_type == ProblemType.TIME_SERIES:
        return run_time_series_algos(
            kwargs["y_train"],
            kwargs["algos"],
            kwargs["hyperparams"],
        )

    raise ValueError("Unsupported ProblemType")
