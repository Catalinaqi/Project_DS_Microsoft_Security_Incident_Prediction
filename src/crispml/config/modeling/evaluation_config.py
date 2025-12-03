# src/crispml/config/modeling/evaluation_config.py

"""
Evaluation configuration for CRISP-ML.

This module defines EvaluationConfig, a small but essential
configuration object that controls how models are evaluated in:

    - Phase 4 (Modeling)
    - Phase 5 (Evaluation & Interpretation)

The configuration is created automatically by the Factory but can
also be customized manually for advanced use-cases.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """
    Defines the evaluation strategy for ML models.

    main_metric:
        The primary metric used for:
            - evaluating performance
            - selecting the best model
            - comparing different algorithms
            - reporting results in Phase 5

    Examples:
        Classification:
            - "accuracy", "precision", "recall", "f1", "roc_auc"

        Regression:
            - "rmse", "mae", "mse", "r2"

        Time Series:
            - "mape", "mae", "rmse"

    When to change:
        - Choose a domain-specific metric
        - Use different metrics for hyperparameter tuning
        - Align metric with business constraints

    Logging:
        Logs the evaluation metric once initialized, so the workflow
        makes clear which metric is being used during the experiment.
    """

    main_metric: Optional[str] = None

    def __post_init__(self):
        logger.info(
            "[EvaluationConfig] Main evaluation metric set to: %s",
            self.main_metric,
        )
