# src/crispml/config/modeling/modeling_config.py

"""
Modeling configuration for CRISP-ML.

This configuration determines which algorithms will be trained
during Phase 4, and with which hyperparameters.

The Factory assigns defaults depending on the ProblemType:
    - CLUSTERING: ["kmeans", "dbscan"]
    - CLASSIFICATION: ["dt", "rf", "svm", "nb", "knn"]
    - REGRESSION: ["linear", "ridge", "lasso", ...]
    - TIME_SERIES: ["arima", "sarima"]

You can extend this module by adding:
    - new algorithm families (e.g. LightGBM, Prophet, LSTM)
    - new hyperparameter grids
    - new modeling strategies per problem type
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelingConfig:
    """
    Contains lists of algorithms configured for each ML task type,
    along with hyperparameters used in grid/random search during Phase 4.

    Parameters:
    -----------
    clustering_algos : list[str]
        Algorithms for unsupervised clustering.

    classification_algos : list[str]
        Algorithms for supervised classification.

    regression_algos : list[str]
        Algorithms for supervised regression.

    ts_algos : list[str]
        Algorithms for time-series modeling (ARIMA, SARIMA, ML-based TS).

    hyperparams : dict[str, dict]
        Hyperparameter grid for each algorithm.
        Example:
            {
                "kmeans": {"n_clusters": [3, 4, 5]},
                "rf": {"n_estimators": [100, 200]},
                "svr": {"C": [0.1, 1, 10]}
            }

    When to modify:
        - Add new algorithms or remove unused ones
        - Expand hyperparameter search grids
        - Introduce new modeling techniques for future courses/projects
    """

    clustering_algos: List[str] = field(default_factory=list)
    classification_algos: List[str] = field(default_factory=list)
    regression_algos: List[str] = field(default_factory=list)
    ts_algos: List[str] = field(default_factory=list)

    hyperparams: Dict[str, Dict] = field(default_factory=dict)

    def __post_init__(self):
        logger.info(
            "[ModelingConfig] Algorithms loaded | clustering=%s | classification=%s | regression=%s | ts=%s",
            self.clustering_algos,
            self.classification_algos,
            self.regression_algos,
            self.ts_algos,
        )
        logger.debug("[ModelingConfig] Hyperparameter settings: %s", self.hyperparams)
