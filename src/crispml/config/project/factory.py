# src/crispml/config/project/factory.py

"""
Factory for constructing a complete ProjectConfig in a clean,
standardized and reproducible way according to the CRISP-ML methodology.

This module builds:
    - DatasetConfig
    - FeatureConfig
    - BigDataConfig
    - ModelingConfig
    - EvaluationConfig
    - TechniquesConfig (Phase 2 + Phase 3)
    - ProjectConfig

Based on the ProblemType (clustering, classification, regression, time series),
the factory automatically selects:
    - relevant preprocessing techniques
    - suitable algorithms
    - default hyperparameter grids
"""

from __future__ import annotations
from typing import List, Optional

# ----------------------------------------------------------------------
# IMPORTS (NOW FIXED AND STRUCTURED BY SUBMODULE)
# ----------------------------------------------------------------------

# CONFIG ENUMS
from src.crispml.config.enums.enums_config import (
    ProblemType,
    DataSourceType,
    FeatureSelectionMode,
)
# CONFIG DATASET
from src.crispml.config.dataset.dataset_config import DatasetConfig
from src.crispml.config.dataset.feature_config import FeatureConfig
from src.crispml.config.dataset.bigdata_config import BigDataConfig
# CONFIG MODELING
from src.crispml.config.modeling.modeling_config import ModelingConfig
from src.crispml.config.modeling.evaluation_config import EvaluationConfig
# CONFIG TECHNIQUES
from src.crispml.config.techniques.techniques import TechniquesConfig
# CONFIG PROJECT CONFIG
from src.crispml.config.project.project_config import ProjectConfig
# COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# FACTORY CLASS (CLEAN, ORGANIZED AND WELL-DOCUMENTED)
# ----------------------------------------------------------------------
def _build_modeling_for(problem_type: ProblemType) -> ModelingConfig:
    """
    Creates a ModelingConfig containing the default algorithms and
    hyperparameter grids for the selected ML task.

    This section is where you can easily add:
        - new algorithms
        - new libraries (LightGBM, Prophet, etc.)
        - custom hyperparameter grids
    """
    logger.info("[factory] Building ModelingConfig for %s", problem_type.name)

    if problem_type == ProblemType.CLUSTERING:
        return ModelingConfig(
            clustering_algos=["kmeans", "dbscan"],
            hyperparams={
                "kmeans": {"n_clusters": [3, 4, 5]},
                "dbscan": {"eps": [0.3, 0.5], "min_samples": [5, 10]},
            },
        )

    if problem_type == ProblemType.CLASSIFICATION:
        return ModelingConfig(
            classification_algos=["dt", "rf", "svm", "nb", "knn"],
            hyperparams={
                "dt": {"max_depth": [3, 5, None]},
                "rf": {"n_estimators": [100, 200]},
                "svm": {"C": [0.1, 1, 10]},
                "knn": {"n_neighbors": [3, 5, 7]},
            },
        )

    if problem_type == ProblemType.REGRESSION:
        return ModelingConfig(
            regression_algos=[
                "linear", "ridge", "lasso", "poly",
                "knn_reg", "tree_reg", "rf_reg",
                "xgboost", "svr", "nn"
            ],
            hyperparams={
                "ridge": {"alpha": [0.1, 1.0, 10.0]},
                "lasso": {"alpha": [0.001, 0.01, 0.1]},
                "knn_reg": {"n_neighbors": [3, 5, 7]},
                "tree_reg": {"max_depth": [3, 5, None]},
                "rf_reg": {"n_estimators": [100, 200]},
                "poly": {"degree": [2, 3]},
                "svr": {"C": [0.1, 1, 10]},
            },
        )

    if problem_type == ProblemType.TIME_SERIES:
        return ModelingConfig(
            ts_algos=["arima", "sarima"],
            hyperparams={
                "arima": {"p": [1, 2], "d": [0, 1], "q": [0, 1]},
                "sarima": {"P": [0, 1], "D": [0, 1], "Q": [0, 1], "m": [12]},
            },
        )

    raise ValueError(f"Unsupported ProblemType: {problem_type}")


# ----------------------------------------------------------------------
# MAIN FACTORY ENTRYPOINT
# ----------------------------------------------------------------------
def make_config(
        name: str,
        problem_type: ProblemType,
        dataset_path: str,
        target_col: Optional[str] = None,
        time_col: Optional[str] = None,
        id_cols: Optional[List[str]] = None,
        feature_mode: FeatureSelectionMode = FeatureSelectionMode.AUTO,
        include_cols: Optional[List[str]] = None,
        exclude_cols: Optional[List[str]] = None,
) -> ProjectConfig:
    """
    Builds a complete ProjectConfig for a CRISP-ML project.

    Parameters:
    ----------
    name : str
        Logical experiment name.
    problem_type : ProblemType
        Determines modeling algorithms, techniques, evaluation.
    dataset_path : str
        Path to CSV dataset (or other formats supported later).
    target_col : str | None
        Target column for supervised tasks.
    time_col : str | None
        Timestamp column for time-series tasks.
    id_cols : list[str] | None
        Columns to ignore for feature modeling.
    feature_mode : FeatureSelectionMode
        AUTO / INCLUDE / EXCLUDE mode.
    include_cols / exclude_cols :
        Lists used when feature_mode != AUTO.
    """

    logger.info(
        "[factory] Creating ProjectConfig '%s' | type=%s | dataset=%s",
        name, problem_type.name, dataset_path
    )

    # --------------------------------------------------------
    # 1) DatasetConfig
    # --------------------------------------------------------
    dataset_cfg = DatasetConfig(
        source_type=DataSourceType.CSV,
        path_or_conn=dataset_path,
        problem_type=problem_type,
        target_col=target_col,
        time_col=time_col,
        id_cols=id_cols or [],
    )
    logger.debug("[factory] DatasetConfig created:\n%s", dataset_cfg)

    # --------------------------------------------------------
    # 2) FeatureConfig
    # --------------------------------------------------------
    features_cfg = FeatureConfig(
        mode=feature_mode,
        include=include_cols or [],
        exclude=exclude_cols or [],
    )
    logger.debug("[factory] FeatureConfig created:\n%s", features_cfg)

    # --------------------------------------------------------
    # 3) BigDataConfig
    # --------------------------------------------------------
    bigdata_cfg = BigDataConfig()
    logger.debug("[factory] BigDataConfig created:\n%s", bigdata_cfg)

    # --------------------------------------------------------
    # 4) ModelingConfig based on ML task
    # --------------------------------------------------------
    modeling_cfg = _build_modeling_for(problem_type)
    logger.debug("[factory] ModelingConfig created:\n%s", modeling_cfg)

    # --------------------------------------------------------
    # 5) EvaluationConfig (basic)
    # --------------------------------------------------------
    evaluation_cfg = EvaluationConfig(main_metric=None)
    logger.debug("[factory] EvaluationConfig created:\n%s", evaluation_cfg)

    # --------------------------------------------------------
    # 6) TechniquesConfig (Phase2 + Phase3)
    # --------------------------------------------------------
    techniques_cfg = TechniquesConfig.for_problem(problem_type)
    logger.debug("[factory] TechniquesConfig created:\n%s", techniques_cfg)

    # --------------------------------------------------------
    # 7) Final ProjectConfig
    # --------------------------------------------------------
    config = ProjectConfig(
        name=name,
        datasetConfig=dataset_cfg,
        featureConfig=features_cfg,
        bigDataConfig=bigdata_cfg,
        modelingConfig=modeling_cfg,
        evaluationConfig=evaluation_cfg,
        techniquesConfig=techniques_cfg,
    )

    logger.info("[factory] ProjectConfig '%s' successfully created.", name)
    logger.debug("[factory] Full ProjectConfig:\n%s", config)

    return config
