# src/crispml/config/project/project_config.py

"""
ProjectConfig – Core configuration object for CRISP-ML projects.

This module defines the unified configuration container that represents
the entire setup of a CRISP-ML workflow. After building a project using
the Factory, all phases (Phase 2 → Phase 5) receive a ProjectConfig
instance to drive their execution.

The ProjectConfig groups together:
    - Dataset configuration
    - Feature selection strategy
    - Big-data sampling preferences
    - Modeling algorithms + hyperparameters
    - Evaluation metric
    - Techniques for Phase 2 and Phase 3

It is intentionally modular and future-proof, allowing seamless
extensions for Phase 4 and Phase 5 technique configurations.
"""

from __future__ import annotations
from dataclasses import dataclass

# CONFIG DATASET
from src.crispml.config.dataset.dataset_config import DatasetConfig
from src.crispml.config.dataset.feature_config import FeatureConfig
from src.crispml.config.dataset.bigdata_config import BigDataConfig
# CONFIG MODELING
from src.crispml.config.modeling.modeling_config import ModelingConfig
from src.crispml.config.modeling.evaluation_config import EvaluationConfig
# CONFIG TECHNIQUES
from src.crispml.config.techniques.techniques import TechniquesConfig


# COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectConfig:
    """
    Central configuration object for CRISP-ML.

    Parameters:
    ----------
    name : str
        Logical name of the project or experiment.
        Appears in log messages, output directories, and reports.

    dataset : DatasetConfig
        Contains dataset location, problem type, schema roles (target/time/id).

    features : FeatureConfig
        Defines feature selection mode and include/exclude lists.

    bigdata : BigDataConfig
        Controls how large datasets are sampled during EDA.

    modeling : ModelingConfig
        Specifies which algorithms to train and their hyperparameters.

    evaluation : EvaluationConfig
        Defines the main evaluation metric for Phases 4 and 5.

    techniques : TechniquesConfig
        Collects all enabled preprocessing techniques for Phase 2 + Phase 3.

    Notes:
    ------
    - This configuration is created automatically by `factory.make_config()`.
    - It is passed to all phase modules to ensure a consistent workflow.
    - Future extensions can easily add more configuration blocks.
    """

    name: str
    datasetConfig: DatasetConfig
    featureConfig: FeatureConfig
    bigDataConfig: BigDataConfig
    modelingConfig: ModelingConfig
    evaluationConfig: EvaluationConfig
    techniquesConfig: TechniquesConfig

    def __post_init__(self):
        logger.info("[ProjectConfig] Project '%s' initialized.", self.name)

        logger.debug("[ProjectConfig] Dataset config:\n%s", self.datasetConfig)
        logger.debug("[ProjectConfig] Feature config:\n%s", self.featureConfig)
        logger.debug("[ProjectConfig] BigData config:\n%s", self.bigDataConfig)
        logger.debug("[ProjectConfig] Modeling config:\n%s", self.modelingConfig)
        logger.debug("[ProjectConfig] Evaluation config:\n%s", self.evaluationConfig)
        logger.debug("[ProjectConfig] Techniques config:\n%s", self.techniquesConfig)

        logger.info("[ProjectConfig] Project '%s' configuration ready.", self.name)
