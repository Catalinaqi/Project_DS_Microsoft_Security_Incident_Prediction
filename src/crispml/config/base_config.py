# src/crispml/config/base_config.py

"""
Base configuration entrypoint for CRISP-ML.

This module exposes a clean and stable import surface for users of the
framework, while the internal implementation remains fully modularized.

Example usage:
    from crispml.config import ProjectConfig, make_config

This design allows:
    - Simple imports for notebooks and pipelines
    - Backward compatibility even if internal structure changes
    - A public API layer decoupled from implementation files
"""

from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)
logger.info("[base_config] CRISP-ML base configuration module loaded.")


# ---------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------
from src.crispml.config.enums.enums import (
    DataSourceType,
    ProblemType,
    FeatureSelectionMode,
)

# ---------------------------------------------------------------------
# DATASET CONFIG
# ---------------------------------------------------------------------
from src.crispml.config.dataset.dataset_config import DatasetConfig
from src.crispml.config.dataset.feature_config import FeatureConfig
from src.crispml.config.dataset.bigdata_config import BigDataConfig

# ---------------------------------------------------------------------
# MODELING CONFIG
# ---------------------------------------------------------------------
from src.crispml.config.modeling.modeling_config import ModelingConfig
from src.crispml.config.modeling.evaluation_config import EvaluationConfig

# ---------------------------------------------------------------------
# TECHNIQUES (PHASE 2 + PHASE 3)
# ---------------------------------------------------------------------
from src.crispml.config.techniques.techniques_phase2 import (
    Phase2DescribeConfig,
    Phase2QualityConfig,
    Phase2EdaConfig,
    Phase2Techniques,
    default_phase2_for_problem,
)

from src.crispml.config.techniques.techniques_phase3 import (
    Phase3CleaningConfig,
    Phase3TransformationConfig,
    Phase3IntegrationConfig,
    Phase3FormattingConfig,
    Phase3Techniques,
    default_phase3_for_problem,
)

from src.crispml.config.techniques.techniques import TechniquesConfig

# ---------------------------------------------------------------------
# PROJECT CONFIG
# ---------------------------------------------------------------------
from src.crispml.config.project.project_config import ProjectConfig


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------
__all__ = [
    # Enums
    "DataSourceType",
    "ProblemType",
    "FeatureSelectionMode",

    # Dataset
    "DatasetConfig",
    "FeatureConfig",
    "BigDataConfig",

    # Modeling
    "ModelingConfig",
    "EvaluationConfig",

    # Phase 2 Techniques
    "Phase2DescribeConfig",
    "Phase2QualityConfig",
    "Phase2EdaConfig",
    "Phase2Techniques",
    "default_phase2_for_problem",

    # Phase 3 Techniques
    "Phase3CleaningConfig",
    "Phase3TransformationConfig",
    "Phase3IntegrationConfig",
    "Phase3FormattingConfig",
    "Phase3Techniques",
    "default_phase3_for_problem",

    # Unified Techniques wrapper
    "TechniquesConfig",

    # ProjectConfig
    "ProjectConfig",
]

logger.info("[base_config] CRISP-ML Base API ready for use.")
