# src/crispml/config/enums/enums.py

"""
Enumeration definitions for CRISP-ML configuration.

This module centralizes all Enum classes used across the configuration
system. Enums ensure that:
    - invalid values cannot be provided by mistake,
    - the configuration remains stable and standardized,
    - each phase can react explicitly to the selected options.

Logging is included for debugging purposes, so that configuration
decisions are easily traceable when constructing a ProjectConfig.
"""

from __future__ import annotations
from enum import Enum, auto

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# DataSourceType
# ----------------------------------------------------------------------
class DataSourceType(Enum):
    """
    Identifies the origin and structural format of the dataset.

    Used by the data loading layer in Phase 2 — Data Understanding.

    Allowed values:
        - CSV     → Standard flat CSV file
        - PARQUET → Columnar format, used for large-scale analytics
        - JSON    → Structured or semi-structured JSON data
        - SQL     → Data obtained from a database query
        - LOG     → Log files (common in monitoring systems)
        - API     → Remote data fetched via REST API

    Change this when:
        - The dataset comes from a different storage format
        - You want to extend the loader to new data sources
    """

    CSV = auto()
    PARQUET = auto()
    JSON = auto()
    SQL = auto()
    LOG = auto()
    API = auto()

    def __str__(self):
        logger.debug("[Enum] DataSourceType resolved: %s", self.name)
        return self.name


# ----------------------------------------------------------------------
# ProblemType
# ----------------------------------------------------------------------
class ProblemType(Enum):
    """
    Defines the type of Machine Learning / Data Mining problem.

    This Enum drives:
        - Technique selection (Phase 2 and 3)
        - Model selection (Phase 4)
        - Evaluation logic (Phase 5)
        - Default preprocessing rules in the Factory

    Allowed values:
        - CLUSTERING   → Unsupervised structure discovery
        - CLASSIFICATION → Predict categorical labels
        - REGRESSION     → Predict continuous numerical values
        - TIME_SERIES    → Forecast or analyze sequential temporal data

    Change this when:
        - You switch to a different ML task
        - You want to extend the framework with a new problem type
    """

    CLUSTERING = auto()
    CLASSIFICATION = auto()
    REGRESSION = auto()
    TIME_SERIES = auto()

    def __str__(self):
        logger.debug("[Enum] ProblemType resolved: %s", self.name)
        return self.name


# ----------------------------------------------------------------------
# FeatureSelectionMode
# ----------------------------------------------------------------------
class FeatureSelectionMode(Enum):
    """
    Strategy for selecting input features for modeling.

    AUTO:
        - Automatically uses all columns except ID, target, and time.
        - Recommended as default.

    INCLUDE:
        - Uses only the columns explicitly listed in FeatureConfig.include[].

    EXCLUDE:
        - Uses all columns except those in FeatureConfig.exclude[].

    Change this when:
        - You want strict control over input features
        - You need to reduce dimensionality manually
        - You want to ignore noisy or irrelevant columns
    """

    AUTO = auto()
    INCLUDE = auto()
    EXCLUDE = auto()

    def __str__(self):
        logger.debug("[Enum] FeatureSelectionMode resolved: %s", self.name)
        return self.name
