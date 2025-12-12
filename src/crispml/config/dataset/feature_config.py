# src/crispml/config/dataset/feature_config.py

"""
Feature configuration module for CRISP-ML.

This module defines how input features are selected during
the preparation and modeling phases.

FeatureConfig is used in:
    - Phase 2 (Data Understanding)
    - Phase 3 (Data Preparation)
    - Factory (ProjectConfig builder)

It defines:
    - feature selection strategy (AUTO, INCLUDE, EXCLUDE)
    - include/exclude column lists
    - max limits for usable features
    - thresholds for categorical encoding
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# package imports CONFIG ENUMS
from src.crispml.config.enums.enums_config import FeatureSelectionMode

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for feature selection strategy.

    Parameters:
    ----------
    mode : FeatureSelectionMode
        Determines how input columns are selected:
            - AUTO:     use all non-ID, non-target, non-time columns
            - INCLUDE:  use only columns listed in include[]
            - EXCLUDE:  use all columns except those in exclude[]

    include : list[str]
        Columns to explicitly include when mode=INCLUDE.

    exclude : list[str]
        Columns to exclude when mode=EXCLUDE.

    max_features : int
        Maximum number of features to retain after preprocessing.

    max_unique_for_cat : int
        Maximum allowed unique values to consider a feature categorical.
        Used in one-hot / ordinal encoding automatically.

    When to modify:
        - to force a restricted feature set
        - to avoid noisy or irrelevant columns
        - to control dimensionality
        - to tune categorical encoding with high-cardinality categories
    """

    mode: FeatureSelectionMode = FeatureSelectionMode.AUTO
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    max_features: int = 50
    max_unique_for_cat: int = 50

    def __post_init__(self):
        logger.info(
            "[FeatureConfig] mode=%s | include=%s | exclude=%s | max_features=%d | max_unique_cat=%d",
            self.mode.name,
            self.include,
            self.exclude,
            self.max_features,
            self.max_unique_for_cat,
        )
