# src/crispml/config/techniques/techniques_phase3.py

"""
Phase 3 Techniques Configuration (DATA PREPARATION)

This module defines all optional techniques that can be activated
or deactivated in Phase 3 of the CRISP-ML workflow. Phase 3 includes:

    3.2 – Data Cleaning
    3.3 – Data Transformation
    3.4 – Data Integration
    3.5 – Data Formatting (train/val/test management)

Each technique is represented as a flag so that the pipeline can execute
only the operations relevant for the selected ML problem type.

The Factory automatically builds a default configuration using
`default_phase3_for_problem`.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# package imports CONFIG ENUMS
from src.crispml.config.enums.enums import ProblemType

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 3.2 – DATA CLEANING
# ---------------------------------------------------------------------
@dataclass
class Phase3CleaningConfig:
    """
    Techniques for detecting and fixing data-quality issues.

    Techniques include:
        - Dropping columns/rows with excessive NaN values
        - Simple and advanced imputation
        - Outlier treatment
        - Robust transformations
        - Rare-category handling
        - Duplicate removal

    These operations strongly affect downstream modeling quality.
    """

    drop_high_nan: bool = True              # Drop rows/columns with too many NaNs
    simple_imputation: bool = True          # Mean/median/mode imputation
    advanced_imputation: bool = False       # KNN/MICE imputation
    outlier_treatment: bool = True          # Winsorization/clipping
    robust_transform: bool = False          # Log/Box-Cox robust transformations
    typos_and_rare_categories: bool = True  # Fix typos and collapse rare categories
    duplicates_handling: bool = True        # Remove duplicate rows

    def __post_init__(self):
        logger.debug("[Phase3CleaningConfig] %s", self)


# ---------------------------------------------------------------------
# 3.3 – DATA TRANSFORMATION
# ---------------------------------------------------------------------
@dataclass
class Phase3TransformationConfig:
    """
    Techniques responsible for transforming and enriching data.

    Includes:
        - scaling (Standard/MinMax/Robust)
        - encoding (OneHot/Ordinal/Target)
        - non-linear transforms (log, sqrt)
        - feature engineering (polynomials, ratios)
        - time-series specific operations (differencing, lag, rolling stats)
    """

    scaling: bool = True                    # Standard/MinMax/Robust scaling
    encoding: bool = True                   # OneHot/Ordinal/Target encoding
    nonlinear_transform: bool = False       # Log/sqrt/box-cox transformations
    feature_engineering: bool = False       # Interactions, polynomials, domain features
    ts_differencing: bool = False           # Make TS stationary
    ts_lag_features: bool = False           # Lag-based features for ML models on TS
    ts_rolling_stats: bool = False          # Rolling mean/variance, etc.

    def __post_init__(self):
        logger.debug("[Phase3TransformationConfig] %s", self)


# ---------------------------------------------------------------------
# 3.4 – DATA INTEGRATION
# ---------------------------------------------------------------------
@dataclass
class Phase3IntegrationConfig:
    """
    Techniques for merging and aligning multiple datasets.

    Includes:
        - merge/join operations
        - alignment by keys or timestamps
        - conflict resolution
        - integration of historical + new incoming data
    """

    join_merge: bool = True                  # Join/Merge tables
    align_on_keys: bool = True               # Align rows by ID/timestamp
    resolve_conflicts: bool = True           # Resolve inconsistent overlapping values
    merge_historical_with_new: bool = False  # Combine old+new TS data

    def __post_init__(self):
        logger.debug("[Phase3IntegrationConfig] %s", self)


# ---------------------------------------------------------------------
# 3.5 – DATA FORMATTING
# ---------------------------------------------------------------------
@dataclass
class Phase3FormattingConfig:
    """
    Formatting operations before modeling.

    Includes:
        - train/val/test splitting
        - temporal split for time-series
        - sliding windows for sequence models
        - tensor formatting for neural networks
    """

    train_val_test_split: bool = True       # Standard randomized split
    use_temporal_split: bool = False        # No shuffle (strict TS split)
    sliding_windows: bool = False           # For LSTM/RNN/TS-ML
    nn_tensors: bool = False                # Prepare 3D tensors

    def __post_init__(self):
        logger.debug("[Phase3FormattingConfig] %s", self)


# ---------------------------------------------------------------------
# GROUPED CONFIGURATION FOR PHASE 3
# ---------------------------------------------------------------------
@dataclass
class Phase3Techniques:
    """
    Container for all Phase 3 technique groups.

    Subsections:
        - cleaning
        - transformation
        - integration
        - formatting

    The Factory uses this structure to decide which preprocessing
    steps to execute depending on the ML problem type.
    """

    cleaning: Phase3CleaningConfig = field(default_factory=Phase3CleaningConfig)
    transformation: Phase3TransformationConfig = field(default_factory=Phase3TransformationConfig)
    integration: Phase3IntegrationConfig = field(default_factory=Phase3IntegrationConfig)
    formatting: Phase3FormattingConfig = field(default_factory=Phase3FormattingConfig)

    def __post_init__(self):
        logger.info("[Phase3Techniques] Phase 3 techniques initialized.")
        logger.debug("[Phase3Techniques] Cleaning:       %s", self.cleaning)
        logger.debug("[Phase3Techniques] Transformation: %s", self.transformation)
        logger.debug("[Phase3Techniques] Integration:    %s", self.integration)
        logger.debug("[Phase3Techniques] Formatting:     %s", self.formatting)


# ---------------------------------------------------------------------
# DEFAULT PRESETS BASED ON PROBLEM TYPE
# ---------------------------------------------------------------------
def default_phase3_for_problem(problem_type: ProblemType) -> Phase3Techniques:
    """
    Build a Phase 3 configuration customized to the selected ML problem type.

    This ensures:
        - Unnecessary preprocessing is disabled
        - Time-series-specific operations are enabled when needed
        - Feature engineering is only used when beneficial
        - Advanced imputation/scaling rules follow best practices

    Logging clearly reports the final configuration chosen.
    """

    logger.info("[Phase3] Building default Phase 3 techniques for %s", problem_type.name)

    cfg = Phase3Techniques()

    # ------------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------------
    if problem_type == ProblemType.CLUSTERING:
        cfg.transformation.scaling = True
        cfg.transformation.feature_engineering = False
        cfg.formatting.use_temporal_split = False

    # ------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------
    elif problem_type == ProblemType.CLASSIFICATION:
        cfg.cleaning.advanced_imputation = True
        cfg.transformation.nonlinear_transform = True
        cfg.transformation.feature_engineering = True

    # ------------------------------------------------------
    # REGRESSION
    # ------------------------------------------------------
    elif problem_type == ProblemType.REGRESSION:
        cfg.cleaning.advanced_imputation = True
        cfg.cleaning.robust_transform = True
        cfg.transformation.nonlinear_transform = True
        cfg.transformation.feature_engineering = True
        cfg.formatting.nn_tensors = True  # useful for advanced NN regressors

    # ------------------------------------------------------
    # TIME SERIES
    # ------------------------------------------------------
    elif problem_type == ProblemType.TIME_SERIES:
        cfg.transformation.scaling = True
        cfg.transformation.encoding = False  # often unnecessary
        cfg.transformation.ts_differencing = True
        cfg.transformation.ts_lag_features = True
        cfg.transformation.ts_rolling_stats = True

        cfg.integration.merge_historical_with_new = True

        cfg.formatting.use_temporal_split = True
        cfg.formatting.sliding_windows = True
        cfg.formatting.nn_tensors = True

    logger.info("[Phase3] Default Phase 3 techniques ready for: %s", problem_type.name)
    logger.debug("[Phase3] Final Phase 3 config:\n%s", cfg)

    return cfg
