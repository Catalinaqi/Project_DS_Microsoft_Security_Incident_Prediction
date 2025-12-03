# src/crispml/config/techniques/techniques_phase2.py

"""
Phase 2 Techniques Configuration (DATA UNDERSTANDING)

This module defines all the configurable techniques that can be activated
or deactivated during Phase 2 of the CRISP-ML methodology. Phase 2 includes:

    2.1 – Collect Initial Data      (handled in DatasetConfig)
    2.2 – Describe Data             (stats, distributions, correlations)
    2.3 – Verify Data Quality       (missing values, outliers, inconsistencies)
    2.4 – Explore Data (EDA)        (scatter matrix, PCA, preliminary models)

Each block is represented by a dedicated dataclass to keep the code
clean, readable, and maintainable. The Factory later builds a full
TechniquesConfig based on the ProblemType (clustering, classification,
regression, time series).
"""

from __future__ import annotations
from dataclasses import dataclass, field

# package imports CONFIG ENUMS
from src.crispml.config.enums.enums import ProblemType

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 2.2 – DESCRIBE THE DATA
# ---------------------------------------------------------------------
@dataclass
class Phase2DescribeConfig:
    """
    Techniques for describing dataset structure and distributions.

    These techniques are used to understand:
        - variable types
        - basic distributions
        - frequency of categories
        - correlation patterns

    All of these are generally safe to enable for any ML task.
    """
    describe_stats: bool = True         # Summary statistics (mean, std, quantiles)
    freq_tables: bool = True            # Frequency tables for categorical variables
    histograms: bool = True             # Distribution histograms
    boxplots: bool = True               # Boxplot visualization
    barplots: bool = True               # Barplots for categorical frequencies
    scatterplots: bool = True           # Basic scatter relationships
    corr_matrix: bool = True            # Correlation matrix heatmap

    def __post_init__(self):
        logger.debug("[Phase2DescribeConfig] %s", self)


# ---------------------------------------------------------------------
# 2.3 – VERIFY DATA QUALITY
# ---------------------------------------------------------------------
@dataclass
class Phase2QualityConfig:
    """
    Techniques for validating data consistency and quality.

    Includes:
        - missing value analysis
        - outlier detection
        - duplicate detection
        - range validation
        - inconsistency checks

    This block is essential for ALL problem types.
    """
    missing_analysis: bool = True
    outlier_detection: bool = True
    duplicates_check: bool = True
    range_check: bool = True
    inconsistencies_check: bool = True

    # NEW: Ranges per column
    ranges: dict = field(default_factory=lambda: {
        # column_name : (min_val, max_val)
        "Age": (0, 120),
        "Year": (1900, 2030),
        "Amount": (0, 1_000_000),
    })

    def __post_init__(self):
        logger.debug("[Phase2QualityConfig] %s", self)


# ---------------------------------------------------------------------
# 2.4 – EXPLORE DATA (EDA)
# ---------------------------------------------------------------------
@dataclass
class Phase2EdaConfig:
    """
    Exploratory Data Analysis techniques.

    These techniques help reveal:
        - structure of numerical relationships
        - dimensionality patterns
        - interactions between features and target
        - time-series behavior
        - cluster shapes (preliminary KMeans/DBSCAN)
        - autocorrelation for time series

    Each technique is selectively enabled depending on the ProblemType.
    """
    scatter_matrix: bool = True         # Multivariate scatter plot grid
    pca: bool = True                    # Dimensionality reduction for visualization
    feature_target_plots: bool = True   # Feature vs Target plots (only for supervised)
    kmeans_prelim: bool = True          # Preliminary clustering test
    dbscan_prelim: bool = True          # Preliminary DBSCAN to detect structure/outliers
    simple_tree_importance: bool = True # Quick feature importance (DecisionTree)
    residual_plots: bool = True         # Regression residual diagnostics
    ts_plot: bool = True                # Time-series line plot
    ts_decomposition: bool = True       # Decompose trend/seasonality
    acf_pacf: bool = True               # Autocorrelation & PACF plots

    def __post_init__(self):
        logger.debug("[Phase2EdaConfig] %s", self)


# ---------------------------------------------------------------------
# GLOBAL PHASE 2 CONFIG WRAPPER
# ---------------------------------------------------------------------
@dataclass
class Phase2Techniques:
    """
    Full set of techniques available for Phase 2 (Data Understanding).

    The configuration is organized into 3 logical blocks:
        - describe  → general statistics + structure
        - quality   → data quality checks
        - eda       → deep exploratory analysis

    The Factory applies presets depending on the ProblemType.
    """
    describe: Phase2DescribeConfig = field(default_factory=Phase2DescribeConfig)
    quality: Phase2QualityConfig = field(default_factory=Phase2QualityConfig)
    eda: Phase2EdaConfig = field(default_factory=Phase2EdaConfig)

    def __post_init__(self):
        logger.info("[Phase2Techniques] Phase 2 techniques initialized.")
        logger.debug("[Phase2Techniques] Describe: %s", self.describe)
        logger.debug("[Phase2Techniques] Quality:  %s", self.quality)
        logger.debug("[Phase2Techniques] EDA:      %s", self.eda)


# ---------------------------------------------------------------------
# PRESETS BASED ON PROBLEM TYPE
# ---------------------------------------------------------------------
def default_phase2_for_problem(problem_type: ProblemType) -> Phase2Techniques:
    """
    Build a Phase 2 configuration tailored for each problem type.

    This function ensures that only relevant EDA techniques are enabled,
    avoiding unnecessary computations and aligning with best practices
    from CRISP-DM.

    Logging:
        - Logs the resulting configuration for transparency
        - Helps trace how the Factory builds the project configuration
    """
    logger.info("[Phase2] Building default Phase 2 techniques for %s", problem_type.name)

    cfg = Phase2Techniques()

    # ------------------------------
    # CLUSTERING
    # ------------------------------
    if problem_type == ProblemType.CLUSTERING:
        cfg.eda.feature_target_plots = False
        cfg.eda.simple_tree_importance = False
        cfg.eda.residual_plots = False
        cfg.eda.ts_plot = False
        cfg.eda.ts_decomposition = False
        cfg.eda.acf_pacf = False

    # ------------------------------
    # CLASSIFICATION
    # ------------------------------
    elif problem_type == ProblemType.CLASSIFICATION:
        cfg.eda.kmeans_prelim = False
        cfg.eda.dbscan_prelim = False
        cfg.eda.ts_plot = False
        cfg.eda.ts_decomposition = False
        cfg.eda.acf_pacf = False

    # ------------------------------
    # REGRESSION
    # ------------------------------
    elif problem_type == ProblemType.REGRESSION:
        cfg.eda.kmeans_prelim = False
        cfg.eda.dbscan_prelim = False
        cfg.eda.feature_target_plots = True
        cfg.eda.residual_plots = True
        cfg.eda.ts_plot = False
        cfg.eda.ts_decomposition = False
        cfg.eda.acf_pacf = False

    # ------------------------------
    # TIME SERIES
    # ------------------------------
    elif problem_type == ProblemType.TIME_SERIES:
        cfg.describe.corr_matrix = False   # often less informative for pure TS
        cfg.eda.kmeans_prelim = False
        cfg.eda.dbscan_prelim = False
        cfg.eda.feature_target_plots = False
        cfg.eda.simple_tree_importance = False
        cfg.eda.residual_plots = False
        cfg.eda.ts_plot = True
        cfg.eda.ts_decomposition = True
        cfg.eda.acf_pacf = True

    logger.info("[Phase2] Default Phase 2 techniques ready for: %s", problem_type.name)
    logger.debug("[Phase2] Final Phase 2 config:\n%s", cfg)

    return cfg
