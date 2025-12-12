# src/crispml/config/enums/enums_config.py

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


class PhaseName(Enum):
    PHASE2_DATA_UNDERSTANDING = auto()
    PHASE3_DATA_PREPARATION = auto()
    PHASE4_MODELLING = auto()
    PHASE5_EVALUATION = auto()

    def __str__(self):
        logger.debug("[Enum] Phase Name: %s", self.name)
        return self.name


class PhaseStep(Enum):
    """
    Singolo step del processo CRISP-ML, con:
        - phase_label  -> "FASE2", "FASE3", ...
        - step_code    -> "2.1", "3.2", ...
        - description  -> descrizione testuale dello step
    """

    # FASE 2 – Data Understanding
    P2_1_COLLECT_INITIAL_DATA = ("FASE2", "2.1", "COLLECT_INITIAL_DATA")
    P2_2_DESCRIBE_DATA        = ("FASE2", "2.2", "DESCRIBE_DATA")
    P2_3_CHECK_DATA_QUALITY   = ("FASE2", "2.3", "CHECK_DATA_QUALITY")
    P2_4_EXPLORE_DATA         = ("FASE2", "2.4", "EXPLORE_DATA")

    # FASE 3 – Data Preparation
    P3_1_SELECT_DATA          = ("FASE3", "3.1", "SELECT_DATA")
    P3_2_DATA_CLEANING        = ("FASE3", "3.2", "DATA_CLEANING")
    P3_3_DATA_TRANSFORMATION  = ("FASE3", "3.3", "DATA_TRANSFORMATION")
    P3_4_DATA_INTEGRATION     = ("FASE3", "3.4", "DATA_INTEGRATION")
    P3_5_DATA_FORMATTING      = ("FASE3", "3.5", "DATA_FORMATTING")

    # FASE 4 – Modeling (Data Mining)
    P4_1_SELECT_TECHNIQUE     = (
        "FASE4",
        "4.1",
        "SELECT_TECHNIQUE",
    )
    P4_2_BUILD_MODEL          = ("FASE4", "4.2", "BUILD_MODEL")
    P4_3_BUILD_TEST_DESIGN    = (
        "FASE4",
        "4.3",
        "BUILD_TEST_DESIGN",
    )
    P4_4_EVALUATE_MODEL       = ("FASE4", "4.4", "EVALUATE_MODEL")

    # FASE 5 – Evaluation (& Interpretation)
    P5_1_EXTRACT_KNOWLEDGE    = ("FASE5", "5.1", "EXTRACT_KNOWLEDGE")
    P5_2_EVALUATE_RESULTS     = ("FASE5", "5.2", "EVALUATE_RESULTS")
    P5_3_REVIEW_PROCESS       = ("FASE5", "5.3", "REVIEW_PROCESS")
    P5_4_DETERMINE_NEXT_STEPS = ("FASE5", "5.4", "DETERMINE_NEXT_STEPS")

    def __init__(self, phase_label: str, step_code: str, description: str):
        """
        Constructor for PhaseStep.
        Purpose:
        - Store the phase label, step code, and human-readable description
          in the instance attributes.
        - These values are later used for logging, file naming, and
          human-readable representations of the step.

        Initialize a PhaseStep instance.
        Flow:
        1. Receive the phase label, step code, and human-readable description
           as arguments.
        2. Store these values in the instance attributes:
           - self.phase_label
           - self.step_code
           - self.description
        3. These attributes will later be used to build log messages
           and string representations for this step.
        """
        self.phase_label = phase_label
        self.step_code = step_code
        self.description = description

    def __str__(self) -> str:
        """
        String representation of the PhaseStep.
        Purpose:
        - Build a formatted label like "[FASE2][2.1] Collect initial data"
          using the stored attributes.
        - Log this label at DEBUG level for traceability.
        - Return the label so that printing the object or using it in
          f-strings produces this standard representation.

        Return a human-readable string representation of this PhaseStep.
        Flow:
        1. Build a formatted string in the form:
             "[<phase_label>][<step_code>] <description>"
           using the instance attributes.
        2. Log this formatted string at DEBUG level so that the resolved
           PhaseStep can be inspected in the logs.
        3. Return the formatted string so that calling str(self) or printing
           the object shows this representation.
        """
        # Esempio: "[FASE2][2.1] Collezionare i dati iniziali"
        text = f"[{self.phase_label}][{self.step_code}] {self.description}"
        logger.debug("[Enum] PhaseStep resolved: %s", text)
        return text


