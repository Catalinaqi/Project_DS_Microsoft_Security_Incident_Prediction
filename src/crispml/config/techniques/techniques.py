# src/crispml/config/techniques/techniques.py

"""
Unified Techniques Configuration for CRISP-ML.

This module acts as the configuration hub for all preprocessing
techniques used in:

    - Phase 2 (Data Understanding)
    - Phase 3 (Data Preparation)

It wraps Phase2Techniques and Phase3Techniques into a single
TechniquesConfig object, which is then embedded inside ProjectConfig.

The design allows the system to automatically adapt the preprocessing
pipeline depending on the ML problem type (clustering, classification,
regression, time-series).
"""

from __future__ import annotations
from dataclasses import dataclass

# package imports CONFIG ENUMS
from src.crispml.config.enums.enums_config import ProblemType
# package imports CONFIG TECHNIQUES
from src.crispml.config.techniques.techniques_phase2 import (
    Phase2Techniques,
    default_phase2_for_problem,
)
# package imports CONFIG TECHNIQUES
from src.crispml.config.techniques.techniques_phase3 import (
    Phase3Techniques,
    default_phase3_for_problem,
)

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TechniquesConfig:
    """
    Container object for all CRISP-ML preprocessing techniques.

    Attributes:
    ----------
    phase2 : Phase2Techniques
        Techniques applied during Phase 2 (Data Understanding).
        Includes: describing data, quality checks, EDA visuals.

    phase3 : Phase3Techniques
        Techniques applied during Phase 3 (Data Preparation).
        Includes: cleaning, transformation, integration, formatting.

    The structure is intentionally modular:
        - easy to extend
        - easy to customize
        - clear separation of concerns

    Future extension points:
        - Phase4Techniques (modeling strategies)
        - Phase5Techniques (evaluation, interpretation)
        - per-algorithm preprocessing flags
    """

    phase2: Phase2Techniques
    phase3: Phase3Techniques

    def __post_init__(self):
        logger.info("[TechniquesConfig] Techniques configuration initialized.")
        logger.debug("[TechniquesConfig] Phase 2 techniques: %s", self.phase2)
        logger.debug("[TechniquesConfig] Phase 3 techniques: %s", self.phase3)

    # ------------------------------------------------------------------
    # Factory method to build a TechniquesConfig based on ProblemType
    # ------------------------------------------------------------------
    @classmethod
    def for_problem(cls, problem_type: ProblemType) -> "TechniquesConfig":
        """
        Build a TechniquesConfig tailored to the ML problem type.

        Parameters:
        -----------
        problem_type : ProblemType
            Determines which preprocessing techniques are most useful.

        How it works:
        - Calls default_phase2_for_problem() → enables/disables Phase 2 techniques
        - Calls default_phase3_for_problem() → enables/disables Phase 3 techniques
        - Builds a unified configuration object

        Example:
            TechniquesConfig.for_problem(ProblemType.CLASSIFICATION)

        Logging:
            Logs every step to trace how the system chooses the
            preprocessing pipeline dynamically.
        """

        logger.info("[TechniquesConfig] Creating configuration for problem type: %s",
                    problem_type.name)

        phase2_cfg = default_phase2_for_problem(problem_type)
        phase3_cfg = default_phase3_for_problem(problem_type)

        logger.info(
            "[TechniquesConfig] Phase 2 ready for %s", problem_type.name
        )
        logger.info(
            "[TechniquesConfig] Phase 3 ready for %s", problem_type.name
        )

        logger.debug("[TechniquesConfig] Phase 2 config:\n%s", phase2_cfg)
        logger.debug("[TechniquesConfig] Phase 3 config:\n%s", phase3_cfg)

        return cls(phase2=phase2_cfg, phase3=phase3_cfg)
