"""
CRISP-ML Pipeline – Phases Module

This package exposes the main functions for each CRISP-DM phase:
    - Phase 2: Data Understanding
    - Phase 3: Data Preparation
    - Phase 4: Modeling
    - Phase 5: Evaluation

Each phase is implemented in a dedicated file and orchestrates
the corresponding operations using the modules in:
    src.crispml.common.*
"""

from .phase2_data_understanding import run_phase2
from .phase3_data_preparation import run_phase3
from .phase4_modeling import run_phase4
from .phase5_evaluation import run_phase5

__all__ = [
    "run_phase2",
    "run_phase3",
    "run_phase4",
    "run_phase5",
]
