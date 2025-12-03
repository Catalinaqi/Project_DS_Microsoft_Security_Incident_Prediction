# src/crispml/config/dataset/dataset_config.py

"""
Dataset configuration module for CRISP-ML.

Defines how the dataset is loaded, what columns have special meaning
(target, time, ID), and how large datasets should be sampled for EDA.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
# package imports CONFIG ENUMS
from src.crispml.config.enums.enums import (
    DataSourceType,
    ProblemType,
)
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """
    Describes the dataset characteristics and structure.

    Parameters:
    ----------
    source_type : DataSourceType
        Defines the type of input (CSV, JSON, SQL, etc.)

    path_or_conn : str
        File path or database/API connection string.

    problem_type : ProblemType
        Type of ML task (clustering, classification, regression, ts).

    target_col : str | None
        Required for supervised learning.

    time_col : str | None
        Required for time-series learning.

    id_cols : list[str]
        Columns to ignore as modeling features.
    """

    source_type: DataSourceType
    path_or_conn: str
    problem_type: ProblemType
    target_col: Optional[str] = None
    time_col: Optional[str] = None
    id_cols: List[str] = field(default_factory=list)

    def __post_init__(self):
        logger.info(
            "[DatasetConfig] source=%s | path=%s | task=%s | target=%s | time=%s | id_cols=%s",
            self.source_type.name,
            self.path_or_conn,
            self.problem_type.name,
            self.target_col,
            self.time_col,
            self.id_cols,
        )
