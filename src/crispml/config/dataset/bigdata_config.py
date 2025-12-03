# src/crispml/config/dataset/bigdata_config.py

"""
Big-data handling configuration for CRISP-ML.

This configuration helps control memory usage and performance when
loading extremely large datasets (CSV, Parquet, Logs, etc.).

It is used mainly in Phase 2 (Data Understanding) during:
    - initial sampling
    - EDA operations
    - plotting
"""

from __future__ import annotations
from dataclasses import dataclass

# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BigDataConfig:
    """
    Controls dataset sampling size during exploratory analysis.

    Parameters:
    ----------
    sample_rows_for_eda : int
        Maximum number of rows loaded when the dataset is extremely large.
        Helps prevent memory exhaustion and performance bottlenecks.

    When to modify:
        - Datasets >= 2–5 million rows → increase or decrease as needed
        - Slow machines → reduce sampling
        - GPU/cluster → increase sampling
    """

    sample_rows_for_eda: int = 50_000

    def __post_init__(self):
        logger.info(
            "[BigDataConfig] EDA sampling rows set to: %d",
            self.sample_rows_for_eda,
        )
