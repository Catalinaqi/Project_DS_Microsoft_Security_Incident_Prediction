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


# @dataclass
# class BigDataConfig:
#     """
#     Controls dataset sampling size during exploratory analysis.
#
#     Parameters:
#     ----------
#     sample_rows_for_eda : int
#         Maximum number of rows loaded when the dataset is extremely large.
#         Helps prevent memory exhaustion and performance bottlenecks.
#
#     When to modify:
#         - Datasets >= 2–5 million rows → increase or decrease as needed
#         - Slow machines → reduce sampling
#         - GPU/cluster → increase sampling
#     """

    # sample_rows_for_eda: int = 50_000
    #
    # def __post_init__(self):
    #     logger.info(
    #         "[BigDataConfig] EDA sampling rows set to: %d",
    #         self.sample_rows_for_eda,
    #     )



@dataclass
class BigDataConfig:
    """
    Controls dataset sampling size during EDA and training.

    sample_rows_for_eda : int
        Max rows for Phase 2 (EDA).
    sample_rows_for_training : int | None
        Max rows for Phase 3 (data preparation / training).
        If None -> use full dataset.
    """

    sample_rows_for_eda: int = 50_000 # max rows for EDA
    sample_rows_for_training: int | None = 300_000  # max rows for training, None = full data

    def __post_init__(self):
        logger.info(
            "[BigDataConfig] EDA sampling rows set to: %d",
            self.sample_rows_for_eda,
        )
        if self.sample_rows_for_training is None:
            logger.info("[BigDataConfig] Training sampling: FULL DATASET")
        else:
            logger.info(
                "[BigDataConfig] Training sampling rows set to: %d",
                self.sample_rows_for_training,
            )

