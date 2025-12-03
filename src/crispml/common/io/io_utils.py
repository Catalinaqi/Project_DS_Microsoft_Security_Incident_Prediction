# src/crispml/common/io/io_utils.py

"""
Unified I/O utilities for CRISP-ML.

This module centralizes all dataset loading logic in a clean and
extensible way. It handles:

    - Reading CSV / Parquet / JSON / SQL / Logs (CSV for now)
    - Applying sampling for EDA (BigDataConfig)
    - Logging every step for auditability
    - Providing a stable API for all Phase 2 and Phase 3 modules

The functions here are intentionally small, focused, and easy to extend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# package imports CONFIG DATASET - DATASET CONFIG
from src.crispml.config.dataset.dataset_config import DatasetConfig
# package imports CONFIG DATASET - BIGDATA CONFIG
from src.crispml.config.dataset.bigdata_config import BigDataConfig
# package imports CONFIG ENUMS
from src.crispml.config.enums.enums import DataSourceType
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# INTERNAL: CSV loader with logging
# ---------------------------------------------------------------------
def _load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Internal helper to load CSV files with logging and error handling.
    """

    logger.info("[io] Loading CSV from %s (nrows=%s)", path, nrows)

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"[io] CSV file not found: {path}")

    df = pd.read_csv(path, nrows=nrows)
    logger.info("[io] CSV loaded successfully with shape: %s", df.shape)

    return df


# ---------------------------------------------------------------------
# INTERNAL: placeholder loaders for future formats
# ---------------------------------------------------------------------
def _load_json(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    raise NotImplementedError("JSON loading not implemented yet.")


def _load_parquet(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    raise NotImplementedError("Parquet loading not implemented yet.")


def _load_sql(query: str, nrows: Optional[int] = None) -> pd.DataFrame:
    raise NotImplementedError("SQL loading not implemented yet.")


# ---------------------------------------------------------------------
# PUBLIC: main dataset loading function
# ---------------------------------------------------------------------
def load_dataset(
        dataset_cfg: DatasetConfig,
        bigdata_cfg: BigDataConfig,
        for_eda: bool = False,
) -> pd.DataFrame:
    """
    Main dataset loader used throughout CRISP-ML.

    Parameters
    ----------
    dataset_cfg : DatasetConfig
        Contains dataset source type, path, target, id columns, etc.

    bigdata_cfg : BigDataConfig
        Sampling configuration for extremely large datasets.

    for_eda : bool
        If True, applies row sampling to speed up exploratory analysis.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.

    Notes
    -----
    - All logging is done here: loading start, sampling applied, final shape.
    - You can extend this function to support remote storage (S3, GCS, APIs).
    """

    logger.info("[io] Starting dataset load (EDA mode: %s)", for_eda)

    # Determine if sampling is needed
    nrows = None
    if for_eda and bigdata_cfg.sample_rows_for_eda:
        nrows = bigdata_cfg.sample_rows_for_eda
        logger.info(
            "[io] Applying EDA sampling: limiting to %d rows",
            nrows,
        )

    # -----------------------------------------------------------------
    # Dispatch by data source type
    # -----------------------------------------------------------------
    if dataset_cfg.source_type == DataSourceType.CSV:
        df = _load_csv(dataset_cfg.path_or_conn, nrows=nrows)

    elif dataset_cfg.source_type == DataSourceType.JSON:
        df = _load_json(dataset_cfg.path_or_conn, nrows=nrows)

    elif dataset_cfg.source_type == DataSourceType.PARQUET:
        df = _load_parquet(dataset_cfg.path_or_conn, nrows=nrows)

    elif dataset_cfg.source_type == DataSourceType.SQL:
        df = _load_sql(dataset_cfg.path_or_conn, nrows=nrows)

    else:
        raise NotImplementedError(
            f"[io] Unsupported DataSourceType: {dataset_cfg.source_type}"
        )

    logger.info("[io] Dataset successfully loaded.")
    logger.debug("[io] Sample head:\n%s", df.head())

    return df
