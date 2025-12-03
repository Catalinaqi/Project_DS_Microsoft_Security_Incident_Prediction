# src/crispml/common/quality/outlier_detection.py

"""
Outlier diagnostics (Phase 2) — detection only.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.output import save_figure
import matplotlib.pyplot as plt

logger = get_logger(__name__)


def detect_outliers_iqr(df: pd.DataFrame, phase_name: str = "phase2_data_understanding"):
    """
    Computes outlier boundaries (IQR) per numeric column.
    Produces a per-column boxplot image.
    """
    num_cols = df.select_dtypes(include=np.number).columns

    logger.info("[QUALITY][OUTLIERS] Starting IQR-based outlier detection on %d columns", len(num_cols))

    fig, ax = plt.subplots(figsize=(10, 6))
    df[num_cols].boxplot(ax=ax)
    ax.set_title("Boxplot – Outlier Detection (IQR)")
    save_figure(fig, "boxplot_outliers.png", phase_name)

    return num_cols.tolist()
