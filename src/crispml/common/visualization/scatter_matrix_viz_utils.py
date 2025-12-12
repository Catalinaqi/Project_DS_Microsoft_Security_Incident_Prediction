# scatter_matrix_viz_utils.py

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def plot_scatter_matrix(
        df: pd.DataFrame,
        numeric_cols: list[str],
        max_cols: int = 6,
        sample_size: int = 50_000,
) -> plt.Figure:
    """
    Generate a readable scatter matrix for exploratory analysis.
    Applies column limiting and row sampling to avoid overplotting.
    """

    logger.info(
        "[VIS][SCATTER] Generating scatter matrix "
        f"(max_cols={max_cols}, sample_size={sample_size})..."
    )

    cols = numeric_cols[:max_cols]

    if len(df) > sample_size:
        df_plot = df[cols].sample(sample_size, random_state=42)
        logger.info(
            f"[VIS][SCATTER] Dataset sampled: {sample_size} / {len(df)} rows."
        )
    else:
        df_plot = df[cols]

    scatter_matrix(
        df_plot,
        figsize=(2.5 * len(cols), 2.5 * len(cols)),
        diagonal="hist",
        alpha=0.5
    )

    plt.tight_layout()
    return plt.gcf()
