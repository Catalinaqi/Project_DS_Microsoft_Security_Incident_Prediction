"""
EDA Visualization Utilities (Phase 2 – Data Understanding)

This module provides reusable plot functions for:
- Histograms
- Boxplots
- Correlation heatmaps
- Scatter matrix
- PCA 2D
- Feature–target plots
- Time-series preview

These plots are DIAGNOSTIC ONLY.
They do not modify the dataset.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.visualization.boxplot_viz_utils import filter_columns_for_boxplot
from src.crispml.common.visualization.scatter_matrix_viz_utils import plot_scatter_matrix

logger = get_logger(__name__)


# ---------------------------------------------------------
# Histograms
# ---------------------------------------------------------
def plot_histograms(df: pd.DataFrame, numeric_cols) -> plt.Figure:
    logger.info("[VIS][HIST] Generating histograms...")
    fig = df[numeric_cols].hist(figsize=(12, 8))
    plt.tight_layout()
    return plt.gcf()


# ---------------------------------------------------------
# Boxplots
# ---------------------------------------------------------
# def plot_boxplots(df: pd.DataFrame, numeric_cols) -> plt.Figure:
#     logger.info("[VIS][BOX] Generating boxplots...")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     df[numeric_cols].boxplot(ax=ax)
#     ax.set_title("Boxplots")
#     plt.tight_layout()
#     return fig

# def plot_boxplots(df: pd.DataFrame, numeric_cols) -> plt.Figure:
#     logger.info("[VIS][BOX] Filtering numeric columns for boxplot...")
#
#     filtered_cols = filter_columns_for_boxplot(df, numeric_cols)
#
#     if not filtered_cols:
#         logger.info("[VIS][BOX] No suitable numeric columns for boxplot. Skipping.")
#         return None
#
#     logger.info(f"[VIS][BOX] Plotting {len(filtered_cols)} filtered numeric columns...")
#
#     fig, ax = plt.subplots(figsize=(12, 6))
#     df[filtered_cols].boxplot(ax=ax)
#     ax.set_title("Filtered Boxplots")
#     # Rotate x-axis labels
#     plt.xticks(rotation=45, ha='right')
#     #plt.xticks(rotation=60, ha='right')
#     plt.tight_layout()
#     return fig

def plot_boxplots(df: pd.DataFrame, numeric_cols: list[str]) -> plt.Figure:
    logger.info("[VIS][BOX] Filtering numeric columns for boxplot...")

    filtered_cols = filter_columns_for_boxplot(df, numeric_cols)

    if not filtered_cols:
        logger.info("[VIS][BOX] No suitable numeric columns for boxplot. Skipping.")
        return None

    logger.info(f"[VIS][BOX] Using FAST boxplot rendering for {len(filtered_cols)} columns...")

    # --------- Precompute boxplot statistics (FAST MODE) ---------
    stats = []
    for col in filtered_cols:
        series = df[col].dropna()

        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # boxplot whiskers (IQR-based)
        whisker_min = q1 - 1.5 * iqr
        whisker_max = q3 + 1.5 * iqr

        stats.append({
            "label": col,
            "whislo": whisker_min,
            "q1": q1,
            "med": q2,
            "q3": q3,
            "whishi": whisker_max,
            #"fliers": []  # no outliers drawn to speed up
            "fliers": series[(series < whisker_min) | (series > whisker_max)].tolist()
        })

    # --------- Render FAST boxplot ---------
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bxp(stats, showfliers=False)
    ax.set_title("Filtered Boxplots (FAST Mode)")

    # Rotate labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    return fig


# ---------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------
def plot_corr_heatmap(df: pd.DataFrame, numeric_cols) -> plt.Figure:
    logger.info("[VIS][CORR] Generating correlation heatmap...")
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90)
    ax.set_yticklabels(numeric_cols)

    ax.set_title("Correlation Heatmap", pad=20)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Scatter Matrix
# ---------------------------------------------------------
# def plot_scatter_matrix(df: pd.DataFrame, numeric_cols) -> plt.Figure:
#     logger.info("[VIS][SCATTER] Generating scatter matrix...")
#     fig = scatter_matrix(df[numeric_cols], figsize=(10, 10), diagonal="kde")
#     plt.tight_layout()
#     return plt.gcf()




def generate_scatter_matrix(df, numeric_cols):
    logger.info("[EDA] Scatter matrix step started.")
    fig = plot_scatter_matrix(df, numeric_cols)
    return fig


# ---------------------------------------------------------
# PCA 2D
# ---------------------------------------------------------
def plot_pca_2d(df: pd.DataFrame, numeric_cols) -> plt.Figure:
    logger.info("[VIS][PCA] Generating PCA 2D plot...")

    data = df[numeric_cols].dropna()
    if data.shape[1] < 2:
        logger.warning("[VIS][PCA] Not enough numeric columns for PCA.")
        return None

    pca = PCA(n_components=2)
    proj = pca.fit_transform(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:, 0], proj[:, 1], alpha=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA (2D Projection)")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Feature–Target Plots
# ---------------------------------------------------------
def plot_feature_target_plots(df: pd.DataFrame, numeric_cols, target_col, problem_type):
    """
    For regression → scatter(x, y)
    For classification → boxplot grouped by class
    """
    logger.info("[VIS][FEAT-TGT] Generating Feature–Target plots...")

    if target_col not in df.columns:
        logger.warning("[VIS][FEAT-TGT] Target column missing.")
        return None

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8, 4 * len(numeric_cols)))
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        if problem_type == "regression":
            ax.scatter(df[col], df[target_col], alpha=0.3)
            ax.set_ylabel(target_col)
        else:
            df.boxplot(column=col, by=target_col, ax=ax)

        ax.set_xlabel(col)
        ax.set_title(f"{col} vs {target_col}")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------
# Time Series Preview
# ---------------------------------------------------------
def plot_time_series_preview(df: pd.DataFrame, dataset_config):
    """
    Very basic time-series preview:
    If dataset_config.date_col exists → simple line plot.

    This is DIAGNOSTIC ONLY.
    """

    date_col = dataset_config.date_col
    target = dataset_config.target_col

    if not date_col or not target:
        logger.warning("[VIS][TS] No date_col or target_col for time-series preview.")
        return None

    if date_col not in df.columns or target not in df.columns:
        logger.warning("[VIS][TS] Required TS columns missing.")
        return None

    logger.info("[VIS][TS] Generating time-series preview...")

    df_plot = df[[date_col, target]].dropna()
    df_plot = df_plot.sort_values(date_col)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_plot[date_col], df_plot[target])
    ax.set_title(f"Time Series Preview: {target}")
    ax.set_xlabel(date_col)
    ax.set_ylabel(target)

    plt.tight_layout()
    return fig
