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
def plot_boxplots(df: pd.DataFrame, numeric_cols) -> plt.Figure:
    logger.info("[VIS][BOX] Generating boxplots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    df[numeric_cols].boxplot(ax=ax)
    ax.set_title("Boxplots")
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
def plot_scatter_matrix(df: pd.DataFrame, numeric_cols) -> plt.Figure:
    logger.info("[VIS][SCATTER] Generating scatter matrix...")
    fig = scatter_matrix(df[numeric_cols], figsize=(10, 10), diagonal="kde")
    plt.tight_layout()
    return plt.gcf()


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
