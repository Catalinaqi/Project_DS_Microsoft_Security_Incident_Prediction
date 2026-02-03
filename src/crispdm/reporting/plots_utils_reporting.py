# src/crispdm/reporting/plots_utils_reporting.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from crispdm.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Plot consistency:
# - Stages produce "visual evidence" (PNG figures) in a consistent style.
# - Centralized plotting avoids duplicated code across stages/tasks.
#
# Program flow:
# - stages call plots_utils_reporting.* to create figures
# - artifacts_service_reporting.save_figure(...) persists them
#
# Design patterns:
# - Utility module (reporting)
# =============================================================================


def plot_missingness_top(
        df: pd.DataFrame,
        *,
        top_n: int = 20,
        title: str = "Top missingness (%)",
):
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)
    miss = miss.iloc[::-1]  # nicer for barh

    fig = plt.figure(figsize=(10, 6))
    plt.barh(miss.index.astype(str), miss.values)
    plt.title(title)
    plt.xlabel("missing %")
    return fig


def plot_target_distribution(
        s: pd.Series,
        *,
        title: str = "Target distribution",
        top_n: int = 30,
):
    vc = s.value_counts(dropna=False).head(top_n)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    return fig


def plot_numeric_hist(
        df: pd.DataFrame,
        col: str,
        *,
        title: Optional[str] = None,
        bins: int = 30,
):
    fig = plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna().values, bins=bins)
    plt.title(title or f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    return fig


def plot_residuals(y_true, y_pred, *, title: str = "Residuals"):
    resid = (y_true - y_pred)
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, s=10)
    plt.axhline(0)
    plt.title(title)
    plt.xlabel("y_pred")
    plt.ylabel("residual (y_true - y_pred)")
    return fig


def plot_confusion_matrix(cm, labels: Sequence[str], *, title: str = "Confusion matrix"):
    # expects cm already computed (e.g. sklearn.metrics.confusion_matrix)
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    return fig


def plot_cluster_sizes(labels, *, title: str = "Cluster sizes"):
    s = pd.Series(labels).value_counts().sort_index()
    fig = plt.figure(figsize=(8, 4))
    plt.bar(s.index.astype(str), s.values)
    plt.title(title)
    plt.xlabel("cluster")
    plt.ylabel("count")
    return fig
