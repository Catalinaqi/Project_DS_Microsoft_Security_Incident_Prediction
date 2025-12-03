# src/crispml/common/evaluation/visualization_regression.py

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    return fig
