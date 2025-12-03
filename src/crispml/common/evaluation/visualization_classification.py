# src/crispml/common/evaluation/visualization_classification.py

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(cm, cmap="Blues")
    fig.colorbar(cax, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")

    return fig


def plot_roc_curve(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        raise ValueError("Model does not support probability/score for ROC.")

    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax.legend()
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")

    logger.info("[eval] ROC AUC=%.4f", auc_val)
    return fig, auc_val
