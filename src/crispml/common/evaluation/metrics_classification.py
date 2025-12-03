# src/crispml/common/evaluation/metrics_classification.py

from __future__ import annotations
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)

def compute_classification_metrics(models, X_test, y_test):
    """
    Computes classification metrics: accuracy, precision, recall, F1 (macro).
    """
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        rows.append({
            "model": name,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        })

        logger.info(
            "[eval] Classifier %s: acc=%.4f f1=%.4f",
            name, acc, f1
        )

    return pd.DataFrame(rows)
