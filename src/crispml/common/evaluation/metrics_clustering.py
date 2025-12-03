# src/crispml/common/evaluation/metrics_clustering.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
# package imports COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)

def compute_clustering_metrics(X, models):
    """
    Computes clustering metrics: silhouette, Davies-Bouldin, inertia.
    """
    rows = []

    for name, model in models.items():
        try:
            labels = getattr(model, "labels_", None)
            if labels is None:
                labels = model.predict(X)
        except Exception:
            logger.warning("[evaluation] Cannot obtain labels for %s", name)
            continue

        if len(set(labels)) <= 1:
            sil = np.nan
            db = np.nan
        else:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)

        inertia = getattr(model, "inertia_", np.nan)

        rows.append({
            "model": name,
            "silhouette": sil,
            "davies_bouldin": db,
            "inertia": inertia,
        })

        logger.info(
            "[eval] Clustering %s: silhouette=%.4f DB=%.4f inertia=%s",
            name, sil, db, inertia
        )

    return pd.DataFrame(rows)
