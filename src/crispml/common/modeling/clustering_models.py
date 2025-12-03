from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

from sklearn.cluster import KMeans, DBSCAN
from src.crispml.common.logging.logging_utils import get_logger

logger = get_logger(__name__)




def _expand(value):
    """Convierte un valor simple en lista si no lo es."""
    return value if isinstance(value, list) else [value]


def run_clustering_algos(
        X: np.ndarray,
        algos: List[str],
        hyperparams: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Ejecuta algoritmos de clustering con soporte para grids de hiperparámetros.
    Soporta:
    - KMeans con lista de n_clusters
    - DBSCAN con grids en eps y min_samples
    """
    models = {}

    for algo in algos:
        hp = hyperparams.get(algo, {})

        # ====================================
        # KMEANS
        # ====================================
        if algo == "kmeans":
            k_list = _expand(hp.get("n_clusters", 3))

            for k in k_list:
                model_name = f"kmeans_k{k}"
                model = KMeans(n_clusters=k, random_state=42)
                model.fit(X)

                models[model_name] = model
                logger.info("[CLUSTERING] Trained %s (k=%d)", model_name, k)

        # ====================================
        # DBSCAN
        # ====================================
        elif algo == "dbscan":
            eps_list = _expand(hp.get("eps", 0.5))
            min_samples_list = _expand(hp.get("min_samples", 5))

            for eps in eps_list:
                for ms in min_samples_list:
                    model_name = f"dbscan_eps{eps}_ms{ms}"
                    model = DBSCAN(eps=eps, min_samples=ms)
                    model.fit(X)

                    models[model_name] = model
                    logger.info(
                        "[CLUSTERING] Trained %s (eps=%.3f, min_samples=%d)",
                        model_name, eps, ms
                    )

        else:
            logger.warning("[CLUSTERING] Unknown algorithm: %s", algo)

    return models

