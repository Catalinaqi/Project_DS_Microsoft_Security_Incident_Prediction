"""
FASE 4 – MODELING (CRISP-DM)

In questa fase si esegue:

    4.1 – Selezionare la tecnica
    4.2 – Costruire il modello
    4.3 – Applicare il progetto di test (X_test, y_test già generati in Fase 3)
    4.4 – Preparare i risultati per la Fase 5 (Evaluation)

Questa fase NON calcola direttamente le metriche finali
(ma può calcolare metriche rapide di diagnostica). Le metriche
principali verranno calcolate in Fase 5 – Evaluation.

La logica di training è completamente delegata ai moduli:
    src.crispml.common.modeling.*
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


# --- CONFIG ---
from src.crispml.config.project.project_config import ProjectConfig
from src.crispml.config.enums.enums import ProblemType

# --- MODELING ENGINE ---
from src.crispml.common.modeling import (
    run_clustering_algos,
    run_classification_algos,
    run_regression_algos,
    run_time_series_algos,
)

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
from src.crispml.common.output import (
    save_figure,
    save_table_as_image,
)


# --- LOGGING ---
from src.crispml.common.logging.logging_utils import get_logger

PHASE_NAME = "phase4_modeling"
logger = get_logger(__name__)


def run_phase4(
        config: ProjectConfig,
        splits: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Esegue la FASE 4 – MODELING.

    Parameters
    ----------
    config : ProjectConfig
        Configurazione complessiva del progetto.
    splits : dict
        Risultati della FASE 3 (X_train, X_val, X_test, y_train, y_test).

    Returns
    -------
    models : dict
        Nome_modello -> modello addestrato.
    extra : dict
        Informazioni addizionali utili in Fase 5.
    """

    logger.info("=== START FASE 4 – MODELING (%s) ===", config.name)

    problem_type = config.dataset.problem_type
    models: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}

    # Estrarre gli split generati in Fase 3
    X_train = splits.get("X_train")
    X_test = splits.get("X_test")
    y_train = splits.get("y_train")
    y_test = splits.get("y_test")

    # ---------------------------------------------------------
    # 4.1 – SELEZIONE DELLA TECNICA
    # ---------------------------------------------------------
    logger.info("[FASE4][4.1] ProblemType rilevato: %s", problem_type)

    # ---------------------------------------------------------
    # 4.2 – COSTRUZIONE E TRAINING DEL MODELLO
    # ---------------------------------------------------------
    if problem_type == ProblemType.CLUSTERING:

        logger.info("[FASE4][CLUSTERING] Algoritmi: %s",
                    config.modeling.clustering_algos)


        full_X = splits.get("X_full")
        if full_X is None:
            raise ValueError("X_full non trovato in splits. Aggiungilo in Fase 3.")

        # --- entrenar modelos ---
        models = run_clustering_algos(
            X=full_X,
            algos=config.modeling.clustering_algos,
            hyperparams=config.modeling.hyperparams,
        )

        # ---------------------------------------------------------
        # NUEVO: generar tabla de resultados de Fase 4
        # ---------------------------------------------------------
        #from src.crispml.common.output.output_utils import save_table_as_image

        rows = []

        for name, model in models.items():

            # obtener labels
            try:
                labels = getattr(model, "labels_", None)
                if labels is None:
                    labels = model.predict(full_X)
            except Exception:
                labels = None

            # metrica básica: número de clusters
            if labels is not None:
                n_clusters = len(set(labels))
                counts = pd.Series(labels).value_counts().to_dict()
            else:
                n_clusters = np.nan
                counts = {}

            inertia = getattr(model, "inertia_", np.nan)

            rows.append({
                "model": name,
                "n_clusters": n_clusters,
                "inertia": inertia,
                "labels_count": str(counts)
            })

        # DataFrame final de fase 4
        modeling_results_df = pd.DataFrame(rows)

        # guardar imagen
        save_table_as_image(
            modeling_results_df,
            filename="01_modeling_clustering_results.png",
            phase_name=PHASE_NAME,
        )

        logger.info(
            "[FASE4][CLUSTERING] Resultados de modelado guardados en 01_modeling_clustering_results.png"
        )

        # guardar datos en extra para Fase 5
        extra["X_for_clustering"] = full_X
        extra["cluster_labels"] = {row["model"]: row["labels_count"] for row in rows}



    elif problem_type == ProblemType.CLASSIFICATION:

        logger.info("[FASE4][CLASSIFICATION] Algoritmi: %s",
                    config.modeling.classification_algos)

        models, _ = run_classification_algos(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            algos=config.modeling.classification_algos,
            hyperparams=config.modeling.hyperparams,
        )

    elif problem_type == ProblemType.REGRESSION:

        logger.info("[FASE4][REGRESSION] Algoritmi: %s",
                    config.modeling.regression_algos)

        models, _ = run_regression_algos(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            algos=config.modeling.regression_algos,
            hyperparams=config.modeling.hyperparams,
        )

    elif problem_type == ProblemType.TIME_SERIES:

        logger.info("[FASE4][TS] Algoritmi: %s",
                    config.modeling.ts_algos)

        if y_train is None:
            raise ValueError("Per il problema TIME_SERIES, y_train deve essere disponibile.")

        models = run_time_series_algos(
            y_train=pd.Series(y_train),
            algos=config.modeling.ts_algos,
            hyperparams=config.modeling.hyperparams,
        )

        # Info necessarie per FASE 5
        extra["y_test"] = y_test

    else:
        raise ValueError(f"Tipo di problema non supportato in Fase 4: {problem_type}")

    # ---------------------------------------------------------
    # 4.4 – OUTPUT PER FASE 5
    # ---------------------------------------------------------
    logger.info("[FASE4] Modelli addestrati: %s", list(models.keys()))
    logger.info("=== END FASE 4 – MODELING ===")

    return models, extra
