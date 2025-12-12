"""
FASE 5 – EVALUATION & INTERPRETATION (CRISP-DM)

In questa fase vengono calcolate tutte le metriche e visualizzazioni
finali del modello addestrato in Fase 4:

    5.1 Estrarre la conoscenza (interpretabilità)
    5.2 Valutare i risultati (metriche)
    5.3 Rivedere il processo (error analysis → lato notebook)
    5.4 Determinare i passi successivi

I risultati vengono salvati in:

    out/phase5_evaluation/

Questo modulo è completamente modulare e delega la logica alle funzioni
in:  src.crispml.common.evaluation.*
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
from src.crispml.config.project.project_config import ProjectConfig
#from src.crispml.config.enums.enums_config import ProblemType

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
from src.crispml.common.evaluation.metrics_classification import (
    compute_classification_metrics,
)
from src.crispml.common.evaluation.metrics_regression import (
    compute_regression_metrics,
)
from src.crispml.common.evaluation.metrics_clustering import (
    compute_clustering_metrics,
)
from src.crispml.common.evaluation.metrics_timeseries import (
    compute_ts_metrics,
)

# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
from src.crispml.common.evaluation.visualization_classification import (
    plot_confusion_matrix as confusion_matrix_figure,
    plot_roc_curve as roc_curve_figure,
)
from src.crispml.common.evaluation.visualization_regression import (
    plot_residuals  as residuals_figure,
)

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
from src.crispml.common.output import (
    save_figure,
    save_table_as_image,
)

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
from src.crispml.common.logging.logging_utils import get_logger


# CONFIG ENUMS
from src.crispml.config.enums.enums_config import ProblemType
from src.crispml.config.enums.enums_config import PhaseName
PHASE_NAME = PhaseName.PHASE5_EVALUATION

#problem_type_str: str = ProblemType.name.lower()

#PHASE_NAME = "phase5_evaluation"
logger = get_logger(__name__)


def run_phase5(
        config: ProjectConfig,
        models: Dict[str, Any],
        splits: Dict[str, Any],
        df_prepared: pd.DataFrame | None = None,
        extra: Dict[str, Any] | None = None,
) -> None:
    """
    Esegue la Fase 5 – Evaluation.

    Parameters
    ----------
    config : ProjectConfig
        Configurazione del progetto.
    models : dict
        Modelli addestrati in Fase 4.
    splits : dict
        Contiene X_test, y_test, ecc. (da Fase 3).
    extra : dict opzionale
        Per time series → contiene y_test.
    """

    logger.info("=== START FASE 5 – EVALUATION (%s) ===", config.name)

    problem_type = config.datasetConfig.problem_type
    extra = extra or {}

    X_test = splits.get("X_test")
    y_test = splits.get("y_test")

    # =========================================================
    # CLUSTERING
    # =========================================================
    if problem_type == ProblemType.CLUSTERING:
        logger.info("[FASE5][CLUSTERING] Calcolo metriche clustering...")

        #metrics_df = compute_clustering_metrics(X_test, models)

        #X = df_prepared if df_prepared is not None else X_test
        X = splits.get("X_full")
        if X is None:
            X = df_prepared   # fallback

        metrics_df = compute_clustering_metrics(X, models)

        save_table_as_image(
            metrics_df,
            f"{problem_type}_01_clustering_metrics.png",
            PHASE_NAME,
        )

    # =========================================================
    # CLASSIFICATION
    # =========================================================
    elif problem_type == ProblemType.CLASSIFICATION:
        logger.info("[FASE5][CLASSIFICATION] Calcolo metriche classificazione...")

        metrics_df = compute_classification_metrics(models, X_test, y_test)

        save_table_as_image(
            metrics_df,
            f"{problem_type}_01_classification_metrics.png",
            PHASE_NAME,
        )

        # Visualizzazioni: confusion matrix + ROC solo per il primo modello
        if models:
            model_name = list(models.keys())[0]
            model = models[model_name]

            # Confusion Matrix
            fig_cm = confusion_matrix_figure(model, X_test, y_test)
            save_figure(fig_cm, f"{problem_type}_02_confusion_matrix_{model_name}.png", PHASE_NAME)

            # ROC curve (se possibile)
            try:
                fig_roc, auc = roc_curve_figure(model, X_test, y_test)
                save_figure(fig_roc, f"{problem_type}_03_roc_curve_{model_name}.png", PHASE_NAME)

                auc_df = pd.DataFrame({"model": [model_name], "AUC": [auc]})
                save_table_as_image(auc_df, f"{problem_type}_04_auc_table_{model_name}.png", PHASE_NAME)

            except Exception as e:
                logger.warning("[FASE5][CLASSIFICATION] ROC non disponibile → %s", e)

    # =========================================================
    # REGRESSION
    # =========================================================
    elif problem_type == ProblemType.REGRESSION:
        logger.info("[FASE5][REGRESSION] Calcolo metriche regressione...")

        metrics_df = compute_regression_metrics(models, X_test, y_test)

        save_table_as_image(
            metrics_df,
            f"{problem_type}_01_regression_metrics.png",
            PHASE_NAME,
        )

        # Residual plot per il miglior modello
        if not metrics_df.empty:
            best = metrics_df.sort_values("R2", ascending=False)["model"].iloc[0]
            best_model = models[best]
            y_pred = best_model.predict(X_test)

            fig_res = residuals_figure(y_test, y_pred)
            save_figure(fig_res, f"{problem_type}_02_residuals_{best}.png", PHASE_NAME)

    # =========================================================
    # TIME SERIES
    # =========================================================
    elif problem_type == ProblemType.TIME_SERIES:
        logger.info("[FASE5][TS] Valutazione serie temporali...")

        y_test = extra.get("y_test", y_test)
        if y_test is None:
            raise ValueError("Per le serie temporali è necessario y_test.")

        if not models:
            raise ValueError("Nessun modello TS ricevuto da Fase 4.")

        model_name = list(models.keys())[0]
        model = models[model_name]

        n_steps = len(y_test)
        y_pred = model.forecast(steps=n_steps)

        # Metriche TS
        metrics_df = compute_ts_metrics(np.asarray(y_test), np.asarray(y_pred))

        save_table_as_image(
            metrics_df,
            f"{problem_type}_01_ts_metrics_{model_name}.png",
            PHASE_NAME,
        )

        # Forecast plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_test, label="Reale")
        ax.plot(y_pred, label="Forecast")
        ax.set_title(f"Time Series – Real vs Forecast ({model_name})")
        ax.legend()
        fig.tight_layout()

        save_figure(fig, f"{problem_type}_02_ts_forecast_{model_name}.png", PHASE_NAME)

    else:
        raise ValueError(f"[FASE5] Tipo di problema non supportato: {problem_type}")

    logger.info("=== END FASE 5 – risultati generati in out/%s ===", PHASE_NAME)
