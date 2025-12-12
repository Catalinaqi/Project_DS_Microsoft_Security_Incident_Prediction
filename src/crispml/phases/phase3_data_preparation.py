"""
FASE 3 – DATA PREPARATION (CRISP-DM)

Questa fase applica SOLO trasformazioni REALI sui dati:

    3.1 – Selezione dei dati (feature selection)
    3.2 – Data Cleaning (imputazione, outlier treatment, duplicati)
    3.3 – Data Transformation (log-transform, encoding, scaling)
    3.4 – Data Integration (placeholder)
    3.5 – Formattazione dei dati (train/val/test split)

Le tecniche sono controllate da:
    config.techniques.phase3.*

Tutti i grafici e tabelle vengono salvati in:
    out/phase3_data_preparation/

Rispetto alla FASE 2:
- Qui SI MODIFICANO i dati
- Qui SI RIPULISCE IL DATASET DEFINTIVO
- Qui si costruiscono X, y e gli split temporali
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.io import load_dataset
from src.crispml.common.feature_selection import (
    select_features_auto,
    select_features_include,
    select_features_exclude,
)

# --- Preprocessing (Phase 3 REAL transformations) ---
from src.crispml.common.preprocessing.missing_values_utils import drop_high_nan_columns, simple_imputation
from src.crispml.common.preprocessing.outlier_utils import treat_outliers
from src.crispml.common.preprocessing.categorical_utils import encode_category_utils
from src.crispml.common.preprocessing.scaling_utils import scale_features
from src.crispml.common.preprocessing.split_utils import train_val_test_split
from src.crispml.common.preprocessing.cleaning_utils import remove_duplicates, apply_log_transform


# --- Output / report images ---
from src.crispml.common.output import (
    save_figure,
    save_table_as_image,
)


# CONFIG
from src.crispml.config.enums.enums_config import ProblemType
from src.crispml.config.enums.enums_config import PhaseName
from src.crispml.config.project.project_config import ProjectConfig
from src.crispml.config.techniques import Phase3Techniques

PHASE_NAME = PhaseName.PHASE3_DATA_PREPARATION

#PHASE_NAME = "phase3_data_preparation"
logger = get_logger(__name__)


# =====================================================================
# SMALL UTILITY: diagnostic plots AFTER CLEANING (not part of Fase 2)
# =====================================================================
def _plot_hist_after_cleaning(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Creates a small set of histograms after cleaning/transformation.
    Only for visual confirmation.
    """
    cols = numeric_cols[:6]
    if not cols:
        return

    logger.info("[FASE3][DIAG] Plotting histograms after cleaning...")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(col)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle("Histograms After Data Cleaning / Transformation")
    fig.tight_layout()

    save_figure(fig,
                #"01_hist_after_cleaning.png",
                f"{ProblemType.name}_01_hist_after_cleaning.png",
                PHASE_NAME)



# =====================================================================
# MAIN FUNCTION
# =====================================================================
def run_phase3(config: ProjectConfig) -> tuple[Dict[str, Any], pd.DataFrame]:
    """
    Esegue l'intera FASE 3 – DATA PREPARATION.

    Returns
    -------
    splits : dict
        X_train, X_val, X_test, y_train, y_val, y_test (+ eventual time index)
    df_prepared : pd.DataFrame
        Dataset finale dopo cleaning + transformation.
    """

    logger.info("=== START PHASE 3 – DATA PREPARATION (%s) ===", config.name)

    problem_type = config.datasetConfig.problem_type

    t: Phase3Techniques = config.techniquesConfig.phase3
    df_trans: pd.DataFrame


    # --------------------------------------------------------
    # 3.1 – Load full dataset + Feature Selection
    # --------------------------------------------------------
    logger.info("[FASE3][3.1] Loading full dataset...")
    df_full = load_dataset(config.datasetConfig, config.bigDataConfig, for_eda=False)

    df_sel, numeric_cols, categorical_cols = select_features_auto(
        df_full, config.datasetConfig, config.featureConfig
    )

    logger.info(
        "[FASE3] After feature selection → shape=%s (num=%d, cat=%d)",
        df_sel.shape,
        len(numeric_cols),
        len(categorical_cols),
    )

    # --------------------------------------------------------
    # 3.2 – DATA CLEANING (transformazioni reali)
    # --------------------------------------------------------
    logger.info("[FASE3][3.2] Running Data Cleaning...")
    df_clean = df_sel.copy()

    # Drop columns with too many NaN
    if t.cleaning.drop_high_nan:
        logger.info("[FASE3][CLEAN] Dropping columns with high NaN ratio...")
        df_clean = drop_high_nan_columns(df_clean, max_nan_ratio=0.8)

    # Simple imputation
    if t.cleaning.simple_imputation:
        logger.info("[FASE3][CLEAN] Applying simple imputation (median / most frequent)...")
        df_clean = simple_imputation(
            df_clean,
            numeric_strategy="median",
            categorical_strategy="most_frequent",
        )

    # Outlier treatment (winsorize)
    if t.cleaning.outlier_treatment:
        logger.info("[FASE3][CLEAN] Treating outliers (winsorization)...")
        df_clean = treat_outliers(df_clean, method="winsorize")

    # Remove duplicates
    if t.cleaning.duplicates_handling:
        logger.info("[FASE3][CLEAN] Removing duplicates...")
        df_clean = remove_duplicates(df_clean)

    # Missing after cleaning report
    missing_after = df_clean.isna().mean().mul(100).round(2)
    rep_missing = (
        missing_after.reset_index()
        .rename(columns={"index": "column", 0: "missing_percent"})
        .sort_values("missing_percent", ascending=False)
    )

    # problem_type
    save_table_as_image(rep_missing,
                        f"{problem_type}_02_missing_after_cleaning.png",
                        PHASE_NAME)

    # --------------------------------------------------------
    # 3.3 – DATA TRANSFORMATION (log, encoding, scaling)
    # --------------------------------------------------------
    logger.info("[FASE3][3.3] Running Data Transformation...")
    df_trans = df_clean.copy()

    # Log-transform (automatic based on skewness)
    if t.transformation.nonlinear_transform and numeric_cols:

        skew = df_trans[numeric_cols].skew().sort_values(ascending=False)
        log_cols = [
            col for col, val in skew.items()
            if abs(val) > 1 and df_trans[col].min() >= 0
        ]
        if log_cols:
            logger.info("[FASE3][TRANSF] Applying log-transform to %d columns", len(log_cols))
        df_trans = apply_log_transform(df_trans, log_cols)

        if log_cols:
            df_log_info = pd.DataFrame(
                {"column": log_cols, "skew": [skew[c] for c in log_cols]}
            )
            save_table_as_image(df_log_info,
                                f"{problem_type}_03_log_transformed_columns.png",
                                PHASE_NAME)

    # Encoding
    if t.transformation.encoding:

        logger.info("[FASE3][TRANSF] Encoding categorical variables (one-hot)...")
        df_encoded = encode_category_utils(
            df_trans,
            encoding="onehot",
            target_col=config.datasetConfig.target_col,
        )
    else:
        df_encoded = df_trans

    # Definire X e y
    target_col = config.datasetConfig.target_col
    if target_col and target_col in df_encoded.columns:
        X = df_encoded.drop(columns=[target_col])
        y = df_encoded[target_col].values
    else:
        X = df_encoded
        y = None

    # Scaling
    if t.transformation.scaling:

        logger.info("[FASE3][TRANSF] Scaling features (StandardScaler)...")
        X_scaled = scale_features(X, scaling="standard")
    else:
        X_scaled = np.asarray(X)

    # Diagnostic histograms after transformation
    _plot_hist_after_cleaning(df_encoded, numeric_cols)

    # --------------------------------------------------------
    # 3.5 – Data Formatting (Train/Val/Test Split)
    # --------------------------------------------------------
    logger.info("[FASE3][3.5] Performing Train/Val/Test Split...")

    time_order = None
    if (
            config.datasetConfig.problem_type == ProblemType.TIME_SERIES and
            config.datasetConfig.time_col and
            config.datasetConfig.time_col in df_full.columns
    ):
        time_order = pd.to_datetime(df_full[config.datasetConfig.time_col], errors="coerce")

    #val_size = 0.1 if t.use_validation_split else 0.0
    val_size = 0.1 if t.formatting.train_val_test_split else 0.0


    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X_scaled,
        y,
        problem_type=config.datasetConfig.problem_type,
        test_size=0.2,
        val_size=val_size,
        random_state=42,
        time_order=time_order,
    )

    splits: Dict[str, Any] = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "time_train": None,
        "time_test": None,
    }

    #splits["X_full"] = X_scaled
    splits["X_full"] = df_encoded.values


    if time_order is not None:
        n = len(time_order)
        n_test = int(n * 0.2)
        n_train_val = n - n_test
        splits["time_train"] = time_order.iloc[:n_train_val]
        splits["time_test"] = time_order.iloc[n_train_val:]

    logger.info("=== END FASE 3 – Results saved in out/%s ===", PHASE_NAME)
    return splits, df_encoded
