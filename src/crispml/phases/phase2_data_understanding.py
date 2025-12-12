"""
FASE 2 – DATA UNDERSTANDING (CRISP-DM)

In questa fase NON si modificano i dati.
Si eseguono solo attività di diagnostica, esplorazione e descrizione:

    2.1 – Collezionare i dati iniziali
    2.2 – Descrivere i dati
    2.3 – Verificare la qualità dei dati
    2.4 – Esplorare i dati (EDA)

Le tecniche eseguite dipendono dai flag in:
    config.techniques.phase2.*

Tutti i grafici e le tabelle vengono salvati in:
    out/phase2_data_understanding/

Questa implementazione è FULL MODULAR:
- Diagnostica → src.crispml.common.quality
- Visualizzazioni → src.crispml.common.visualization
- Feature selection → src.crispml.common.feature_selection
- Output image/table → src.crispml.common.output
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import pandas as pd

# CONFIG PROJECT
from src.crispml.config.project.project_config import ProjectConfig

# COMMON IO
from src.crispml.common.io import load_dataset
# COMMON FEATURE SELECTION
from src.crispml.common.feature_selection import (
    select_features_auto,
    select_features_include,
    select_features_exclude,
)
# COMMON QUALITY
# --- Quality diagnostics (Phase 2) ---
from src.crispml.common.quality import (
    analyze_missing_values,
    detect_outliers_iqr,
    analyze_duplicates,
    detect_inconsistencies,
    validate_range,
)
# COMMON VISUALIZATION
# --- Visualization (Phase 2 EDA) ---
from src.crispml.common.visualization.eda_plots_viz_utils import (
    plot_histograms,
    plot_boxplots,
    plot_corr_heatmap,
    plot_scatter_matrix,
    plot_pca_2d,
    plot_feature_target_plots,
    plot_time_series_preview,
)
# COMMON OUTPUT
# --- Output (figures / tables) ---
from src.crispml.common.output import (
    save_table_as_image,
    save_figure,
    save_table_as_csv,
)
# COMMON LOGGING
from src.crispml.common.logging.logging_utils import get_logger

# CONFIG ENUMS
# from src.crispml.config.enums.enums_config import ProblemType
from src.crispml.config.enums.enums_config import PhaseName
from src.crispml.config.enums.enums_config import PhaseStep

PHASE_NAME = PhaseName.PHASE2_DATA_UNDERSTANDING

#problem_type_str: str = ProblemType.name.lower()

logger = get_logger(__name__)


# ============================================================
# 2.1 – COLLEZIONARE I DATI INIZIALI
# ============================================================
def _collect_initial_data(config: ProjectConfig) -> pd.DataFrame:
    """
    Reads dataset applying BigData sampling if configured.
    """
    step = PhaseStep.P2_1_COLLECT_INITIAL_DATA

    logger.info(
        "%s... Start loading initial dataset: %s",
        step,
        config.datasetConfig.problem_type.name,
    )

    df = load_dataset(config.datasetConfig, config.bigDataConfig, for_eda=True)
    logger.info(
        "%s... End dataset loaded, shape: %s",
        step,
        df.shape,
    )
    return df


# ============================================================
# 2.2 – DESCRIVERE I DATI (STATISTICHE + GRAFICI)
# ============================================================
def _describe_data(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                   config: ProjectConfig) -> None:
    """
    Runs descriptive statistics and basic visualizations.
    All heavy logic is delegated to visualization utils.
    """
    step = PhaseStep.P2_2_DESCRIBE_DATA
    t = config.techniquesConfig.phase2
    problem_type = config.datasetConfig.problem_type.name

    logger.info(
        "%s... Start describing data: %s",
        step,
        df.shape,
    )

    # --- Descriptive statistics (numeric) ---
    if t.describe.describe_stats and numeric_cols:
        desc = df[numeric_cols].describe().T
        desc.reset_index(inplace=True)
        desc.rename(columns={"index": "column"}, inplace=True)
        save_table_as_csv(desc, f"{problem_type}_{step}_01_stats_numeric.csv", PHASE_NAME)
        save_table_as_image(desc, f"{problem_type}_{step}_01_stats_numeric.png", PHASE_NAME)

    # --- Categorical statistics ---
    if t.describe.freq_tables and categorical_cols:
        desc_cat = df[categorical_cols].describe(include="all").T
        desc_cat.reset_index(inplace=True)
        desc_cat.rename(columns={"index": "column"}, inplace=True)
        save_table_as_csv(desc_cat, f"{problem_type}_{step}_02_stats_categorical.csv", PHASE_NAME)
        save_table_as_image(desc_cat, f"{problem_type}_{step}_02_stats_categorical.png", PHASE_NAME)

    # --- Histograms ---
    if t.describe.histograms:
        fig = plot_histograms(df, numeric_cols)
        save_figure(fig, f"{problem_type}_{step}_03_histograms.png", PHASE_NAME)

    # --- Boxplots ---
    if t.describe.boxplots:
        fig = plot_boxplots(df, numeric_cols)
        save_figure(fig, f"{problem_type}_{step}_04_boxplots.png", PHASE_NAME)

    # --- Correlation Heatmap ---
    if t.describe.corr_matrix:
        fig = plot_corr_heatmap(df, numeric_cols)
        save_figure(fig, f"{problem_type}_{step}_05_corr_heatmap.png", PHASE_NAME)

    logger.info(
        "%s... End describing data: %s",
        step,
        df.shape,
    )

# ============================================================
# 2.3 – QUALITY DIAGNOSTICS (NO TRANSFORMAZIONE)
# ============================================================
def _check_quality(df: pd.DataFrame, numeric_cols: List[str],
                   config: ProjectConfig,
                   range_checks: Optional[Dict[str, Tuple[float, float]]]) -> None:

    t = config.techniquesConfig.phase2
    problem_type = config.datasetConfig.problem_type.name
    step = PhaseStep.P2_3_CHECK_DATA_QUALITY

    #logger.info("[FASE2][2.3] Checking data quality...")
    logger.info(
        "%s... Start Checking data quality: %s",
        step,
        df.shape,
    )
    # Missing values
    if t.quality.missing_analysis:
        rep = analyze_missing_values(df)
        save_table_as_image(rep, f"{problem_type}_{step}_06_missing_values.png", PHASE_NAME)

    # Outliers (IQR detection)
    if t.quality.outlier_detection:
        rep = detect_outliers_iqr(df, PHASE_NAME, problem_type)
        #save_table_as_image(rep, f"{problem_type}_{step}_boxplot_outliers.png", PHASE_NAME)
        # rep saved by visualization inside module

    # Duplicates
    if t.quality.duplicates_check:
        rep = analyze_duplicates(df)

        if "info" not in rep.columns:   # hay duplicados de verdad
            sample = rep.head(10)
            problem_type_str = config.datasetConfig.problem_type.name
            save_table_as_image(
                # sample,
                # f"{problem_type_str}_{step}_08_duplicates.png",
                # PHASE_NAME,
                sample,
                filename=f"{problem_type_str}_{step}_08_duplicates_sample.png",
                phase_name=PHASE_NAME,
                title="Sample of duplicated rows (first 10 shown)",
            )
        #save_table_as_image(rep, f"{problem_type}_{step}_07_duplicates.png", PHASE_NAME)

    # Inconsistencies
    # if t.quality.inconsistencies_check:
    #     rep = detect_inconsistencies(df)
    #     if not rep.empty:
    #         save_table_as_image(rep, f"{problem_type}_{step}_09_inconsistencies.png", PHASE_NAME)

    if t.quality.inconsistencies_check:
        rep = detect_inconsistencies(
            df,
            rules=t.quality.inconsistency_rules
        )

        if not rep.empty:
            save_table_as_image(
                rep,
                f"{problem_type}_{step}_09_inconsistencies.png",
                PHASE_NAME
            )


    # Range validation
    # --- Range validation ---
    if t.quality.range_check:

        logger.info(
            "%s... Start Checking data quality: %s",
            step,
            df.shape,
        )

        ranges = t.quality.ranges # dict: {column: (min, max)}

        for col, (min_v, max_v) in ranges.items():

            if col not in df.columns:
                logger.warning("[QUALITY][RANGE] Column '%s' not found in dataset, skippi"
                               "ng.", col)
                continue

            # Correct call with all 3 required parameters
            # df_range = validate_range(df[[col]], min_v, max_v)
            # Correct call: validate_range(df, column, min_val, max_val, phase_name)
            df_range = validate_range(
                df,
                column=col,
                min_val=min_v,
                max_val=max_v,
                phase_name=PHASE_NAME
            )


            if df_range is not None and not df_range.empty:
                logger.info(
                    "[QUALITY][RANGE] Out-of-range values detected in column '%s' (%d rows)",
                    col, df_range.shape[0]
                )

                save_table_as_image(
                    df_range.head(20),
                    f"{problem_type}_{step}_range_{col}.png",
                    PHASE_NAME
                )
            else:
                logger.info("[QUALITY][RANGE] Column '%s' OK.", col)

    logger.info(
        "%s... End Checking data quality: %s",
        step,
        df.shape,
    )


# ============================================================
# 2.4 – EDA: SCATTER MATRIX, PCA, FEATURE-TARGET, TS
# ============================================================
def _explore_eda(df: pd.DataFrame, numeric_cols: List[str], config: ProjectConfig,
                 sample_for_plots: int) -> None:

    step = PhaseStep.P2_4_EXPLORE_DATA
    #logger.info("[FASE2][2.4] Running EDA visual exploration...")
    logger.info(
        "%s... Start Running EDA visual exploration: %s",
        step,
        df.shape,
    )
    problem_type = config.datasetConfig.problem_type.name

    # Optional sampling for heavy plots
    df_plot = df.sample(sample_for_plots, random_state=42) if len(df) > sample_for_plots else df

    t = config.techniquesConfig.phase2

    # Scatter matrix
    if t.eda.scatter_matrix:
        fig = plot_scatter_matrix(df_plot, numeric_cols)
        save_figure(fig, f"{problem_type}_{step}_10_scatter_matrix.png", PHASE_NAME)

    # PCA 2D
    if t.eda.pca:
        fig = plot_pca_2d(df_plot, numeric_cols)
        save_figure(fig, f"{problem_type}_{step}_11_pca_2d.png", PHASE_NAME)

    # Feature–Target plots
    target = config.datasetConfig.target_col
    if t.eda.feature_target_plots and target:
        fig = plot_feature_target_plots(df_plot, numeric_cols, target, config.datasetConfig.problem_type)
        if fig:
            save_figure(fig, f"{problem_type}_{step}_12_feature_target.png", PHASE_NAME)

    # Time Series preview
    if t.eda.ts_plot:
        fig = plot_time_series_preview(df_plot, config.datasetConfig)
        if fig:
            save_figure(fig, f"{problem_type}_{step}_13_timeseries_preview.png", PHASE_NAME)

    logger.info(
        "%s... End Running EDA visual exploration: %s",
        step,
        df.shape,
    )

# ============================================================
# FUNZIONE PRINCIPALE
# ============================================================
def run_phase2(config: ProjectConfig,
               range_checks: Optional[Dict[str, Tuple[float, float]]] = None,
               sample_for_plots: int = 5000) -> pd.DataFrame:
    """
    Esegue l'intera FASE 2 – DATA UNDERSTANDING.

    Parameters
    ----------
    config : ProjectConfig
        Configurazione completa del progetto.
    range_checks : dict
        Controlli espliciti di range (solo diagnostica).
    sample_for_plots : int
        Numero massimo di righe da usare per i grafici pesanti.

    Returns
    -------
    df_eda : pd.DataFrame
        Dataset (eventualmente campionato) usato per EDA.
    """
    logger.info("=== START PHASE 2 – DATA UNDERSTANDING (%s) ===", config.name)

    # 2.1 – Load dataset
    df_raw = _collect_initial_data(config)

    # Feature selection
    df_eda, numeric_cols, categorical_cols = select_features_auto(
        df_raw, config.datasetConfig, config.featureConfig
    )
    logger.info("[FASE2] After feature selection: shape=%s (num=%d, cat=%d)",
                df_eda.shape, len(numeric_cols), len(categorical_cols))

    # 2.2 – Descriptive statistics + plots
    _describe_data(df_eda, numeric_cols, categorical_cols, config)

    # 2.3 – Quality diagnostics
    _check_quality(df_eda, numeric_cols, config, range_checks)

    # 2.4 – EDA plots
    _explore_eda(df_eda, numeric_cols, config, sample_for_plots)

    logger.info("=== END PHASE 2 – RESULTS IN out/%s ===", PHASE_NAME)
    return df_eda
