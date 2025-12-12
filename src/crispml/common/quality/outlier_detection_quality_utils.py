from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.common.output import save_figure
from src.crispml.common.visualization.boxplot_viz_utils import filter_columns_for_boxplot
from src.crispml.config.enums.enums_config import PhaseName, ProblemType, PhaseStep

phase_name_num = PhaseName.PHASE2_DATA_UNDERSTANDING
logger = get_logger(__name__)


def detect_outliers_iqr(
        df: pd.DataFrame,
        phase_name: PhaseName = phase_name_num,
        problem_type: ProblemType | str | None = None,
) -> list[str]:
    """
    Detect outliers using IQR for suitable numeric columns.
    Only columns that pass filtering (non-ID, non-degenerate, low cardinality)
    are considered for IQR diagnostics.
    """

    # Step name for output files
    step = PhaseStep.P2_3_CHECK_DATA_QUALITY


    # All numeric columns
    num_cols = list(df.select_dtypes(include=np.number).columns)

    # Apply filtering logic
    logger.info("[QUALITY][OUTLIERS] Filtering numeric columns before IQR check...")
    filtered_cols = filter_columns_for_boxplot(df, num_cols)

    logger.info(
        "[QUALITY][OUTLIERS] Numeric columns: %d → Filtered for IQR: %d",
        len(num_cols),
        len(filtered_cols),
    )

    if not filtered_cols:
        logger.warning("[QUALITY][OUTLIERS] No suitable numeric columns for IQR detection.")
        return []

    # Resolve problem type string
    problem_type_str = (
        problem_type.name
        if isinstance(problem_type, ProblemType)
        else str(problem_type or "UNKNOWN")
    )

    outlier_cols: list[str] = []

    # ---- Perform IQR analysis ONLY on filtered columns --------------------
    logger.info(
        "[QUALITY][OUTLIERS] Starting IQR detection on filtered columns: %s",
        filtered_cols,
    )

    for col in filtered_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask_outliers = (series < lower) | (series > upper)

        if mask_outliers.any():
            outlier_cols.append(col)
            logger.debug(
                "[QUALITY][OUTLIERS][%s] q1=%.3f q3=%.3f iqr=%.3f "
                "lower=%.3f upper=%.3f num_outliers=%d",
                col,
                q1, q3, iqr, lower, upper,
                int(mask_outliers.sum()),
            )

    # ---- Create visualization only for filtered columns -------------------
    # fig, ax = plt.subplots(figsize=(18, 6))
    # df[filtered_cols].boxplot(ax=ax, showfliers=False)
    # ax.set_title("Boxplot – Outlier Detection (IQR) – Filtered Numeric Columns")
    # # Rotate x-axis labels
    # plt.xticks(rotation=45, ha='right')

    # ---- FAST BOXPLOT FOR OUTLIER DIAGNOSTICS -------------------------
    logger.info("[QUALITY][OUTLIERS] Generating FAST boxplot visualization...")
    #
    # stats = []
    # for col in filtered_cols:
    #     series = df[col].dropna()
    #
    #     q1 = series.quantile(0.25)
    #     q2 = series.quantile(0.50)
    #     q3 = series.quantile(0.75)
    #     iqr = q3 - q1
    #
    #     whisker_min = q1 - 1.5 * iqr
    #     whisker_max = q3 + 1.5 * iqr
    #
    #     stats.append({
    #         "label": col,
    #         "whislo": whisker_min,
    #         "q1": q1,
    #         "med": q2,
    #         "q3": q3,
    #         "whishi": whisker_max,
    #         "fliers": []
    #     })

    stats = []
    for col in filtered_cols:
        series = df[col].dropna()

        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        whisker_min = q1 - 1.5 * iqr
        whisker_max = q3 + 1.5 * iqr

        # TRUE IQR outliers (for visual clarity)
        fliers = series[(series < whisker_min) | (series > whisker_max)].tolist()

        stats.append({
            "label": col,
            "whislo": whisker_min,
            "q1": q1,
            "med": q2,
            "q3": q3,
            "whishi": whisker_max,
            "fliers": fliers  # <- now visible
        })

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bxp(stats, showfliers=True)
    ax.set_title("Boxplot – Outlier Detection (FAST Mode)",
                 fontsize=12,
                 fontweight="bold",
                 color="darkred")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()




    filename = f"{problem_type_str}_{step}_07_boxplot_outliers.png"
    save_figure(fig=fig, filename=filename, phase_name=phase_name)

    logger.info(
        "[QUALITY][OUTLIERS] IQR detection completed. Columns with outliers: %s",
        outlier_cols,
    )

    return outlier_cols
