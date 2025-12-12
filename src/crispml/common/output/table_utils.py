# ---------------------------------------------------------
# src/crispml/common/output/table_utils.py
# Improved table export utilities for Phase 2 and Phase 3
# ---------------------------------------------------------

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt

from .path_utils import get_output_dir
from .style_utils import (
    HEADER_BG,
    HEADER_TEXT_COLOR,
    FONT_SMALL,
    FONT_MEDIUM,
    FONT_LARGE,
)
from src.crispml.common.logging.logging_utils import get_logger
from src.crispml.config.enums.enums_config import PhaseName
logger = get_logger(__name__)


# =========================================================
#  🔹 CORE FUNCTION: SAVE ANY TABLE AS IMAGE
# =========================================================
def save_table_as_image(
        df: pd.DataFrame,
        filename: str,
        phase_name: PhaseName, # src -> enums
        dpi: int = 300,
        max_rows: int = 100,
        align: str = "left",
        index_name: str | None = None,
        title: str | None = None
):
    """
    Save a DataFrame as a high-quality table image
    formatted like the example (gray header, full borders, clean layout).
    """

    # ===========================
    # Fix double extension
    # ===========================
    if filename.endswith(".png"):
        filename = filename[:-4]

    # ===========================
    # Row truncation
    # ===========================
    if len(df) > max_rows:
        df_to_plot = df.head(max_rows).copy()
        df_to_plot.loc["..."] = ["..."] * df.shape[1]
    else:
        df_to_plot = df.copy()

    # Optional index name
    if index_name is not None:
        df_to_plot.rename_axis(index_name, inplace=True)

    # ===========================
    # Output folder
    # ===========================
    out_dir = get_output_dir(phase_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{filename}.png"

    # ===========================
    # Insert sequential index column "No."
    # ===========================
    df_to_plot.insert(0, "No.", range(1, len(df_to_plot) + 1))

    # ===========================
    # Adaptive figure size
    # ===========================
    n_rows, n_cols = df_to_plot.shape

    col_width = 1.0
    row_height = 0.35

    fig_width = max(6, n_cols * col_width)
    fig_height = max(1.5, n_rows * row_height)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.axis("off")

    # ===========================
    # Table creation
    # ===========================
    align_map = {"left": "left", "center": "center", "right": "right"}

    table = ax.table(
        cellText=df_to_plot.values,
        colLabels=df_to_plot.columns,
        cellLoc=align_map.get(align, "left"),
        loc="center"
    )

    # ===========================
    # Style: gray header + borders
    # ===========================
    for (r, c), cell in table.get_celld().items():

        # Header row
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(weight="bold", color=HEADER_TEXT_COLOR)

        # Borders on every cell
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

    # ===========================
    # Adaptive font size
    # ===========================
    if n_cols <= 6:
        font_size = FONT_LARGE
    elif n_cols <= 10:
        font_size = FONT_MEDIUM
    else:
        font_size = FONT_SMALL

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Auto-adjust column widths
    try:
        table.auto_set_column_width(col=list(range(n_cols)))
    except Exception:
        pass

    table.scale(1.0, 1.20)

    # ===========================
    # Margins
    # ===========================
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.05
    )

    # ===========================
    # Optional title
    # ===========================
    if title:
        fig.suptitle(
            title,
            fontsize=14,
            fontweight="bold",
            y=0.98
        )

    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    logger.info("[table_utils] Image saved → %s", path)
    return path

# =========================================================
#  🔹 FUNCTION: SAVE TABLE AS CSV
# =========================================================
def save_table_as_csv(
        df: pd.DataFrame,
        filename: str,
        phase_name: PhaseName,
        include_index: bool = False,
        sep: str = ",",
        encoding: str = "utf-8"
):
    """
    Save a DataFrame as a CSV file inside the correct /out/<phase> folder.

    Parameters
    ----------
    df : pd.DataFrame
        Table to export.
    filename : str
        File name without extension.
    phase_name : PhaseName
        Determines output folder location.
    include_index : bool
        Whether to save the dataframe index.
    sep : str
        Delimiter used in CSV.
    encoding : str
        Output text encoding.
    """

    # Ensure correct extension
    if filename.endswith(".csv"):
        filename = filename[:-4]

    # Output folder
    out_dir = get_output_dir(phase_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{filename}.csv"

    # Export
    df.to_csv(path, index=include_index, sep=sep, encoding=encoding)

    logger.info("[table_utils] CSV saved → %s", path)
    return path
