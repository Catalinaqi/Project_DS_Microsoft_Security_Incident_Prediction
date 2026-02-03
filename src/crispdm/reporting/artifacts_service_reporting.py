# src/crispdm/reporting/artifacts_service_reporting.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.enums_utils_config import ProblemType

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This module implements the "artifact policy" of the project:
# - Every run must produce a reproducible, navigable folder structure.
# - Each stage writes its own tables + figures, separated for traceability.
# - Root of the run keeps key outputs: models/, metrics.json, logs.txt, config_used.yml
#
# Program flow:
# - run_facade_api.run_pipeline(...) creates run_dir
# - pipeline runners call stages
# - each stage writes:
#     run_dir/<stage_name>/tables_png/*.png
#     run_dir/<stage_name>/figures/*.png
#     run_dir/<stage_name>/stage_report.json
#
# Design patterns:
# - Artifact Repository (filesystem-based)
# - Convention over configuration
# =============================================================================

STAGE_DIRS = [
    "stage2_understanding",
    "stage3_preparation",
    "stage4_modeling",
    "stage5_evaluation",
]


def make_timestamp(ts: Optional[str] = None) -> str:
    return ts or datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(
        *,
        output_root: Path | str,
        task: ProblemType,
        dataset_key: str,
        timestamp: Optional[str] = None,
) -> Path:
    """
    Create:
      out/runs/<task>/<dataset_key>/<timestamp>/
        models/
        metrics.json  (later)
        logs.txt      (written by logging)
        <stage>/figures/
        <stage>/tables_png/
    """
    out_root = Path(output_root)
    ts = make_timestamp(timestamp)
    run_dir = out_root / "runs" / task.value / dataset_key / ts

    # Root dirs
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    # Stage dirs
    for s in STAGE_DIRS:
        (run_dir / s / "figures").mkdir(parents=True, exist_ok=True)
        (run_dir / s / "tables_png").mkdir(parents=True, exist_ok=True)

    log.info("[artifacts] run_dir created: %s", run_dir)
    return run_dir


def stage_dir(run_dir: Path, stage_name: str) -> Path:
    d = run_dir / stage_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(exist_ok=True)
    (d / "tables_png").mkdir(exist_ok=True)
    return d


def save_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("[artifacts] json saved: %s", path)
    return path


def save_stage_report(run_dir: Path, stage_name: str, payload: Dict[str, Any]) -> Path:
    d = stage_dir(run_dir, stage_name)
    return save_json(d / "stage_report.json", payload)


def save_metrics(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    return save_json(run_dir / "metrics.json", metrics)


def save_table_png(
        df: pd.DataFrame,
        *,
        out_path: Path,
        title: Optional[str] = None,
        max_rows: int = 30,
) -> Path:
    """
    Save a pandas DataFrame as a PNG using matplotlib table.
    Keeps the project "everything visible as PNG" rule.
    """
    if df is None:
        raise ValueError("df must not be None")

    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    fig, ax = plt.subplots(figsize=(12, 0.4 * (len(df2) + 2)))
    ax.axis("off")
    if title:
        ax.set_title(title)

    tbl = ax.table(
        cellText=df2.values,
        colLabels=list(df2.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    log.info("[artifacts] table png saved: %s", out_path)
    return out_path


def save_figure(fig, *, out_path: Path) -> Path:
    """
    Save a matplotlib figure. Closes it to avoid memory leaks in notebooks.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("[artifacts] figure saved: %s", out_path)
    return out_path


def save_model_pickle(run_dir: Path, model: Any, filename: str = "model.pkl") -> Path:
    """
    Basic filesystem model persistence via pickle.
    Good enough for sklearn-style models.
    """
    import pickle
    out_path = run_dir / "models" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pickle.dumps(model))
    log.info("[artifacts] model saved: %s", out_path)
    return out_path
