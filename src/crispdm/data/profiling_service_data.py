from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from crispdm.core.logging_utils_core import get_logger
from crispdm.data.load_utils_data import load_csv_by_strategy
from typing import List
import pandas as pd
import re
log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Implements Stage 2 (Data Understanding / Profiling) for the PREVIEW flow.
# Key outputs:
# - dataset structure summary (dtypes, missing, cardinality)
# - lightweight EDA summaries
# - candidate suggestions for:
#   - target_col (classification/regression)
#   - time_col (timeseries)
#   - id_cols  (technical identifiers)
#
# Program flow:
# - preview_facade_api.run_preview()
#   -> build_preview_config()
#   -> run_stage2_preview(config)
#   -> returns suggestions dict (used by notebook to set target/time/id)
#
# Design patterns
# - GoF: none
# - Enterprise/Architectural:
#   - Service Layer (profiling service)
#   - Heuristic recommendation engine (lightweight)
# =============================================================================



_ID_PATTERNS = [
    re.compile(r"^id$", re.IGNORECASE),                          # Id
    re.compile(r"(^|_)(uuid|guid)($|_)", re.IGNORECASE),         # uuid / guid
    re.compile(r"(^|_)(id)($|_)", re.IGNORECASE),                # _id or id token in snake_case
    re.compile(r"(?:^|[a-z0-9])id$", re.IGNORECASE),             # endswith Id (camelCase) like IncidentId, AlertId
]

_TARGETISH = re.compile(r"(target|label|class|grade)\b", re.IGNORECASE)

@dataclass(frozen=True)
class Stage2Suggestions:
    target_candidates: List[str]
    time_candidates: List[str]
    id_candidates: List[str]


def _ensure_dir(root: Path, rel: str) -> Path:
    p = root / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_table_png(df: pd.DataFrame, out_path: Path, *, title: str = "", dpi: int = 150) -> None:
    """
    Save a small dataframe as a PNG table (for reporting).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep it readable: limit rows/cols for the PNG
    df2 = df.copy()
    if len(df2) > 30:
        df2 = df2.head(30)

    fig, ax = plt.subplots(figsize=(14, 0.4 * max(8, len(df2))))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    tbl = ax.table(cellText=df2.values, colLabels=df2.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _suggest_time_cols(df: pd.DataFrame) -> List[str]:
    """
    Heuristics:
    - column name contains date/time/timestamp
    - convertible to datetime with reasonable success rate (on sample)
    """
    candidates: List[str] = []
    name_hits = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["time", "date", "timestamp", "datetime"]):
            name_hits.append(c)

    # Try parse on a few columns only (avoid heavy work)
    parse_hits = []
    for c in df.columns[: min(len(df.columns), 50)]:
        if c in name_hits:
            parse_hits.append(c)
            continue

        # Solo stringhe/object (evita int/float che spesso sono ID)
        if df[c].dtype == "object":
            s = df[c].dropna().astype(str).head(10_000)
            if len(s) == 0:
                continue

            # --------
            # Heuristica veloce: se è quasi tutto numerico, probabilmente NON è datetime
            numeric_ratio = s.str.fullmatch(r"\d+(\.\d+)?").mean()
            if numeric_ratio >= 0.98:
                continue
            # --------
            # Parsing silenzioso (evita warning "Could not infer format")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format.*")
                try:
                    # Se hai pandas 2.x, questa è spesso la scelta migliore:
                    parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=False)
                except TypeError:
                    # pandas vecchio: fallback classico
                    parsed = pd.to_datetime(s, errors="coerce", utc=False)

            rate = float(parsed.notna().mean())

            #parsed = pd.to_datetime(s, errors="coerce", utc=False)
            #rate = float(parsed.notna().mean())
            if rate >= 0.7:
                parse_hits.append(c)
                log.debug("time-col candidate: %s rate=%.3f", c, rate)

    # Prioritize name hits first
    seen = set()
    for c in name_hits + parse_hits:
        if c not in seen:
            candidates.append(c)
            seen.add(c)
    return candidates[:10]

def _looks_like_id_col(name: str) -> bool:
    # protect target-like columns from being misclassified as id
    if _TARGETISH.search(name):
        return False
    return any(p.search(name) for p in _ID_PATTERNS)

def _suggest_id_cols(df: pd.DataFrame) -> List[str]:
    """
    Heuristics:
    - name looks like id/guid/uuid (token/suffix, not substring inside words)
    - OR very high cardinality (almost unique)
    """
    n = len(df)
    candidates: List[str] = []

    # 1) name-based
    for c in df.columns:
        if _looks_like_id_col(c):
            candidates.append(c)

    # 2) cardinality-based (limit columns to keep it cheap)
    for c in df.columns[: min(len(df.columns), 80)]:
        if c in candidates:
            continue
        # again: never treat target-ish columns as ids
        if _TARGETISH.search(c):
            continue
        try:
            nunq = df[c].nunique(dropna=True)
        except Exception:
            continue
        if n > 0 and nunq / n >= 0.98:
            candidates.append(c)

    return candidates[:10]


# def _suggest_target_cols(
#         df: pd.DataFrame,
#         id_cols: List[str],
#         time_cols: List[str],
#         max_missing_rate: float = 0.30,   # <- nuevo
# ) -> List[str]:
    """
    Heuristics:
    - prefer columns named like label/target/class/grade
    - prefer categorical-like columns (low/moderate cardinality)
    - exclude obvious id/time
    - exclude columns with high missing rate
    """
    excluded = set(id_cols + time_cols)

    # def ok_missing(c: str) -> bool:
        # mean() de boolean -> ratio de NaN
        # return df[c].isna().mean() <= max_missing_rate

    # 1) Name-priority (pero igual filtramos missing)
    # name_priority = []
    # for c in df.columns:
    #     if c in excluded:
    #         continue
    #     if not ok_missing(c):
    #         continue
    #     if _TARGETISH.search(c):
    #         name_priority.append(c)

    # 2) Categorical-like por cardinalidad (también filtramos missing)
    # n = len(df)
    # cat_like = []
    # upper = min(200, max(5, int(0.05 * max(1, n))))
    # for c in df.columns:
    #     if c in excluded:
    #         continue
    #     if not ok_missing(c):
    #         continue
    #     try:
    #         nunq = df[c].nunique(dropna=True)
    #     except Exception:
    #         continue
    #     if nunq <= 1:
    #         continue
    #     if 2 <= nunq <= upper:
    #         cat_like.append(c)

    # Merge: name_priority first
    # out = []
    # seen = set()
    # for c in name_priority + cat_like:
    #     if c not in seen:
    #         out.append(c)
    #         seen.add(c)
    #
    # return out[:10]

def _suggest_target_cols(
        df: pd.DataFrame,
        id_cols: List[str],
        time_cols: List[str],
        max_missing_rate: float = 0.30,
) -> List[str]:
    """
    Heuristics:
    - prefer columns named like label/target/class/grade
    - prefer categorical-like columns (low/moderate cardinality)
    - exclude obvious id/time
    - exclude columns with high missing rate
    - exclude identifier-like columns (e.g., *Id) unless target-ish
    """
    excluded = set(id_cols + time_cols)

    def ok_missing(c: str) -> bool:
        return float(df[c].isna().mean()) <= max_missing_rate

    def identifier_like(c: str) -> bool:
        # si es target-ish (grade/label/target/class) NO lo bloquees
        if _TARGETISH.search(c):
            return False
        return _looks_like_id_col(c)  # <- clave: bloquea OAuthApplicationId, ApplicationId, etc.

    # 1) Name priority (aplica missing)
    name_priority = []
    for c in df.columns:
        if c in excluded:
            continue
        if not ok_missing(c):
            continue
        if _TARGETISH.search(c):
            name_priority.append(c)

    # 2) Cat-like (aplica missing + bloqueo id-like)
    n = len(df)
    upper = min(200, max(5, int(0.05 * max(1, n))))
    cat_like = []
    for c in df.columns:
        if c in excluded:
            continue
        if not ok_missing(c):
            continue
        if identifier_like(c):
            continue
        try:
            nunq = df[c].nunique(dropna=True)
        except Exception:
            continue
        if nunq <= 1:
            continue
        if 2 <= nunq <= upper:
            cat_like.append(c)

    out, seen = [], set()
    for c in name_priority + cat_like:
        if c not in seen:
            out.append(c)
            seen.add(c)

    return out[:10]


def run_stage2_preview(
        *,
        project_config: Any,
) -> Dict[str, Any]:
    """
    Execute Stage2 profiling in preview mode and return suggestions.

    project_config is expected to be schema_dto_config.ProjectConfig
    with:
      project_config.pipeline.variables["dataset_path"]
      project_config.runtime.output_root
      project_config.stages.stage2_understanding.dataset_input/csv_params/read_strategy/output_policy
    """
    log.info("[run_stage2_preview] START")

    s2 = project_config.stages.stage2_understanding
    runtime = project_config.runtime
    vars_ = project_config.pipeline.variables

    dataset_path = vars_.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path missing in config variables (Stage2 requires it).")

    csv_params = (s2.dataset_input or {}).get("csv_params") or {}
    read_strategy = (s2.dataset_input or {}).get("read_strategy") or {}
    output_policy = s2.output_policy or {}

    output_root = Path(runtime.output_root)
    figures_dir = _ensure_dir(output_root, output_policy.get("figures_dir", "figures/stage2"))
    tables_dir = _ensure_dir(output_root, output_policy.get("tables_png_dir", "tables_png/stage2"))
    dpi = int(output_policy.get("dpi", 150))

    # Load sample/full/chunks:
    df, chunks, norm_strategy = load_csv_by_strategy(
        dataset_path,
        csv_params=csv_params,
        strategy=read_strategy,
    )

    if df is None and chunks is not None:
        # For chunked mode, just take first chunk for "suggestions" and basic summary.
        log.info("[run_stage2_preview] chunked mode: taking first chunk for quick profiling")
        df = next(chunks)

    assert df is not None

    log.info("[run_stage2_preview] profiling rows=%d cols=%d", len(df), df.shape[1])

    # Basic profiling table
    prof = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
        "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values(by=["missing_rate", "nunique"], ascending=[False, False])

    _save_table_png(prof.head(25), tables_dir / "stage2_profile_top25.png", title="Stage2 - Profile (Top 25)", dpi=dpi)

    # Suggestions
    time_candidates = _suggest_time_cols(df)
    id_candidates = _suggest_id_cols(df)
    target_candidates = _suggest_target_cols(df, id_candidates, time_candidates)

    sugg = Stage2Suggestions(
        target_candidates=target_candidates,
        time_candidates=time_candidates,
        id_candidates=id_candidates,
    )

    log.info("[run_stage2_preview] suggestions target=%s", sugg.target_candidates[:5])
    log.info("[run_stage2_preview] suggestions time=%s", sugg.time_candidates[:5])
    log.info("[run_stage2_preview] suggestions id=%s", sugg.id_candidates[:5])

    # Return a dict that your notebook can display/consume.
    result = {
        "dataset_path": str(dataset_path),
        "rows_profiled": int(len(df)),
        "cols": int(df.shape[1]),
        "suggestions": {
            "target_candidates": sugg.target_candidates,
            "time_candidates": sugg.time_candidates,
            "id_candidates": sugg.id_candidates,
        },
        "artifacts": {
            "profile_table_png": str((tables_dir / "stage2_profile_top25.png").as_posix()),
        },
    }
    log.info("result = %s", result)
    log.info("[run_stage2_preview] DONE")
    return result
