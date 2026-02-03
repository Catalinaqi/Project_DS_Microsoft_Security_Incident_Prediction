# src/crispdm/stage/stage2_understanding_runner_stage.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.schema_dto_config import ProjectConfig

from crispdm.data.load_utils_data import load_csv_by_strategy
from crispdm.data.quality_rules_utils_data import apply_quality_rules, violations_to_df

from crispdm.reporting.artifacts_service_reporting import (
    stage_dir,
    save_stage_report,
    save_table_png,
    save_figure,
)
from crispdm.reporting.plots_utils_reporting import (
    plot_missingness_top,
    plot_numeric_hist,
)

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stage 2 - Data Understanding (CRISP-ML Phase 2)
# - Report-only: MUST NOT modify data
# - Executes YAML-defined steps/methods for Stage 2
# - Persists artifacts using project artifact policy:
#     run_dir/stage2_understanding/figures/*.png
#     run_dir/stage2_understanding/tables_png/*.png
#     run_dir/stage2_understanding/stage_report.json
#
# Design patterns:
# - Enterprise/Architectural: Stage Runner + Reporting/Artifact Repository
# =============================================================================

STAGE_NAME = "stage2_understanding"


def _enabled(node: Any, default: bool = True) -> bool:
    return bool(node.get("enabled", default)) if isinstance(node, dict) else default


def _dget(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _schema_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "n_null": [int(df[c].isna().sum()) for c in df.columns],
        "null_pct": [float(df[c].isna().mean() * 100.0) for c in df.columns],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values(["null_pct", "n_unique"], ascending=[False, False])


def _describe_table(df: pd.DataFrame, include: Any = "all") -> pd.DataFrame:
    desc = df.describe(include=include).transpose()
    desc.insert(0, "column", desc.index.astype(str))
    return desc.reset_index(drop=True)


def _min_max_mean_std(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    cols = _numeric_cols(df) if numeric_only else list(df.columns)
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "column": c,
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
        })
    return pd.DataFrame(rows)


def _duplicates_summary(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
    dup_mask = df.duplicated(subset=subset, keep=keep)
    return pd.DataFrame([{
        "rows": int(len(df)),
        "duplicates": int(dup_mask.sum()),
        "dup_pct": float(dup_mask.mean() * 100.0) if len(df) else 0.0,
        "subset": str(subset),
        "keep": str(keep),
    }])


def run_stage2_understanding_runner_stage(
        *,
        cfg: ProjectConfig,
        run_dir: Path,
        ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes Stage 2 for ANY pipeline (clustering/classification/regression/timeseries).
    Stage2 is task-agnostic: it only profiles & reports.
    """
    s2 = cfg.stages.stage2_understanding
    if not s2.enabled:
        log.info("[Stage2] SKIP (disabled)")
        return ctx

    steps_cfg = s2.steps or {}
    output_policy = s2.output_policy or {}

    # Artifact policy (single source of truth)
    sdir = stage_dir(run_dir, STAGE_NAME)
    figures_dir = sdir / "figures"
    tables_dir = sdir / "tables_png"

    log.info("[Stage2] START run_dir=%s stage_dir=%s", run_dir, sdir)
    log.debug("[Stage2] output_policy(from yml)=%s", output_policy)
    if output_policy.get("figures_dir") or output_policy.get("tables_png_dir"):
        log.warning(
            "[Stage2] YAML output_policy dirs ignored. Using artifact policy: %s/{figures,tables_png}",
            sdir
        )

    # ---------------------------
    # Step 2.1 Data acquisition
    # ---------------------------
    step21 = steps_cfg.get("step_2_1_data_acquisition", {}) or {}
    if not _enabled(step21, default=True):
        raise ValueError("Stage2 requires step_2_1_data_acquisition.enabled=true")

    methods21 = step21.get("methods") or {}
    load_csv_cfg = methods21.get("load_csv") or {}
    if not _enabled(load_csv_cfg, default=True):
        raise ValueError("Stage2 requires step_2_1_data_acquisition.methods.load_csv.enabled=true")

    dataset_input = s2.dataset_input or {}
    if not isinstance(dataset_input, dict):
        raise ValueError("stage2_understanding.dataset_input must be a dict")

    path = dataset_input.get("path")
    if not path:
        raise ValueError("dataset_input.path missing (placeholder not resolved?)")

    csv_params = dataset_input.get("csv_params") or {}
    if not isinstance(csv_params, dict):
        csv_params = {}

    # ✅ tu convención: read_strategy está dentro de dataset_input
    read_strategy = dataset_input.get("read_strategy") or {}
    if not isinstance(read_strategy, dict):
        read_strategy = {}

    log.info("[Stage2][2.1] load_csv START path=%s", path)
    log.debug("[Stage2][2.1] csv_params=%s", csv_params)
    log.debug("[Stage2][2.1] read_strategy=%s", read_strategy)

    df, chunks_gen, norm = load_csv_by_strategy(str(path), csv_params=csv_params, strategy=read_strategy)

    # Si es chunked, toma el primer chunk para reporte (Stage2 es report-only)
    if df is None and chunks_gen is not None:
        log.info("[Stage2][2.1] mode=chunked -> using first chunk for reporting")
        try:
            df = next(chunks_gen)
        except StopIteration:
            df = pd.DataFrame()

    ctx["df_stage2"] = df
    log.info("[Stage2][2.1] load_csv END mode=%s rows=%s cols=%s",
             norm.mode.value, int(df.shape[0]), int(df.shape[1]))

    # ---------------------------
    # Step 2.2 Describe data
    # ---------------------------
    step22 = steps_cfg.get("step_2_2_describe_data", {}) or {}
    if _enabled(step22, default=True):
        log.info("[Stage2][2.2] START describe_data")
        techniques = step22.get("techniques") or {}

        ds = techniques.get("descriptive_statistics", {}) or {}
        if _enabled(ds, default=True):
            m = ds.get("methods") or {}

            describe_cfg = m.get("describe", {}) or {}
            if _enabled(describe_cfg, default=True):
                params = describe_cfg.get("params") or {}
                include = _dget(params, "include", "all")
                desc_tbl = _describe_table(df, include=include)
                save_table_png(desc_tbl, out_path=tables_dir / "describe.png", title="Stage2 - describe()")

            mms_cfg = m.get("min_max_mean_std", {}) or {}
            if _enabled(mms_cfg, default=True):
                params = mms_cfg.get("params") or {}
                numeric_only = bool(_dget(params, "numeric_only", True))
                mm_tbl = _min_max_mean_std(df, numeric_only=numeric_only)
                save_table_png(mm_tbl, out_path=tables_dir / "min_max_mean_std.png", title="Stage2 - min/max/mean/std")

        si = techniques.get("schema_inspection", {}) or {}
        if _enabled(si, default=True):
            m = si.get("methods") or {}

            if _enabled(m.get("dtype_analysis", {}) or {}, default=True):
                schema_tbl = _schema_table(df)
                save_table_png(schema_tbl.head(300), out_path=tables_dir / "schema_dtype_null_unique.png",
                               title="Stage2 - schema/dtypes/nulls/unique")

            card_cfg = m.get("cardinality_count", {}) or {}
            if _enabled(card_cfg, default=True):
                params = card_cfg.get("params") or {}
                max_unique = int(_dget(params, "max_unique_to_report", 50))
                card = pd.DataFrame({
                    "column": df.columns,
                    "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
                }).sort_values("n_unique", ascending=False)
                save_table_png(card.head(max_unique), out_path=tables_dir / "cardinality_top.png",
                               title=f"Stage2 - cardinality top {max_unique}")

            if _enabled(m.get("null_count", {}) or {}, default=True):
                nulls = pd.DataFrame({
                    "column": df.columns,
                    "n_null": [int(df[c].isna().sum()) for c in df.columns],
                    "null_pct": [float(df[c].isna().mean() * 100.0) for c in df.columns],
                }).sort_values("null_pct", ascending=False)
                save_table_png(nulls.head(300), out_path=tables_dir / "null_count.png", title="Stage2 - null count")

        log.info("[Stage2][2.2] END describe_data")

    # ---------------------------
    # Step 2.3 Data quality assessment
    # ---------------------------
    step23 = steps_cfg.get("step_2_3_data_quality_assessment", {}) or {}
    quality_report: Dict[str, Any] = {"total_rules_violated": 0, "total_violations": 0, "violations": []}

    if _enabled(step23, default=True):
        log.info("[Stage2][2.3] START data_quality_assessment")
        methods = step23.get("methods") or {}

        miss_cfg = methods.get("missing_analysis", {}) or {}
        if _enabled(miss_cfg, default=True):
            params = miss_cfg.get("params") or {}
            top_n = int(_dget(params, "show_top_columns", 30))
            fig = plot_missingness_top(df, top_n=top_n, title=f"Stage2 - missingness top {top_n}")
            save_figure(fig, out_path=figures_dir / "missingness_top.png")

        dup_cfg = methods.get("duplicate_detection", {}) or {}
        if _enabled(dup_cfg, default=True):
            params = dup_cfg.get("params") or {}
            subset = params.get("subset", None)
            keep = _dget(params, "keep", "first")
            dup_tbl = _duplicates_summary(df, subset=subset, keep=keep)
            save_table_png(dup_tbl, out_path=tables_dir / "duplicates.png", title="Stage2 - duplicates")

        # Reglas externas: usamos tu engine real
        ranges_rules_path = None
        logic_rules_path = None
        business_rules_path = None

        range_cfg = methods.get("range_validation", {}) or {}
        if _enabled(range_cfg, default=True):
            params = range_cfg.get("params") or {}
            ranges_rules_path = params.get("rules_file")

        logic_cfg = methods.get("inconsistency_checks", {}) or {}
        if _enabled(logic_cfg, default=True):
            params = logic_cfg.get("params") or {}
            logic_rules_path = params.get("rules_file")

        business_cfg = methods.get("business_kpi_rules", {}) or {}
        if _enabled(business_cfg, default=False):
            params = business_cfg.get("params") or {}
            business_rules_path = params.get("rules_file")

        if ranges_rules_path or logic_rules_path or business_rules_path:
            log.info("[Stage2][2.3] apply_quality_rules START")
            log.debug("[Stage2][2.3] ranges_rules_path=%s logic_rules_path=%s business_rules_path=%s",
                      ranges_rules_path, logic_rules_path, business_rules_path)

            quality_report = apply_quality_rules(
                df,
                ranges_rules_path=ranges_rules_path,
                logic_rules_path=logic_rules_path,
                business_rules_path=business_rules_path,
            )

            vio_df = violations_to_df(quality_report)
            if len(vio_df):
                save_table_png(
                    vio_df.head(300),
                    out_path=tables_dir / "quality_rules_violations.png",
                    title="Stage2 - quality rules violations",
                )

            log.info(
                "[Stage2][2.3] apply_quality_rules END total_violations=%s total_rules_violated=%s",
                quality_report.get("total_violations"),
                quality_report.get("total_rules_violated"),
            )
        else:
            log.info("[Stage2][2.3] rules skipped (no rules_file configured)")

        log.info("[Stage2][2.3] END data_quality_assessment")

    ctx["stage2_quality_report"] = quality_report

    # ---------------------------
    # Step 2.4 EDA
    # ---------------------------
    step24 = steps_cfg.get("step_2_4_eda", {}) or {}
    if _enabled(step24, default=True):
        log.info("[Stage2][2.4] START eda")
        methods = step24.get("methods") or {}

        hist_cfg = methods.get("histograms", {}) or {}
        if _enabled(hist_cfg, default=True):
            params = hist_cfg.get("params") or {}
            numeric_only = bool(_dget(params, "numeric_only", True))
            max_cols = int(_dget(params, "max_columns", 20))
            bins = int(_dget(params, "bins", 30))

            cols = _numeric_cols(df) if numeric_only else list(df.columns)
            for c in cols[:max_cols]:
                fig = plot_numeric_hist(df, c, bins=bins, title=f"Stage2 - hist: {c}")
                save_figure(fig, out_path=figures_dir / f"hist_{c}.png")

        log.info("[Stage2][2.4] END eda")

    # ---------------------------
    # stage_report.json
    # ---------------------------
    report = {
        "stage": STAGE_NAME,
        "task": cfg.pipeline.task.value,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dataset_path": str(path),
        "read_mode": norm.mode.value,
        "quality_total_violations": int(quality_report.get("total_violations", 0)),
        "quality_total_rules_violated": int(quality_report.get("total_rules_violated", 0)),
    }
    save_stage_report(run_dir, STAGE_NAME, report)

    log.info("[Stage2] END rows=%s cols=%s", int(df.shape[0]), int(df.shape[1]))
    return ctx
