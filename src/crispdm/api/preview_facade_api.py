# src/crispdm/api/preview_facade_api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from crispdm.core.logging_utils_core import init_logging, get_logger
from crispdm.config.load_loader_config import load_yaml, load_and_resolve
from crispdm.config.schema_dto_config import ProjectConfig
from crispdm.config.validate_validator_config import validate_config
from crispdm.config.build_factory_config import build_run_context
from crispdm.data.load_utils_data import resolve_path, load_csv

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Enterprise / Architectural patterns:
# - Facade: single entrypoint to orchestrate cross-cutting concerns
#   (logging + config + validation + stage orchestration).
# Not a GoF Facade strictly, but same intent (simplify subsystem usage).
# ---------------------------------------------------------------------


def save_config_used_unique(
        resolved_config: Dict[str, Any],
        *,
        task: str,
        audit_dir: Path,
        run_ts: str
) -> Path:
    """
    Save resolved config snapshot for audit:
      out/audit/config_used__<task>__<YYYYMMDD_HHMMSS>.yml
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / f"config_used__{task}__{run_ts}.yml"
    out_path.write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    return out_path


def preview(
        *,
        pipeline_yaml_path: str | Path,
        dataset_yaml_path: str | Path,
        dataset_id: str,
        project_root: Path,
        runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Preview flow (Stage2 only conceptually):
      1) Load dataset_config.yml
      2) Merge runtime vars (dataset_path/target/time/id/output_root)
      3) Load+resolve pipeline YAML
      4) Validate in preview mode (target/time can be None)
      5) Build DTO + RunContext
      6) Save config_used snapshot
      7) Load sample CSV for Stage2 profiling

    Returns a dict with:
      - cfg (typed DTO)
      - run_context (paths)
      - df_head (small preview)
      - meta (read strategy meta)
    """
    runtime_overrides = runtime_overrides or {}

    # ---- Load dataset config ----
    ds_cfg = load_yaml(dataset_yaml_path)
    ds = ds_cfg["datasets"][dataset_id]

    # Resolve dataset paths relative to project root
    train_path = resolve_path(ds["paths"]["train"], project_root=project_root)

    # Stage2 read strategy (big CSV -> sample/chunked)
    stage2_strategy = ds.get("stage2_read_strategy", {})
    csv_params = ds.get("csv_params", {})

    # ---- runtime vars for placeholder resolution ----
    runtime_vars = {
        "dataset_path": str(train_path),
        "target_col": ds.get("columns_hint", {}).get("target_col"),
        "time_col": ds.get("columns_hint", {}).get("time_col"),
        "id_cols": ds.get("columns_hint", {}).get("id_cols", []),
        # IMPORTANT: force absolute out root in YAML via ${output_root}
        "output_root": str(project_root / "out"),
    }
    runtime_vars.update(runtime_overrides)  # notebook wins

    # ---- Load + resolve pipeline YAML ----
    loaded = load_and_resolve(pipeline_yaml_path, runtime_vars=runtime_vars)

    # ---- Validate (preview mode) ----
    vr = validate_config(loaded.resolved, mode="preview")
    if not vr.ok:
        raise ValueError("Config validation failed (preview): " + " | ".join(vr.errors))

    # ---- DTO ----
    cfg = ProjectConfig.from_dict(loaded.resolved)

    # ---- Build RunContext + init logging for this run ----
    rc = build_run_context(cfg, project_root=project_root, run_name=f"{cfg.pipeline.task.value}_preview")
    init_logging(rc.log_file, level="DEBUG")
    _log = get_logger(__name__)  # re-acquire after init_logging
    _log.info("preview: start task=%s pipeline=%s", cfg.pipeline.task.value, cfg.pipeline.name)

    # ---- Save audit config_used with unique name ----
    config_used_path = save_config_used_unique(
        loaded.resolved,
        task=cfg.pipeline.task.value,
        audit_dir=rc.audit_dir,
        run_ts=rc.run_ts
    )
    _log.info("preview: saved config_used=%s", config_used_path)

    # ---- Load CSV sample for Stage2 ----
    df, meta = load_csv(csv_path=train_path, csv_params=csv_params, strategy=stage2_strategy)
    _log.info("preview: loaded df rows=%d cols=%d mode=%s", len(df), df.shape[1], meta.get("mode"))

    return {
        "cfg": cfg,
        "run_context": rc,
        "config_used_path": str(config_used_path),
        "df_head": df.head(5),
        "read_meta": meta,
        "validation_warnings": vr.warnings,
    }
