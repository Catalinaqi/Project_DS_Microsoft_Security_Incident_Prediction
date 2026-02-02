from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from crispdm.core.logging_utils_core import get_logger, init_logging, build_log_file
from crispdm.config.build_factory_config import build_preview_config, BuiltConfig
from crispdm.data.profiling_service_data import run_stage2_preview

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Facade entrypoint for the notebook PREVIEW flow.
# The notebook should call a single function, not multiple low-level services.
#
# Program flow:
# - Notebook -> preview_facade_api.run_preview(...)
#   -> build_factory_config.build_preview_config(...)
#   -> init_logging(...) (one log file per execution)
#   -> profiling_service_data.run_stage2_preview(...)
#   -> returns suggestions for target/time/id
#
# Design patterns
# - GoF:
#   - Facade (single entrypoint for notebook)
# - Enterprise/Architectural:
#   - Application Service / Orchestrator (thin)
# =============================================================================


@dataclass(frozen=True)
class PreviewResult:
    config: Any
    suggestions: Dict[str, Any] # Dictionary with suggested target/time/id columns
    audit_config_path: str
    log_file: str


def run_preview(
        *,
        pipeline_config_path: str | Path,
        dataset_config_path: str | Path,
        dataset_key: str,
        notebook_vars: Optional[Dict[str, Any]] = None,
) -> PreviewResult:
    """
    Build config + run Stage2 preview to suggest target/time/id columns.
    """
    notebook_vars = notebook_vars or {}
    log.info("Start [build_factory_config.build_preview_config]...with params: "
             "pipeline_config_path=%s dataset_config_path=%s dataset_key=%s notebook_vars=%s",
             pipeline_config_path,dataset_config_path,dataset_key,notebook_vars)
    built: BuiltConfig = build_preview_config(
        pipeline_config_path=pipeline_config_path,
        dataset_config_path=dataset_config_path,
        dataset_key=dataset_key,
        notebook_vars=notebook_vars,
    )

    cfg = built.project_config
    # Create a unique run name for logging/output:
    #run_name = f"preview_{cfg.pipeline.task.value}_{cfg.pipeline.name}"
    run_name = f"preview_{cfg.pipeline.name}"

    # Initialize logging ONCE per execution:
    log_file = build_log_file(cfg.runtime.output_root, run_name=run_name)
    init_logging(log_file=log_file, level=cfg.runtime.log_level)

    log.info("[run_preview] START task=%s dataset=%s",
             cfg.pipeline.task.value,
             cfg.pipeline.variables.get("dataset_path"))

    suggestions = run_stage2_preview(project_config=cfg)

    log.info("[run_preview] DONE")
    return PreviewResult(
        config=cfg,
        suggestions=suggestions,
        audit_config_path=str(built.audit_path),
        log_file=str(log_file),
    )
