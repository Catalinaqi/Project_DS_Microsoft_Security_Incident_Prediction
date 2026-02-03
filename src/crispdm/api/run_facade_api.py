# src/crispdm/api/run_facade_api.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from crispdm.core.logging_utils_core import get_logger, build_log_file, init_logging
from crispdm.config.build_factory_config import build_run_config
from crispdm.config.enums_utils_config import ProblemType
from crispdm.reporting.artifacts_service_reporting import create_run_dir

from crispdm.pipeline.clustering_runner_pipeline import run_clustering_runner_pipeline
#from crispdm.pipeline.classification_runner_pipeline import run_classification_runner_pipeline
#from crispdm.pipeline.regression_runner_pipeline import run_regression_runner_pipeline
#from crispdm.pipeline.timeseries_runner_pipeline import run_timeseries_runner_pipeline

log = get_logger(__name__)

# =============================================================================
# Facade: builds config + initializes run_dir + dispatches to pipeline runner
# =============================================================================

def run_pipeline(
        *,
        pipeline_config_path: str | Path,
        dataset_config_path: str | Path,
        dataset_key: str,
        notebook_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    notebook_vars = notebook_vars or {}

    built = build_run_config(
        pipeline_config_path=pipeline_config_path,
        dataset_config_path=dataset_config_path,
        dataset_key=dataset_key,
        notebook_vars=notebook_vars,
    )
    cfg = built.project_config

    # Init logging for this run
    log_file = build_log_file(
        output_root=cfg.runtime.output_root,
        run_name=f"{cfg.pipeline.task.value}_run"
    )
    init_logging(log_file, level=str(cfg.runtime.log_level))

    log.info("=== RUN START task=%s dataset_key=%s ===", cfg.pipeline.task.value, dataset_key)
    log.info("audit_config_used=%s", built.audit_path)

    # Create run directory (artifact policy)
    run_dir = create_run_dir(
        output_root=cfg.runtime.output_root,
        task=cfg.pipeline.task,
        dataset_key=dataset_key,
    )
    log.info("run_dir=%s", run_dir)

    # Dispatch
    if cfg.pipeline.task == ProblemType.CLUSTERING:
        return run_clustering_runner_pipeline(cfg=cfg, run_dir=run_dir)

    #if cfg.pipeline.task == ProblemType.CLASSIFICATION:
        #return run_classification_runner_pipeline(cfg=cfg, run_dir=run_dir)

    # if cfg.pipeline.task == ProblemType.REGRESSION:
    #     return run_regression_runner_pipeline(cfg=cfg, run_dir=run_dir)
    #
    # if cfg.pipeline.task == ProblemType.TIMESERIES:
    #     return run_timeseries_runner_pipeline(cfg=cfg, run_dir=run_dir)

    raise ValueError(f"Unsupported task: {cfg.pipeline.task}")
