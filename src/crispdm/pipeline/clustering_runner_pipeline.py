# src/crispdm/pipelines/clustering_runner_pipelines.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.schema_dto_config import ProjectConfig

from crispdm.stage.stage2_understanding_runner_stage import run_stage2_understanding_runner_stage
#from crispdm.stage.stage3_preparation_runner_stage import run_stage3_preparation_runner_stage
#from crispdm.stage.stage4_modeling_runner_stage import run_stage4_modeling_runner_stage
#from crispdm.stage.stage5_evaluation_runner_stage import run_stage5_evaluation_runner_stage

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Pipeline runner for clustering:
# - Orchestrates Stage2->Stage5 for CLUSTERING task
#
# Design patterns
# - Enterprise/Architectural: Pipeline Runner (application orchestrator)
# =============================================================================


def run_clustering_runner_pipeline(*, cfg: ProjectConfig, run_dir: Path) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}

    log.info("[clustering_runner_pipelines] START run_dir=%s", run_dir)

    if cfg.stages.stage2_understanding.enabled:
        ctx = run_stage2_understanding_runner_stage(cfg=cfg, run_dir=run_dir, ctx=ctx)

    # cuando implementes stage3-5:
    # if hasattr(cfg.stages, "stage3_preparation") and cfg.stages.stage3_preparation.enabled:
    #     ctx = run_stage3_preparation_runner_stage(cfg=cfg, run_dir=run_dir, ctx=ctx)
    #
    # if hasattr(cfg.stages, "stage4_modeling") and cfg.stages.stage4_modeling.enabled:
    #     ctx = run_stage4_modeling_runner_stage(cfg=cfg, run_dir=run_dir, ctx=ctx)
    #
    # if hasattr(cfg.stages, "stage5_evaluation") and cfg.stages.stage5_evaluation.enabled:
    #     ctx = run_stage5_evaluation_runner_stage(cfg=cfg, run_dir=run_dir, ctx=ctx)

    log.info("[clustering_runner_pipelines] END")
    return ctx
