# src/crispdm/config/schema_dto_config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.enums_utils_config import ProblemType, normalize_problem_type, LogLevel

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This is the "schema.py" layer:
# - Converts resolved YAML dicts into typed Python objects (DTOs)
# - Provides a stable configuration contract for the rest of the pipeline
#
# Program flow:
# - load_loader_config.load_and_resolve() -> dict
# - ProjectConfig.from_dict(resolved_dict) -> ProjectConfig (typed)
# - stages/pipelines consume ProjectConfig, never raw YAML
#
# Design patterns
# - GoF: none.
# - Enterprise/Architectural:
#   - DTO / Schema layer
#   - Configuration Object (single source of truth)
# =============================================================================


@dataclass(frozen=True)
class PipelineMeta:
    """
    Describes the pipeline as a product: name, task, objective, and variables.
    """
    name: str
    task: ProblemType
    objective: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Global runtime settings for the whole run (not "theory CRISP", but execution ops).
    """
    random_seed: int = 42
    output_root: Path = Path("out")
    overwrite_artifacts: bool = True
    log_level: str = LogLevel.DEBUG


@dataclass(frozen=True)
class Stage2Config:
    """
    Stage 2 (Data Understanding) configuration.
    Stage 2 MUST be report-only (no data modification).
    """
    enabled: bool
    objective: str
    dataset_input: Dict[str, Any] = field(default_factory=dict)
    output_policy: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StagesConfig:
    """
    Container for stage configs. We start with Stage2 and extend later.
    """
    stage2_understanding: Stage2Config


@dataclass(frozen=True)
class ProjectConfig:
    """
    Root configuration object used by the entire program.
    """
    version: str
    pipeline: PipelineMeta
    runtime: RuntimeConfig
    stages: StagesConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProjectConfig":
        """
        Build a typed ProjectConfig from a resolved YAML dictionary.

        Why strict parsing?
        - It catches config drift early (typos, missing sections).
        - It keeps the rest of the codebase stable.
        """
        log.info("ProjectConfig.from_dict: start")
        log.debug("ProjectConfig.from_dict: top_keys=%s", list(d.keys()))

        version = str(d.get("version", "1.0"))

        pipe = d.get("pipeline", {})
        runtime = d.get("runtime", {})
        stages = d.get("stages", {})

        if not isinstance(pipe, dict):
            raise ValueError("pipeline must be a dict")
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be a dict")
        if not isinstance(stages, dict):
            raise ValueError("stages must be a dict")

        pipeline_meta = PipelineMeta(
            name=str(pipe.get("name", "")),
            task=normalize_problem_type(pipe.get("task", "")),
            objective=str(pipe.get("objective", "") or ""),
            variables=pipe.get("variables", {}) if isinstance(pipe.get("variables", {}), dict) else {},
        )

        runtime_cfg = RuntimeConfig(
            random_seed=int(runtime.get("random_seed", 42)),
            output_root=Path(str(runtime.get("output_root", "out"))),
            overwrite_artifacts=bool(runtime.get("overwrite_artifacts", True)),
        )

        # Stage 2
        s2 = stages.get("stage2_understanding", {})
        if not isinstance(s2, dict):
            raise ValueError("stages.stage2_understanding must be a dict")

        stage2_cfg = Stage2Config(
            enabled=bool(s2.get("enabled", True)),
            objective=str(s2.get("objective", "") or ""),
            dataset_input=s2.get("dataset_input", {}) if isinstance(s2.get("dataset_input", {}), dict) else {},
            output_policy=s2.get("output_policy", {}) if isinstance(s2.get("output_policy", {}), dict) else {},
            steps=s2.get("steps", {}) if isinstance(s2.get("steps", {}), dict) else {},
        )

        cfg = ProjectConfig(
            version=version,
            pipeline=pipeline_meta,
            runtime=runtime_cfg,
            stages=StagesConfig(stage2_understanding=stage2_cfg),
        )

        log.info("ProjectConfig.from_dict: done pipeline=%s task=%s output_root=%s",
                 cfg.pipeline.name, cfg.pipeline.task.value, cfg.runtime.output_root)
        return cfg
