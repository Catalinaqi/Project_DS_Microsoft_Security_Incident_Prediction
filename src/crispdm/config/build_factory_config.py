
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.load_loader_config import load_and_resolve, load_yaml
from crispdm.config.schema_dto_config import ProjectConfig
from crispdm.config.validate_validator_config import validate_config_dict
from crispdm.config.enums_utils_config import ProblemType, normalize_problem_type
from crispdm.reporting.audit_service_reporting import save_config_used

import json

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This module builds the final "resolved configuration" used by the program.
# It is the assembly point for:
# - pipeline config YAML
# - dataset config YAML
# - notebook runtime variables overrides
#
# Program flow:
# - preview_facade_api -> build_preview_config(...)
#   -> load YAMLs
#   -> compute runtime_vars (dataset_path, etc.)
#   -> resolve ${vars} in pipeline YAML
#   -> validate (preview mode)
#   -> convert to DTO (ProjectConfig)
#   -> save audit snapshot (config_used.yml)
#
# Design patterns
# - GoF:
#   - Factory (creates final ProjectConfig)
# - Enterprise/Architectural:
#   - Builder (assemble config from multiple sources)
#   - Single Source of Truth (typed ProjectConfig)
# =============================================================================


@dataclass(frozen=True)
class BuiltConfig:
    """
    Returned by builder functions to keep both typed config and raw dicts.
    """
    project_config: ProjectConfig
    resolved_dict: Dict[str, Any]
    audit_path: Path


def _select_dataset_path(dataset_cfg: Dict[str, Any], *, split: str = "train") -> str:
    """
    Dataset config format expected:
      datasets:
        <key>:
          paths:
            train: "..."
            test: "..."
    """
    log.info("Start [_select_dataset_path] for split='%s' and dataset_cfg='%s'", split, dataset_cfg)
    log.info("dataset_cfg transformed to json: %s", json.dumps(dataset_cfg, indent=2,ensure_ascii=False))
    datasets = dataset_cfg.get("datasets") or {} # no exist en el dict
    log.info("Start [_select_dataset_path] for dataset_cfg='%s'", dataset_cfg.get("datasets"))
    log.info("Start [_select_dataset_path] for datasets='%s'", datasets)
    #t) or not datasets:
    #    raise ValueError("dataset_config.yml must contain datasets:<key>:... mapping.")

    # If only one dataset, use it; otherwise caller must choose key earlier.
    # Here dataset_cfg is already filtered to a single dataset entry by build_*.
    paths = dataset_cfg.get("paths") or {}
    log.info("Selecting dataset path for split='%s' from paths: %s", split, paths)
    if not isinstance(paths, dict):
        raise ValueError("dataset entry must contain paths:{train:,test:} mapping.")
    p = paths.get(split)
    if not p:
        raise ValueError(f"Dataset paths missing '{split}'.")
    log.info("End [_select_dataset_path]: selected dataset path='%s'", p)
    return str(p)


def _load_dataset_entry(dataset_config_path: Path, dataset_key: str) -> Dict[str, Any]:
    """
    Load a single dataset entry from dataset_config.yml.
    Raises KeyError if dataset_key not found.
    :param dataset_config_path:
    :param dataset_key:
    :return:
    """
    log.info("Start [_load_dataset_entry]: Loading dataset entry for key='%s' and path='%s'", dataset_key, dataset_config_path)
    log.info("Start [load_yaml.load_loader_config] Loading yaml dataset config from %s", dataset_config_path)
    raw = load_yaml(dataset_config_path)
    log.info("End [load_yaml.load_loader_config] Loaded keys: %s", list(raw.keys()))
    datasets = raw.get("datasets") or {}
    if dataset_key not in datasets:
        raise KeyError(f"dataset_key '{dataset_key}' not found in {dataset_config_path}. Available: {list(datasets.keys())}")
    entry = datasets[dataset_key] or {}
    if not isinstance(entry, dict):
        raise ValueError(f"datasets.{dataset_key} must be a dict.")
    elif dataset_key in datasets:
        log.info("End [_load_dataset_entry]: Loading dataset entry for key='%s'", dataset_key)

    return {"version": raw.get("version", "1.0"), "key": dataset_key, **entry}


def build_preview_config(
        *,
        pipeline_config_path: str | Path,
        dataset_config_path: str | Path,
        dataset_key: str,
        notebook_vars: Optional[Dict[str, Any]] = None, # target_col, time_col, id_cols
) -> BuiltConfig:
    """
    Build config for the PREVIEW flow:
    - Must load CSV (dataset_path required)
    - Can leave target_col/time_col empty (Stage2 will suggest)
    """
    notebook_vars = notebook_vars or {}
    pipeline_config_path = Path(pipeline_config_path)
    dataset_config_path = Path(dataset_config_path)

    log.info("Start [build_preview_config] notebooks_vars=%s pipeline=%s dataset=%s key=%s",
             notebook_vars, pipeline_config_path, dataset_config_path, dataset_key)

    dataset_entry = _load_dataset_entry(dataset_config_path, dataset_key)

    dataset_path = notebook_vars.get("dataset_path") or _select_dataset_path(dataset_entry, split="train")


    # Vars injected into pipeline YAML placeholders:
    runtime_vars: Dict[str, Any] = {
        "dataset_path": dataset_path,
        "target_col": notebook_vars.get("target_col"),
        "time_col": notebook_vars.get("time_col"),
        "id_cols": notebook_vars.get("id_cols"),
        "output_root": notebook_vars.get("output_root"),
    }

    log.info("Start [load_and_resolve.load_loader_config] with path='%s' and runtime_vars=%s", pipeline_config_path,runtime_vars)
    loaded = load_and_resolve(pipeline_config_path, runtime_vars=runtime_vars)
    resolved = loaded.resolved
    log.info("End [load_and_resolve.load_loader_config] ...) Resolved keys: %s", list(resolved.keys()))

    # Validate in preview mode:
    log.info("Start [validate_validator_config.validate_config_dict] in preview mode")
    vr = validate_config_dict(resolved, mode="preview")
    vr.raise_if_invalid()

    # Convert into typed DTO:
    cfg = ProjectConfig.from_dict(resolved)

    # Save audit snapshot:
    audit_path = save_config_used(
        resolved,
        task=cfg.pipeline.task,
        output_root=cfg.runtime.output_root,
    )

    log.info("[build_preview_config] DONE task=%s audit=%s",
             cfg.pipeline.task.value, audit_path)
    return BuiltConfig(project_config=cfg, resolved_dict=resolved, audit_path=audit_path)
