
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
        "output_root": notebook_vars.get("output_root") or "out",

    }

    log.info("Start [load_and_resolve.load_loader_config] with path='%s' and runtime_vars=%s", pipeline_config_path,runtime_vars)
    loaded = load_and_resolve(pipeline_config_path, runtime_vars=runtime_vars)
    resolved = loaded.resolved
    log.info("End [load_and_resolve.load_loader_config] ...) Resolved keys: %s", list(resolved.keys()))

    # Validate in preview mode:
    log.info("Start [validate_validator_config.validate_config_dict] in preview mode")
    vr = validate_config_dict(resolved, mode="preview")
    log.info("End [validate_validator_config.validate_config_dict] ... Validation ok=%s errors=%d warnings=%d",
             vr.ok, len(vr.errors), len(vr.warnings))

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



def build_run_config(
        *,
        pipeline_config_path: str | Path,
        dataset_config_path: str | Path,
        dataset_key: str,
        notebook_vars: Optional[Dict[str, Any]] = None,
) -> BuiltConfig:
    """
    RUN flow:
    - Requires required columns depending on task (validator en mode='run')
    - Validates with mode='run'
    - Builds full ProjectConfig for stages 2–5
    """
    notebook_vars = notebook_vars or {}
    pipeline_config_path = Path(pipeline_config_path)
    dataset_config_path = Path(dataset_config_path)

    dataset_entry = _load_dataset_entry(dataset_config_path, dataset_key)
    runtime_vars = _build_runtime_vars(dataset_entry, notebook_vars)

    log.info(
        "build_run_config: pipeline=%s dataset=%s key=%s runtime_vars=%s",
        pipeline_config_path, dataset_config_path, dataset_key, runtime_vars
    )

    loaded = load_and_resolve(pipeline_config_path, runtime_vars=runtime_vars)
    resolved = loaded.resolved

    resolved = apply_dataset_defaults(resolved, dataset_entry)

    vr = validate_config_dict(resolved, mode="run")
    vr.raise_if_invalid()

    cfg = ProjectConfig.from_dict(resolved)

    audit_path = save_config_used(
        resolved,
        task=cfg.pipeline.task,
        output_root=cfg.runtime.output_root,
    )

    return BuiltConfig(project_config=cfg, resolved_dict=resolved, audit_path=audit_path)


def apply_dataset_defaults(resolved: Dict[str, Any], dataset_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply dataset_config.yml defaults into the resolved pipeline dict.
    Only fills missing sections (does NOT override explicit pipeline config).

    Dataset entry expected (examples):
    - csv_params
    - stage2_read_strategy
    - stage3_read_strategy
    - columns_hint
    - suggestion_policy
    - test_has_target
    """
    stages = _ensure_dict(resolved.get("stages"))
    s2 = _ensure_dict(stages.get("stage2_understanding"))
    s3 = _ensure_dict(stages.get("stage3_preparation"))

    # -------- Stage2 defaults --------
    #dataset_input = _ensure_dict(s2.get("dataset_input"))
    #_set_if_missing(dataset_input, "csv_params", _ensure_dict(dataset_entry.get("csv_params")))
    #s2["dataset_input"] = dataset_input

    #_set_if_missing(s2, "read_strategy", _ensure_dict(dataset_entry.get("stage2_read_strategy")))
    dataset_input = _ensure_dict(s2.get("dataset_input"))
    _set_if_missing(dataset_input, "csv_params", _ensure_dict(dataset_entry.get("csv_params")))
    _set_if_missing(dataset_input, "read_strategy", _ensure_dict(dataset_entry.get("stage2_read_strategy")))
    s2["dataset_input"] = dataset_input




    # hints for stage2 logic (suggestions)
    s2_steps = _ensure_dict(s2.get("steps"))
    _set_if_missing(s2_steps, "columns_hint", _ensure_dict(dataset_entry.get("columns_hint")))
    _set_if_missing(s2_steps, "suggestion_policy", _ensure_dict(dataset_entry.get("suggestion_policy")))
    _set_if_missing(s2_steps, "test_has_target", bool(dataset_entry.get("test_has_target", False)))

    s2["steps"] = s2_steps
    stages["stage2_understanding"] = s2



    # -------- Stage3 defaults --------
    #_set_if_missing(s3, "read_strategy", _ensure_dict(dataset_entry.get("stage3_read_strategy")))
    s3 = _ensure_dict(stages.get("stage3_preparation"))
    s3_input = _ensure_dict(s3.get("dataset_input"))
    _set_if_missing(s3_input, "read_strategy", _ensure_dict(dataset_entry.get("stage3_read_strategy")))
    s3["dataset_input"] = s3_input
    stages["stage3_preparation"] = s3

    #stages["stage3_preparation"] = s3

    # ---------------------------------

    resolved["stages"] = stages
    return resolved


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _set_if_missing(d: Dict[str, Any], key: str, value: Any) -> None:
    if key not in d or d.get(key) in (None, "", {}, []):
        d[key] = value

def _build_runtime_vars(dataset_entry: Dict[str, Any], notebook_vars: Dict[str, Any]) -> Dict[str, Any]:
    """
    Centralize runtime vars used to resolve ${...} placeholders in pipeline YAML.
    """
    dataset_path = notebook_vars.get("dataset_path") or _select_dataset_path(dataset_entry, split="train")

    return {
        "dataset_path": dataset_path,
        "target_col": notebook_vars.get("target_col"),
        "time_col": notebook_vars.get("time_col"),
        "id_cols": notebook_vars.get("id_cols"),
        "output_root": notebook_vars.get("output_root") or "out",
    }
