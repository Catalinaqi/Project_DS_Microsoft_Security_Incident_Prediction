# src/crispdm/config/validate_validator_config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.enums_utils_config import ProblemType, normalize_problem_type

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Enterprise / Architectural patterns:
# - Validator component: centralizes config validation rules.
# - Fail-fast configuration: detect drift/missing required fields early.
# Not a GoF pattern.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def _as_list(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def validate_config(resolved_yaml: Dict[str, Any], *, mode: str = "preview") -> ValidationResult:
    """
    Validate resolved YAML config.

    mode="preview":
      - allows target_col/time_col to be None (Stage2 will suggest candidates)
    mode="run":
      - enforces required fields for Stage3+ execution
    """
    mode = (mode or "").strip().lower()
    log.info("validate_config: start mode=%s", mode)

    errors: List[str] = []
    warnings: List[str] = []

    if mode not in {"preview", "run"}:
        return ValidationResult(False, [f"Invalid mode='{mode}'. Use 'preview' or 'run'."], [])

    pipeline = resolved_yaml.get("pipeline", {})
    runtime = resolved_yaml.get("runtime", {})
    stages = resolved_yaml.get("stages", {})

    if not isinstance(pipeline, dict):
        errors.append("Missing or invalid 'pipeline' section (must be dict).")
        return ValidationResult(False, errors, warnings)

    task_raw = pipeline.get("task")
    if not task_raw:
        errors.append("Missing pipeline.task (clustering/classification/regression/timeseries).")
        return ValidationResult(False, errors, warnings)

    try:
        task = normalize_problem_type(task_raw)
    except Exception:
        errors.append(f"Invalid pipeline.task='{task_raw}'.")
        return ValidationResult(False, errors, warnings)

    variables = pipeline.get("variables", {})
    if not isinstance(variables, dict):
        errors.append("pipeline.variables must be a dict.")
        return ValidationResult(False, errors, warnings)

    dataset_path = variables.get("dataset_path")
    target_col = variables.get("target_col")
    time_col = variables.get("time_col")
    id_cols = _as_list(variables.get("id_cols"))

    # ---- dataset_path ----
    if not dataset_path:
        errors.append("pipeline.variables.dataset_path is required.")
    else:
        p = Path(str(dataset_path))
        if not p.exists():
            msg = f"dataset_path does not exist: {p}"
            # In preview we allow relative/wrong paths (user may adjust), but in run we must fail.
            (errors if mode == "run" else warnings).append(msg)

    # ---- task rules ----
    if task == ProblemType.CLUSTERING:
        if target_col is not None:
            errors.append("clustering: target_col must be null/None.")
        # time_col optional, id_cols optional

    elif task in {ProblemType.CLASSIFICATION, ProblemType.REGRESSION}:
        if mode == "run" and not target_col:
            errors.append(f"{task.value}: target_col is required in run mode.")
        if mode == "preview" and not target_col:
            warnings.append(f"{task.value}: target_col is None in preview mode (Stage2 should suggest candidates).")

    elif task == ProblemType.TIMESERIES:
        if mode == "run":
            if not time_col:
                errors.append("timeseries: time_col is required in run mode.")
            if not target_col:
                errors.append("timeseries: target_col is required in run mode.")
        else:
            if not time_col:
                warnings.append("timeseries: time_col is None in preview mode (Stage2 should suggest candidates).")
            if not target_col:
                warnings.append("timeseries: target_col is None in preview mode (Stage2 should suggest candidates).")

    # ---- optional sanity ----
    if id_cols:
        if target_col and target_col in id_cols:
            warnings.append("id_cols includes target_col (usually wrong).")
        if time_col and time_col in id_cols:
            warnings.append("id_cols includes time_col (usually wrong).")

    # ---- stage presence (preview flow requires stage2) ----
    if not isinstance(stages, dict):
        errors.append("Missing or invalid 'stages' section (must be dict).")
    else:
        if "stage2_understanding" not in stages:
            warnings.append("stages.stage2_understanding not found (preview flow expects it).")

    ok = len(errors) == 0
    if ok:
        log.info("validate_config: OK task=%s mode=%s", task.value, mode)
    else:
        log.error("validate_config: FAILED task=%s mode=%s errors=%d", task.value, mode, len(errors))
        for e in errors:
            log.error("  - %s", e)

    for w in warnings:
        log.warning("  - %s", w)

    return ValidationResult(ok=ok, errors=errors, warnings=warnings)
