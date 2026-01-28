# src/crispdm/config/build_factory_config.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from crispdm.core.logging_utils_core import get_logger, build_log_file
from crispdm.config.schema_dto_config import ProjectConfig

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Enterprise / Architectural patterns:
# - Builder/Factory for "runtime context": compute absolute paths, timestamps,
#   audit/log filenames (single place, reproducible runs).
# Not strictly GoF Factory Method (no inheritance), but same intent.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    run_ts: str
    task: str
    pipeline_name: str

    out_root: Path
    audit_dir: Path
    logs_dir: Path
    figures_dir: Path
    tables_png_dir: Path
    metrics_dir: Path
    models_dir: Path

    log_file: Path


def _abs_under(project_root: Path, maybe_rel: Path) -> Path:
    return maybe_rel if maybe_rel.is_absolute() else (project_root / maybe_rel)


def build_run_context(
        cfg: ProjectConfig,
        *,
        project_root: Path,
        run_name: Optional[str] = None,
        timestamp: Optional[str] = None,
) -> RunContext:
    """
    Build a consistent set of per-run paths.
    For now we keep a simple structure (your current out/ folders).
    Later you can switch to out/runs/<task>/<dataset>/<ts>/ without changing callers.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    task = cfg.pipeline.task.value
    pipe_name = cfg.pipeline.name

    out_root = _abs_under(project_root, Path(str(cfg.runtime.output_root)))

    audit_dir = out_root / "audit"
    logs_dir = out_root / "logs"
    figures_dir = out_root / "figures"
    tables_png_dir = out_root / "tables_png"
    metrics_dir = out_root / "metrics"
    models_dir = out_root / "models"

    # Ensure dirs exist (so later services can just write)
    for d in [audit_dir, logs_dir, figures_dir, tables_png_dir, metrics_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    effective_run_name = run_name or f"{task}_preview"
    log_file = build_log_file(output_root=out_root, run_name=effective_run_name, timestamp=ts)

    log.debug(
        "build_run_context: out_root=%s audit=%s logs=%s",
        out_root, audit_dir, logs_dir
    )

    return RunContext(
        project_root=project_root,
        run_ts=ts,
        task=task,
        pipeline_name=pipe_name,
        out_root=out_root,
        audit_dir=audit_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        tables_png_dir=tables_png_dir,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
        log_file=log_file,
    )
