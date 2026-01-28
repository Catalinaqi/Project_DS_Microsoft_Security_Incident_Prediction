# src/crispdm/reporting/audit_service_reporting.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.enums_utils_config import ProblemType

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# This module implements the "audit snapshot" capability:
# - Persists the EXACT resolved configuration used for a run.
# - Enables reproducibility and traceability (what config produced what artifacts).
#
# Program flow:
# - build_factory_config.build_preview_config/build_run_config
#   -> calls save_config_used(resolved_config, ...)
#   -> writes out/audit/config_used__<task>__<timestamp>.yml
#
# Design patterns
# - GoF: none
# - Enterprise/Architectural:
#   - Audit Trail / Configuration Snapshot
#   - Reproducibility support (ML governance basic)
# =============================================================================


def save_config_used(
        resolved_config: Dict[str, Any],
        *,
        task: ProblemType,
        output_root: Path | str,
        timestamp: Optional[str] = None,
) -> Path:
    """
    Save the resolved config used for a run as:
      out/audit/config_used__<task>__<YYYYMMDD_HHMMSS>.yml

    Notes:
    - Does NOT overwrite previous runs (timestamped filename).
    - Called by facades/builders (preview/run), not by stage services.
    """
    out_root = Path(output_root)
    audit_dir = out_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_used__{task.value}__{ts}.yml"
    out_path = audit_dir / filename

    out_path.write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )

    log.info("[audit] config_used saved: %s", out_path)
    return out_path
