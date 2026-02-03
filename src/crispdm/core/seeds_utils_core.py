# src/crispdm/core/seeds_utils_core.py
from __future__ import annotations

import os
import random
from typing import Optional

from crispdm.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Reproducibility is a core requirement in ML pipelines:
# - Same config + same seed => same splits, same metrics (within reason).
# - Ensures stable experiments and supports audit/compliance.
#
# Program flow (typical):
# - run_facade_api.run_pipeline(...)
#   -> seeds_utils_core.set_global_seed(cfg.runtime.seed)
#   -> pipeline runners + stages
#
# Design patterns:
# - Cross-cutting concern utility (reproducibility)
# =============================================================================


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """
    Set global random seeds for common libraries:
    - Python's random
    - NumPy (if installed)
    - PyTorch (if installed)

    deterministic=True:
    - For libraries like PyTorch, tries to enable deterministic algorithms.
    - Note: full determinism can still vary across OS/BLAS/hardware.
    """
    if seed is None:
        raise ValueError("seed must not be None")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy (optional)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception as e:
        log.debug("[seed] numpy not set (missing or error): %s", e)

    # PyTorch (optional)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        log.debug("[seed] torch not set (missing or error): %s", e)

    log.info("[seed] Global seed set: seed=%s deterministic=%s", seed, deterministic)
