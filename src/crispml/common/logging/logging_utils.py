# src/crispml/common/logging_utils.py

"""
Centralized logging system for the CRISP-ML framework.

This module configures a single root logger called `crispml`, and all
modules inside the library obtain child loggers via get_logger(__name__).

Benefits:
---------
- Prevents duplicated logging configuration across modules.
- Ensures consistent formatting (console + file).
- Automatically creates a clean logs folder `out/logs`.
- Designed for notebooks, terminal, and long experiments.
"""

from __future__ import annotations
import logging
from pathlib import Path


# ---------------------------------------------------------------------
# INTERNAL: configure the root CRISP-ML logger once
# ---------------------------------------------------------------------
def _configure_root_logger() -> None:
    """
    Configures the root logger 'crispml' with:
        - Console handler   (INFO level)
        - File handler      (INFO level)

    This function is executed only once, even if called multiple times.
    """
    root_logger = logging.getLogger("crispml")

    # Avoid configuring multiple times
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)

    # Build log directory: <project_root>/out/logs
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "out" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "crispml.log"

    # ------------------------
    # Console handler
    # ------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # ------------------------
    # File handler
    # ------------------------
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.info("CRISP-ML logging initialized. Log file: %s", log_file)


# ---------------------------------------------------------------------
# PUBLIC: obtain a logger for a module
# ---------------------------------------------------------------------
def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a properly configured logger for a given module.

    Parameters
    ----------
    module_name : str
        Usually __name__ from the calling module.

    Returns
    -------
    logging.Logger
        A child logger of `crispml` with consistent formatting and handlers.
    """
    _configure_root_logger()
    return logging.getLogger(f"crispml.{module_name}")
