from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import pandas as pd

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.enums_utils_config import ReadMode, normalize_read_mode

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Data loading utilities focused on CSV (your scenario).
# Supports strategies for large files:
# - sample: read only a small subset (fast for Stage2 profiling)
# - chunked: iterate through chunks to scan full file without full memory load
# - full: load entire file into memory
#
# Program flow:
# - profiling_service_data (Stage2) calls load_csv_* based on read strategy
#
# Design patterns
# - GoF: none
# - Enterprise/Architectural:
#   - Data Access Layer (thin)
#   - Strategy (ReadMode selects reading behavior)
# =============================================================================


@dataclass(frozen=True)
class CsvReadStrategy:
    mode: ReadMode = ReadMode.SAMPLE
    sample_rows: int = 200_000
    chunksize: int = 200_000
    random_state: int = 42


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()

def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = find_project_root()
    return root / p

def load_csv_sample(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        sample_rows: int = 200_000,
) -> pd.DataFrame:
    """
    Load a sample of the CSV quickly. For huge CSVs, nrows is the most practical
    approach (reads first N rows).
    """
    p = Path(path)
    log.info("[load_csv_sample] path=%s sample_rows=%d", p, sample_rows)
    params = dict(csv_params or {})
    log.info("[load_csv_sample] csv_params=%s", json.dumps(params, indent=2, ensure_ascii=False))

    #log.info("starting to load CSV with nrows=%d path=%d params=%d", sample_rows, p, params)
    #df = pd.read_csv(p, nrows=sample_rows, **params)
    #log.info("ended loading CSV")

    #log.info("[load_csv_sample] cwd=%s", Path.cwd())
    #log.info("[load_csv_sample] abs_path=%s exists=%s", p.resolve(), p.exists())


    try:
        p = resolve_path(path)
        log.info(
            "starting to load CSV with nrows=%d path=%s params=%s",
            sample_rows, str(p), json.dumps(params, ensure_ascii=False)
        )
        df = pd.read_csv(p, nrows=sample_rows, **params)
        log.info("ended loading CSV")
    except Exception:
        log.exception("[load_csv_sample] read_csv failed path=%s nrows=%d params=%s", p, sample_rows, params)
        raise

    log.info("[load_csv_sample] loaded rows=%d cols=%d", len(df), df.shape[1])
    return df


def iter_csv_chunks(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        chunksize: int = 200_000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield DataFrame chunks for scanning a very large CSV.
    """
    p = Path(path)
    log.info("[iter_csv_chunks] path=%s chunksize=%d", p, chunksize)
    params = dict(csv_params or {})
    for chunk in pd.read_csv(p, chunksize=chunksize, **params):
        yield chunk


def load_csv_full(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load full CSV into memory (use with caution for 2GB files).
    """
    p = Path(path)
    log.info("[load_csv_full] path=%s", p)
    params = dict(csv_params or {})
    df = pd.read_csv(p, **params)
    log.info("[load_csv_full] loaded rows=%d cols=%d", len(df), df.shape[1])
    return df


def load_csv_by_strategy(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Generator[pd.DataFrame, None, None]], CsvReadStrategy]:
    """
    Unified loader:
    Returns:
      (df_sample_or_full, chunks_generator_or_none, normalized_strategy)
    """
    strategy = strategy or {}
    log.info("[load_csv_by_strategy] path=%s strategy=%s", path, strategy)
    log.info("[load_csv_by_strategy] json to strategy=%s", json.dumps(strategy, indent=2,ensure_ascii=False))
    default=ReadMode.SAMPLE
    #log.info("[load_csv_by_strategy] read ReadMode.SAMPLE=%s", ReadMode.SAMPLE)
    log.info("[load_csv_by_strategy] read without value for default =%s", default)
    mode = normalize_read_mode(strategy.get("mode"))
    log.info("[load_csv_by_strategy] normalized without value for mode=%s", mode)
    #log.info("[load_csv_by_strategy] normalized mode=%s", mode.value)
    sample_rows = int(strategy.get("sample_rows", 200_000))
    chunksize = int(strategy.get("chunksize", 200_000))
    random_state = int(strategy.get("random_state", 42))

    norm = CsvReadStrategy(mode=mode, sample_rows=sample_rows, chunksize=chunksize, random_state=random_state)
    log.info("[load_csv_by_strategy] mode=%s", norm.mode.value)

    if norm.mode == ReadMode.SAMPLE:
        log.info("[load_csv_by_strategy] loading sample mode")
        return load_csv_sample(path, csv_params=csv_params, sample_rows=norm.sample_rows), None, norm
    if norm.mode == ReadMode.CHUNKED:
        log.info("[load_csv_by_strategy] loading chunked mode")
        return None, iter_csv_chunks(path, csv_params=csv_params, chunksize=norm.chunksize), norm
    # FULL:
    return load_csv_full(path, csv_params=csv_params), None, norm
