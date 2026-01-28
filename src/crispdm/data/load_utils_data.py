# src/crispdm/data/load_utils_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from crispdm.core.logging_utils_core import get_logger
from crispdm.config.enums_utils_config import ReadMode

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Enterprise / Architectural patterns:
# - Data Access Utility (I/O boundary): isolates Pandas read_csv and strategies.
# - Keeps pipelines/stages clean (separation of concerns).
# Not a GoF pattern.
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReadStrategy:
    mode: ReadMode = ReadMode.SAMPLE
    sample_rows: int = 200_000
    sample_frac: Optional[float] = None
    random_state: int = 42
    chunksize: int = 200_000


def resolve_path(path: str | Path, *, project_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root / p)


def _parse_strategy(d: Dict[str, Any] | None) -> ReadStrategy:
    d = d or {}
    mode = ReadMode(str(d.get("mode", "sample")).strip().lower())
    return ReadStrategy(
        mode=mode,
        sample_rows=int(d.get("sample_rows", 200_000)),
        sample_frac=d.get("sample_frac"),
        random_state=int(d.get("random_state", 42)),
        chunksize=int(d.get("chunksize", 200_000)),
    )


def load_csv(
        *,
        csv_path: Path,
        csv_params: Dict[str, Any] | None = None,
        strategy: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load CSV with memory-friendly strategy.

    Returns:
      df: DataFrame (full or sample)
      meta: info about how it was read (mode, rows, path)
    """
    csv_params = csv_params or {}
    strat = _parse_strategy(strategy)

    log.info("load_csv: start path=%s mode=%s", csv_path, strat.mode.value)
    log.debug("load_csv: csv_params=%s", csv_params)
    log.debug("load_csv: strategy=%s", strat)

    # Pandas params
    sep = csv_params.get("sep", ",")
    encoding = csv_params.get("encoding", "utf-8")
    decimal = csv_params.get("decimal", ".")
    low_memory = bool(csv_params.get("low_memory", True))

    if strat.mode == ReadMode.FULL:
        df = pd.read_csv(csv_path, sep=sep, encoding=encoding, decimal=decimal, low_memory=low_memory)
        meta = {"mode": "full", "rows": len(df), "path": str(csv_path)}
        log.info("load_csv: done rows=%d", len(df))
        return df, meta

    if strat.mode == ReadMode.SAMPLE:
        # For big CSV, nrows is the simplest and most stable
        nrows = strat.sample_rows
        df = pd.read_csv(csv_path, sep=sep, encoding=encoding, decimal=decimal, low_memory=low_memory, nrows=nrows)
        meta = {"mode": "sample", "rows": len(df), "nrows": nrows, "path": str(csv_path)}
        log.info("load_csv: done rows=%d (sample)", len(df))
        return df, meta

    # CHUNKED: read first N rows via iterator (still returns a sample df)
    if strat.mode == ReadMode.CHUNKED:
        target_rows = strat.sample_rows
        chunksize = strat.chunksize
        acc = []
        read_rows = 0
        for chunk in pd.read_csv(
                csv_path,
                sep=sep,
                encoding=encoding,
                decimal=decimal,
                low_memory=low_memory,
                chunksize=chunksize
        ):
            acc.append(chunk)
            read_rows += len(chunk)
            if read_rows >= target_rows:
                break
        df = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()
        df = df.iloc[:target_rows].copy()
        meta = {"mode": "chunked", "rows": len(df), "chunksize": chunksize, "target_rows": target_rows, "path": str(csv_path)}
        log.info("load_csv: done rows=%d (chunked sample)", len(df))
        return df, meta

    raise ValueError(f"Unsupported read mode: {strat.mode}")
