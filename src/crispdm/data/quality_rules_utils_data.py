# src/crispdm/data/quality_rules_utils_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from crispdm.core.logging_utils_core import get_logger

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Data quality gates are a CRISP-ML requirement:
# - Externalize validation rules (YAML) so notebooks/code don't change.
# - Produce a structured report + violations table for reporting/audit.
#
# Program flow:
# - Stage2 loads df
# - Stage2 calls apply_quality_rules(df, cfg.rules.*)
# - Stage2 writes tables/figures of violations
#
# Design patterns:
# - Rule Engine (minimal)
# - External configuration (YAML-driven)
# =============================================================================


def load_rules_yml(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rules file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data


def apply_quality_rules(
        df: pd.DataFrame,
        *,
        ranges_rules_path: Optional[Path | str] = None,
        logic_rules_path: Optional[Path | str] = None,
        business_rules_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Apply up to 3 rule sets.
    Returns a dict with:
      - summary counters
      - violations table (as records)
    """
    all_violations: List[Dict[str, Any]] = []

    if ranges_rules_path:
        rules = load_rules_yml(ranges_rules_path)
        all_violations += _apply_ranges(df, rules)

    if logic_rules_path:
        rules = load_rules_yml(logic_rules_path)
        all_violations += _apply_logic(df, rules)

    if business_rules_path:
        rules = load_rules_yml(business_rules_path)
        all_violations += _apply_business(df, rules)

    vio_df = pd.DataFrame(all_violations) if all_violations else pd.DataFrame(
        columns=["rule_set", "rule_name", "column", "violation_count", "example_values"]
    )

    report = {
        "total_rules_violated": int((vio_df["violation_count"] > 0).sum()) if len(vio_df) else 0,
        "total_violations": int(vio_df["violation_count"].sum()) if len(vio_df) else 0,
        "violations": vio_df.to_dict(orient="records"),
    }
    log.info("[quality] violations=%s", report["total_violations"])
    return report


def violations_to_df(report: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report.get("violations", []))


# ---------------------------- Internal rule handlers ----------------------------

def _apply_ranges(df: pd.DataFrame, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expected YAML structure (example):
    rules:
      - name: "age_range"
        column: "age"
        min: 0
        max: 120
      - name: "country_allowed"
        column: "country"
        allowed_values: ["IT", "ES", "PE"]
    """
    out = []
    for r in (rules.get("rules") or []):
        name = r.get("name", "unnamed_range_rule")
        col = r.get("column")
        if not col or col not in df.columns:
            out.append(_vio("ranges", name, col or "", 0, ["SKIPPED: missing column"]))
            continue

        s = df[col]
        mask = pd.Series([False] * len(df), index=df.index)

        if "min" in r:
            mask |= s < r["min"]
        if "max" in r:
            mask |= s > r["max"]
        if "allowed_values" in r:
            allowed = set(r["allowed_values"])
            mask |= ~s.isna() & ~s.isin(allowed)
        if "regex" in r:
            mask |= ~s.astype(str).str.match(r["regex"], na=False)

        vio_count = int(mask.sum())
        examples = s[mask].dropna().astype(str).head(5).tolist()
        out.append(_vio("ranges", name, col, vio_count, examples))
    return out


def _apply_logic(df: pd.DataFrame, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Minimal logic rules:
    rules:
      - name: "start_before_end"
        expr: "start_date <= end_date"
    expr is evaluated with pandas.DataFrame.eval, so keep expressions simple.
    """
    out = []
    for r in (rules.get("rules") or []):
        name = r.get("name", "unnamed_logic_rule")
        expr = r.get("expr")
        if not expr:
            out.append(_vio("logic", name, "", 0, ["SKIPPED: missing expr"]))
            continue

        try:
            ok = df.eval(expr)
            # ok expected boolean Series; violations are ~ok
            mask = ~ok.fillna(False)
            vio_count = int(mask.sum())
            out.append(_vio("logic", name, "expr", vio_count, [expr]))
        except Exception as e:
            out.append(_vio("logic", name, "expr", 0, [f"ERROR: {e}"]))
    return out


def _apply_business(df: pd.DataFrame, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Business KPI rules (minimal):
    rules:
      - name: "non_negative_revenue"
        column: "revenue"
        min: 0
    (Reuse range semantics; keeps it simple.)
    """
    # for now business rules reuse range handler
    # but keep a different rule_set name for reporting clarity
    out = []
    for r in (rules.get("rules") or []):
        r2 = dict(r)
        name = r2.get("name", "unnamed_business_rule")
        col = r2.get("column")
        if not col:
            out.append(_vio("business", name, "", 0, ["SKIPPED: missing column"]))
            continue
        # reuse range checks
        tmp = {"rules": [r2]}
        v = _apply_ranges(df, tmp)[0]
        v["rule_set"] = "business"
        out.append(v)
    return out


def _vio(rule_set: str, rule_name: str, column: str, violation_count: int, example_values: List[Any]) -> Dict[str, Any]:
    return {
        "rule_set": rule_set,
        "rule_name": rule_name,
        "column": column,
        "violation_count": int(violation_count),
        "example_values": example_values,
    }
