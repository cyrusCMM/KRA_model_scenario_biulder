# -*- coding: utf-8 -*-
"""
SIMULATION ENGINE V1
Scenario simulation layer for KRA forecasting model

Design
------
- receives shocked_macro_df from upstream
- runs annual shocked tax engine
- computes annual delta vs baseline
- allocates annual delta to months
- rebuilds annual scenario totals from monthly path
- preserves bridge heads (VAT, imports / Import Excise Duty)
- supports:
    * current FY open-year remaining-month allocation
    * future FY full-year allocation
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import pandas as pd

from tax_engine_v2 import run_tax_engine
from monthly_engine_v2 import (
    allocate_annual_delta_to_months,
    build_scenario_monthly_path,
    rebuild_annual_from_monthly_path,
)


class SimulationEngineError(Exception):
    """Raised when scenario simulation logic cannot be executed consistently."""
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if pd.notna(v) else default
    except Exception:
        return default


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise SimulationEngineError(f"{label} is missing or not a DataFrame.")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SimulationEngineError(f"{label} missing required columns: {missing}")


def _get_selected_year(data: Dict[str, Any]) -> str:
    selected_year = _clean(data.get("rolling_control", {}).get("selected_year", ""))
    if selected_year not in {"2025/26", "2026/27", "2027/28"}:
        raise SimulationEngineError(
            f"Invalid selected year '{selected_year}'. Expected one of: 2025/26, 2026/27, 2027/28."
        )
    return selected_year


def _get_year_status_2025_26(data: Dict[str, Any]) -> str:
    return _clean(data.get("rolling_control", {}).get("year_status_2025_26", "OPEN")).upper()


def _get_actual_months_loaded(data: Dict[str, Any]) -> int:
    v = data.get("rolling_control", {}).get("actual_months_loaded", 0)
    try:
        return int(v)
    except Exception:
        return 0


def _get_current_fiscal_year(data: Dict[str, Any]) -> str:
    return _clean(data.get("rolling_control", {}).get("current_fiscal_year", "2025/26")) or "2025/26"


def _get_scenario_allocation_mode(data: Dict[str, Any]) -> str:
    return _clean(
        data.get("rolling_control", {}).get("scenario_allocation_mode", "remaining_only")
    ).lower()


# ============================================================
# DETAIL-LEVEL COMPARISON
# ============================================================

def build_annual_scenario_delta(
    baseline_detail_df: pd.DataFrame,
    shocked_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_detail_df,
        ["Internal Tax Head", "Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "baseline_detail_df",
    )
    _require_columns(
        shocked_detail_df,
        ["Internal Tax Head", "Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "shocked_detail_df",
    )

    left = baseline_detail_df.copy()
    right = shocked_detail_df.copy()

    for df in (left, right):
        df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)
        df["Annex Tax Head"] = df["Annex Tax Head"].map(_clean)

    left = left.rename(columns={
        "Opening Base": "Baseline Opening Base",
        "Baseline Forecast": "Baseline Annual Baseline Forecast",
        "Policy Adjustment": "Baseline Policy Adjustment",
        "Final Forecast": "Baseline Final Forecast",
    })

    right = right.rename(columns={
        "Opening Base": "Shocked Opening Base",
        "Baseline Forecast": "Shocked Annual Baseline Forecast",
        "Policy Adjustment": "Shocked Policy Adjustment",
        "Final Forecast": "Shocked Final Forecast",
    })

    merged = left.merge(
        right,
        on=["Internal Tax Head", "Annex Tax Head"],
        how="outer",
        suffixes=("", "_r"),
    )

    numeric_cols = [
        "Baseline Opening Base",
        "Baseline Annual Baseline Forecast",
        "Baseline Policy Adjustment",
        "Baseline Final Forecast",
        "Shocked Opening Base",
        "Shocked Annual Baseline Forecast",
        "Shocked Policy Adjustment",
        "Shocked Final Forecast",
    ]
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    merged["Scenario Impact"] = (
        merged["Shocked Final Forecast"] - merged["Baseline Final Forecast"]
    )
    merged["Structural Increment vs Baseline"] = (
        merged["Shocked Annual Baseline Forecast"] - merged["Baseline Annual Baseline Forecast"]
    )
    merged["Policy Increment vs Baseline"] = (
        merged["Shocked Policy Adjustment"] - merged["Baseline Policy Adjustment"]
    )
    merged["Macro Increment vs Baseline"] = (
        merged["Scenario Impact"] - merged["Policy Increment vs Baseline"]
    )

    merged["Scenario Impact %"] = merged.apply(
        lambda r: r["Scenario Impact"] / r["Baseline Final Forecast"]
        if _to_float(r["Baseline Final Forecast"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    keep_cols = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Baseline Opening Base",
        "Baseline Annual Baseline Forecast",
        "Baseline Policy Adjustment",
        "Baseline Final Forecast",
        "Shocked Annual Baseline Forecast",
        "Shocked Policy Adjustment",
        "Shocked Final Forecast",
        "Scenario Impact",
        "Scenario Impact %",
        "Structural Increment vs Baseline",
        "Macro Increment vs Baseline",
        "Policy Increment vs Baseline",
    ]
    return merged[keep_cols].sort_values("Internal Tax Head").reset_index(drop=True)


# ============================================================
# MONTHLY ALLOCATION RULES
# ============================================================

def resolve_monthly_allocation_rule(
    data: Dict[str, Any],
    scenario_duration_months: Optional[int] = None,
) -> Dict[str, Any]:
    selected_year = _get_selected_year(data)
    current_fiscal_year = _get_current_fiscal_year(data)
    year_status_2025_26 = _get_year_status_2025_26(data)
    months_loaded = _get_actual_months_loaded(data)
    allocation_mode = _get_scenario_allocation_mode(data)

    if selected_year == current_fiscal_year and year_status_2025_26 == "OPEN":
        allocate_remaining_only = True
        effective_duration = scenario_duration_months
        effective_months_loaded = months_loaded
    else:
        allocate_remaining_only = False
        effective_duration = scenario_duration_months
        effective_months_loaded = 0

    if allocation_mode == "full_year":
        allocate_remaining_only = False

    return {
        "selected_year": selected_year,
        "current_fiscal_year": current_fiscal_year,
        "year_status_2025_26": year_status_2025_26,
        "months_loaded": effective_months_loaded,
        "allocate_remaining_only": allocate_remaining_only,
        "scenario_duration_months": effective_duration,
    }


# ============================================================
# SCENARIO DETAIL REBUILD
# ============================================================

def rebuild_scenario_detail_from_monthly(
    shocked_annual_detail_df: pd.DataFrame,
    scenario_annual_rebuild_df: pd.DataFrame,
    allocated_heads: Set[str],
) -> pd.DataFrame:
    """
    Rebuild scenario detail from shocked annual detail.

    Only heads that actually received monthly delta allocation AND are not bridge heads
    get their Final Forecast overwritten by the monthly rebuilt annual total.

    Bridge heads keep their shocked annual Final Forecast.
    """
    _require_columns(
        shocked_annual_detail_df,
        ["Internal Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "shocked_annual_detail_df",
    )
    _require_columns(
        scenario_annual_rebuild_df,
        ["Internal Tax Head", "Scenario Annual Rebuild"],
        "scenario_annual_rebuild_df",
    )

    detail = shocked_annual_detail_df.copy()
    annual = scenario_annual_rebuild_df.copy()

    detail["Internal Tax Head"] = detail["Internal Tax Head"].map(_clean)
    annual["Internal Tax Head"] = annual["Internal Tax Head"].map(_clean)
    annual["Scenario Annual Rebuild"] = pd.to_numeric(
        annual["Scenario Annual Rebuild"], errors="coerce"
    ).fillna(0.0)

    out = detail.merge(
        annual[["Internal Tax Head", "Scenario Annual Rebuild"]],
        on="Internal Tax Head",
        how="left",
    )

    allocated_heads_clean = {_clean(h) for h in allocated_heads}

    bridge_mask = (
        out.get("Logic Source", pd.Series([""] * len(out))).astype(str).str.strip().str.lower().eq("bridge")
        | out.get("Formula Type", pd.Series([""] * len(out))).astype(str).str.strip().str.lower().eq("bridge")
        | out["Internal Tax Head"].isin(["VAT, imports", "Import Excise Duty"])
    )

    original_final = pd.to_numeric(out["Final Forecast"], errors="coerce").fillna(0.0)
    rebuilt_final = pd.to_numeric(out["Scenario Annual Rebuild"], errors="coerce").fillna(original_final)

    use_rebuild = out["Internal Tax Head"].isin(allocated_heads_clean) & (~bridge_mask)

    out["Macro Contribution"] = 0.0
    out.loc[use_rebuild, "Macro Contribution"] = (
        rebuilt_final[use_rebuild] - original_final[use_rebuild]
    )
    out.loc[use_rebuild, "Final Forecast"] = rebuilt_final[use_rebuild]

    out = out.drop(columns=["Scenario Annual Rebuild"])
    return out.sort_values("Internal Tax Head").reset_index(drop=True)


# ============================================================
# SUMMARIES
# ============================================================

def build_annex_summary(detail_df: pd.DataFrame, selected_year: str) -> pd.DataFrame:
    _require_columns(
        detail_df,
        ["Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "detail_df",
    )

    agg_cols = ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"]
    if "Macro Contribution" in detail_df.columns:
        agg_cols.append("Macro Contribution")
    if "Target" in detail_df.columns:
        agg_cols.append("Target")

    out = (
        detail_df.groupby("Annex Tax Head", as_index=False)[agg_cols]
        .sum()
        .rename(columns={
            "Annex Tax Head": "Tax head",
            "Final Forecast": f"Projected Collection {selected_year}",
        })
    )
    return out


def build_department_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        detail_df,
        ["Department", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "detail_df",
    )

    agg_cols = ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"]
    if "Macro Contribution" in detail_df.columns:
        agg_cols.append("Macro Contribution")
    if "Target" in detail_df.columns:
        agg_cols.append("Target")

    out = detail_df.groupby("Department", as_index=False)[agg_cols].sum()
    return out


def build_total_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        detail_df,
        ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "detail_df",
    )

    opening = pd.to_numeric(detail_df["Opening Base"], errors="coerce").fillna(0.0).sum()
    baseline = pd.to_numeric(detail_df["Baseline Forecast"], errors="coerce").fillna(0.0).sum()
    policy = pd.to_numeric(detail_df["Policy Adjustment"], errors="coerce").fillna(0.0).sum()
    final = pd.to_numeric(detail_df["Final Forecast"], errors="coerce").fillna(0.0).sum()
    macro = (
        pd.to_numeric(detail_df["Macro Contribution"], errors="coerce").fillna(0.0).sum()
        if "Macro Contribution" in detail_df.columns else 0.0
    )

    return pd.DataFrame([{
        "Opening Base": opening,
        "Baseline Forecast": baseline,
        "Macro Contribution": macro,
        "Policy Adjustment": policy,
        "Final Forecast": final,
        "Forecast Growth": ((final - opening) / opening) if opening != 0 else 0.0,
    }])


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def run_simulation_engine(
    data: Dict[str, Any],
    baseline_outputs: Dict[str, Any],
    shocked_macro_df: pd.DataFrame,
    scenario_duration_months: Optional[int] = None,
) -> Dict[str, Any]:
    selected_year = _get_selected_year(data)

    if "detail" not in baseline_outputs:
        raise SimulationEngineError("baseline_outputs missing 'detail'.")
    if "monthly_outputs" not in baseline_outputs:
        raise SimulationEngineError(
            "baseline_outputs missing 'monthly_outputs'. Run rolling baseline with monthly layer first."
        )

    if shocked_macro_df is None or not isinstance(shocked_macro_df, pd.DataFrame) or shocked_macro_df.empty:
        raise SimulationEngineError("shocked_macro_df is missing or empty.")

    # 1. Annual shocked run
    shocked_annual_outputs = run_tax_engine(data=data, macro_df=shocked_macro_df)
    shocked_annual_detail = shocked_annual_outputs["detail"].copy()

    # 2. Annual delta vs baseline
    annual_delta = build_annual_scenario_delta(
        baseline_detail_df=baseline_outputs["detail"],
        shocked_detail_df=shocked_annual_detail,
    )

    # 3. Monthly allocation rule
    allocation = resolve_monthly_allocation_rule(
        data=data,
        scenario_duration_months=scenario_duration_months,
    )

    monthly_shares_df = baseline_outputs["monthly_outputs"]["monthly_shares"]
    baseline_monthly_path_df = baseline_outputs["monthly_outputs"]["baseline_monthly_path"]

    monthly_delta = allocate_annual_delta_to_months(
        annual_delta_df=annual_delta[["Internal Tax Head", "Scenario Impact"]],
        monthly_shares_df=monthly_shares_df,
        fiscal_year=selected_year,
        months_loaded=allocation["months_loaded"],
        allocate_remaining_only=allocation["allocate_remaining_only"],
        scenario_duration_months=allocation["scenario_duration_months"],
        annual_delta_col="Scenario Impact",
    )

    allocated_heads = set(monthly_delta["Internal Tax Head"].astype(str).str.strip().unique().tolist())

    # 4. Scenario monthly path
    scenario_monthly_path = build_scenario_monthly_path(
        baseline_monthly_path_df=baseline_monthly_path_df,
        monthly_delta_df=monthly_delta,
    )

    scenario_annual_rebuild = rebuild_annual_from_monthly_path(
        scenario_monthly_path,
        value_col="Scenario Monthly Forecast",
        annual_col_name="Scenario Annual Rebuild",
    )

    # 5. Rebuild scenario detail
    scenario_detail = rebuild_scenario_detail_from_monthly(
        shocked_annual_detail_df=shocked_annual_detail,
        scenario_annual_rebuild_df=scenario_annual_rebuild,
        allocated_heads=allocated_heads,
    )

    # 6. Summaries
    annex_summary = build_annex_summary(scenario_detail, selected_year)
    department_summary = build_department_summary(scenario_detail)
    total_summary = build_total_summary(scenario_detail)

    return {
        "annual_delta": annual_delta,
        "monthly_delta": monthly_delta,
        "scenario_monthly_path": scenario_monthly_path,
        "scenario_annual_rebuild": scenario_annual_rebuild,
        "detail": scenario_detail,
        "annex_summary": annex_summary,
        "department_summary": department_summary,
        "total_summary": total_summary,
        "allocation_metadata": allocation,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("SIMULATION ENGINE V1 REVISED LOADED")
    print("=" * 90)