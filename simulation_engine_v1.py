# -*- coding: utf-8 -*-
"""
SIMULATION ENGINE V6
Scenario simulation layer for KRA forecasting model.

Design
------
- annual tax engine remains annual
- annual scenario delta is computed first
- current-FY effect is scaled to the affected months only
- monthly delta is allocated from scenario start month and duration
- annual scenario totals are rebuilt from monthly paths
- all allocated heads are rebuilt consistently from monthly results
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
    pass


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
        raise SimulationEngineError(f"Invalid selected year '{selected_year}'.")
    return selected_year


def _next_fiscal_year(selected_year: str) -> Optional[str]:
    order = ["2025/26", "2026/27", "2027/28"]
    if selected_year not in order:
        return None
    idx = order.index(selected_year)
    return order[idx + 1] if idx + 1 < len(order) else None


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
    )

    numeric_cols = [c for c in merged.columns if c not in {"Internal Tax Head", "Annex Tax Head"}]
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    merged["Scenario Impact"] = merged["Shocked Final Forecast"] - merged["Baseline Final Forecast"]
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

    return merged[[
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
    ]].sort_values("Internal Tax Head").reset_index(drop=True)


def _duration_split(
    scenario_start_month: int,
    scenario_duration_months: Optional[int],
    carryover_to_next_fy: bool,
) -> Dict[str, int]:
    start = int(scenario_start_month)
    duration = int(scenario_duration_months or 0)

    if duration <= 0:
        return {
            "start_month": start,
            "duration_total": 0,
            "months_remaining_current_fy": max(0, 12 - start + 1),
            "months_current_fy": 0,
            "months_next_fy": 0,
        }

    months_remaining_current_fy = max(0, 12 - start + 1)
    months_current_fy = min(duration, months_remaining_current_fy)
    months_next_fy = max(0, duration - months_current_fy) if carryover_to_next_fy else 0

    return {
        "start_month": start,
        "duration_total": duration,
        "months_remaining_current_fy": months_remaining_current_fy,
        "months_current_fy": months_current_fy,
        "months_next_fy": months_next_fy,
    }


def _scale_delta_for_partial_current_fy(
    annual_delta_df: pd.DataFrame,
    monthly_shares_df: pd.DataFrame,
    scenario_start_month: int,
    months_current_fy: int,
) -> pd.DataFrame:
    _require_columns(annual_delta_df, ["Internal Tax Head", "Scenario Impact"], "annual_delta_df")
    _require_columns(monthly_shares_df, ["Internal Tax Head", "Month Index", "Monthly Share"], "monthly_shares_df")

    shares = monthly_shares_df.copy()
    shares["Internal Tax Head"] = shares["Internal Tax Head"].map(_clean)
    shares["Month Index"] = pd.to_numeric(shares["Month Index"], errors="coerce").fillna(0).astype(int)
    shares["Monthly Share"] = pd.to_numeric(shares["Monthly Share"], errors="coerce").fillna(0.0)

    if months_current_fy <= 0:
        out = annual_delta_df.copy()
        out["Scenario Impact"] = 0.0
        return out

    eligible_end_month = scenario_start_month + max(months_current_fy - 1, 0)

    eligible_share = (
        shares.loc[
            (shares["Month Index"] >= int(scenario_start_month)) &
            (shares["Month Index"] <= int(eligible_end_month))
        ]
        .groupby("Internal Tax Head", as_index=False)["Monthly Share"]
        .sum()
        .rename(columns={"Monthly Share": "Eligible Year Share"})
    )

    out = annual_delta_df.copy()
    out["Internal Tax Head"] = out["Internal Tax Head"].map(_clean)
    out = out.merge(eligible_share, on="Internal Tax Head", how="left")
    out["Eligible Year Share"] = out["Eligible Year Share"].fillna(0.0)
    out["Scenario Impact"] = pd.to_numeric(out["Scenario Impact"], errors="coerce").fillna(0.0) * out["Eligible Year Share"]

    return out.drop(columns=["Eligible Year Share"])


def rebuild_scenario_detail_from_monthly(
    baseline_detail_df: pd.DataFrame,
    shocked_annual_detail_df: pd.DataFrame,
    scenario_annual_rebuild_df: pd.DataFrame,
    annual_delta_df: pd.DataFrame,
    allocated_heads: Set[str],
) -> pd.DataFrame:
    _require_columns(
        baseline_detail_df,
        ["Internal Tax Head", "Final Forecast"],
        "baseline_detail_df",
    )
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
    _require_columns(
        annual_delta_df,
        ["Internal Tax Head", "Macro Increment vs Baseline", "Policy Increment vs Baseline"],
        "annual_delta_df",
    )

    base = baseline_detail_df.copy()
    detail = shocked_annual_detail_df.copy()
    annual = scenario_annual_rebuild_df.copy()
    delta = annual_delta_df.copy()

    for df in (base, detail, annual, delta):
        df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)

    base = base[["Internal Tax Head", "Final Forecast"]].rename(columns={"Final Forecast": "Baseline Final Forecast"})
    annual["Scenario Annual Rebuild"] = pd.to_numeric(annual["Scenario Annual Rebuild"], errors="coerce").fillna(0.0)
    delta["Macro Increment vs Baseline"] = pd.to_numeric(delta["Macro Increment vs Baseline"], errors="coerce").fillna(0.0)
    delta["Policy Increment vs Baseline"] = pd.to_numeric(delta["Policy Increment vs Baseline"], errors="coerce").fillna(0.0)

    out = detail.merge(base, on="Internal Tax Head", how="left")
    out = out.merge(annual[["Internal Tax Head", "Scenario Annual Rebuild"]], on="Internal Tax Head", how="left")
    out = out.merge(
        delta[["Internal Tax Head", "Macro Increment vs Baseline", "Policy Increment vs Baseline"]],
        on="Internal Tax Head",
        how="left",
    )

    allocated_heads_clean = {_clean(h) for h in allocated_heads}

    original_final = pd.to_numeric(out["Final Forecast"], errors="coerce").fillna(0.0)
    rebuilt_final = pd.to_numeric(out["Scenario Annual Rebuild"], errors="coerce").fillna(original_final)

    # Critical fix: rebuild every allocated head from the monthly rebuild
    use_rebuild = out["Internal Tax Head"].isin(allocated_heads_clean)
    out.loc[use_rebuild, "Final Forecast"] = rebuilt_final[use_rebuild]

    out["Baseline Final Forecast"] = pd.to_numeric(out["Baseline Final Forecast"], errors="coerce").fillna(0.0)
    out["Policy Increment vs Baseline"] = pd.to_numeric(out["Policy Increment vs Baseline"], errors="coerce").fillna(0.0)

    out["Macro Contribution"] = (
        pd.to_numeric(out["Final Forecast"], errors="coerce").fillna(0.0)
        - out["Baseline Final Forecast"]
        - out["Policy Increment vs Baseline"]
    )

    out = out.drop(columns=["Scenario Annual Rebuild", "Baseline Final Forecast"])
    return out.sort_values("Internal Tax Head").reset_index(drop=True)


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

    return (
        detail_df.groupby("Annex Tax Head", as_index=False)[agg_cols]
        .sum()
        .rename(columns={
            "Annex Tax Head": "Tax head",
            "Final Forecast": f"Projected Collection {selected_year}",
        })
    )


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

    return detail_df.groupby("Department", as_index=False)[agg_cols].sum()


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


def run_simulation_engine(
    data: Dict[str, Any],
    baseline_outputs: Dict[str, Any],
    shocked_macro_df: pd.DataFrame,
    scenario_start_month: int = 10,
    scenario_duration_months: Optional[int] = None,
    carryover_to_next_fy: bool = False,
    recovery_profile: str = "",
    scenario_type: str = "",
    severity: str = "",
) -> Dict[str, Any]:
    selected_year = _get_selected_year(data)
    next_year = _next_fiscal_year(selected_year)

    if "detail" not in baseline_outputs:
        raise SimulationEngineError("baseline_outputs missing 'detail'.")
    if "monthly_outputs" not in baseline_outputs:
        raise SimulationEngineError("baseline_outputs missing 'monthly_outputs'.")
    if shocked_macro_df is None or not isinstance(shocked_macro_df, pd.DataFrame) or shocked_macro_df.empty:
        raise SimulationEngineError("shocked_macro_df is missing or empty.")

    shocked_annual_outputs = run_tax_engine(data=data, macro_df=shocked_macro_df)
    shocked_annual_detail = shocked_annual_outputs["detail"].copy()

    annual_delta_full = build_annual_scenario_delta(
        baseline_detail_df=baseline_outputs["detail"],
        shocked_detail_df=shocked_annual_detail,
    )

    monthly_shares_df = baseline_outputs["monthly_outputs"]["monthly_shares"]
    baseline_monthly_path_df = baseline_outputs["monthly_outputs"]["baseline_monthly_path"]

    split = _duration_split(
        scenario_start_month=scenario_start_month,
        scenario_duration_months=scenario_duration_months,
        carryover_to_next_fy=carryover_to_next_fy,
    )

    annual_delta_current = _scale_delta_for_partial_current_fy(
        annual_delta_df=annual_delta_full[["Internal Tax Head", "Scenario Impact"]],
        monthly_shares_df=monthly_shares_df,
        scenario_start_month=split["start_month"],
        months_current_fy=split["months_current_fy"],
    )

    monthly_delta = allocate_annual_delta_to_months(
        annual_delta_df=annual_delta_current,
        monthly_shares_df=monthly_shares_df,
        fiscal_year=selected_year,
        months_loaded=0,
        allocate_remaining_only=False,
        scenario_duration_months=split["months_current_fy"],
        annual_delta_col="Scenario Impact",
        scenario_start_month=split["start_month"],
    )

    allocated_heads = set(monthly_delta["Internal Tax Head"].astype(str).str.strip().unique().tolist())

    scenario_monthly_path = build_scenario_monthly_path(
        baseline_monthly_path_df=baseline_monthly_path_df,
        monthly_delta_df=monthly_delta,
    )

    scenario_annual_rebuild = rebuild_annual_from_monthly_path(
        scenario_monthly_path,
        value_col="Scenario Monthly Forecast",
        annual_col_name="Scenario Annual Rebuild",
    )

    annual_delta_effective_full = annual_delta_full.copy()
    scenario_rebuild_by_head = scenario_annual_rebuild.copy()
    scenario_rebuild_by_head["Internal Tax Head"] = scenario_rebuild_by_head["Internal Tax Head"].map(_clean)

    if "Fiscal Year" in scenario_rebuild_by_head.columns:
        scenario_rebuild_by_head = scenario_rebuild_by_head.loc[
            scenario_rebuild_by_head["Fiscal Year"].map(_clean) == selected_year
        ].copy()

    scenario_rebuild_by_head = scenario_rebuild_by_head[["Internal Tax Head", "Scenario Annual Rebuild"]].copy()

    annual_delta_effective_full = annual_delta_effective_full.merge(
        scenario_rebuild_by_head,
        on="Internal Tax Head",
        how="left",
    )

    annual_delta_effective_full["Scenario Annual Rebuild"] = pd.to_numeric(
        annual_delta_effective_full["Scenario Annual Rebuild"], errors="coerce"
    ).fillna(
        pd.to_numeric(annual_delta_effective_full["Baseline Final Forecast"], errors="coerce").fillna(0.0)
    )

    annual_delta_effective_full["Scenario Impact"] = (
        annual_delta_effective_full["Scenario Annual Rebuild"]
        - pd.to_numeric(annual_delta_effective_full["Baseline Final Forecast"], errors="coerce").fillna(0.0)
    )
    annual_delta_effective_full["Macro Increment vs Baseline"] = (
        annual_delta_effective_full["Scenario Impact"]
        - pd.to_numeric(annual_delta_effective_full["Policy Increment vs Baseline"], errors="coerce").fillna(0.0)
    )

    scenario_detail = rebuild_scenario_detail_from_monthly(
        baseline_detail_df=baseline_outputs["detail"],
        shocked_annual_detail_df=shocked_annual_detail,
        scenario_annual_rebuild_df=scenario_annual_rebuild,
        annual_delta_df=annual_delta_effective_full,
        allocated_heads=allocated_heads,
    )

    annex_summary = build_annex_summary(scenario_detail, selected_year)
    department_summary = build_department_summary(scenario_detail)
    total_summary = build_total_summary(scenario_detail)

    return {
        "annual_delta": annual_delta_effective_full,
        "monthly_delta": monthly_delta,
        "scenario_monthly_path": scenario_monthly_path,
        "scenario_annual_rebuild": scenario_annual_rebuild,
        "detail": scenario_detail,
        "annex_summary": annex_summary,
        "department_summary": department_summary,
        "total_summary": total_summary,
        "allocation_metadata": {
            "selected_year": selected_year,
            "next_year": next_year,
            "scenario_start_month": scenario_start_month,
            "scenario_duration_months": scenario_duration_months,
            "carryover_to_next_fy": carryover_to_next_fy,
            "recovery_profile": recovery_profile,
            "scenario_type": scenario_type,
            "severity": severity,
            "months_current_fy": split["months_current_fy"],
            "months_next_fy": split["months_next_fy"],
            "allocation_mode": "current_year_partial_effect_reconciled",
        },
    }