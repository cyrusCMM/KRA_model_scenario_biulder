# -*- coding: utf-8 -*-
"""
SCENARIO RUNNER V2
Fresh orchestration layer
Scope:
- run baseline rolling pipeline
- run shocked scenario pipeline
- compare baseline vs scenario
- return clean package for dashboard/export/app
"""

from typing import Any, Dict, Optional

import pandas as pd

from rolling_engine_v1 import run_rolling_engine
from simulation_engine_v1 import run_simulation_engine


class ScenarioRunnerError(Exception):
    """Raised when scenario orchestration fails."""
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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ScenarioRunnerError(f"{label} missing required columns: {missing}")


def _get_selected_year(data: Dict[str, Any]) -> str:
    selected_year = _clean(data.get("rolling_control", {}).get("selected_year", ""))
    if selected_year not in {"2025/26", "2026/27", "2027/28"}:
        raise ScenarioRunnerError(
            f"Invalid selected year '{selected_year}'. Expected one of: 2025/26, 2026/27, 2027/28."
        )
    return selected_year


def _get_projection_column(selected_year: str) -> str:
    return f"Projected Collection {selected_year}"


# ============================================================
# COMPARISON TABLES
# ============================================================

def build_total_comparison(
    baseline_total_df: pd.DataFrame,
    scenario_total_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(baseline_total_df, ["Final Forecast"], "baseline_total_df")
    _require_columns(scenario_total_df, ["Final Forecast"], "scenario_total_df")

    baseline_opening = _to_float(
        pd.to_numeric(baseline_total_df.get("Opening Base", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[0],
        0.0,
    )
    baseline_final = _to_float(
        pd.to_numeric(baseline_total_df["Final Forecast"], errors="coerce").fillna(0.0).iloc[0],
        0.0,
    )
    baseline_structural = _to_float(
        pd.to_numeric(baseline_total_df.get("Baseline Forecast", pd.Series([baseline_final])), errors="coerce")
        .fillna(baseline_final).iloc[0],
        baseline_final,
    )
    baseline_policy = _to_float(
        pd.to_numeric(baseline_total_df.get("Policy Adjustment", pd.Series([0.0])), errors="coerce")
        .fillna(0.0).iloc[0],
        0.0,
    )

    scenario_final = _to_float(
        pd.to_numeric(scenario_total_df["Final Forecast"], errors="coerce").fillna(0.0).iloc[0],
        0.0,
    )
    scenario_structural = _to_float(
        pd.to_numeric(scenario_total_df.get("Baseline Forecast", pd.Series([scenario_final])), errors="coerce")
        .fillna(scenario_final).iloc[0],
        scenario_final,
    )
    scenario_policy = _to_float(
        pd.to_numeric(scenario_total_df.get("Policy Adjustment", pd.Series([0.0])), errors="coerce")
        .fillna(0.0).iloc[0],
        0.0,
    )
    scenario_macro = _to_float(
        pd.to_numeric(scenario_total_df.get("Macro Contribution", pd.Series([0.0])), errors="coerce")
        .fillna(0.0).iloc[0],
        0.0,
    )

    impact = scenario_final - baseline_final
    impact_pct = impact / baseline_final if baseline_final != 0 else 0.0

    return pd.DataFrame([{
        "Opening Base": baseline_opening,
        "Baseline Final": baseline_final,
        "Scenario Final": scenario_final,
        "Scenario Impact": impact,
        "Scenario Impact %": impact_pct,
        "Baseline Structural": baseline_structural,
        "Scenario Structural": scenario_structural,
        "Structural Increment vs Baseline": scenario_structural - baseline_structural,
        "Baseline Policy": baseline_policy,
        "Scenario Policy": scenario_policy,
        "Policy Increment vs Baseline": scenario_policy - baseline_policy,
        "Scenario Macro Contribution": scenario_macro,
    }])


def build_tax_head_comparison(
    baseline_detail_df: pd.DataFrame,
    scenario_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_detail_df,
        ["Internal Tax Head", "Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "baseline_detail_df",
    )
    _require_columns(
        scenario_detail_df,
        ["Internal Tax Head", "Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "scenario_detail_df",
    )

    left = baseline_detail_df.copy()
    right = scenario_detail_df.copy()

    left["Internal Tax Head"] = left["Internal Tax Head"].map(_clean)
    right["Internal Tax Head"] = right["Internal Tax Head"].map(_clean)

    left = left.rename(columns={
        "Opening Base": "Baseline Opening Base",
        "Baseline Forecast": "Baseline Structural",
        "Policy Adjustment": "Baseline Policy",
        "Final Forecast": "Baseline Final",
    })

    right = right.rename(columns={
        "Opening Base": "Scenario Opening Base",
        "Baseline Forecast": "Scenario Structural",
        "Policy Adjustment": "Scenario Policy",
        "Final Forecast": "Scenario Final",
    })

    keep_left = ["Internal Tax Head", "Annex Tax Head", "Baseline Opening Base", "Baseline Structural", "Baseline Policy", "Baseline Final"]
    keep_right = ["Internal Tax Head", "Annex Tax Head", "Scenario Opening Base", "Scenario Structural", "Scenario Policy", "Scenario Final"]

    if "Macro Contribution" in right.columns:
        keep_right.append("Macro Contribution")

    out = left[keep_left].merge(
        right[keep_right],
        on=["Internal Tax Head", "Annex Tax Head"],
        how="outer",
    )

    numeric_cols = [c for c in out.columns if c not in {"Internal Tax Head", "Annex Tax Head"}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["Scenario Impact"] = out["Scenario Final"] - out["Baseline Final"]
    out["Scenario Impact %"] = out.apply(
        lambda r: r["Scenario Impact"] / r["Baseline Final"]
        if _to_float(r["Baseline Final"], 0.0) != 0.0 else 0.0,
        axis=1,
    )
    out["Structural Increment vs Baseline"] = out["Scenario Structural"] - out["Baseline Structural"]
    out["Policy Increment vs Baseline"] = out["Scenario Policy"] - out["Baseline Policy"]

    if "Macro Contribution" not in out.columns:
        out["Macro Contribution"] = out["Scenario Impact"] - out["Policy Increment vs Baseline"]

    out["Macro Share of Scenario Impact"] = out.apply(
        lambda r: r["Macro Contribution"] / r["Scenario Impact"]
        if _to_float(r["Scenario Impact"], 0.0) != 0.0 else 0.0,
        axis=1,
    )
    out["Policy Share of Scenario Impact"] = out.apply(
        lambda r: r["Policy Increment vs Baseline"] / r["Scenario Impact"]
        if _to_float(r["Scenario Impact"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    order_cols = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Baseline Opening Base",
        "Baseline Structural",
        "Baseline Policy",
        "Baseline Final",
        "Scenario Structural",
        "Scenario Policy",
        "Scenario Final",
        "Scenario Impact",
        "Scenario Impact %",
        "Structural Increment vs Baseline",
        "Macro Contribution",
        "Policy Increment vs Baseline",
        "Macro Share of Scenario Impact",
        "Policy Share of Scenario Impact",
    ]
    return out[order_cols].sort_values("Scenario Impact", ascending=False).reset_index(drop=True)


def build_annex_comparison(
    baseline_annex_df: pd.DataFrame,
    scenario_annex_df: pd.DataFrame,
    selected_year: str,
) -> pd.DataFrame:
    col = _get_projection_column(selected_year)

    _require_columns(baseline_annex_df, ["Tax head", col], "baseline_annex_df")
    _require_columns(scenario_annex_df, ["Tax head", col], "scenario_annex_df")

    left = baseline_annex_df[["Tax head", col]].copy().rename(columns={col: "Baseline"})
    right = scenario_annex_df[["Tax head", col]].copy().rename(columns={col: "Scenario"})

    out = left.merge(right, on="Tax head", how="outer")
    out["Baseline"] = pd.to_numeric(out["Baseline"], errors="coerce").fillna(0.0)
    out["Scenario"] = pd.to_numeric(out["Scenario"], errors="coerce").fillna(0.0)
    out["Impact"] = out["Scenario"] - out["Baseline"]
    out["Impact %"] = out.apply(
        lambda r: r["Impact"] / r["Baseline"] if _to_float(r["Baseline"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    return out.sort_values("Impact", ascending=False).reset_index(drop=True)


def build_top_movers(
    tax_head_comparison_df: pd.DataFrame,
    n: int = 10,
) -> Dict[str, pd.DataFrame]:
    _require_columns(tax_head_comparison_df, ["Internal Tax Head", "Scenario Impact"], "tax_head_comparison_df")

    work = tax_head_comparison_df.copy()
    work["Scenario Impact"] = pd.to_numeric(work["Scenario Impact"], errors="coerce").fillna(0.0)

    gainers = work.sort_values("Scenario Impact", ascending=False).head(n).reset_index(drop=True)
    losers = work.sort_values("Scenario Impact", ascending=True).head(n).reset_index(drop=True)

    return {
        "top_gainers": gainers,
        "top_losers": losers,
    }


# ============================================================
# MASTER RUNNERS
# ============================================================

def run_baseline_only(
    data: Dict[str, Any],
    baseline_macro_df: pd.DataFrame,
) -> Dict[str, Any]:
    baseline_outputs = run_rolling_engine(
        data=data,
        macro_df=baseline_macro_df,
    )

    return {
        "baseline": baseline_outputs,
    }


def run_baseline_and_scenario(
    data: Dict[str, Any],
    baseline_macro_df: pd.DataFrame,
    shocked_macro_df: pd.DataFrame,
    scenario_name: str,
    scenario_start_month: int,
    scenario_duration_months: int,
    carryover_to_next_fy: bool,
    recovery_profile: str,
    scenario_type: str,
    severity: str,
) -> Dict[str, Any]:
    selected_year = _get_selected_year(data)

    baseline_outputs = run_rolling_engine(
        data=data,
        macro_df=baseline_macro_df,
    )

    scenario_outputs = run_simulation_engine(
        data=data,
        baseline_outputs=baseline_outputs,
        shocked_macro_df=shocked_macro_df,
        scenario_start_month=scenario_start_month,
        scenario_duration_months=scenario_duration_months,
        carryover_to_next_fy=carryover_to_next_fy,
        recovery_profile=recovery_profile,
        scenario_type=scenario_type,
        severity=severity,
    )

    total_comparison = build_total_comparison(
        baseline_total_df=baseline_outputs["total_summary"],
        scenario_total_df=scenario_outputs["total_summary"],
    )

    tax_head_comparison = build_tax_head_comparison(
        baseline_detail_df=baseline_outputs["detail"],
        scenario_detail_df=scenario_outputs["detail"],
    )

    annex_comparison = build_annex_comparison(
        baseline_annex_df=baseline_outputs["annex_summary"],
        scenario_annex_df=scenario_outputs["annex_summary"],
        selected_year=selected_year,
    )

    movers = build_top_movers(
        tax_head_comparison_df=tax_head_comparison,
        n=10,
    )

    return {
        "metadata": {
            "selected_year": selected_year,
            "scenario_name": scenario_name,
            "scenario_start_month": scenario_start_month,
            "scenario_duration_months": scenario_duration_months,
            "carryover_to_next_fy": carryover_to_next_fy,
            "recovery_profile": recovery_profile,
            "scenario_type": scenario_type,
            "severity": severity,
        },
        "baseline": baseline_outputs,
        "scenario": scenario_outputs,
        "comparisons": {
            "total_comparison": total_comparison,
            "tax_head_comparison": tax_head_comparison,
            "annex_comparison": annex_comparison,
            **movers,
        },
    }