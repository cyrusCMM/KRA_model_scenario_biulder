# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:27:52 2026

@author: hp
"""

from __future__ import annotations

# ============================================================
# DECOMPOSITION ENGINE V1
# Fresh decomposition layer
# Scope:
# - baseline decomposition
# - scenario vs baseline decomposition
# - total and tax-head summaries
# - optional monthly decomposition summaries
# ============================================================

from typing import Any, Dict

import pandas as pd


class DecompositionEngineError(Exception):
    """Raised when decomposition tables cannot be built consistently."""
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
        raise DecompositionEngineError(f"{label} missing required columns: {missing}")


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


# ============================================================
# BASELINE DECOMPOSITION
# ============================================================

def decompose_baseline_detail(
    baseline_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_detail_df,
        [
            "Internal Tax Head",
            "Annex Tax Head",
            "Opening Base",
            "Baseline Forecast",
            "Policy Adjustment",
            "Final Forecast",
        ],
        "baseline_detail_df",
    )

    df = baseline_detail_df.copy()
    df["Opening Base"] = pd.to_numeric(df["Opening Base"], errors="coerce").fillna(0.0)
    df["Baseline Forecast"] = pd.to_numeric(df["Baseline Forecast"], errors="coerce").fillna(0.0)
    df["Policy Adjustment"] = pd.to_numeric(df["Policy Adjustment"], errors="coerce").fillna(0.0)
    df["Final Forecast"] = pd.to_numeric(df["Final Forecast"], errors="coerce").fillna(0.0)

    df["Baseline Structural Change"] = df["Baseline Forecast"] - df["Opening Base"]
    df["Total Change"] = df["Final Forecast"] - df["Opening Base"]

    df["Baseline Growth Rate"] = df.apply(
        lambda r: (r["Baseline Forecast"] - r["Opening Base"]) / r["Opening Base"]
        if _to_float(r["Opening Base"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    df["Final Growth Rate"] = df.apply(
        lambda r: (r["Final Forecast"] - r["Opening Base"]) / r["Opening Base"]
        if _to_float(r["Opening Base"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    df["Policy Share of Total Change"] = df.apply(
        lambda r: r["Policy Adjustment"] / r["Total Change"]
        if _to_float(r["Total Change"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    cols = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Opening Base",
        "Baseline Forecast",
        "Baseline Structural Change",
        "Policy Adjustment",
        "Final Forecast",
        "Total Change",
        "Baseline Growth Rate",
        "Final Growth Rate",
        "Policy Share of Total Change",
    ]
    return df[cols].sort_values("Total Change", ascending=False).reset_index(drop=True)


def build_baseline_decomposition_summary(
    baseline_decomp_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_decomp_df,
        [
            "Opening Base",
            "Baseline Forecast",
            "Baseline Structural Change",
            "Policy Adjustment",
            "Final Forecast",
            "Total Change",
        ],
        "baseline_decomp_df",
    )

    opening = _safe_series(baseline_decomp_df, "Opening Base").sum()
    structural = _safe_series(baseline_decomp_df, "Baseline Structural Change").sum()
    policy = _safe_series(baseline_decomp_df, "Policy Adjustment").sum()
    final_total = _safe_series(baseline_decomp_df, "Final Forecast").sum()
    total_change = _safe_series(baseline_decomp_df, "Total Change").sum()
    baseline_forecast = _safe_series(baseline_decomp_df, "Baseline Forecast").sum()

    return pd.DataFrame([{
        "Opening Base": opening,
        "Baseline Forecast": baseline_forecast,
        "Baseline Structural Change": structural,
        "Policy Adjustment": policy,
        "Final Forecast": final_total,
        "Total Change": total_change,
        "Baseline Growth Rate": ((baseline_forecast - opening) / opening) if opening != 0 else 0.0,
        "Final Growth Rate": ((final_total - opening) / opening) if opening != 0 else 0.0,
        "Policy Share of Total Change": (policy / total_change) if total_change != 0 else 0.0,
    }])


# ============================================================
# SCENARIO DECOMPOSITION
# ============================================================

def decompose_scenario_against_baseline(
    tax_head_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        tax_head_comparison_df,
        [
            "Internal Tax Head",
            "Annex Tax Head",
            "Baseline Final",
            "Scenario Final",
            "Scenario Impact",
            "Structural Increment vs Baseline",
            "Macro Contribution",
            "Policy Increment vs Baseline",
        ],
        "tax_head_comparison_df",
    )

    df = tax_head_comparison_df.copy()

    numeric_cols = [
        "Baseline Final",
        "Scenario Final",
        "Scenario Impact",
        "Structural Increment vs Baseline",
        "Macro Contribution",
        "Policy Increment vs Baseline",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["Macro Share of Scenario Impact"] = df.apply(
        lambda r: r["Macro Contribution"] / r["Scenario Impact"]
        if _to_float(r["Scenario Impact"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    df["Policy Share of Scenario Impact"] = df.apply(
        lambda r: r["Policy Increment vs Baseline"] / r["Scenario Impact"]
        if _to_float(r["Scenario Impact"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    cols = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Baseline Final",
        "Scenario Final",
        "Scenario Impact",
        "Structural Increment vs Baseline",
        "Macro Contribution",
        "Policy Increment vs Baseline",
        "Macro Share of Scenario Impact",
        "Policy Share of Scenario Impact",
    ]
    return df[cols].sort_values("Scenario Impact", ascending=False).reset_index(drop=True)


def build_scenario_decomposition_summary(
    scenario_decomp_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        scenario_decomp_df,
        [
            "Baseline Final",
            "Scenario Final",
            "Scenario Impact",
            "Structural Increment vs Baseline",
            "Macro Contribution",
            "Policy Increment vs Baseline",
        ],
        "scenario_decomp_df",
    )

    baseline_total = _safe_series(scenario_decomp_df, "Baseline Final").sum()
    scenario_total = _safe_series(scenario_decomp_df, "Scenario Final").sum()
    impact = _safe_series(scenario_decomp_df, "Scenario Impact").sum()
    structural = _safe_series(scenario_decomp_df, "Structural Increment vs Baseline").sum()
    macro = _safe_series(scenario_decomp_df, "Macro Contribution").sum()
    policy = _safe_series(scenario_decomp_df, "Policy Increment vs Baseline").sum()

    return pd.DataFrame([{
        "Baseline Final": baseline_total,
        "Scenario Final": scenario_total,
        "Scenario Impact": impact,
        "Structural Increment vs Baseline": structural,
        "Macro Contribution": macro,
        "Policy Increment vs Baseline": policy,
        "Macro Share of Scenario Impact": (macro / impact) if impact != 0 else 0.0,
        "Policy Share of Scenario Impact": (policy / impact) if impact != 0 else 0.0,
    }])


# ============================================================
# ANNEX DECOMPOSITION
# ============================================================

def build_annex_decomposition(
    annex_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        annex_comparison_df,
        ["Tax head", "Baseline", "Scenario", "Impact", "Impact %"],
        "annex_comparison_df",
    )

    df = annex_comparison_df.copy()
    for c in ["Baseline", "Scenario", "Impact", "Impact %"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df.sort_values("Impact", ascending=False).reset_index(drop=True)


# ============================================================
# MONTHLY DECOMPOSITION
# ============================================================

def build_monthly_decomposition(
    monthly_total_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        monthly_total_comparison_df,
        [
            "Fiscal Year",
            "Month Index",
            "Month Name",
            "Baseline Monthly Forecast",
            "Scenario Monthly Forecast",
            "Monthly Impact",
            "Monthly Impact %",
        ],
        "monthly_total_comparison_df",
    )

    df = monthly_total_comparison_df.copy()

    for c in [
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
        "Monthly Impact",
        "Monthly Impact %",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df.sort_values(["Fiscal Year", "Month Index"]).reset_index(drop=True)


def build_monthly_tax_head_decomposition(
    monthly_tax_head_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        monthly_tax_head_comparison_df,
        [
            "Internal Tax Head",
            "Fiscal Year",
            "Month Index",
            "Month Name",
            "Baseline Monthly Forecast",
            "Scenario Monthly Forecast",
            "Monthly Impact",
            "Monthly Impact %",
        ],
        "monthly_tax_head_comparison_df",
    )

    df = monthly_tax_head_comparison_df.copy()

    for c in [
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
        "Monthly Impact",
        "Monthly Impact %",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)


def build_monthly_decomposition_summary(
    monthly_total_decomp_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        monthly_total_decomp_df,
        ["Baseline Monthly Forecast", "Scenario Monthly Forecast", "Monthly Impact"],
        "monthly_total_decomp_df",
    )

    baseline_total = _safe_series(monthly_total_decomp_df, "Baseline Monthly Forecast").sum()
    scenario_total = _safe_series(monthly_total_decomp_df, "Scenario Monthly Forecast").sum()
    impact_total = _safe_series(monthly_total_decomp_df, "Monthly Impact").sum()

    return pd.DataFrame([{
        "Baseline Monthly Total": baseline_total,
        "Scenario Monthly Total": scenario_total,
        "Monthly Impact Total": impact_total,
        "Monthly Impact %": (impact_total / baseline_total) if baseline_total != 0 else 0.0,
    }])


# ============================================================
# HIGH-LEVEL PACKAGE BUILDER
# ============================================================

def build_decomposition_pack(
    dashboard_pack: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Accepts output from dashboard_builder_v1.build_dashboard_pack()
    and produces decomposition tables.
    """
    required_keys = [
        "total_comparison",
        "tax_head_comparison",
        "annex_comparison",
        "monthly_total_comparison",
        "monthly_tax_head_comparison",
    ]
    missing = [k for k in required_keys if k not in dashboard_pack]
    if missing:
        raise DecompositionEngineError(
            f"dashboard_pack missing required keys: {missing}"
        )

    total_comparison = dashboard_pack["total_comparison"]
    tax_head_comparison = dashboard_pack["tax_head_comparison"]
    annex_comparison = dashboard_pack["annex_comparison"]
    monthly_total_comparison = dashboard_pack["monthly_total_comparison"]
    monthly_tax_head_comparison = dashboard_pack["monthly_tax_head_comparison"]

    scenario_decomp = decompose_scenario_against_baseline(tax_head_comparison)
    scenario_summary = build_scenario_decomposition_summary(scenario_decomp)

    annex_decomp = build_annex_decomposition(annex_comparison)

    monthly_total_decomp = build_monthly_decomposition(monthly_total_comparison)
    monthly_tax_head_decomp = build_monthly_tax_head_decomposition(monthly_tax_head_comparison)
    monthly_summary = build_monthly_decomposition_summary(monthly_total_decomp)

    return {
        "scenario_decomposition": scenario_decomp,
        "scenario_decomposition_summary": scenario_summary,
        "annex_decomposition": annex_decomp,
        "monthly_total_decomposition": monthly_total_decomp,
        "monthly_tax_head_decomposition": monthly_tax_head_decomp,
        "monthly_decomposition_summary": monthly_summary,
    }


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    print("=" * 90)
    print("DECOMPOSITION ENGINE V1 TEST")
    print("=" * 90)
    print("This test uses synthetic data only.")
    print("=" * 90)

    dashboard_pack = {
        "total_comparison": pd.DataFrame([{
            "Opening Base": 2200.0,
            "Baseline Final": 2410.0,
            "Scenario Final": 2350.0,
            "Scenario Impact": -60.0,
            "Scenario Impact %": -60.0 / 2410.0,
        }]),
        "tax_head_comparison": pd.DataFrame({
            "Internal Tax Head": ["PAYE", "Domestic VAT"],
            "Annex Tax Head": ["PAYE", "Domestic VAT"],
            "Baseline Final": [1090.0, 1320.0],
            "Scenario Final": [1060.0, 1290.0],
            "Scenario Impact": [-30.0, -30.0],
            "Structural Increment vs Baseline": [-20.0, -25.0],
            "Macro Contribution": [-30.0, -30.0],
            "Policy Increment vs Baseline": [0.0, 0.0],
        }),
        "annex_comparison": pd.DataFrame({
            "Tax head": ["PAYE", "Domestic VAT"],
            "Baseline": [1090.0, 1320.0],
            "Scenario": [1060.0, 1290.0],
            "Impact": [-30.0, -30.0],
            "Impact %": [-30 / 1090.0, -30 / 1320.0],
        }),
        "monthly_total_comparison": pd.DataFrame({
            "Fiscal Year": ["2025/26", "2025/26"],
            "Month Index": [9, 10],
            "Month Name": ["Mar", "Apr"],
            "Baseline Monthly Forecast": [200.0, 200.0],
            "Scenario Monthly Forecast": [190.0, 190.0],
            "Monthly Impact": [-10.0, -10.0],
            "Monthly Impact %": [-0.05, -0.05],
        }),
        "monthly_tax_head_comparison": pd.DataFrame({
            "Internal Tax Head": ["PAYE", "Domestic VAT"],
            "Fiscal Year": ["2025/26", "2025/26"],
            "Month Index": [9, 9],
            "Month Name": ["Mar", "Mar"],
            "Baseline Monthly Forecast": [90.0, 110.0],
            "Scenario Monthly Forecast": [85.0, 105.0],
            "Monthly Impact": [-5.0, -5.0],
            "Monthly Impact %": [-5 / 90.0, -5 / 110.0],
        }),
    }

    pack = build_decomposition_pack(dashboard_pack)

    print("\n[1] SCENARIO DECOMPOSITION")
    print(pack["scenario_decomposition"])

    print("\n[2] SCENARIO DECOMPOSITION SUMMARY")
    print(pack["scenario_decomposition_summary"])

    print("\n[3] MONTHLY TOTAL DECOMPOSITION")
    print(pack["monthly_total_decomposition"])

    print("\n[4] MONTHLY DECOMPOSITION SUMMARY")
    print(pack["monthly_decomposition_summary"])

    print("\nDECOMPOSITION ENGINE V1 TEST PASSED.")