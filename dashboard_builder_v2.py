from __future__ import annotations

"""
============================================================
DASHBOARD BUILDER V2
============================================================

Purpose:
- Build presentation-ready dashboard tables
- Enforce fiscal month ordering
- Keep chart tables numeric and complete
- Avoid sparse/blank Streamlit charts
============================================================
"""

from typing import Any, Dict, Tuple
import pandas as pd


class DashboardBuilderError(Exception):
    pass


FISCAL_ORDER = [
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun"
]


# ------------------------------------------------------------
# BASIC HELPERS
# ------------------------------------------------------------
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
        raise DashboardBuilderError(f"{label} missing required columns: {missing}")


def _safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _sort_desc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df.copy().reset_index(drop=True)
    return df.sort_values(col, ascending=False, kind="stable").reset_index(drop=True)


# ------------------------------------------------------------
# EXECUTIVE TABLES
# ------------------------------------------------------------
def build_executive_summary_table(total_comparison_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        total_comparison_df,
        ["Opening Base", "Baseline Final", "Scenario Final", "Scenario Impact", "Scenario Impact %"],
        "total_comparison_df",
    )

    row = total_comparison_df.iloc[0]

    return pd.DataFrame([{
        "Opening Base": _to_float(row["Opening Base"], 0.0),
        "Baseline Final": _to_float(row["Baseline Final"], 0.0),
        "Scenario Final": _to_float(row["Scenario Final"], 0.0),
        "Scenario Impact": _to_float(row["Scenario Impact"], 0.0),
        "Scenario Impact %": _to_float(row["Scenario Impact %"], 0.0),
        "Structural Increment vs Baseline": _to_float(row.get("Structural Increment vs Baseline", 0.0), 0.0),
        "Policy Increment vs Baseline": _to_float(row.get("Policy Increment vs Baseline", 0.0), 0.0),
        "Scenario Macro Contribution": _to_float(row.get("Scenario Macro Contribution", 0.0), 0.0),
    }])


def build_contribution_summary_table(tax_head_comparison_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        tax_head_comparison_df,
        ["Scenario Impact", "Macro Contribution", "Policy Increment vs Baseline", "Structural Increment vs Baseline"],
        "tax_head_comparison_df",
    )

    impact = _safe_series(tax_head_comparison_df, "Scenario Impact").sum()
    macro = _safe_series(tax_head_comparison_df, "Macro Contribution").sum()
    policy = _safe_series(tax_head_comparison_df, "Policy Increment vs Baseline").sum()
    structural = _safe_series(tax_head_comparison_df, "Structural Increment vs Baseline").sum()

    return pd.DataFrame([{
        "Scenario Impact": impact,
        "Macro Contribution": macro,
        "Policy Increment vs Baseline": policy,
        "Structural Increment vs Baseline": structural,
        "Macro Share of Scenario Impact": (macro / impact) if impact != 0 else 0.0,
        "Policy Share of Scenario Impact": (policy / impact) if impact != 0 else 0.0,
    }])


def build_minister_brief_table(
    executive_summary_df: pd.DataFrame,
    contribution_summary_df: pd.DataFrame,
    top_gainers_df: pd.DataFrame,
    top_losers_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    _require_columns(executive_summary_df, ["Scenario Impact"], "executive_summary_df")
    _require_columns(contribution_summary_df, ["Macro Contribution", "Policy Increment vs Baseline"], "contribution_summary_df")
    _require_columns(top_gainers_df, ["Internal Tax Head", "Scenario Impact"], "top_gainers_df")
    _require_columns(top_losers_df, ["Internal Tax Head", "Scenario Impact"], "top_losers_df")

    gainers = ", ".join(top_gainers_df["Internal Tax Head"].astype(str).head(top_n).tolist())
    losers = ", ".join(top_losers_df["Internal Tax Head"].astype(str).head(top_n).tolist())

    return pd.DataFrame([{
        "Scenario Impact": _to_float(executive_summary_df["Scenario Impact"].iloc[0], 0.0),
        "Macro Contribution": _to_float(contribution_summary_df["Macro Contribution"].iloc[0], 0.0),
        "Policy Increment vs Baseline": _to_float(contribution_summary_df["Policy Increment vs Baseline"].iloc[0], 0.0),
        "Top Positive Movers": gainers,
        "Top Negative Movers": losers,
    }])


# ------------------------------------------------------------
# TOP MOVERS
# ------------------------------------------------------------
def build_top_gainers_and_losers(
    tax_head_comparison_df: pd.DataFrame,
    n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(tax_head_comparison_df, ["Internal Tax Head", "Scenario Impact"], "tax_head_comparison_df")

    work = tax_head_comparison_df.copy()
    work["Scenario Impact"] = pd.to_numeric(work["Scenario Impact"], errors="coerce").fillna(0.0)

    gainers = work.sort_values("Scenario Impact", ascending=False, kind="stable").head(n).reset_index(drop=True)
    losers = work.sort_values("Scenario Impact", ascending=True, kind="stable").head(n).reset_index(drop=True)

    return gainers, losers


# ------------------------------------------------------------
# MONTHLY COMPARISON TABLES
# ------------------------------------------------------------
def build_monthly_total_comparison(
    baseline_monthly_path_df: pd.DataFrame,
    scenario_monthly_path_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_monthly_path_df,
        ["Fiscal Year", "Month Index", "Month Name", "Baseline Monthly Forecast"],
        "baseline_monthly_path_df",
    )
    _require_columns(
        scenario_monthly_path_df,
        ["Fiscal Year", "Month Index", "Month Name", "Scenario Monthly Forecast"],
        "scenario_monthly_path_df",
    )

    base = (
        baseline_monthly_path_df.groupby(["Fiscal Year", "Month Index", "Month Name"], as_index=False)[
            ["Baseline Monthly Forecast"]
        ].sum()
    )

    scen = (
        scenario_monthly_path_df.groupby(["Fiscal Year", "Month Index", "Month Name"], as_index=False)[
            ["Scenario Monthly Forecast"]
        ].sum()
    )

    out = base.merge(
        scen,
        on=["Fiscal Year", "Month Index", "Month Name"],
        how="outer",
    )

    out["Baseline Monthly Forecast"] = pd.to_numeric(out["Baseline Monthly Forecast"], errors="coerce").fillna(0.0)
    out["Scenario Monthly Forecast"] = pd.to_numeric(out["Scenario Monthly Forecast"], errors="coerce").fillna(0.0)
    out["Monthly Impact"] = out["Scenario Monthly Forecast"] - out["Baseline Monthly Forecast"]
    out["Monthly Impact %"] = out.apply(
        lambda r: r["Monthly Impact"] / r["Baseline Monthly Forecast"]
        if _to_float(r["Baseline Monthly Forecast"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    out["Month Name"] = pd.Categorical(out["Month Name"], categories=FISCAL_ORDER, ordered=True)
    return out.sort_values(["Fiscal Year", "Month Name"]).reset_index(drop=True)


def build_monthly_tax_head_comparison(
    baseline_monthly_path_df: pd.DataFrame,
    scenario_monthly_path_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_monthly_path_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Baseline Monthly Forecast"],
        "baseline_monthly_path_df",
    )
    _require_columns(
        scenario_monthly_path_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Scenario Monthly Forecast"],
        "scenario_monthly_path_df",
    )

    out = baseline_monthly_path_df[
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Baseline Monthly Forecast"]
    ].merge(
        scenario_monthly_path_df[
            ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Scenario Monthly Forecast"]
        ],
        on=["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name"],
        how="outer",
    )

    out["Baseline Monthly Forecast"] = pd.to_numeric(out["Baseline Monthly Forecast"], errors="coerce").fillna(0.0)
    out["Scenario Monthly Forecast"] = pd.to_numeric(out["Scenario Monthly Forecast"], errors="coerce").fillna(0.0)
    out["Monthly Impact"] = out["Scenario Monthly Forecast"] - out["Baseline Monthly Forecast"]
    out["Monthly Impact %"] = out.apply(
        lambda r: r["Monthly Impact"] / r["Baseline Monthly Forecast"]
        if _to_float(r["Baseline Monthly Forecast"], 0.0) != 0.0 else 0.0,
        axis=1,
    )

    out["Month Name"] = pd.Categorical(out["Month Name"], categories=FISCAL_ORDER, ordered=True)
    return out.sort_values(["Internal Tax Head", "Month Name"]).reset_index(drop=True)


# ------------------------------------------------------------
# CHART TABLES
# ------------------------------------------------------------
def build_monthly_chart_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chart-ready table with full baseline/scenario trends and actual overlay.
    This avoids blank charts.
    """
    required = [
        "Month Index",
        "Month Name",
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
        "Actual Monthly",
    ]
    missing = [c for c in required if c not in monthly_df.columns]
    if missing:
        raise DashboardBuilderError(f"Monthly chart input missing columns: {missing}")

    df = (
        monthly_df.groupby(["Month Index", "Month Name"], as_index=False)
        .agg({
            "Baseline Monthly Forecast": "sum",
            "Scenario Monthly Forecast": "sum",
            "Actual Monthly": "sum",
        })
        .sort_values("Month Index")
        .reset_index(drop=True)
    )

    df["Month Name"] = pd.Categorical(df["Month Name"], categories=FISCAL_ORDER, ordered=True)
    df = df.sort_values("Month Name").reset_index(drop=True)

    out = pd.DataFrame(index=df["Month Name"].astype(str))
    out["Actual"] = pd.to_numeric(df["Actual Monthly"], errors="coerce").fillna(0.0)
    out["Baseline Forecast"] = pd.to_numeric(df["Baseline Monthly Forecast"], errors="coerce").fillna(0.0)
    out["Scenario Forecast"] = pd.to_numeric(df["Scenario Monthly Forecast"], errors="coerce").fillna(0.0)

    return out


def build_monthly_impact_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scenario minus baseline by month.
    """
    required = [
        "Month Index",
        "Month Name",
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
    ]
    missing = [c for c in required if c not in monthly_df.columns]
    if missing:
        raise DashboardBuilderError(f"Monthly impact input missing columns: {missing}")

    df = (
        monthly_df.groupby(["Month Index", "Month Name"], as_index=False)
        .agg({
            "Baseline Monthly Forecast": "sum",
            "Scenario Monthly Forecast": "sum",
        })
        .sort_values("Month Index")
        .reset_index(drop=True)
    )

    df["Monthly Impact"] = (
        pd.to_numeric(df["Scenario Monthly Forecast"], errors="coerce").fillna(0.0)
        - pd.to_numeric(df["Baseline Monthly Forecast"], errors="coerce").fillna(0.0)
    )

    df["Month Name"] = pd.Categorical(df["Month Name"], categories=FISCAL_ORDER, ordered=True)
    df = df.sort_values("Month Name").reset_index(drop=True)

    out = pd.DataFrame(index=df["Month Name"].astype(str))
    out["Monthly Impact"] = df["Monthly Impact"].fillna(0.0)

    return out


# ------------------------------------------------------------
# PRESENTATION TABLES
# ------------------------------------------------------------
def build_tax_head_summary_table(tax_head_comparison_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        tax_head_comparison_df,
        ["Internal Tax Head", "Annex Tax Head", "Baseline Final", "Scenario Final", "Scenario Impact", "Scenario Impact %"],
        "tax_head_comparison_df",
    )

    cols = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Baseline Final",
        "Scenario Final",
        "Scenario Impact",
        "Scenario Impact %",
        "Structural Increment vs Baseline",
        "Macro Contribution",
        "Policy Increment vs Baseline",
        "Macro Share of Scenario Impact",
        "Policy Share of Scenario Impact",
    ]
    existing = [c for c in cols if c in tax_head_comparison_df.columns]
    return _sort_desc(tax_head_comparison_df[existing].copy(), "Scenario Impact")


def build_annex_summary_table(annex_comparison_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        annex_comparison_df,
        ["Tax head", "Baseline", "Scenario", "Impact", "Impact %"],
        "annex_comparison_df",
    )
    return _sort_desc(annex_comparison_df.copy(), "Impact")


# ------------------------------------------------------------
# ACCURACY TABLES
# ------------------------------------------------------------
def build_accuracy_dashboard_tables(baseline_outputs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    if "forecast_accuracy_by_head" in baseline_outputs:
        tables["forecast_accuracy_by_head"] = baseline_outputs["forecast_accuracy_by_head"].copy()

    if "forecast_accuracy_summary" in baseline_outputs:
        tables["forecast_accuracy_summary"] = baseline_outputs["forecast_accuracy_summary"].copy()

    if "base_switch_table" in baseline_outputs:
        tables["base_switch_table"] = baseline_outputs["base_switch_table"].copy()

    return tables


# ------------------------------------------------------------
# HIGH-LEVEL PACKAGE BUILDER
# ------------------------------------------------------------
def build_dashboard_pack(scenario_package: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    if "baseline" not in scenario_package:
        raise DashboardBuilderError("scenario_package missing 'baseline'.")
    if "scenario" not in scenario_package:
        raise DashboardBuilderError("scenario_package missing 'scenario'.")
    if "comparisons" not in scenario_package:
        raise DashboardBuilderError("scenario_package missing 'comparisons'.")

    baseline = scenario_package["baseline"]
    scenario = scenario_package["scenario"]
    comparisons = scenario_package["comparisons"]

    total_comparison = comparisons["total_comparison"].copy()
    tax_head_comparison = comparisons["tax_head_comparison"].copy()
    annex_comparison = comparisons["annex_comparison"].copy()

    executive_summary = build_executive_summary_table(total_comparison)
    contribution_summary = build_contribution_summary_table(tax_head_comparison)

    top_gainers, top_losers = build_top_gainers_and_losers(
        tax_head_comparison_df=tax_head_comparison,
        n=10,
    )

    minister_brief = build_minister_brief_table(
        executive_summary_df=executive_summary,
        contribution_summary_df=contribution_summary,
        top_gainers_df=top_gainers,
        top_losers_df=top_losers,
        top_n=5,
    )

    baseline_monthly_path = baseline["monthly_outputs"]["baseline_monthly_path"].copy()
    scenario_monthly_path = scenario["scenario_monthly_path"].copy()

    monthly_total_comparison = build_monthly_total_comparison(
        baseline_monthly_path_df=baseline_monthly_path,
        scenario_monthly_path_df=scenario_monthly_path,
    )

    actual_monthly = (
        baseline_monthly_path.groupby(["Fiscal Year", "Month Index", "Month Name"], as_index=False)[["Actual Monthly"]]
        .sum()
    )

    monthly_total_comparison = monthly_total_comparison.merge(
        actual_monthly,
        on=["Fiscal Year", "Month Index", "Month Name"],
        how="left",
    )

    monthly_total_comparison["Actual Monthly"] = pd.to_numeric(
        monthly_total_comparison["Actual Monthly"], errors="coerce"
    ).fillna(0.0)

    monthly_tax_head_comparison = build_monthly_tax_head_comparison(
        baseline_monthly_path_df=baseline_monthly_path,
        scenario_monthly_path_df=scenario_monthly_path,
    )

    tax_head_summary = build_tax_head_summary_table(tax_head_comparison)
    annex_summary = build_annex_summary_table(annex_comparison)

    accuracy_tables = build_accuracy_dashboard_tables(baseline)

    out: Dict[str, pd.DataFrame] = {
        "executive_summary": executive_summary,
        "contribution_summary": contribution_summary,
        "minister_brief": minister_brief,
        "total_comparison": total_comparison,
        "tax_head_comparison": tax_head_comparison,
        "tax_head_summary": tax_head_summary,
        "annex_comparison": annex_comparison,
        "annex_summary": annex_summary,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "monthly_total_comparison": monthly_total_comparison,
        "monthly_tax_head_comparison": monthly_tax_head_comparison,
    }

    out.update(accuracy_tables)
    return out


# ------------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 90)
    print("DASHBOARD BUILDER V2 TEST")
    print("=" * 90)

    monthly_total = pd.DataFrame({
        "Fiscal Year": ["2025/26"] * 4,
        "Month Index": [1, 2, 9, 10],
        "Month Name": ["Jul", "Aug", "Mar", "Apr"],
        "Baseline Monthly Forecast": [100, 120, 130, 140],
        "Scenario Monthly Forecast": [100, 120, 110, 115],
        "Actual Monthly": [100, 120, 0, 0],
    })

    print(build_monthly_chart_table(monthly_total))
    print(build_monthly_impact_table(monthly_total))

    print("\nDASHBOARD BUILDER V2 TEST PASSED.")
