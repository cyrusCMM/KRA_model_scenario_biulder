from __future__ import annotations

"""
============================================================
DASHBOARD BUILDER V1
============================================================

Purpose:
- Build presentation-ready dashboard tables from scenario outputs
- Keep chart logic out of Streamlit
- Enforce fiscal month ordering
- Separate actual months from forecast months
- Support baseline vs scenario analysis cleanly
============================================================
"""

from typing import Any, Dict, Tuple
import pandas as pd


class DashboardBuilderError(Exception):
    """Raised when dashboard tables cannot be built consistently."""
    pass


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
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


def _find_projection_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if _clean(c).lower().startswith("projected collection"):
            return c
    raise DashboardBuilderError("Could not find projected collection column.")


# ------------------------------------------------------------
# EXECUTIVE TABLES
# ------------------------------------------------------------
def build_executive_summary_table(
    total_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        total_comparison_df,
        [
            "Opening Base",
            "Baseline Final",
            "Scenario Final",
            "Scenario Impact",
            "Scenario Impact %",
        ],
        "total_comparison_df",
    )

    row = total_comparison_df.iloc[0]

    out = pd.DataFrame([{
        "Opening Base": _to_float(row["Opening Base"], 0.0),
        "Baseline Final": _to_float(row["Baseline Final"], 0.0),
        "Scenario Final": _to_float(row["Scenario Final"], 0.0),
        "Scenario Impact": _to_float(row["Scenario Impact"], 0.0),
        "Scenario Impact %": _to_float(row["Scenario Impact %"], 0.0),
        "Structural Increment vs Baseline": _to_float(row.get("Structural Increment vs Baseline", 0.0), 0.0),
        "Policy Increment vs Baseline": _to_float(row.get("Policy Increment vs Baseline", 0.0), 0.0),
        "Scenario Macro Contribution": _to_float(row.get("Scenario Macro Contribution", 0.0), 0.0),
    }])

    return out


def build_contribution_summary_table(
    tax_head_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        tax_head_comparison_df,
        [
            "Scenario Impact",
            "Macro Contribution",
            "Policy Increment vs Baseline",
            "Structural Increment vs Baseline",
        ],
        "tax_head_comparison_df",
    )

    impact = _safe_series(tax_head_comparison_df, "Scenario Impact").sum()
    macro = _safe_series(tax_head_comparison_df, "Macro Contribution").sum()
    policy = _safe_series(tax_head_comparison_df, "Policy Increment vs Baseline").sum()
    structural = _safe_series(tax_head_comparison_df, "Structural Increment vs Baseline").sum()

    out = pd.DataFrame([{
        "Scenario Impact": impact,
        "Macro Contribution": macro,
        "Policy Increment vs Baseline": policy,
        "Structural Increment vs Baseline": structural,
        "Macro Share of Scenario Impact": (macro / impact) if impact != 0 else 0.0,
        "Policy Share of Scenario Impact": (policy / impact) if impact != 0 else 0.0,
    }])

    return out


def build_minister_brief_table(
    executive_summary_df: pd.DataFrame,
    contribution_summary_df: pd.DataFrame,
    top_gainers_df: pd.DataFrame,
    top_losers_df: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    _require_columns(executive_summary_df, ["Scenario Impact"], "executive_summary_df")
    _require_columns(
        contribution_summary_df,
        ["Macro Contribution", "Policy Increment vs Baseline"],
        "contribution_summary_df",
    )
    _require_columns(top_gainers_df, ["Internal Tax Head", "Scenario Impact"], "top_gainers_df")
    _require_columns(top_losers_df, ["Internal Tax Head", "Scenario Impact"], "top_losers_df")

    gainers = ", ".join(top_gainers_df["Internal Tax Head"].astype(str).head(top_n).tolist())
    losers = ", ".join(top_losers_df["Internal Tax Head"].astype(str).head(top_n).tolist())

    out = pd.DataFrame([{
        "Scenario Impact": _to_float(executive_summary_df["Scenario Impact"].iloc[0], 0.0),
        "Macro Contribution": _to_float(contribution_summary_df["Macro Contribution"].iloc[0], 0.0),
        "Policy Increment vs Baseline": _to_float(contribution_summary_df["Policy Increment vs Baseline"].iloc[0], 0.0),
        "Top Positive Movers": gainers,
        "Top Negative Movers": losers,
    }])

    return out


# ------------------------------------------------------------
# TOP MOVERS
# ------------------------------------------------------------
def build_top_gainers_and_losers(
    tax_head_comparison_df: pd.DataFrame,
    n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        tax_head_comparison_df,
        ["Internal Tax Head", "Scenario Impact"],
        "tax_head_comparison_df",
    )

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
        baseline_monthly_path_df.groupby(
            ["Fiscal Year", "Month Index", "Month Name"], as_index=False
        )[["Baseline Monthly Forecast"]]
        .sum()
    )

    scen = (
        scenario_monthly_path_df.groupby(
            ["Fiscal Year", "Month Index", "Month Name"], as_index=False
        )[["Scenario Monthly Forecast"]]
        .sum()
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

    out["Month Name"] = pd.Categorical(
        out["Month Name"],
        categories=FISCAL_ORDER,
        ordered=True,
    )

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

    out["Month Name"] = pd.Categorical(
        out["Month Name"],
        categories=FISCAL_ORDER,
        ordered=True,
    )

    return out.sort_values(["Internal Tax Head", "Month Name"]).reset_index(drop=True)


# ------------------------------------------------------------
# CHART TABLES
# ------------------------------------------------------------
def build_monthly_chart_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds chart-ready monthly table:
    - Actuals only for loaded months
    - Forecasts only for remaining months
    - Correct fiscal ordering
    """
    required = [
        "Month Index",
        "Month Name",
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
        "Actual Monthly",
        "Is Actual",
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
            "Is Actual": "max",
        })
        .sort_values("Month Index")
        .reset_index(drop=True)
    )

    df["Month Name"] = pd.Categorical(
        df["Month Name"],
        categories=FISCAL_ORDER,
        ordered=True,
    )

    df = df.sort_values("Month Name").reset_index(drop=True)

    out = pd.DataFrame(index=df["Month Name"])
    out["Actual"] = df["Actual Monthly"].where(df["Is Actual"] == True)
    out["Baseline Forecast"] = df["Baseline Monthly Forecast"].where(df["Is Actual"] == False)
    out["Scenario Forecast"] = df["Scenario Monthly Forecast"].where(df["Is Actual"] == False)

    return out


def build_monthly_impact_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds scenario-minus-baseline impact table for forecast months only.
    """
    required = [
        "Month Index",
        "Month Name",
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
        "Is Actual",
    ]
    missing = [c for c in required if c not in monthly_df.columns]
    if missing:
        raise DashboardBuilderError(f"Monthly impact input missing columns: {missing}")

    df = (
        monthly_df.groupby(["Month Index", "Month Name"], as_index=False)
        .agg({
            "Baseline Monthly Forecast": "sum",
            "Scenario Monthly Forecast": "sum",
            "Is Actual": "max",
        })
        .sort_values("Month Index")
        .reset_index(drop=True)
    )

    df["Impact"] = df["Scenario Monthly Forecast"] - df["Baseline Monthly Forecast"]

    df["Month Name"] = pd.Categorical(
        df["Month Name"],
        categories=FISCAL_ORDER,
        ordered=True,
    )

    df = df.sort_values("Month Name").reset_index(drop=True)

    out = pd.DataFrame(index=df["Month Name"])
    out["Monthly Impact"] = df["Impact"].where(df["Is Actual"] == False)

    return out


# ------------------------------------------------------------
# PRESENTATION TABLES
# ------------------------------------------------------------
def build_tax_head_summary_table(
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
            "Scenario Impact %",
        ],
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


def build_annex_summary_table(
    annex_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        annex_comparison_df,
        ["Tax head", "Baseline", "Scenario", "Impact", "Impact %"],
        "annex_comparison_df",
    )

    return _sort_desc(annex_comparison_df.copy(), "Impact")


# ------------------------------------------------------------
# ACCURACY TABLES
# ------------------------------------------------------------
def build_accuracy_dashboard_tables(
    baseline_outputs: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
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
def build_dashboard_pack(
    scenario_package: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Accepts output from scenario_runner_v1.run_baseline_and_scenario()
    and produces dashboard-ready tables.
    """
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

    # Build Is Actual if not already present
    if "Is Actual" not in baseline_monthly_path.columns:
        if "Source" in baseline_monthly_path.columns:
            baseline_monthly_path["Is Actual"] = baseline_monthly_path["Source"].astype(str).str.lower().eq("actual")
        else:
            baseline_monthly_path["Is Actual"] = False

    if "Actual Monthly" not in baseline_monthly_path.columns:
        if "Actual Monthly" not in baseline_monthly_path.columns:
            baseline_monthly_path["Actual Monthly"] = 0.0

    monthly_total_comparison = build_monthly_total_comparison(
        baseline_monthly_path_df=baseline_monthly_path,
        scenario_monthly_path_df=scenario_monthly_path,
    )

    # Add actual support columns for charting
    actual_monthly = (
        baseline_monthly_path.groupby(
            ["Fiscal Year", "Month Index", "Month Name"], as_index=False
        )[["Actual Monthly"]]
        .sum()
    )
    is_actual = (
        baseline_monthly_path.groupby(
            ["Fiscal Year", "Month Index", "Month Name"], as_index=False
        )[["Is Actual"]]
        .max()
    )

    monthly_total_comparison = monthly_total_comparison.merge(
        actual_monthly,
        on=["Fiscal Year", "Month Index", "Month Name"],
        how="left",
    ).merge(
        is_actual,
        on=["Fiscal Year", "Month Index", "Month Name"],
        how="left",
    )

    monthly_total_comparison["Actual Monthly"] = pd.to_numeric(
        monthly_total_comparison["Actual Monthly"], errors="coerce"
    ).fillna(0.0)
    monthly_total_comparison["Is Actual"] = monthly_total_comparison["Is Actual"].fillna(False).astype(bool)

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
    print("DASHBOARD BUILDER V1 TEST")
    print("=" * 90)

    total_comparison = pd.DataFrame([{
        "Opening Base": 2200.0,
        "Baseline Final": 2410.0,
        "Scenario Final": 2350.0,
        "Scenario Impact": -60.0,
        "Scenario Impact %": -60.0 / 2410.0,
        "Structural Increment vs Baseline": -45.0,
        "Policy Increment vs Baseline": 0.0,
        "Scenario Macro Contribution": -60.0,
    }])

    tax_head_comparison = pd.DataFrame({
        "Internal Tax Head": ["PAYE", "Domestic VAT"],
        "Annex Tax Head": ["PAYE", "Domestic VAT"],
        "Baseline Final": [1090.0, 1320.0],
        "Scenario Final": [1060.0, 1290.0],
        "Scenario Impact": [-30.0, -30.0],
        "Scenario Impact %": [-30 / 1090.0, -30 / 1320.0],
        "Structural Increment vs Baseline": [-20.0, -25.0],
        "Macro Contribution": [-30.0, -30.0],
        "Policy Increment vs Baseline": [0.0, 0.0],
        "Macro Share of Scenario Impact": [1.0, 1.0],
        "Policy Share of Scenario Impact": [0.0, 0.0],
    })

    annex_comparison = pd.DataFrame({
        "Tax head": ["PAYE", "Domestic VAT"],
        "Baseline": [1090.0, 1320.0],
        "Scenario": [1060.0, 1290.0],
        "Impact": [-30.0, -30.0],
        "Impact %": [-30 / 1090.0, -30 / 1320.0],
    })

    baseline_monthly = pd.DataFrame({
        "Internal Tax Head": ["PAYE", "PAYE", "Domestic VAT", "Domestic VAT"],
        "Fiscal Year": ["2025/26"] * 4,
        "Month Index": [9, 10, 9, 10],
        "Month Name": ["Mar", "Apr", "Mar", "Apr"],
        "Baseline Monthly Forecast": [90.0, 90.0, 110.0, 110.0],
        "Actual Monthly": [0.0, 0.0, 0.0, 0.0],
        "Source": ["forecast"] * 4,
        "Is Actual": [False] * 4,
    })

    scenario_monthly = pd.DataFrame({
        "Internal Tax Head": ["PAYE", "PAYE", "Domestic VAT", "Domestic VAT"],
        "Fiscal Year": ["2025/26"] * 4,
        "Month Index": [9, 10, 9, 10],
        "Month Name": ["Mar", "Apr", "Mar", "Apr"],
        "Scenario Monthly Forecast": [85.0, 85.0, 105.0, 105.0],
    })

    scenario_package = {
        "baseline": {
            "monthly_outputs": {
                "baseline_monthly_path": baseline_monthly,
            }
        },
        "scenario": {
            "scenario_monthly_path": scenario_monthly,
        },
        "comparisons": {
            "total_comparison": total_comparison,
            "tax_head_comparison": tax_head_comparison,
            "annex_comparison": annex_comparison,
        },
    }

    pack = build_dashboard_pack(scenario_package)

    print("\n[1] EXECUTIVE SUMMARY")
    print(pack["executive_summary"])

    print("\n[2] MONTHLY TOTAL COMPARISON")
    print(pack["monthly_total_comparison"])

    print("\n[3] MONTHLY CHART TABLE")
    print(build_monthly_chart_table(pack["monthly_total_comparison"]))

    print("\n[4] MONTHLY IMPACT TABLE")
    print(build_monthly_impact_table(pack["monthly_total_comparison"]))

    print("\nDASHBOARD BUILDER V1 TEST PASSED.")