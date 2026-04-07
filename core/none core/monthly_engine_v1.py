# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:35:43 2026

@author: hp
"""

from __future__ import annotations

# ============================================================
# MONTHLY ENGINE V1
# Fresh monthly engine for:
# - mapping monthly collections to model tax heads
# - validating monthly shares
# - extracting actual YTD
# - building baseline monthly paths
# - rebuilding annual values from monthly paths
# - allocating annual scenario deltas to months
# ============================================================

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class MonthlyEngineError(Exception):
    """Raised when monthly logic cannot be executed consistently."""
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

MONTH_NAME_MAP = {
    1: "Jul",
    2: "Aug",
    3: "Sep",
    4: "Oct",
    5: "Nov",
    6: "Dec",
    7: "Jan",
    8: "Feb",
    9: "Mar",
    10: "Apr",
    11: "May",
    12: "Jun",
}


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


def _require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MonthlyEngineError(f"{label} missing required columns: {missing}")


def _copy_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df.copy()


# ============================================================
# VALIDATION
# ============================================================

def validate_monthly_mapping(mapping_df: pd.DataFrame) -> None:
    """
    Ensures:
    - required columns exist
    - no duplicate active monthly labels
    - signs are numeric
    """
    _require_columns(
        mapping_df,
        [
            "Internal Tax Head",
            "Monthly Label",
            "Weight",
            "Sign",
            "Use_for_actual",
            "Use_for_share",
        ],
        "mapping_df",
    )

    work = mapping_df.copy()
    work["Internal Tax Head"] = work["Internal Tax Head"].map(_clean)
    work["Monthly Label"] = work["Monthly Label"].map(_clean)
    work["Weight"] = pd.to_numeric(work["Weight"], errors="coerce")
    work["Sign"] = pd.to_numeric(work["Sign"], errors="coerce")

    if work["Internal Tax Head"].eq("").any():
        raise MonthlyEngineError("mapping_df contains blank Internal Tax Head values.")

    if work["Monthly Label"].eq("").any():
        raise MonthlyEngineError("mapping_df contains blank Monthly Label values.")

    if work["Weight"].isna().any():
        raise MonthlyEngineError("mapping_df contains non-numeric Weight values.")

    if work["Sign"].isna().any():
        raise MonthlyEngineError("mapping_df contains non-numeric Sign values.")

    active = work.loc[work["Use_for_actual"] | work["Use_for_share"]].copy()
    dups = active["Monthly Label"].duplicated()
    if dups.any():
        dup_vals = active.loc[dups, "Monthly Label"].unique().tolist()
        raise MonthlyEngineError(
            f"mapping_df contains duplicate active Monthly Label values: {dup_vals}"
        )


def validate_monthly_collections(monthly_df: pd.DataFrame) -> None:
    _require_columns(
        monthly_df,
        ["Department", "Monthly Label", "Fiscal Year", "Month Index", "Month Name", "Collection"],
        "monthly_df",
    )

    work = monthly_df.copy()
    work["Month Index"] = pd.to_numeric(work["Month Index"], errors="coerce")
    work["Collection"] = pd.to_numeric(work["Collection"], errors="coerce")

    if work["Month Index"].isna().any():
        raise MonthlyEngineError("monthly_df contains non-numeric Month Index values.")

    bad_months = work.loc[(work["Month Index"] < 1) | (work["Month Index"] > 12)]
    if not bad_months.empty:
        raise MonthlyEngineError("monthly_df contains Month Index outside 1..12.")

    if work["Collection"].isna().any():
        raise MonthlyEngineError("monthly_df contains non-numeric Collection values.")


def validate_monthly_shares(shares_df: pd.DataFrame, tolerance: float = 1e-4) -> pd.DataFrame:
    _require_columns(
        shares_df,
        [
            "Internal Tax Head",
            "Reference Fiscal Year",
            "Month Index",
            "Month Name",
            "Reference Monthly Value",
            "Reference Annual Total",
            "Monthly Share",
        ],
        "shares_df",
    )

    work = shares_df.copy()
    work["Month Index"] = pd.to_numeric(work["Month Index"], errors="coerce")
    work["Monthly Share"] = pd.to_numeric(work["Monthly Share"], errors="coerce")

    if work["Month Index"].isna().any():
        raise MonthlyEngineError("shares_df contains non-numeric Month Index values.")

    if work["Monthly Share"].isna().any():
        raise MonthlyEngineError("shares_df contains non-numeric Monthly Share values.")

    sums = (
        work.groupby(["Internal Tax Head", "Reference Fiscal Year"], as_index=False)["Monthly Share"]
        .sum()
        .rename(columns={"Monthly Share": "Share Sum"})
    )

    bad = sums.loc[(sums["Share Sum"] - 1.0).abs() > tolerance].copy()
    if not bad.empty:
        raise MonthlyEngineError(
            "Monthly shares do not sum to 1 for some tax heads: "
            f"{bad.to_dict(orient='records')}"
        )

    return sums


# ============================================================
# MONTHLY MAPPING AND AGGREGATION
# ============================================================

def aggregate_monthly_to_tax_heads(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    use_flag: str = "actual",
) -> pd.DataFrame:
    """
    Maps raw monthly collections to Internal Tax Head using Monthly_Mapping.

    Parameters
    ----------
    use_flag : {"actual", "share"}
        - "actual" uses Use_for_actual == True
        - "share"  uses Use_for_share == True
    """
    validate_monthly_collections(monthly_collections_df)
    validate_monthly_mapping(monthly_mapping_df)

    if use_flag not in {"actual", "share"}:
        raise MonthlyEngineError("use_flag must be either 'actual' or 'share'.")

    flag_col = "Use_for_actual" if use_flag == "actual" else "Use_for_share"

    monthly = monthly_collections_df.copy()
    mapping = monthly_mapping_df.copy()

    monthly["Department"] = monthly["Department"].map(_clean)
    monthly["Monthly Label"] = monthly["Monthly Label"].map(_clean)
    monthly["Fiscal Year"] = monthly["Fiscal Year"].map(_clean)
    monthly["Month Name"] = monthly["Month Name"].map(_clean)
    monthly["Month Index"] = pd.to_numeric(monthly["Month Index"], errors="coerce").astype(int)
    monthly["Collection"] = pd.to_numeric(monthly["Collection"], errors="coerce").fillna(0.0)

    mapping = mapping.loc[mapping[flag_col]].copy()
    mapping["Department"] = mapping.get("Department", "").map(_clean) if "Department" in mapping.columns else ""
    mapping["Internal Tax Head"] = mapping["Internal Tax Head"].map(_clean)
    mapping["Monthly Label"] = mapping["Monthly Label"].map(_clean)
    mapping["Weight"] = pd.to_numeric(mapping["Weight"], errors="coerce").fillna(1.0)
    mapping["Sign"] = pd.to_numeric(mapping["Sign"], errors="coerce").fillna(1.0)

    merged = monthly.merge(
        mapping[
            ["Internal Tax Head", "Monthly Label", "Weight", "Sign"] +
            (["Department"] if "Department" in mapping.columns else [])
        ],
        on="Monthly Label",
        how="inner",
        suffixes=("", "_map"),
    )

    # If Department exists in mapping, enforce it only when not blank
    if "Department_map" in merged.columns:
        mask = merged["Department_map"].eq("") | (merged["Department"] == merged["Department_map"])
        merged = merged.loc[mask].copy()

    merged["Mapped Monthly Collection"] = (
        merged["Collection"] * merged["Weight"] * merged["Sign"]
    )

    out = (
        merged.groupby(
            ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name"],
            as_index=False
        )["Mapped Monthly Collection"]
        .sum()
    )

    out = out.sort_values(["Internal Tax Head", "Fiscal Year", "Month Index"]).reset_index(drop=True)
    return out


def build_monthly_taxhead_actuals(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    actual_months_loaded: Optional[int] = None,
    fiscal_year: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produces Internal Tax Head monthly actuals with Load Flag.
    """
    out = aggregate_monthly_to_tax_heads(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        use_flag="actual",
    )

    if fiscal_year is not None:
        out = out.loc[out["Fiscal Year"].map(_clean) == _clean(fiscal_year)].copy()

    out["Load Flag"] = 1

    if actual_months_loaded is not None:
        out["Load Flag"] = (out["Month Index"] <= int(actual_months_loaded)).astype(int)
        out.loc[out["Month Index"] > int(actual_months_loaded), "Mapped Monthly Collection"] = 0.0

    return out.reset_index(drop=True)


# ============================================================
# MONTHLY SHARES
# ============================================================

def compute_monthly_shares_from_collections(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    reference_fiscal_year: str,
) -> pd.DataFrame:
    """
    Computes monthly shares directly from monthly raw collections using share mapping.
    """
    share_base = aggregate_monthly_to_tax_heads(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        use_flag="share",
    )

    share_base = share_base.loc[
        share_base["Fiscal Year"].map(_clean) == _clean(reference_fiscal_year)
    ].copy()

    if share_base.empty:
        raise MonthlyEngineError(
            f"No mapped monthly collections found for reference fiscal year '{reference_fiscal_year}'."
        )

    annual_totals = (
        share_base.groupby(["Internal Tax Head", "Fiscal Year"], as_index=False)["Mapped Monthly Collection"]
        .sum()
        .rename(columns={"Mapped Monthly Collection": "Reference Annual Total"})
    )

    out = share_base.merge(
        annual_totals,
        on=["Internal Tax Head", "Fiscal Year"],
        how="left",
    )

    out = out.rename(columns={
        "Fiscal Year": "Reference Fiscal Year",
        "Mapped Monthly Collection": "Reference Monthly Value",
    })

    out["Monthly Share"] = out.apply(
        lambda r: 0.0
        if _to_float(r["Reference Annual Total"], 0.0) == 0.0
        else _to_float(r["Reference Monthly Value"], 0.0) / _to_float(r["Reference Annual Total"], 0.0),
        axis=1,
    )

    out = out[
        [
            "Internal Tax Head",
            "Reference Fiscal Year",
            "Month Index",
            "Month Name",
            "Reference Monthly Value",
            "Reference Annual Total",
            "Monthly Share",
        ]
    ].copy()

    out = out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)

    validate_monthly_shares(out)
    return out


def get_remaining_share_table(
    monthly_shares_df: pd.DataFrame,
    months_loaded: int,
) -> pd.DataFrame:
    """
    Computes:
    - share loaded to date
    - share remaining
    - normalized remaining share weights
    """
    validate_monthly_shares(monthly_shares_df)

    shares = monthly_shares_df.copy()
    shares["Month Index"] = pd.to_numeric(shares["Month Index"], errors="coerce").astype(int)
    shares["Monthly Share"] = pd.to_numeric(shares["Monthly Share"], errors="coerce").fillna(0.0)

    share_loaded = (
        shares.loc[shares["Month Index"] <= months_loaded]
        .groupby("Internal Tax Head", as_index=False)["Monthly Share"]
        .sum()
        .rename(columns={"Monthly Share": "Loaded Share"})
    )

    share_remaining = (
        shares.loc[shares["Month Index"] > months_loaded]
        .groupby("Internal Tax Head", as_index=False)["Monthly Share"]
        .sum()
        .rename(columns={"Monthly Share": "Remaining Share"})
    )

    out = shares.merge(share_loaded, on="Internal Tax Head", how="left")
    out = out.merge(share_remaining, on="Internal Tax Head", how="left")

    out["Loaded Share"] = out["Loaded Share"].fillna(0.0)
    out["Remaining Share"] = out["Remaining Share"].fillna(0.0)

    out["Remaining Month Normalized Share"] = out.apply(
        lambda r: 0.0
        if r["Month Index"] <= months_loaded or _to_float(r["Remaining Share"], 0.0) == 0.0
        else _to_float(r["Monthly Share"], 0.0) / _to_float(r["Remaining Share"], 0.0),
        axis=1,
    )

    return out


# ============================================================
# ACTUAL YTD
# ============================================================

def compute_actual_ytd(
    monthly_taxhead_actuals_df: pd.DataFrame,
    fiscal_year: str,
    months_loaded: int,
) -> pd.DataFrame:
    _require_columns(
        monthly_taxhead_actuals_df,
        [
            "Internal Tax Head",
            "Fiscal Year",
            "Month Index",
            "Mapped Monthly Collection",
            "Load Flag",
        ],
        "monthly_taxhead_actuals_df",
    )

    work = monthly_taxhead_actuals_df.copy()
    work["Fiscal Year"] = work["Fiscal Year"].map(_clean)
    work["Month Index"] = pd.to_numeric(work["Month Index"], errors="coerce").astype(int)
    work["Mapped Monthly Collection"] = pd.to_numeric(work["Mapped Monthly Collection"], errors="coerce").fillna(0.0)
    work["Load Flag"] = pd.to_numeric(work["Load Flag"], errors="coerce").fillna(0).astype(int)

    work = work.loc[work["Fiscal Year"] == _clean(fiscal_year)].copy()
    work = work.loc[(work["Load Flag"] == 1) & (work["Month Index"] <= int(months_loaded))].copy()

    out = (
        work.groupby("Internal Tax Head", as_index=False)["Mapped Monthly Collection"]
        .sum()
        .rename(columns={"Mapped Monthly Collection": "Actual YTD"})
    )

    return out


# ============================================================
# BASELINE MONTHLY PATH
# ============================================================

def build_baseline_monthly_path(
    annual_detail_df: pd.DataFrame,
    monthly_shares_df: pd.DataFrame,
    monthly_taxhead_actuals_df: pd.DataFrame,
    fiscal_year: str,
    months_loaded: int,
    annual_value_col: str = "Final Forecast",
) -> pd.DataFrame:
    """
    Builds current-year monthly path:
    - loaded months use actuals
    - remaining months use annual forecast * normalized remaining shares
    """
    _require_columns(annual_detail_df, ["Internal Tax Head", annual_value_col], "annual_detail_df")
    validate_monthly_shares(monthly_shares_df)

    annual = annual_detail_df.copy()
    annual["Internal Tax Head"] = annual["Internal Tax Head"].map(_clean)
    annual[annual_value_col] = pd.to_numeric(annual[annual_value_col], errors="coerce").fillna(0.0)

    shares = get_remaining_share_table(monthly_shares_df, months_loaded).copy()
    shares["Internal Tax Head"] = shares["Internal Tax Head"].map(_clean)

    actuals = monthly_taxhead_actuals_df.copy()
    _require_columns(
        actuals,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Mapped Monthly Collection", "Load Flag"],
        "monthly_taxhead_actuals_df",
    )
    actuals["Internal Tax Head"] = actuals["Internal Tax Head"].map(_clean)
    actuals["Fiscal Year"] = actuals["Fiscal Year"].map(_clean)
    actuals["Month Index"] = pd.to_numeric(actuals["Month Index"], errors="coerce").astype(int)
    actuals["Mapped Monthly Collection"] = pd.to_numeric(actuals["Mapped Monthly Collection"], errors="coerce").fillna(0.0)
    actuals["Load Flag"] = pd.to_numeric(actuals["Load Flag"], errors="coerce").fillna(0).astype(int)
    actuals = actuals.loc[actuals["Fiscal Year"] == _clean(fiscal_year)].copy()

    actual_ytd = compute_actual_ytd(actuals, fiscal_year=fiscal_year, months_loaded=months_loaded)

    path = shares.merge(
        annual[["Internal Tax Head", annual_value_col]],
        on="Internal Tax Head",
        how="left",
    ).rename(columns={annual_value_col: "Annual Final Forecast"})

    path = path.merge(
        actual_ytd,
        on="Internal Tax Head",
        how="left",
    )

    path["Annual Final Forecast"] = path["Annual Final Forecast"].fillna(0.0)
    path["Actual YTD"] = path["Actual YTD"].fillna(0.0)

    # Merge actual monthly values
    actual_months = actuals.rename(columns={"Mapped Monthly Collection": "Actual Monthly"})
    path = path.merge(
        actual_months[
            ["Internal Tax Head", "Month Index", "Month Name", "Actual Monthly", "Load Flag"]
        ],
        on=["Internal Tax Head", "Month Index", "Month Name"],
        how="left",
    )

    path["Actual Monthly"] = path["Actual Monthly"].fillna(0.0)
    path["Load Flag"] = path["Load Flag"].fillna(0).astype(int)

    path["Remaining Forecast Total"] = path["Annual Final Forecast"] - path["Actual YTD"]

    path["Baseline Monthly Forecast"] = path.apply(
        lambda r: _to_float(r["Actual Monthly"], 0.0)
        if int(r["Month Index"]) <= int(months_loaded)
        else _to_float(r["Remaining Forecast Total"], 0.0) * _to_float(r["Remaining Month Normalized Share"], 0.0),
        axis=1,
    )

    path["Source"] = path["Month Index"].apply(lambda m: "actual" if int(m) <= int(months_loaded) else "forecast")
    path["Fiscal Year"] = fiscal_year

    keep_cols = [
        "Internal Tax Head",
        "Fiscal Year",
        "Month Index",
        "Month Name",
        "Monthly Share",
        "Loaded Share",
        "Remaining Share",
        "Remaining Month Normalized Share",
        "Annual Final Forecast",
        "Actual YTD",
        "Remaining Forecast Total",
        "Actual Monthly",
        "Baseline Monthly Forecast",
        "Source",
    ]
    out = path[keep_cols].copy()
    out = out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)

    return out


def rebuild_annual_from_monthly_path(
    monthly_path_df: pd.DataFrame,
    value_col: str = "Baseline Monthly Forecast",
    annual_col_name: str = "Rebuilt Annual Forecast",
) -> pd.DataFrame:
    _require_columns(monthly_path_df, ["Internal Tax Head", "Fiscal Year", value_col], "monthly_path_df")

    work = monthly_path_df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    out = (
        work.groupby(["Internal Tax Head", "Fiscal Year"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: annual_col_name})
    )
    return out


# ============================================================
# SCENARIO MONTHLY ALLOCATION
# ============================================================

def allocate_annual_delta_to_months(
    annual_delta_df: pd.DataFrame,
    monthly_shares_df: pd.DataFrame,
    fiscal_year: str,
    months_loaded: int = 0,
    allocate_remaining_only: bool = False,
    scenario_duration_months: Optional[int] = None,
    annual_delta_col: str = "Scenario Impact",
) -> pd.DataFrame:
    """
    Allocates annual scenario delta to months.

    Cases:
    - future year full allocation across all 12 months
    - current year allocation across remaining months only
    - optional duration window allocation across first N eligible months
    """
    _require_columns(annual_delta_df, ["Internal Tax Head", annual_delta_col], "annual_delta_df")
    validate_monthly_shares(monthly_shares_df)

    annual = annual_delta_df.copy()
    annual["Internal Tax Head"] = annual["Internal Tax Head"].map(_clean)
    annual[annual_delta_col] = pd.to_numeric(annual[annual_delta_col], errors="coerce").fillna(0.0)

    shares = monthly_shares_df.copy()
    shares["Internal Tax Head"] = shares["Internal Tax Head"].map(_clean)
    shares["Month Index"] = pd.to_numeric(shares["Month Index"], errors="coerce").astype(int)
    shares["Monthly Share"] = pd.to_numeric(shares["Monthly Share"], errors="coerce").fillna(0.0)

    path = shares.merge(
        annual[["Internal Tax Head", annual_delta_col]],
        on="Internal Tax Head",
        how="left",
    ).rename(columns={annual_delta_col: "Annual Delta"})

    path["Annual Delta"] = path["Annual Delta"].fillna(0.0)

    # Eligibility
    if allocate_remaining_only:
        path["Eligible"] = path["Month Index"] > int(months_loaded)
    else:
        path["Eligible"] = True

    # Optional duration window over eligible months
    if scenario_duration_months is not None:
        scenario_duration_months = int(scenario_duration_months)
        path = path.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)

        path["Eligible Rank"] = (
            path.loc[path["Eligible"]]
            .groupby("Internal Tax Head")
            .cumcount() + 1
        )
        path["Eligible Rank"] = path["Eligible Rank"].fillna(0).astype(int)
        path["Eligible"] = path["Eligible"] & (path["Eligible Rank"] <= scenario_duration_months)

    eligible_share = (
        path.loc[path["Eligible"]]
        .groupby("Internal Tax Head", as_index=False)["Monthly Share"]
        .sum()
        .rename(columns={"Monthly Share": "Eligible Share"})
    )
    path = path.merge(eligible_share, on="Internal Tax Head", how="left")
    path["Eligible Share"] = path["Eligible Share"].fillna(0.0)

    path["Allocation Weight"] = path.apply(
        lambda r: 0.0
        if (not bool(r["Eligible"])) or _to_float(r["Eligible Share"], 0.0) == 0.0
        else _to_float(r["Monthly Share"], 0.0) / _to_float(r["Eligible Share"], 0.0),
        axis=1,
    )

    path["Monthly Delta"] = path["Annual Delta"] * path["Allocation Weight"]
    path["Fiscal Year"] = fiscal_year

    keep_cols = [
        "Internal Tax Head",
        "Fiscal Year",
        "Month Index",
        "Month Name",
        "Monthly Share",
        "Eligible",
        "Eligible Share",
        "Allocation Weight",
        "Annual Delta",
        "Monthly Delta",
    ]
    out = path[keep_cols].copy()
    out = out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)
    return out


def build_scenario_monthly_path(
    baseline_monthly_path_df: pd.DataFrame,
    monthly_delta_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        baseline_monthly_path_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Baseline Monthly Forecast"],
        "baseline_monthly_path_df",
    )
    _require_columns(
        monthly_delta_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Monthly Delta"],
        "monthly_delta_df",
    )

    base = baseline_monthly_path_df.copy()
    delta = monthly_delta_df.copy()

    out = base.merge(
        delta[["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Monthly Delta"]],
        on=["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name"],
        how="left",
    )

    out["Monthly Delta"] = pd.to_numeric(out["Monthly Delta"], errors="coerce").fillna(0.0)
    out["Scenario Monthly Forecast"] = (
        pd.to_numeric(out["Baseline Monthly Forecast"], errors="coerce").fillna(0.0)
        + out["Monthly Delta"]
    )

    return out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)


# ============================================================
# CONSISTENCY CHECKS
# ============================================================

def validate_rebuild_consistency(
    annual_detail_df: pd.DataFrame,
    monthly_path_df: pd.DataFrame,
    annual_value_col: str = "Final Forecast",
    monthly_value_col: str = "Baseline Monthly Forecast",
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    _require_columns(annual_detail_df, ["Internal Tax Head", annual_value_col], "annual_detail_df")
    _require_columns(monthly_path_df, ["Internal Tax Head", monthly_value_col], "monthly_path_df")

    annual = annual_detail_df.copy()
    annual["Internal Tax Head"] = annual["Internal Tax Head"].map(_clean)
    annual[annual_value_col] = pd.to_numeric(annual[annual_value_col], errors="coerce").fillna(0.0)

    rebuilt = (
        monthly_path_df.groupby("Internal Tax Head", as_index=False)[monthly_value_col]
        .sum()
        .rename(columns={monthly_value_col: "Rebuilt"})
    )

    check = annual[["Internal Tax Head", annual_value_col]].merge(
        rebuilt,
        on="Internal Tax Head",
        how="left",
    )
    check["Rebuilt"] = check["Rebuilt"].fillna(0.0)
    check["Gap"] = check["Rebuilt"] - check[annual_value_col]
    check["Pass"] = check["Gap"].abs() <= tolerance

    return check.sort_values("Internal Tax Head").reset_index(drop=True)


# ============================================================
# HIGH-LEVEL ENTRY POINTS
# ============================================================

def run_monthly_baseline_pipeline(
    annual_detail_df: pd.DataFrame,
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    reference_fiscal_year: str,
    current_fiscal_year: str,
    months_loaded: int,
    annual_value_col: str = "Final Forecast",
) -> Dict[str, pd.DataFrame]:
    monthly_taxhead_actuals = build_monthly_taxhead_actuals(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        actual_months_loaded=months_loaded,
        fiscal_year=current_fiscal_year,
    )

    monthly_shares = compute_monthly_shares_from_collections(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        reference_fiscal_year=reference_fiscal_year,
    )

    actual_ytd = compute_actual_ytd(
        monthly_taxhead_actuals_df=monthly_taxhead_actuals,
        fiscal_year=current_fiscal_year,
        months_loaded=months_loaded,
    )

    baseline_monthly_path = build_baseline_monthly_path(
        annual_detail_df=annual_detail_df,
        monthly_shares_df=monthly_shares,
        monthly_taxhead_actuals_df=monthly_taxhead_actuals,
        fiscal_year=current_fiscal_year,
        months_loaded=months_loaded,
        annual_value_col=annual_value_col,
    )

    annual_rebuild = rebuild_annual_from_monthly_path(
        baseline_monthly_path,
        value_col="Baseline Monthly Forecast",
        annual_col_name="Rebuilt Annual Forecast",
    )

    rebuild_check = validate_rebuild_consistency(
        annual_detail_df=annual_detail_df,
        monthly_path_df=baseline_monthly_path,
        annual_value_col=annual_value_col,
        monthly_value_col="Baseline Monthly Forecast",
    )

    return {
        "monthly_taxhead_actuals": monthly_taxhead_actuals,
        "monthly_shares": monthly_shares,
        "actual_ytd": actual_ytd,
        "baseline_monthly_path": baseline_monthly_path,
        "annual_rebuild": annual_rebuild,
        "rebuild_check": rebuild_check,
    }


def run_monthly_scenario_pipeline(
    baseline_monthly_path_df: pd.DataFrame,
    annual_delta_df: pd.DataFrame,
    monthly_shares_df: pd.DataFrame,
    fiscal_year: str,
    months_loaded: int = 0,
    allocate_remaining_only: bool = False,
    scenario_duration_months: Optional[int] = None,
    annual_delta_col: str = "Scenario Impact",
) -> Dict[str, pd.DataFrame]:
    monthly_delta = allocate_annual_delta_to_months(
        annual_delta_df=annual_delta_df,
        monthly_shares_df=monthly_shares_df,
        fiscal_year=fiscal_year,
        months_loaded=months_loaded,
        allocate_remaining_only=allocate_remaining_only,
        scenario_duration_months=scenario_duration_months,
        annual_delta_col=annual_delta_col,
    )

    scenario_monthly_path = build_scenario_monthly_path(
        baseline_monthly_path_df=baseline_monthly_path_df,
        monthly_delta_df=monthly_delta,
    )

    scenario_annual_rebuild = rebuild_annual_from_monthly_path(
        scenario_monthly_path,
        value_col="Scenario Monthly Forecast",
        annual_col_name="Scenario Annual Rebuild",
    )

    return {
        "monthly_delta": monthly_delta,
        "scenario_monthly_path": scenario_monthly_path,
        "scenario_annual_rebuild": scenario_annual_rebuild,
    }


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    print("=" * 90)
    print("MONTHLY ENGINE V1 TEST")
    print("=" * 90)
    print("This test uses synthetic data only.")
    print("It checks mapping, shares, baseline path, and scenario allocation.")
    print("=" * 90)

    # --------------------------------------------------------
    # Synthetic raw monthly collections
    # --------------------------------------------------------
    monthly_collections = pd.DataFrame({
        "Department": ["DTD"] * 24,
        "Monthly Label": ["PAYE"] * 12 + ["VAT Domestic"] * 12,
        "Fiscal Year": ["2024/25"] * 12 + ["2024/25"] * 12,
        "Month Index": list(range(1, 13)) + list(range(1, 13)),
        "Month Name": [MONTH_NAME_MAP[i] for i in range(1, 13)] * 2,
        "Collection": [10] * 12 + [20] * 12,
    })

    # --------------------------------------------------------
    # Synthetic mapping
    # --------------------------------------------------------
    monthly_mapping = pd.DataFrame({
        "Internal Tax Head": ["PAYE", "Domestic VAT"],
        "Monthly Label": ["PAYE", "VAT Domestic"],
        "Weight": [1.0, 1.0],
        "Sign": [1.0, 1.0],
        "Use_for_actual": [True, True],
        "Use_for_share": [True, True],
    })

    # --------------------------------------------------------
    # Annual detail
    # --------------------------------------------------------
    annual_detail = pd.DataFrame({
        "Internal Tax Head": ["PAYE", "Domestic VAT"],
        "Final Forecast": [180.0, 300.0],
    })

    # --------------------------------------------------------
    # Run baseline pipeline
    # --------------------------------------------------------
    baseline = run_monthly_baseline_pipeline(
        annual_detail_df=annual_detail,
        monthly_collections_df=monthly_collections,
        monthly_mapping_df=monthly_mapping,
        reference_fiscal_year="2024/25",
        current_fiscal_year="2024/25",
        months_loaded=8,
        annual_value_col="Final Forecast",
    )

    print("\n[1] SHARE CHECK")
    print(
        baseline["monthly_shares"]
        .groupby("Internal Tax Head", as_index=False)["Monthly Share"]
        .sum()
    )

    print("\n[2] ACTUAL YTD")
    print(baseline["actual_ytd"])

    print("\n[3] REBUILD CHECK")
    print(baseline["rebuild_check"])

    # --------------------------------------------------------
    # Scenario delta
    # --------------------------------------------------------
    annual_delta = pd.DataFrame({
        "Internal Tax Head": ["PAYE", "Domestic VAT"],
        "Scenario Impact": [12.0, -24.0],
    })

    scenario = run_monthly_scenario_pipeline(
        baseline_monthly_path_df=baseline["baseline_monthly_path"],
        annual_delta_df=annual_delta,
        monthly_shares_df=baseline["monthly_shares"],
        fiscal_year="2024/25",
        months_loaded=8,
        allocate_remaining_only=True,
        scenario_duration_months=None,
        annual_delta_col="Scenario Impact",
    )

    print("\n[4] MONTHLY DELTA TOTALS")
    print(
        scenario["monthly_delta"]
        .groupby("Internal Tax Head", as_index=False)["Monthly Delta"]
        .sum()
    )

    print("\n[5] SCENARIO ANNUAL REBUILD")
    print(scenario["scenario_annual_rebuild"])

    print("\nMONTHLY ENGINE V1 TEST PASSED.")