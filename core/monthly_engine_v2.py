from __future__ import annotations

"""
============================================================
MONTHLY ENGINE V2
Revised for:
- current open FY: actuals + remaining-month forecast
- future FY: pure forecast, no actuals required
- bridge-aware monthly aggregation
============================================================
"""

from typing import Any, Dict, List, Optional

import pandas as pd


class MonthlyEngineError(Exception):
    """Raised when monthly logic cannot be executed consistently."""
    pass


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

FISCAL_ORDER = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]

AGGREGATE_HEADS = {
    "Import Excise Duty": ["Excise Duty Oil", "Excise Duty Ordinary"],
    "VAT, imports": ["VAT Imports Ordinary", "VAT Imports Oil"],
}


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


def _require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MonthlyEngineError(f"{label} missing required columns: {missing}")


def _empty_month_frame(
    heads: List[str],
    fiscal_year: str,
) -> pd.DataFrame:
    rows = []
    for head in heads:
        for m in range(1, 13):
            rows.append({
                "Internal Tax Head": _clean(head),
                "Fiscal Year": _clean(fiscal_year),
                "Month Index": m,
                "Month Name": MONTH_NAME_MAP[m],
                "Mapped Monthly Collection": 0.0,
                "Load Flag": 0,
            })
    return pd.DataFrame(rows)


def _append_aggregate_rows(
    df: pd.DataFrame,
    value_cols: List[str],
    group_cols: List[str],
    head_col: str = "Internal Tax Head",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    work = df.copy()
    work[head_col] = work[head_col].map(_clean)

    rows = []
    for agg_head, components in AGGREGATE_HEADS.items():
        comp = work.loc[work[head_col].isin([_clean(x) for x in components])].copy()
        if comp.empty:
            continue

        grouped = comp.groupby(group_cols, as_index=False)[value_cols].sum()
        grouped[head_col] = agg_head
        rows.append(grouped)

    if not rows:
        return work

    return pd.concat([work] + rows, ignore_index=True, sort=False)


# ============================================================
# VALIDATION
# ============================================================

def validate_monthly_mapping(mapping_df: pd.DataFrame) -> None:
    _require_columns(
        mapping_df,
        ["Internal Tax Head", "Monthly Label", "Weight", "Sign", "Use_for_actual", "Use_for_share"],
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
            f"Monthly shares do not sum to 1 for some tax heads: {bad.to_dict(orient='records')}"
        )

    return sums


# ============================================================
# MAPPING TO TAX HEADS
# ============================================================

def aggregate_monthly_to_tax_heads(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    use_flag: str = "actual",
) -> pd.DataFrame:
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
    mapping["Internal Tax Head"] = mapping["Internal Tax Head"].map(_clean)
    mapping["Monthly Label"] = mapping["Monthly Label"].map(_clean)
    mapping["Weight"] = pd.to_numeric(mapping["Weight"], errors="coerce").fillna(1.0)
    mapping["Sign"] = pd.to_numeric(mapping["Sign"], errors="coerce").fillna(1.0)

    keep_cols = ["Internal Tax Head", "Monthly Label", "Weight", "Sign"]
    if "Department" in mapping.columns:
        mapping["Department"] = mapping["Department"].map(_clean)
        keep_cols.append("Department")

    merged = monthly.merge(mapping[keep_cols], on="Monthly Label", how="inner", suffixes=("", "_map"))

    if "Department_map" in merged.columns:
        mask = merged["Department_map"].eq("") | (merged["Department"] == merged["Department_map"])
        merged = merged.loc[mask].copy()

    merged["Mapped Monthly Collection"] = merged["Collection"] * merged["Weight"] * merged["Sign"]

    out = (
        merged.groupby(["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name"], as_index=False)[
            "Mapped Monthly Collection"
        ]
        .sum()
        .sort_values(["Internal Tax Head", "Fiscal Year", "Month Index"])
        .reset_index(drop=True)
    )

    return out


def build_monthly_taxhead_actuals(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    actual_months_loaded: Optional[int] = None,
    fiscal_year: Optional[str] = None,
    all_heads: Optional[List[str]] = None,
) -> pd.DataFrame:
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

    if (out.empty or fiscal_year is not None) and all_heads is not None and fiscal_year is not None:
        # future year / no actuals path
        out = _empty_month_frame(all_heads, fiscal_year)

    out = append_aggregate_monthly_taxhead_actuals(out)
    return out.reset_index(drop=True)


# ============================================================
# AGGREGATE BUILDERS
# ============================================================

def append_aggregate_monthly_taxhead_actuals(monthly_taxhead_actuals_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        monthly_taxhead_actuals_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Mapped Monthly Collection", "Load Flag"],
        "monthly_taxhead_actuals_df",
    )

    out = _append_aggregate_rows(
        df=monthly_taxhead_actuals_df,
        value_cols=["Mapped Monthly Collection", "Load Flag"],
        group_cols=["Fiscal Year", "Month Index", "Month Name"],
        head_col="Internal Tax Head",
    )

    out["Load Flag"] = pd.to_numeric(out["Load Flag"], errors="coerce").fillna(0.0)
    out["Load Flag"] = (out["Load Flag"] > 0).astype(int)

    return out.sort_values(["Internal Tax Head", "Fiscal Year", "Month Index"]).reset_index(drop=True)


def append_aggregate_monthly_shares(monthly_shares_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        monthly_shares_df,
        [
            "Internal Tax Head",
            "Reference Fiscal Year",
            "Month Index",
            "Month Name",
            "Reference Monthly Value",
            "Reference Annual Total",
            "Monthly Share",
        ],
        "monthly_shares_df",
    )

    out = _append_aggregate_rows(
        df=monthly_shares_df,
        value_cols=["Reference Monthly Value"],
        group_cols=["Reference Fiscal Year", "Month Index", "Month Name"],
        head_col="Internal Tax Head",
    )

    annual_totals = (
        out.groupby(["Internal Tax Head", "Reference Fiscal Year"], as_index=False)["Reference Monthly Value"]
        .sum()
        .rename(columns={"Reference Monthly Value": "Reference Annual Total"})
    )

    out = out.drop(columns=["Reference Annual Total", "Monthly Share"], errors="ignore")
    out = out.merge(annual_totals, on=["Internal Tax Head", "Reference Fiscal Year"], how="left")

    out["Monthly Share"] = out.apply(
        lambda r: 0.0
        if _to_float(r["Reference Annual Total"], 0.0) == 0.0
        else _to_float(r["Reference Monthly Value"], 0.0) / _to_float(r["Reference Annual Total"], 0.0),
        axis=1,
    )

    out = out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)
    validate_monthly_shares(out)
    return out


def append_aggregate_monthly_path(monthly_path_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    _require_columns(
        monthly_path_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", value_col],
        "monthly_path_df",
    )

    numeric_candidates = [
        c for c in monthly_path_df.columns
        if c not in {"Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Source"}
        and pd.api.types.is_numeric_dtype(monthly_path_df[c])
    ]

    out = _append_aggregate_rows(
        df=monthly_path_df,
        value_cols=numeric_candidates,
        group_cols=["Fiscal Year", "Month Index", "Month Name"],
        head_col="Internal Tax Head",
    )

    if "Source" in out.columns:
        out["Source"] = out["Source"].fillna("derived")

    return out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)


def append_aggregate_monthly_delta(monthly_delta_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        monthly_delta_df,
        [
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
        ],
        "monthly_delta_df",
    )

    out = _append_aggregate_rows(
        df=monthly_delta_df,
        value_cols=["Monthly Delta"],
        group_cols=["Fiscal Year", "Month Index", "Month Name"],
        head_col="Internal Tax Head",
    )

    annual_delta = (
        out.groupby(["Internal Tax Head", "Fiscal Year"], as_index=False)["Monthly Delta"]
        .sum()
        .rename(columns={"Monthly Delta": "Annual Delta"})
    )

    out = out.drop(columns=["Annual Delta", "Eligible Share", "Allocation Weight"], errors="ignore")
    out = out.merge(annual_delta, on=["Internal Tax Head", "Fiscal Year"], how="left")

    if "Monthly Share" not in out.columns:
        out["Monthly Share"] = 0.0
    if "Eligible" not in out.columns:
        out["Eligible"] = True
    if "Eligible Share" not in out.columns:
        out["Eligible Share"] = 0.0
    if "Allocation Weight" not in out.columns:
        out["Allocation Weight"] = 0.0

    out["Monthly Share"] = pd.to_numeric(out["Monthly Share"], errors="coerce").fillna(0.0)
    out["Eligible"] = out["Eligible"].astype("boolean").fillna(True)
    out["Eligible Share"] = pd.to_numeric(out["Eligible Share"], errors="coerce").fillna(0.0)
    out["Allocation Weight"] = pd.to_numeric(out["Allocation Weight"], errors="coerce").fillna(0.0)

    mask_agg = out["Internal Tax Head"].isin(list(AGGREGATE_HEADS.keys()))
    out.loc[mask_agg, "Allocation Weight"] = out.loc[mask_agg].apply(
        lambda r: 0.0 if _to_float(r["Annual Delta"], 0.0) == 0.0 else _to_float(r["Monthly Delta"], 0.0) / _to_float(r["Annual Delta"], 0.0),
        axis=1,
    )

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
    return out[keep_cols].sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)


# ============================================================
# SHARES
# ============================================================

def compute_monthly_shares_from_collections(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    reference_fiscal_year: str,
) -> pd.DataFrame:
    share_base = aggregate_monthly_to_tax_heads(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        use_flag="share",
    )

    share_base = share_base.loc[share_base["Fiscal Year"].map(_clean) == _clean(reference_fiscal_year)].copy()

    if share_base.empty:
        raise MonthlyEngineError(
            f"No mapped monthly collections found for reference fiscal year '{reference_fiscal_year}'."
        )

    annual_totals = (
        share_base.groupby(["Internal Tax Head", "Fiscal Year"], as_index=False)["Mapped Monthly Collection"]
        .sum()
        .rename(columns={"Mapped Monthly Collection": "Reference Annual Total"})
    )

    out = share_base.merge(annual_totals, on=["Internal Tax Head", "Fiscal Year"], how="left")
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
    out = append_aggregate_monthly_shares(out)
    validate_monthly_shares(out)
    return out


def get_remaining_share_table(monthly_shares_df: pd.DataFrame, months_loaded: int) -> pd.DataFrame:
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
        if int(r["Month Index"]) <= months_loaded or _to_float(r["Remaining Share"], 0.0) == 0.0
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
    if monthly_taxhead_actuals_df is None or monthly_taxhead_actuals_df.empty:
        return pd.DataFrame(columns=["Internal Tax Head", "Actual YTD"])

    _require_columns(
        monthly_taxhead_actuals_df,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Mapped Monthly Collection", "Load Flag"],
        "monthly_taxhead_actuals_df",
    )

    work = monthly_taxhead_actuals_df.copy()
    work["Fiscal Year"] = work["Fiscal Year"].map(_clean)
    work["Month Index"] = pd.to_numeric(work["Month Index"], errors="coerce").astype(int)
    work["Mapped Monthly Collection"] = pd.to_numeric(work["Mapped Monthly Collection"], errors="coerce").fillna(0.0)
    work["Load Flag"] = pd.to_numeric(work["Load Flag"], errors="coerce").fillna(0).astype(int)

    work = work.loc[work["Fiscal Year"] == _clean(fiscal_year)].copy()
    work = work.loc[(work["LoadFlag"] == 1) & (work["Month Index"] <= int(months_loaded))] if "LoadFlag" in work.columns else work.loc[(work["Load Flag"] == 1) & (work["Month Index"] <= int(months_loaded))]

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
    use_actuals: bool = True,
) -> pd.DataFrame:
    _require_columns(annual_detail_df, ["Internal Tax Head", annual_value_col], "annual_detail_df")
    validate_monthly_shares(monthly_shares_df)

    annual = annual_detail_df.copy()
    annual["Internal Tax Head"] = annual["Internal Tax Head"].map(_clean)
    annual[annual_value_col] = pd.to_numeric(annual[annual_value_col], errors="coerce").fillna(0.0)

    if use_actuals:
        shares = get_remaining_share_table(monthly_shares_df, months_loaded).copy()
    else:
        shares = monthly_shares_df.copy()
        shares["Loaded Share"] = 0.0
        shares["Remaining Share"] = 1.0
        shares["Remaining Month Normalized Share"] = pd.to_numeric(shares["Monthly Share"], errors="coerce").fillna(0.0)

    shares["Internal Tax Head"] = shares["Internal Tax Head"].map(_clean)

    if use_actuals and monthly_taxhead_actuals_df is not None and not monthly_taxhead_actuals_df.empty:
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

        actual_months = actuals.rename(columns={"Mapped Monthly Collection": "Actual Monthly"})
        actual_months = actual_months[["Internal Tax Head", "Month Index", "Month Name", "Actual Monthly", "Load Flag"]].copy()
    else:
        actual_ytd = pd.DataFrame({"Internal Tax Head": annual["Internal Tax Head"].unique(), "Actual YTD": 0.0})
        actual_months = _empty_month_frame(annual["Internal Tax Head"].unique().tolist(), fiscal_year)
        actual_months = actual_months.rename(columns={"Mapped Monthly Collection": "Actual Monthly"})

    path = shares.merge(
        annual[["Internal Tax Head", annual_value_col]],
        on="Internal Tax Head",
        how="left",
    ).rename(columns={annual_value_col: "Annual Final Forecast"})

    path = path.merge(actual_ytd, on="Internal Tax Head", how="left")
    path["Annual Final Forecast"] = path["Annual Final Forecast"].fillna(0.0)
    path["Actual YTD"] = path["Actual YTD"].fillna(0.0)

    path = path.merge(
        actual_months[["Internal Tax Head", "Month Index", "Month Name", "Actual Monthly", "Load Flag"]],
        on=["Internal Tax Head", "Month Index", "Month Name"],
        how="left",
    )

    path["Actual Monthly"] = pd.to_numeric(path["Actual Monthly"], errors="coerce").fillna(0.0)
    path["Load Flag"] = pd.to_numeric(path["Load Flag"], errors="coerce").fillna(0).astype(int)

    path["Remaining Forecast Total"] = path["Annual Final Forecast"] - path["Actual YTD"]

    if use_actuals:
        path["Baseline Monthly Forecast"] = path.apply(
            lambda r: _to_float(r["Actual Monthly"], 0.0)
            if int(r["Month Index"]) <= int(months_loaded)
            else _to_float(r["Remaining Forecast Total"], 0.0) * _to_float(r["Remaining Month Normalized Share"], 0.0),
            axis=1,
        )
        path["Source"] = path["Month Index"].apply(lambda m: "actual" if int(m) <= int(months_loaded) else "forecast")
    else:
        path["Baseline Monthly Forecast"] = path["Annual Final Forecast"] * pd.to_numeric(path["Monthly Share"], errors="coerce").fillna(0.0)
        path["Source"] = "forecast"

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
    out = append_aggregate_monthly_path(out, value_col="Baseline Monthly Forecast")

    for col in ["Annual Final Forecast", "Actual YTD", "Remaining Forecast Total"]:
        if col in out.columns:
            agg = out.groupby(["Internal Tax Head", "Fiscal Year"], as_index=False)[col].max()
            out = out.drop(columns=[col]).merge(agg, on=["Internal Tax Head", "Fiscal Year"], how="left")

    return out.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)


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
    _require_columns(annual_delta_df, ["Internal Tax Head", annual_delta_col], "annual_delta_df")
    validate_monthly_shares(monthly_shares_df)

    annual = annual_delta_df.copy()
    annual["Internal Tax Head"] = annual["Internal Tax Head"].map(_clean)
    annual = annual.loc[~annual["Internal Tax Head"].isin(list(AGGREGATE_HEADS.keys()))].copy()
    annual[annual_delta_col] = pd.to_numeric(annual[annual_delta_col], errors="coerce").fillna(0.0)

    shares = monthly_shares_df.copy()
    shares["Internal Tax Head"] = shares["Internal Tax Head"].map(_clean)
    shares = shares.loc[~shares["Internal Tax Head"].isin(list(AGGREGATE_HEADS.keys()))].copy()
    shares["Month Index"] = pd.to_numeric(shares["Month Index"], errors="coerce").astype(int)
    shares["Monthly Share"] = pd.to_numeric(shares["Monthly Share"], errors="coerce").fillna(0.0)

    path = shares.merge(
        annual[["Internal Tax Head", annual_delta_col]],
        on="Internal Tax Head",
        how="left",
    ).rename(columns={annual_delta_col: "Annual Delta"})

    path["Annual Delta"] = path["Annual Delta"].fillna(0.0)

    if allocate_remaining_only:
        path["Eligible"] = path["Month Index"] > int(months_loaded)
    else:
        path["Eligible"] = True

    if scenario_duration_months is not None:
        scenario_duration_months = int(scenario_duration_months)
        path = path.sort_values(["Internal Tax Head", "Month Index"]).reset_index(drop=True)
        path["Eligible Rank"] = 0
        eligible_idx = path["Eligible"]
        path.loc[eligible_idx, "Eligible Rank"] = (
            path.loc[eligible_idx].groupby("Internal Tax Head").cumcount() + 1
        )
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
    out = append_aggregate_monthly_delta(out)
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

    check = annual[["Internal Tax Head", annual_value_col]].merge(rebuilt, on="Internal Tax Head", how="left")
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
    use_actuals: bool = True,
) -> Dict[str, pd.DataFrame]:
    heads = annual_detail_df["Internal Tax Head"].astype(str).str.strip().unique().tolist()

    monthly_taxhead_actuals = build_monthly_taxhead_actuals(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        actual_months_loaded=months_loaded if use_actuals else 0,
        fiscal_year=current_fiscal_year,
        all_heads=heads,
    )

    monthly_shares = compute_monthly_shares_from_collections(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        reference_fiscal_year=reference_fiscal_year,
    )

    actual_ytd = compute_actual_ytd(
        monthly_taxhead_actuals_df=monthly_taxhead_actuals if use_actuals else pd.DataFrame(),
        fiscal_year=current_fiscal_year,
        months_loaded=months_loaded if use_actuals else 0,
    )

    baseline_monthly_path = build_baseline_monthly_path(
        annual_detail_df=annual_detail_df,
        monthly_shares_df=monthly_shares,
        monthly_taxhead_actuals_df=monthly_taxhead_actuals,
        fiscal_year=current_fiscal_year,
        months_loaded=months_loaded if use_actuals else 0,
        annual_value_col=annual_value_col,
        use_actuals=use_actuals,
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


if __name__ == "__main__":
    print("=" * 90)
    print("MONTHLY ENGINE V2 TEST")
    print("=" * 90)
    print("Module loaded successfully.")