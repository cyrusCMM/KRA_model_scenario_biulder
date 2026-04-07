# -*- coding: utf-8 -*-
"""
ROLLING ENGINE V1
Baseline rolling orchestration engine
Revised to support:
- current FY open-year actuals + forecast
- future FY full 12-month synthetic forecast
- correct monthly input key usage
"""

from typing import Any, Dict
import pandas as pd

from tax_engine_v2 import run_tax_engine
from monthly_engine_v2 import run_monthly_baseline_pipeline


class RollingEngineError(Exception):
    """Raised when rolling forecast logic cannot be executed consistently."""
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
        raise RollingEngineError(f"{label} is missing or not a DataFrame.")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RollingEngineError(f"{label} missing required columns: {missing}")


def _get_selected_year(data: Dict[str, Any]) -> str:
    selected_year = _clean(data.get("rolling_control", {}).get("selected_year", ""))
    if selected_year not in {"2025/26", "2026/27", "2027/28"}:
        raise RollingEngineError(
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


def _get_reference_share_year(data: Dict[str, Any]) -> str:
    return _clean(data.get("rolling_control", {}).get("reference_share_year", "2024/25")) or "2024/25"


def _get_current_fiscal_year(data: Dict[str, Any]) -> str:
    return _clean(data.get("rolling_control", {}).get("current_fiscal_year", "2025/26")) or "2025/26"


# ============================================================
# ANNUAL DETAIL REBUILD
# ============================================================

def apply_rebuilt_annual_to_detail(
    annual_detail_df: pd.DataFrame,
    rebuilt_annual_df: pd.DataFrame,
    annual_col_name: str,
    keep_policy_adjustment: bool = True,
) -> pd.DataFrame:
    _require_columns(
        annual_detail_df,
        ["Internal Tax Head", "Opening Base", "Baseline Forecast", "Final Forecast"],
        "annual_detail_df",
    )
    _require_columns(
        rebuilt_annual_df,
        ["Internal Tax Head", annual_col_name],
        "rebuilt_annual_df",
    )

    detail = annual_detail_df.copy()
    rebuilt = rebuilt_annual_df.copy()

    detail["Internal Tax Head"] = detail["Internal Tax Head"].map(_clean)
    rebuilt["Internal Tax Head"] = rebuilt["Internal Tax Head"].map(_clean)
    rebuilt[annual_col_name] = pd.to_numeric(rebuilt[annual_col_name], errors="coerce").fillna(0.0)

    out = detail.merge(
        rebuilt[["Internal Tax Head", annual_col_name]],
        on="Internal Tax Head",
        how="left",
    )

    out[annual_col_name] = pd.to_numeric(out[annual_col_name], errors="coerce").fillna(out["Final Forecast"])
    out["Baseline Forecast"] = out[annual_col_name]
    out["Final Forecast"] = out[annual_col_name]

    if not keep_policy_adjustment and "Policy Adjustment" in out.columns:
        out["Policy Adjustment"] = 0.0

    out = out.drop(columns=[annual_col_name])
    return out


# ============================================================
# SUMMARIES
# ============================================================

def build_annex_summary(detail_df: pd.DataFrame, selected_year: str) -> pd.DataFrame:
    _require_columns(
        detail_df,
        ["Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"],
        "detail_df",
    )

    out = (
        detail_df.groupby("Annex Tax Head", as_index=False)[
            ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"]
        ]
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
        ["Department", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"],
        "detail_df",
    )

    out = (
        detail_df.groupby("Department", as_index=False)[
            ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"]
        ]
        .sum()
    )
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

    return pd.DataFrame([{
        "Opening Base": opening,
        "Baseline Forecast": baseline,
        "Policy Adjustment": policy,
        "Final Forecast": final,
        "Forecast Growth": ((final - opening) / opening) if opening != 0 else 0.0,
    }])


# ============================================================
# MONTHLY BASELINE ORCHESTRATION
# ============================================================

def run_monthly_baseline_pipeline_from_data(
    annual_detail_df: pd.DataFrame,
    data: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    selected_year = _get_selected_year(data)
    current_fiscal_year = _get_current_fiscal_year(data)
    reference_share_year = _get_reference_share_year(data)
    months_loaded = _get_actual_months_loaded(data)

    # Current FY gets actuals blended in; future years are full forecast only
    use_actuals = selected_year == current_fiscal_year

    monthly_collections_df = data.get("monthly_collections_normalized")
    monthly_mapping_df = data.get("monthly_mapping")

    if monthly_collections_df is None or not isinstance(monthly_collections_df, pd.DataFrame):
        raise RollingEngineError("data['monthly_collections_normalized'] is missing or not a DataFrame.")
    if monthly_mapping_df is None or not isinstance(monthly_mapping_df, pd.DataFrame):
        raise RollingEngineError("data['monthly_mapping'] is missing or not a DataFrame.")

    monthly_outputs = run_monthly_baseline_pipeline(
        annual_detail_df=annual_detail_df,
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        reference_fiscal_year=reference_share_year,
        current_fiscal_year=selected_year,
        months_loaded=months_loaded if use_actuals else 0,
        annual_value_col="Final Forecast",
        use_actuals=use_actuals,
    )

    return monthly_outputs


# ============================================================
# MASTER RUNNER
# ============================================================

def run_rolling_engine(
    data: Dict[str, Any],
    macro_df: pd.DataFrame,
) -> Dict[str, Any]:
    selected_year = _get_selected_year(data)
    year_status_2025_26 = _get_year_status_2025_26(data)

    baseline_outputs = run_tax_engine(data=data, macro_df=macro_df)
    detail = baseline_outputs["detail"].copy()

    outputs: Dict[str, Any] = {
        "selected_year": selected_year,
        "year_status_2025_26": year_status_2025_26,
        "baseline_detail_initial": detail.copy(),
    }

    # Always run monthly pipeline:
    # - current FY => actuals + remaining months
    # - future FY => full synthetic 12-month path
    monthly_outputs = run_monthly_baseline_pipeline_from_data(
        annual_detail_df=detail,
        data=data,
    )

    rebuilt_annual = monthly_outputs["annual_rebuild"].rename(
        columns={"Rebuilt Annual Forecast": f"Rebuilt Final {selected_year}"}
    )

    detail = apply_rebuilt_annual_to_detail(
        annual_detail_df=detail,
        rebuilt_annual_df=rebuilt_annual,
        annual_col_name=f"Rebuilt Final {selected_year}",
        keep_policy_adjustment=True,
    )

    outputs["monthly_outputs"] = monthly_outputs
    outputs["baseline_detail_final"] = detail.copy()

    final_detail = outputs["baseline_detail_final"].copy()
    final_annex_summary = build_annex_summary(final_detail, selected_year)
    final_department_summary = build_department_summary(final_detail)
    final_total_summary = build_total_summary(final_detail)

    outputs["detail"] = final_detail
    outputs["annex_summary"] = final_annex_summary
    outputs["department_summary"] = final_department_summary
    outputs["total_summary"] = final_total_summary

    return outputs


if __name__ == "__main__":
    print("=" * 90)
    print("ROLLING ENGINE V1 (REVISED) LOADED")
    print("=" * 90)