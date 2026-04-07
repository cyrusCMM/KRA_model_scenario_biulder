# -*- coding: utf-8 -*-
"""
VALIDATION ENGINE V2
Validation layer for workbook-driven KRA forecasting pipeline

Scope
-----
- validate loaded input tables
- validate annual baseline outputs
- validate monthly outputs
- validate rolling outputs
- validate scenario outputs
- validate dashboard pack
- validate decomposition pack
"""

from typing import Any, Dict, List, Optional

import pandas as pd


class ValidationEngineError(Exception):
    """Raised when any pipeline object fails structural validation."""
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


def _require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationEngineError(f"{label} missing required columns: {missing}")


def _require_key(d: Dict[str, Any], key: str, label: str) -> None:
    if key not in d:
        raise ValidationEngineError(f"{label} missing required key: '{key}'")


def _require_dataframe(df: Any, label: str) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValidationEngineError(f"{label} is not a DataFrame.")


def _require_nonempty_df(df: pd.DataFrame, label: str) -> None:
    _require_dataframe(df, label)
    if df.empty:
        raise ValidationEngineError(f"{label} is empty.")


def _find_projection_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if _clean(c).lower().startswith("projected collection "):
            return c
    return None


def _assert_no_duplicates(df: pd.DataFrame, column: str, label: str) -> None:
    if column not in df.columns:
        raise ValidationEngineError(f"{label} missing duplicate-check column '{column}'.")
    work = df[column].astype(str).str.strip()
    dups = work.duplicated()
    if dups.any():
        vals = work.loc[dups].unique().tolist()
        raise ValidationEngineError(f"{label} contains duplicate values in '{column}': {vals}")


def _assert_month_index_valid(df: pd.DataFrame, label: str) -> None:
    if "Month Index" not in df.columns:
        raise ValidationEngineError(f"{label} missing 'Month Index'.")
    m = pd.to_numeric(df["Month Index"], errors="coerce")
    if m.isna().any():
        raise ValidationEngineError(f"{label} contains non-numeric Month Index.")
    if ((m < 1) | (m > 12)).any():
        raise ValidationEngineError(f"{label} contains Month Index values outside 1..12.")


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_loaded_inputs(data: Dict[str, Any]) -> None:
    required_keys = [
        "rolling_control",
        "macro",
        "elasticities",
        "elasticities_dict",
        "tax_heads_input",
        "policy_measures",
        "targets",
        "monthly_collections_normalized",
        "monthly_mapping",
        "monthly_taxhead_actuals",
        "monthly_shares",
        "tax_heads_rolling",
        "forecast_accuracy_2025_26",
    ]
    for k in required_keys:
        _require_key(data, k, "loaded_inputs")

    rolling_control = data["rolling_control"]
    selected_year = _clean(rolling_control.get("selected_year", ""))
    if selected_year not in {"2025/26", "2026/27", "2027/28"}:
        raise ValidationEngineError(
            f"rolling_control selected_year invalid: '{selected_year}'"
        )

    year_status = _clean(rolling_control.get("year_status_2025_26", "")).upper()
    if year_status not in {"OPEN", "CLOSED"}:
        raise ValidationEngineError(
            f"rolling_control year_status_2025_26 invalid: '{year_status}'"
        )

    months_loaded = rolling_control.get("actual_months_loaded", 0)
    try:
        months_loaded = int(months_loaded)
    except Exception:
        raise ValidationEngineError("rolling_control actual_months_loaded is not an integer.")
    if not (0 <= months_loaded <= 12):
        raise ValidationEngineError("rolling_control actual_months_loaded must be between 0 and 12.")

    # Macro
    macro = data["macro"]
    _require_nonempty_df(macro, "macro")
    if "Variable" not in macro.columns and "year" not in macro.columns:
        raise ValidationEngineError("macro must contain either 'Variable' or 'year' column.")

    # Elasticities
    elasticities = data["elasticities"]
    _require_nonempty_df(elasticities, "elasticities")
    _require_columns(elasticities, ["Parameter", "Value"], "elasticities")

    # Tax heads input
    tax_heads_input = data["tax_heads_input"]
    _require_nonempty_df(tax_heads_input, "tax_heads_input")
    _require_columns(
        tax_heads_input,
        ["Department", "Revenue Type", "Internal Tax Head", "Annex Tax Head", "Actual 2024/25"],
        "tax_heads_input",
    )
    _assert_no_duplicates(tax_heads_input, "Internal Tax Head", "tax_heads_input")

    # Policy
    policy = data["policy_measures"]
    _require_nonempty_df(policy, "policy_measures")
    _require_columns(policy, ["Internal Tax Head"], "policy_measures")

    # Monthly collections
    monthly_collections = data["monthly_collections_normalized"]
    _require_nonempty_df(monthly_collections, "monthly_collections_normalized")
    _require_columns(
        monthly_collections,
        ["Department", "Monthly Label", "Fiscal Year", "Month Index", "Month Name", "Collection"],
        "monthly_collections_normalized",
    )
    _assert_month_index_valid(monthly_collections, "monthly_collections_normalized")

    # Monthly mapping
    monthly_mapping = data["monthly_mapping"]
    _require_nonempty_df(monthly_mapping, "monthly_mapping")
    _require_columns(
        monthly_mapping,
        ["Internal Tax Head", "Department", "Monthly Label", "Weight", "Sign", "Use_for_actual", "Use_for_share"],
        "monthly_mapping",
    )

    # Monthly taxhead actuals
    monthly_tax_actuals = data["monthly_taxhead_actuals"]
    _require_nonempty_df(monthly_tax_actuals, "monthly_taxhead_actuals")
    _require_columns(
        monthly_tax_actuals,
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Month Name", "Mapped Monthly Collection", "Load Flag"],
        "monthly_taxhead_actuals",
    )
    _assert_month_index_valid(monthly_tax_actuals, "monthly_taxhead_actuals")

    # Monthly shares
    monthly_shares = data["monthly_shares"]
    _require_nonempty_df(monthly_shares, "monthly_shares")
    _require_columns(
        monthly_shares,
        ["Internal Tax Head", "Reference Fiscal Year", "Month Index", "Month Name", "Reference Monthly Value", "Reference Annual Total", "Monthly Share"],
        "monthly_shares",
    )
    _assert_month_index_valid(monthly_shares, "monthly_shares")

    share_sums = (
        monthly_shares.groupby(["Internal Tax Head", "Reference Fiscal Year"], as_index=False)["Monthly Share"]
        .sum()
    )
    bad_share = share_sums.loc[(share_sums["Monthly Share"] - 1.0).abs() > 1e-4]
    if not bad_share.empty:
        raise ValidationEngineError(
            f"monthly_shares do not sum to 1 for some tax heads: {bad_share.to_dict(orient='records')}"
        )

    # Tax heads rolling
    tax_heads_rolling = data["tax_heads_rolling"]
    _require_nonempty_df(tax_heads_rolling, "tax_heads_rolling")
    _require_columns(
        tax_heads_rolling,
        [
            "Internal Tax Head",
            "Annex Tax Head",
            "Opening Base 2025/26",
            "Opening Base 2026/27",
            "Opening Base 2027/28",
        ],
        "tax_heads_rolling",
    )

    # Forecast accuracy
    fa = data["forecast_accuracy_2025_26"]
    _require_nonempty_df(fa, "forecast_accuracy_2025_26")
    if "Internal Tax Head" not in fa.columns:
        raise ValidationEngineError("forecast_accuracy_2025_26 missing 'Internal Tax Head'.")


# ============================================================
# ANNUAL OUTPUT VALIDATION
# ============================================================

def validate_tax_engine_outputs(outputs: Dict[str, pd.DataFrame]) -> None:
    required_keys = ["detail", "annex_summary", "department_summary", "total_summary"]
    for k in required_keys:
        _require_key(outputs, k, "tax_engine_outputs")
        _require_nonempty_df(outputs[k], f"tax_engine_outputs['{k}']")

    detail = outputs["detail"]
    annex = outputs["annex_summary"]
    total = outputs["total_summary"]

    _require_columns(
        detail,
        ["Internal Tax Head", "Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "tax_engine detail",
    )
    _assert_no_duplicates(detail, "Internal Tax Head", "tax_engine detail")

    proj_col = _find_projection_column(annex)
    if proj_col is None:
        raise ValidationEngineError("tax_engine annex_summary missing projected collection column.")

    _require_columns(
        total,
        ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"],
        "tax_engine total_summary",
    )

    detail_final = pd.to_numeric(detail["Final Forecast"], errors="coerce").fillna(0.0).sum()
    annex_final = pd.to_numeric(annex[proj_col], errors="coerce").fillna(0.0).sum()
    total_final = pd.to_numeric(total["Final Forecast"], errors="coerce").fillna(0.0).iloc[0]

    if abs(detail_final - annex_final) > 1e-4:
        raise ValidationEngineError(f"tax_engine detail vs annex mismatch: {detail_final} != {annex_final}")
    if abs(detail_final - total_final) > 1e-4:
        raise ValidationEngineError(f"tax_engine detail vs total mismatch: {detail_final} != {total_final}")


# ============================================================
# MONTHLY OUTPUT VALIDATION
# ============================================================

def validate_monthly_pipeline_outputs(outputs: Dict[str, pd.DataFrame]) -> None:
    required_keys = [
        "monthly_taxhead_actuals",
        "monthly_shares",
        "actual_ytd",
        "baseline_monthly_path",
        "annual_rebuild",
        "rebuild_check",
    ]
    for k in required_keys:
        _require_key(outputs, k, "monthly_pipeline_outputs")

    # Must exist and be non-empty
    for k in ["monthly_taxhead_actuals", "monthly_shares", "baseline_monthly_path", "annual_rebuild", "rebuild_check"]:
        _require_nonempty_df(outputs[k], f"monthly_pipeline_outputs['{k}']")

    # actual_ytd may be empty when months_loaded == 0
    actual_ytd = outputs["actual_ytd"]
    _require_dataframe(actual_ytd, "monthly_pipeline_outputs['actual_ytd']")

    monthly_path = outputs["baseline_monthly_path"]
    _require_columns(
        monthly_path,
        [
            "Internal Tax Head",
            "Fiscal Year",
            "Month Index",
            "Month Name",
            "Baseline Monthly Forecast",
            "Source",
        ],
        "baseline_monthly_path",
    )
    _assert_month_index_valid(monthly_path, "baseline_monthly_path")

    annual_rebuild = outputs["annual_rebuild"]
    _require_columns(
        annual_rebuild,
        ["Internal Tax Head"],
        "annual_rebuild",
    )

    rebuild_check = outputs["rebuild_check"]
    _require_columns(
        rebuild_check,
        ["Internal Tax Head", "Rebuilt", "Gap", "Pass"],
        "rebuild_check",
    )
    if (~rebuild_check["Pass"].astype(bool)).any():
        bad = rebuild_check.loc[~rebuild_check["Pass"].astype(bool)]
        raise ValidationEngineError(
            f"monthly rebuild check failed for some tax heads: {bad.to_dict(orient='records')}"
        )


# ============================================================
# ROLLING OUTPUT VALIDATION
# ============================================================

def validate_rolling_outputs(outputs: Dict[str, Any]) -> None:
    required_keys = ["detail", "annex_summary", "department_summary", "total_summary"]
    for k in required_keys:
        _require_key(outputs, k, "rolling_outputs")
        _require_nonempty_df(outputs[k], f"rolling_outputs['{k}']")

    validate_tax_engine_outputs({
        "detail": outputs["detail"],
        "annex_summary": outputs["annex_summary"],
        "department_summary": outputs["department_summary"],
        "total_summary": outputs["total_summary"],
    })

    if "monthly_outputs" in outputs:
        validate_monthly_pipeline_outputs(outputs["monthly_outputs"])

    if "forecast_accuracy_by_head" in outputs:
        fa = outputs["forecast_accuracy_by_head"]
        _require_nonempty_df(fa, "forecast_accuracy_by_head")

    if "base_switch_table" in outputs:
        bst = outputs["base_switch_table"]
        _require_nonempty_df(bst, "base_switch_table")


# ============================================================
# SCENARIO OUTPUT VALIDATION
# ============================================================

def validate_simulation_outputs(outputs: Dict[str, Any]) -> None:
    required_keys = [
        "annual_delta",
        "monthly_delta",
        "scenario_monthly_path",
        "scenario_annual_rebuild",
        "detail",
        "annex_summary",
        "department_summary",
        "total_summary",
        "allocation_metadata",
    ]
    for k in required_keys:
        _require_key(outputs, k, "simulation_outputs")

    for k in [
        "annual_delta",
        "monthly_delta",
        "scenario_monthly_path",
        "scenario_annual_rebuild",
        "detail",
        "annex_summary",
        "department_summary",
        "total_summary",
    ]:
        _require_nonempty_df(outputs[k], f"simulation_outputs['{k}']")

    _require_columns(
        outputs["annual_delta"],
        ["Internal Tax Head", "Annex Tax Head", "Scenario Impact"],
        "annual_delta",
    )
    _require_columns(
        outputs["monthly_delta"],
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Monthly Delta"],
        "monthly_delta",
    )
    _require_columns(
        outputs["scenario_monthly_path"],
        ["Internal Tax Head", "Fiscal Year", "Month Index", "Scenario Monthly Forecast"],
        "scenario_monthly_path",
    )
    _require_columns(
        outputs["scenario_annual_rebuild"],
        ["Internal Tax Head", "Scenario Annual Rebuild"],
        "scenario_annual_rebuild",
    )

    validate_tax_engine_outputs({
        "detail": outputs["detail"],
        "annex_summary": outputs["annex_summary"],
        "department_summary": outputs["department_summary"],
        "total_summary": outputs["total_summary"],
    })

        # Consistency: compare only allocatable (non-bridge) heads
    bridge_heads = {"Import Excise Duty", "VAT, imports"}

    annual_delta = outputs["annual_delta"].copy()
    monthly_delta = outputs["monthly_delta"].copy()

    annual_delta["Internal Tax Head"] = annual_delta["Internal Tax Head"].astype(str).str.strip()
    monthly_delta["Internal Tax Head"] = monthly_delta["Internal Tax Head"].astype(str).str.strip()

    annual_delta_sum = pd.to_numeric(
        annual_delta.loc[~annual_delta["Internal Tax Head"].isin(bridge_heads), "Scenario Impact"],
        errors="coerce"
    ).fillna(0.0).sum()

    monthly_delta_sum = pd.to_numeric(
        monthly_delta.loc[~monthly_delta["Internal Tax Head"].isin(bridge_heads), "Monthly Delta"],
        errors="coerce"
    ).fillna(0.0).sum()

    if abs(annual_delta_sum - monthly_delta_sum) > 1e-4:
        raise ValidationEngineError(
            f"simulation annual delta vs monthly delta mismatch (non-bridge heads): {annual_delta_sum} != {monthly_delta_sum}"
        ) 


# ============================================================
# SCENARIO PACKAGE VALIDATION
# ============================================================

def validate_scenario_runner_package(package: Dict[str, Any]) -> None:
    required_keys = ["metadata", "baseline", "scenario", "comparisons"]
    for k in required_keys:
        _require_key(package, k, "scenario_runner_package")

    validate_rolling_outputs(package["baseline"])
    validate_simulation_outputs(package["scenario"])

    comps = package["comparisons"]
    for k in ["total_comparison", "tax_head_comparison", "annex_comparison", "top_gainers", "top_losers"]:
        _require_key(comps, k, "scenario_runner_package['comparisons']")
        _require_nonempty_df(comps[k], f"comparisons['{k}']")


# ============================================================
# DASHBOARD PACK VALIDATION
# ============================================================

def validate_dashboard_pack(pack: Dict[str, pd.DataFrame]) -> None:
    required_keys = [
        "executive_summary",
        "contribution_summary",
        "minister_brief",
        "total_comparison",
        "tax_head_comparison",
        "tax_head_summary",
        "annex_comparison",
        "annex_summary",
        "top_gainers",
        "top_losers",
        "monthly_total_comparison",
        "monthly_tax_head_comparison",
    ]
    for k in required_keys:
        _require_key(pack, k, "dashboard_pack")
        _require_nonempty_df(pack[k], f"dashboard_pack['{k}']")


# ============================================================
# DECOMPOSITION PACK VALIDATION
# ============================================================

def validate_decomposition_pack(pack: Dict[str, pd.DataFrame]) -> None:
    required_keys = [
        "scenario_decomposition",
        "scenario_decomposition_summary",
        "annex_decomposition",
        "monthly_total_decomposition",
        "monthly_tax_head_decomposition",
        "monthly_decomposition_summary",
    ]
    for k in required_keys:
        _require_key(pack, k, "decomposition_pack")
        _require_nonempty_df(pack[k], f"decomposition_pack['{k}']")


# ============================================================
# MASTER VALIDATION SHORTCUTS
# ============================================================

def run_full_output_validation(
    loaded_inputs: Optional[Dict[str, Any]] = None,
    tax_engine_outputs: Optional[Dict[str, pd.DataFrame]] = None,
    monthly_outputs: Optional[Dict[str, pd.DataFrame]] = None,
    rolling_outputs: Optional[Dict[str, Any]] = None,
    simulation_outputs: Optional[Dict[str, Any]] = None,
    scenario_runner_package: Optional[Dict[str, Any]] = None,
    dashboard_pack: Optional[Dict[str, pd.DataFrame]] = None,
    decomposition_pack: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    if loaded_inputs is not None:
        validate_loaded_inputs(loaded_inputs)

    if tax_engine_outputs is not None:
        validate_tax_engine_outputs(tax_engine_outputs)

    if monthly_outputs is not None:
        validate_monthly_pipeline_outputs(monthly_outputs)

    if rolling_outputs is not None:
        validate_rolling_outputs(rolling_outputs)

    if simulation_outputs is not None:
        validate_simulation_outputs(simulation_outputs)

    if scenario_runner_package is not None:
        validate_scenario_runner_package(scenario_runner_package)

    if dashboard_pack is not None:
        validate_dashboard_pack(dashboard_pack)

    if decomposition_pack is not None:
        validate_decomposition_pack(decomposition_pack)