# -*- coding: utf-8 -*-
"""
ROLLING LOADER V4
Workbook-governed loader for KRA forecasting model

Purpose
-------
- Make Control the governing entry sheet
- Read raw workbook inputs
- Derive dynamic monthly objects from raw monthly sheets
- Stop trusting workbook-derived sheets as primary runtime truth
- Preserve legacy objects for audit/reference where useful
"""


from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rolling_loader_v3 import (
    FORECAST_YEARS,
    RollingLoaderError,
    _bool_like,
    _clean,
    _find_first_existing_column,
    _read_key_value_sheet,
    _read_sheet,
    _require_columns,
    _standardize_columns,
    _to_num_series,
    load_control,
    load_elasticities,
    load_elasticities_dict,
    load_forecast_accuracy,
    load_macro,
    load_policy_measures,
    load_scenarios,
    load_targets,
    load_tax_heads_input,
    load_tax_heads_rolling,
)
from monthly_engine_v2 import (
    build_monthly_taxhead_actuals,
    compute_monthly_shares_from_collections,
    validate_monthly_collections,
    validate_monthly_mapping,
    validate_monthly_shares,
)
from validation_engine_v1 import validate_loaded_inputs


# ============================================================
# CONTROL GOVERNANCE
# ============================================================

def load_control_v4(file_path: str | Path) -> Dict[str, Any]:
    """
    Official governing entry sheet.

    Expected to contain at minimum:
    - Selected Year
    - Selected Scenario
    - Mode

    Preferred additional controls:
    - Current Fiscal Year
    - Reference Share Year
    - Scenario Allocation Mode
    - Scenario Duration Months
    - Lock Actual Months
    """
    df = _read_sheet(file_path, "Control")
    control_map = _read_key_value_sheet(df, "Control")

    selected_year = _clean(control_map.get("Selected Year", ""))
    scenario = _clean(control_map.get("Selected Scenario", "Baseline"))
    mode = _clean(control_map.get("Mode", "baseline")).lower()

    current_fiscal_year = (
        _clean(control_map.get("Current Fiscal Year", ""))
        or selected_year
        or "2025/26"
    )

    reference_share_year = (
        _clean(control_map.get("Reference Share Year", ""))
        or "2024/25"
    )

    scenario_allocation_mode = (
        _clean(control_map.get("Scenario Allocation Mode", "remaining_only")).lower()
        or "remaining_only"
    )

    scenario_duration_raw = control_map.get("Scenario Duration Months", None)
    scenario_duration_months: Optional[int]
    try:
        scenario_duration_months = int(pd.to_numeric(pd.Series([scenario_duration_raw]), errors="coerce").iloc[0])
        if pd.isna(scenario_duration_months):
            scenario_duration_months = None
    except Exception:
        scenario_duration_months = None

    lock_actual_months = _bool_like(control_map.get("Lock Actual Months", "Y"))

    return {
        "selected_year": selected_year,
        "scenario": scenario,
        "mode": mode,
        "current_fiscal_year": current_fiscal_year,
        "reference_share_year": reference_share_year,
        "scenario_allocation_mode": scenario_allocation_mode,
        "scenario_duration_months": scenario_duration_months,
        "lock_actual_months": lock_actual_months,
        "raw": control_map,
    }


# ============================================================
# RAW MONTHLY SHEETS
# ============================================================

def load_monthly_mapping_raw(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Monthly_Mapping"))

    required = [
        "Internal Tax Head",
        "Department",
        "Monthly Label",
        "Weight",
        "Sign",
        "Use_for_actual",
        "Use_for_share",
    ]
    _require_columns(df, required, "Monthly_Mapping")

    df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)
    df["Department"] = df["Department"].map(_clean)
    df["Monthly Label"] = df["Monthly Label"].map(_clean)

    df["Weight"] = _to_num_series(df["Weight"]).fillna(1.0)
    df["Sign"] = _to_num_series(df["Sign"]).fillna(1.0)

    df["Use_for_actual"] = df["Use_for_actual"].map(_bool_like)
    df["Use_for_share"] = df["Use_for_share"].map(_bool_like)

    validate_monthly_mapping(df)
    return df


def load_monthly_collections_normalized_raw(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Monthly_Collections_Normalized"))

    base_required = [
        "Department",
        "Monthly Label",
        "Fiscal Year",
        "Month Index",
        "Month Name",
    ]
    _require_columns(df, base_required, "Monthly_Collections_Normalized")

    value_col = _find_first_existing_column(
        df,
        ["Collection", "Monthly Collection", "Mapped Monthly Collection", "Value", "Amount", "Collections"],
        "Monthly_Collections_Normalized",
    )

    text_cols = ["Department", "Monthly Label", "Fiscal Year", "Month Name"]
    for c in text_cols:
        df[c] = df[c].map(_clean)

    df["Month Index"] = _to_num_series(df["Month Index"])
    df["Collection"] = _to_num_series(df[value_col]).fillna(0.0)

    out = df[["Department", "Monthly Label", "Fiscal Year", "Month Index", "Month Name", "Collection"]].copy()
    validate_monthly_collections(out)
    return out


# ============================================================
# DERIVED MONTHLY OBJECTS
# ============================================================

def derive_actual_months_loaded_from_raw(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    current_fiscal_year: str,
) -> int:
    """
    Derives loaded months from raw mapped actuals.

    Rule:
    - Build taxhead actuals for the current fiscal year without forced truncation
    - A month counts as loaded if mapped collections exist in that month
    - Return max loaded month index, else 0
    """
    mapped = build_monthly_taxhead_actuals(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        actual_months_loaded=None,
        fiscal_year=current_fiscal_year,
    )

    if mapped.empty:
        return 0

    work = mapped.copy()
    work["Month Index"] = pd.to_numeric(work["Month Index"], errors="coerce")
    work["Mapped Monthly Collection"] = pd.to_numeric(work["Mapped Monthly Collection"], errors="coerce").fillna(0.0)

    loaded = work.loc[work["Mapped Monthly Collection"] != 0.0, "Month Index"]
    if loaded.empty:
        return 0

    return int(loaded.max())


def derive_monthly_taxhead_actuals(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    actual_months_loaded: int,
    current_fiscal_year: str,
) -> pd.DataFrame:
    return build_monthly_taxhead_actuals(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        actual_months_loaded=actual_months_loaded,
        fiscal_year=current_fiscal_year,
    )


def derive_monthly_shares(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    reference_share_year: str,
) -> pd.DataFrame:
    shares = compute_monthly_shares_from_collections(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        reference_fiscal_year=reference_share_year,
    )
    validate_monthly_shares(shares)
    return shares


# ============================================================
# ROLLING CONTROL SYNTHESIS
# ============================================================

def build_rolling_control_v4(
    control: Dict[str, Any],
    actual_months_loaded: int,
) -> Dict[str, Any]:
    """
    Backward-compatible rolling control object used by downstream engines.
    """
    selected_year = control["selected_year"]
    if selected_year not in FORECAST_YEARS:
        raise RollingLoaderError(
            f"Selected Year '{selected_year}' invalid. Allowed: {FORECAST_YEARS}"
        )

    year_status_2025_26 = "OPEN" if selected_year == "2025/26" else "CLOSED"

    return {
        "selected_year": selected_year,
        "current_fiscal_year": control["current_fiscal_year"],
        "reference_share_year": control["reference_share_year"],
        "actual_months_loaded": int(actual_months_loaded),
        "year_status_2025_26": year_status_2025_26,
        "roll_mode": "rolling",
        "lock_actual_months": bool(control["lock_actual_months"]),
        "scenario_allocation_mode": control["scenario_allocation_mode"],
        "scenario_duration_months": control["scenario_duration_months"],
        "scenario": control["scenario"],
        "mode": control["mode"],
        "raw": control["raw"],
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_loaded_data_v4(data: Dict[str, Any]) -> None:
    rolling = data["rolling_control"]

    if rolling["selected_year"] not in FORECAST_YEARS:
        raise RollingLoaderError(
            f"Selected forecast year '{rolling['selected_year']}' is invalid. Allowed: {FORECAST_YEARS}"
        )

    months_loaded = int(rolling["actual_months_loaded"])
    if not (0 <= months_loaded <= 12):
        raise RollingLoaderError("Derived actual_months_loaded must be between 0 and 12.")

    if rolling["scenario_allocation_mode"] not in {"remaining_only", "full_year"}:
        raise RollingLoaderError(
            "Scenario Allocation Mode must be either 'remaining_only' or 'full_year'."
        )

    scenarios_df = data["scenarios"]
    if "Scenario" not in scenarios_df.columns:
        raise RollingLoaderError("CGE_Scenarios missing 'Scenario' column.")

    scenario_name = _clean(data["control"]["scenario"])
    available = set(scenarios_df["Scenario"].astype(str).str.strip().tolist())
    if scenario_name and scenario_name not in available:
        raise RollingLoaderError(
            f"Selected Scenario '{scenario_name}' not found in CGE_Scenarios. Available: {sorted(available)}"
        )


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def load_all_inputs(file_path: str | Path, validate: bool = True) -> Dict[str, Any]:
    file_path = Path(file_path).resolve()

    # --------------------------------------------------------
    # Official control
    # --------------------------------------------------------
    control = load_control_v4(file_path)

    # --------------------------------------------------------
    # Raw core sheets
    # --------------------------------------------------------
    macro = load_macro(file_path)
    elasticities = load_elasticities(file_path)
    elasticities_dict = load_elasticities_dict(file_path)
    targets = load_targets(file_path)
    scenarios = load_scenarios(file_path)
    tax_heads_input = load_tax_heads_input(file_path)
    policy_measures = load_policy_measures(file_path)
    tax_heads_rolling = load_tax_heads_rolling(file_path)
    forecast_accuracy_2025_26 = load_forecast_accuracy(file_path)

    # --------------------------------------------------------
    # Raw monthly sheets
    # --------------------------------------------------------
    monthly_collections_normalized = load_monthly_collections_normalized_raw(file_path)
    monthly_mapping = load_monthly_mapping_raw(file_path)

    # --------------------------------------------------------
    # Derived runtime monthly objects
    # --------------------------------------------------------
    actual_months_loaded = derive_actual_months_loaded_from_raw(
        monthly_collections_df=monthly_collections_normalized,
        monthly_mapping_df=monthly_mapping,
        current_fiscal_year=control["current_fiscal_year"],
    )

    monthly_taxhead_actuals = derive_monthly_taxhead_actuals(
        monthly_collections_df=monthly_collections_normalized,
        monthly_mapping_df=monthly_mapping,
        actual_months_loaded=actual_months_loaded,
        current_fiscal_year=control["current_fiscal_year"],
    )

    monthly_shares = derive_monthly_shares(
        monthly_collections_df=monthly_collections_normalized,
        monthly_mapping_df=monthly_mapping,
        reference_share_year=control["reference_share_year"],
    )

    # --------------------------------------------------------
    # Backward-compatible rolling control
    # --------------------------------------------------------
    rolling_control = build_rolling_control_v4(
        control=control,
        actual_months_loaded=actual_months_loaded,
    )

    data: Dict[str, Any] = {
        "control": control,
        "rolling_control": rolling_control,
        "macro": macro,
        "elasticities": elasticities,
        "elasticities_dict": elasticities_dict,
        "targets": targets,
        "scenarios": scenarios,
        "tax_heads_input": tax_heads_input,
        "policy_measures": policy_measures,
        "tax_heads_rolling": tax_heads_rolling,
        "forecast_accuracy_2025_26": forecast_accuracy_2025_26,
        "monthly_collections_normalized": monthly_collections_normalized,
        "monthly_mapping": monthly_mapping,
        "monthly_taxhead_actuals": monthly_taxhead_actuals,
        "monthly_shares": monthly_shares,
    }

    if validate:
        validate_loaded_data_v4(data)
        validate_loaded_inputs(data)

    return data


if __name__ == "__main__":
    workbook = Path("kra_forecast_input_template_final.xlsx").resolve()
    loaded = load_all_inputs(workbook, validate=False)

    print("=" * 90)
    print("ROLLING LOADER V4 TEST")
    print("=" * 90)
    print("Workbook:", workbook)
    print("Selected year:", loaded["rolling_control"]["selected_year"])
    print("Selected scenario:", loaded["control"]["scenario"])
    print("Current fiscal year:", loaded["rolling_control"]["current_fiscal_year"])
    print("Reference share year:", loaded["rolling_control"]["reference_share_year"])
    print("Derived actual months loaded:", loaded["rolling_control"]["actual_months_loaded"])
    print("Monthly taxhead actuals rows:", len(loaded["monthly_taxhead_actuals"]))
    print("Monthly shares rows:", len(loaded["monthly_shares"]))