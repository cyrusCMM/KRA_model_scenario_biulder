# -*- coding: utf-8 -*-
"""
ROLLING LOADER V4
Working fallback-safe loader for KRA forecasting model
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class RollingLoaderError(Exception):
    """Raised when the rolling input workbook cannot be read consistently."""
    pass


FORECAST_YEARS = ["2025/26", "2026/27", "2027/28"]
FISCAL_MONTHS = [
    (1, "Jul"), (2, "Aug"), (3, "Sep"), (4, "Oct"), (5, "Nov"), (6, "Dec"),
    (7, "Jan"), (8, "Feb"), (9, "Mar"), (10, "Apr"), (11, "May"), (12, "Jun"),
]


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _to_num_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool_like(x: Any) -> bool:
    s = _clean(x).lower()
    return s in {"1", "true", "t", "yes", "y"}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_clean(c) for c in out.columns]
    return out


def _sheet_exists(file_path: str | Path, sheet_name: str) -> bool:
    xl = pd.ExcelFile(file_path)
    return sheet_name in xl.sheet_names


def _read_sheet(file_path: str | Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError as exc:
        raise RollingLoaderError(f"Sheet '{sheet_name}' not found in workbook.") from exc
    except FileNotFoundError as exc:
        raise RollingLoaderError(f"Workbook not found: {file_path}") from exc


def _require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RollingLoaderError(f"{label} missing required columns: {missing}")


def _find_first_existing_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RollingLoaderError(f"{label} missing required column. Expected one of: {candidates}")


def _read_key_value_sheet(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    df = _standardize_columns(df)

    if {"Control Item", "Value"}.issubset(df.columns):
        key_col = "Control Item"
        val_col = "Value"
    elif {"Setting", "Value"}.issubset(df.columns):
        key_col = "Setting"
        val_col = "Value"
    elif df.shape[1] >= 2:
        key_col = df.columns[0]
        val_col = df.columns[1]
    else:
        raise RollingLoaderError(f"{label} must contain at least two columns.")

    return dict(zip(df[key_col].map(_clean), df[val_col]))


# ============================================================
# CONTROL
# ============================================================

def load_control_v4(file_path: str | Path) -> Dict[str, Any]:
    df = _read_sheet(file_path, "Control")
    control_map = _read_key_value_sheet(df, "Control")

    selected_year = _clean(control_map.get("Selected Year", "2025/26")) or "2025/26"
    scenario = _clean(control_map.get("Selected Scenario", "Baseline")) or "Baseline"
    mode = _clean(control_map.get("Mode", "baseline")).lower() or "baseline"

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
    try:
        temp = pd.to_numeric(pd.Series([scenario_duration_raw]), errors="coerce").iloc[0]
        scenario_duration_months = None if pd.isna(temp) else int(temp)
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
# CORE SHEETS
# ============================================================

def load_macro(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Macro_Baseline"))
    _require_columns(df, ["Variable"], "Macro_Baseline")

    for y in FORECAST_YEARS:
        if y not in df.columns:
            raise RollingLoaderError(f"Macro_Baseline missing forecast year column '{y}'.")

    df["Variable"] = df["Variable"].map(_clean)

    for y in FORECAST_YEARS:
        df[y] = _to_num_series(df[y])

    return df


def load_elasticities(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Elasticities"))
    _require_columns(df, ["Parameter", "Value"], "Elasticities")

    df["Parameter"] = df["Parameter"].map(_clean)
    df["Value"] = _to_num_series(df["Value"])
    return df


def load_elasticities_dict(file_path: str | Path) -> Dict[str, float]:
    df = load_elasticities(file_path)
    return {
        _clean(r["Parameter"]): 0.0 if pd.isna(r["Value"]) else float(r["Value"])
        for _, r in df.iterrows()
    }


def load_targets(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Targets"))

    if "Annex Tax Head" in df.columns:
        df["Annex Tax Head"] = df["Annex Tax Head"].map(_clean)

    for col in ["Target 2025/26", "Target 2026/27", "Target 2027/28"]:
        if col in df.columns:
            df[col] = _to_num_series(df[col])

    return df


def load_scenarios(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "CGE_Scenarios"))
    _require_columns(df, ["Scenario"], "CGE_Scenarios")

    df["Scenario"] = df["Scenario"].map(_clean)
    for c in df.columns:
        if c != "Scenario":
            df[c] = _to_num_series(df[c]).fillna(0.0)

    return df


def load_tax_heads_input(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Tax_Heads_Input"))

    required = [
        "Department",
        "Revenue Type",
        "Internal Tax Head",
        "Annex Tax Head",
        "Actual 2024/25",
    ]
    _require_columns(df, required, "Tax_Heads_Input")

    for c in df.columns:
        if c != "Actual 2024/25":
            df[c] = df[c].map(_clean)

    df["Actual 2024/25"] = _to_num_series(df["Actual 2024/25"]).fillna(0.0)
    return df


def load_policy_measures(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Policy_Measures"))

    if "Internal Tax Head" not in df.columns:
        raise RollingLoaderError("Policy_Measures missing 'Internal Tax Head' column.")

    df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)

    for y in FORECAST_YEARS:
        fa = _to_num_series(df.get(f"FA {y}", pd.Series([0] * len(df)))).fillna(0.0)
        admin = _to_num_series(df.get(f"Admin {y}", pd.Series([0] * len(df)))).fillna(0.0)
        eaccma = _to_num_series(df.get(f"EACCMA {y}", pd.Series([0] * len(df)))).fillna(0.0)
        df[f"Policy Total {y}"] = fa + admin + eaccma

    return df


# ============================================================
# FALLBACK BUILDERS
# ============================================================

def build_tax_heads_rolling_fallback(tax_heads_input: pd.DataFrame) -> pd.DataFrame:
    """
    Safe fallback when Tax_Heads_Rolling is missing.
    Uses Actual 2024/25 as the opening base for all years.
    """
    df = tax_heads_input.copy()

    out = pd.DataFrame({
        "Internal Tax Head": df["Internal Tax Head"].map(_clean),
        "Annex Tax Head": df["Annex Tax Head"].map(_clean),
        "Actual 2024/25": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
        "Opening Base 2025/26": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
        "Formula Forecast 2025/26": 0.0,
        "Final 2025/26": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
        "Opening Base 2026/27": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
        "Formula Forecast 2026/27": 0.0,
        "Final 2026/27": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
        "Opening Base 2027/28": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
        "Formula Forecast 2027/28": 0.0,
        "Final 2027/28": pd.to_numeric(df["Actual 2024/25"], errors="coerce").fillna(0.0),
    })

    return out


def load_tax_heads_rolling(file_path: str | Path, tax_heads_input: pd.DataFrame) -> pd.DataFrame:
    if not _sheet_exists(file_path, "Tax_Heads_Rolling"):
        return build_tax_heads_rolling_fallback(tax_heads_input)

    df = _standardize_columns(_read_sheet(file_path, "Tax_Heads_Rolling"))

    required = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Opening Base 2025/26",
        "Opening Base 2026/27",
        "Opening Base 2027/28",
    ]
    _require_columns(df, required, "Tax_Heads_Rolling")

    text_cols = ["Internal Tax Head", "Annex Tax Head", "Department", "Revenue Type"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(_clean)

    for c in df.columns:
        if c not in text_cols:
            df[c] = _to_num_series(df[c])

    return df


def build_monthly_mapping_fallback(tax_heads_input: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tax_heads_input.iterrows():
        head = _clean(r["Internal Tax Head"])
        dept = _clean(r["Department"])
        rows.append({
            "Internal Tax Head": head,
            "Department": dept,
            "Monthly Label": head,
            "Weight": 1.0,
            "Sign": 1.0,
            "Use_for_actual": True,
            "Use_for_share": True,
        })
    return pd.DataFrame(rows)


def load_monthly_mapping_raw(file_path: str | Path, tax_heads_input: pd.DataFrame) -> pd.DataFrame:
    if not _sheet_exists(file_path, "Monthly_Mapping"):
        return build_monthly_mapping_fallback(tax_heads_input)

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

    return df


def build_monthly_collections_fallback(
    tax_heads_input: pd.DataFrame,
    reference_share_year: str = "2024/25",
    current_fiscal_year: str = "2025/26",
) -> pd.DataFrame:
    """
    Builds synthetic monthly collections using equal shares from Actual 2024/25.
    This keeps the monthly engine runnable.
    """
    rows = []

    for _, r in tax_heads_input.iterrows():
        dept = _clean(r["Department"])
        head = _clean(r["Internal Tax Head"])
        annual = float(pd.to_numeric(pd.Series([r["Actual 2024/25"]]), errors="coerce").fillna(0.0).iloc[0])
        monthly = annual / 12.0

        for year in [reference_share_year, current_fiscal_year]:
            for month_index, month_name in FISCAL_MONTHS:
                rows.append({
                    "Department": dept,
                    "Monthly Label": head,
                    "Fiscal Year": year,
                    "Month Index": month_index,
                    "Month Name": month_name,
                    "Collection": monthly if year == reference_share_year else 0.0,
                })

    return pd.DataFrame(rows)


def load_monthly_collections_normalized_raw(
    file_path: str | Path,
    tax_heads_input: pd.DataFrame,
    reference_share_year: str,
    current_fiscal_year: str,
) -> pd.DataFrame:
    if not _sheet_exists(file_path, "Monthly_Collections_Normalized"):
        return build_monthly_collections_fallback(
            tax_heads_input=tax_heads_input,
            reference_share_year=reference_share_year,
            current_fiscal_year=current_fiscal_year,
        )

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

    for c in ["Department", "Monthly Label", "Fiscal Year", "Month Name"]:
        df[c] = df[c].map(_clean)

    df["Month Index"] = _to_num_series(df["Month Index"])
    df["Collection"] = _to_num_series(df[value_col]).fillna(0.0)

    return df[["Department", "Monthly Label", "Fiscal Year", "Month Index", "Month Name", "Collection"]].copy()


def build_forecast_accuracy_fallback(tax_heads_input: pd.DataFrame) -> pd.DataFrame:
    actual = pd.to_numeric(tax_heads_input["Actual 2024/25"], errors="coerce").fillna(0.0)
    heads = tax_heads_input["Internal Tax Head"].map(_clean)

    return pd.DataFrame({
        "Internal Tax Head": heads,
        "Year Status": "OPEN",
        "Forecast Final 2025/26": actual,
        "Frozen Forecast for Accuracy": actual,
        "Actual Final 2025/26": actual,
        "Forecast Error": 0.0,
        "APE": 0.0,
        "PE": 0.0,
        "Opening Base Applied 2026/27": actual,
        "Base Source 2026/27": "fallback_actual_2024_25",
        "Accuracy Ready?": "N",
    })


def load_forecast_accuracy(file_path: str | Path, tax_heads_input: pd.DataFrame) -> pd.DataFrame:
    if not _sheet_exists(file_path, "Forecast_Accuracy_2025_26"):
        return build_forecast_accuracy_fallback(tax_heads_input)

    df = _standardize_columns(_read_sheet(file_path, "Forecast_Accuracy_2025_26"))
    if "Internal Tax Head" in df.columns:
        df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)
    return df


# ============================================================
# DERIVED MONTHLY OBJECTS
# ============================================================

def derive_actual_months_loaded_from_raw(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    current_fiscal_year: str,
) -> int:
    if monthly_collections_df.empty or monthly_mapping_df.empty:
        return 0

    valid_labels = set(
        monthly_mapping_df.loc[monthly_mapping_df["Use_for_actual"], "Monthly Label"]
        .astype(str).str.strip().tolist()
    )

    work = monthly_collections_df.copy()
    work["Fiscal Year"] = work["Fiscal Year"].map(_clean)
    work["Monthly Label"] = work["Monthly Label"].map(_clean)
    work["Month Index"] = pd.to_numeric(work["Month Index"], errors="coerce")
    work["Collection"] = pd.to_numeric(work["Collection"], errors="coerce").fillna(0.0)

    work = work.loc[
        (work["Fiscal Year"] == _clean(current_fiscal_year))
        & (work["Monthly Label"].isin(valid_labels))
        & (work["Collection"] != 0.0)
    ].copy()

    if work.empty:
        return 0

    return int(work["Month Index"].max())


def derive_monthly_taxhead_actuals(
    monthly_collections_df: pd.DataFrame,
    monthly_mapping_df: pd.DataFrame,
    actual_months_loaded: int,
    current_fiscal_year: str,
) -> pd.DataFrame:
    from monthly_engine_v2 import build_monthly_taxhead_actuals
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
    from monthly_engine_v2 import compute_monthly_shares_from_collections
    return compute_monthly_shares_from_collections(
        monthly_collections_df=monthly_collections_df,
        monthly_mapping_df=monthly_mapping_df,
        reference_fiscal_year=reference_share_year,
    )


# ============================================================
# ROLLING CONTROL
# ============================================================

def build_rolling_control_v4(control: Dict[str, Any], actual_months_loaded: int) -> Dict[str, Any]:
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

    scenario_name = _clean(data["control"]["scenario"])
    available = set(data["scenarios"]["Scenario"].astype(str).str.strip().tolist())
    if scenario_name and scenario_name not in available:
        raise RollingLoaderError(
            f"Selected Scenario '{scenario_name}' not found in CGE_Scenarios. Available: {sorted(available)}"
        )


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def load_all_inputs(file_path: str | Path, validate: bool = True) -> Dict[str, Any]:
    file_path = Path(file_path).resolve()

    control = load_control_v4(file_path)

    macro = load_macro(file_path)
    elasticities = load_elasticities(file_path)
    elasticities_dict = load_elasticities_dict(file_path)
    targets = load_targets(file_path)
    scenarios = load_scenarios(file_path)
    tax_heads_input = load_tax_heads_input(file_path)
    policy_measures = load_policy_measures(file_path)

    tax_heads_rolling = load_tax_heads_rolling(file_path, tax_heads_input)
    forecast_accuracy_2025_26 = load_forecast_accuracy(file_path, tax_heads_input)

    monthly_mapping = load_monthly_mapping_raw(file_path, tax_heads_input)
    monthly_collections_normalized = load_monthly_collections_normalized_raw(
        file_path=file_path,
        tax_heads_input=tax_heads_input,
        reference_share_year=control["reference_share_year"],
        current_fiscal_year=control["current_fiscal_year"],
    )

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

    return data


if __name__ == "__main__":
    workbook = Path("kra_forecast_input_template_final.xlsx").resolve()
    loaded = load_all_inputs(workbook, validate=True)

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