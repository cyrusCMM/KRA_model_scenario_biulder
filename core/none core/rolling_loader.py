from __future__ import annotations

# ============================================================
# ROLLING LOADER V2
# Fresh full loader written against the final rolling workbook
# Saved in same folder as the workbook for direct testing
# ============================================================

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class RollingLoaderError(Exception):
    """Raised when the rolling input workbook cannot be read consistently."""
    pass


# ============================================================
# CONSTANTS
# ============================================================

FORECAST_YEARS = ["2025/26", "2026/27", "2027/28"]


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


# ============================================================
# CONTROL SHEETS
# ============================================================

def load_control(file_path: str | Path) -> Dict[str, Any]:
    df = _standardize_columns(_read_sheet(file_path, "Control"))
    _require_columns(df, ["Control Item", "Value"], "Control")

    control_map = dict(zip(df["Control Item"].map(_clean), df["Value"]))

    return {
        "selected_year": _clean(control_map.get("Selected Year", "")),
        "scenario": _clean(control_map.get("Selected Scenario", "")),
        "mode": _clean(control_map.get("Mode", "")).lower(),
        "raw": control_map,
    }


def load_rolling_control(file_path: str | Path) -> Dict[str, Any]:
    df = _standardize_columns(_read_sheet(file_path, "Rolling_Control"))

    if {"Control Item", "Value"}.issubset(df.columns):
        key_col = "Control Item"
        val_col = "Value"
    elif df.shape[1] >= 2:
        key_col = df.columns[0]
        val_col = df.columns[1]
    else:
        raise RollingLoaderError("Rolling_Control sheet must contain at least two columns.")

    rc = dict(zip(df[key_col].map(_clean), df[val_col]))

    selected_year = _clean(rc.get("Selected Forecast Year", "")) or _clean(rc.get("Selected Year", ""))
    current_fiscal_year = _clean(rc.get("Current Fiscal Year", "")) or selected_year
    reference_share_year = _clean(rc.get("Reference Year for Shares", "")) or "2024/25"

    actual_months_loaded_raw = rc.get("Actual Months Loaded", rc.get("Current Fiscal Month", ""))
    actual_months_loaded = pd.to_numeric(pd.Series([actual_months_loaded_raw]), errors="coerce").iloc[0]
    if pd.isna(actual_months_loaded):
        actual_months_loaded = 0
    actual_months_loaded = int(actual_months_loaded)

    year_status_2025_26 = _clean(
        rc.get("2025/26 Year Status", rc.get("Year Status", "OPEN"))
    ).upper()

    roll_mode = _clean(rc.get("Roll Mode", "rolling")).lower()
    lock_actual_months = _bool_like(rc.get("Lock Actual Months", "Y"))
    scenario_allocation_mode = _clean(
        rc.get("Scenario Allocation Mode", "remaining_only")
    ).lower()

    return {
        "selected_year": selected_year,
        "current_fiscal_year": current_fiscal_year,
        "reference_share_year": reference_share_year,
        "actual_months_loaded": actual_months_loaded,
        "year_status_2025_26": year_status_2025_26,
        "roll_mode": roll_mode,
        "lock_actual_months": lock_actual_months,
        "scenario_allocation_mode": scenario_allocation_mode,
        "raw": rc,
    }


# ============================================================
# MACRO, ELASTICITIES, TARGETS, SCENARIOS
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


def load_macro_dict(file_path: str | Path) -> Dict[str, Dict[str, float]]:
    df = load_macro(file_path)
    result: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        var = _clean(row["Variable"])
        result[var] = {}
        for y in FORECAST_YEARS:
            val = pd.to_numeric(pd.Series([row[y]]), errors="coerce").iloc[0]
            result[var][y] = 0.0 if pd.isna(val) else float(val)

    return result


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

    if "Scenario" in df.columns:
        df["Scenario"] = df["Scenario"].map(_clean)

    for c in df.columns:
        if c != "Scenario":
            df[c] = _to_num_series(df[c]).fillna(0.0)

    return df


# ============================================================
# TAX HEADS AND POLICY
# ============================================================

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

    text_cols = [
        "Department",
        "Revenue Type",
        "Internal Tax Head",
        "Annex Tax Head",
        "Formula Type",
        "Driver 1",
        "Elasticity 1",
        "Driver 2",
        "Elasticity 2",
        "Driver 3",
        "Elasticity 3",
        "Post Multiplier Driver",
        "Post Multiplier Elasticity",
        "Additive Driver",
        "Additive Elasticity",
        "Improved Driver 1",
        "Improved Elasticity 1",
        "Improved Driver 2",
        "Improved Elasticity 2",
    ]

    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(_clean)

    df["Actual 2024/25"] = _to_num_series(df["Actual 2024/25"])

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


def load_tax_heads_rolling(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Tax_Heads_Rolling"))

    required = [
        "Internal Tax Head",
        "Annex Tax Head",
        "Actual 2024/25",
        "Opening Base 2025/26",
        "Formula Forecast 2025/26",
        "Final 2025/26",
        "Opening Base 2026/27",
        "Formula Forecast 2026/27",
        "Final 2026/27",
        "Opening Base 2027/28",
        "Formula Forecast 2027/28",
        "Final 2027/28",
    ]
    _require_columns(df, required, "Tax_Heads_Rolling")

    text_cols = [
        "Department",
        "Revenue Type",
        "Internal Tax Head",
        "Annex Tax Head",
        "Base Source 2026/27",
        "Base Source 2027/28",
    ]

    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].map(_clean)

    numeric_cols = [c for c in df.columns if c not in text_cols]
    for c in numeric_cols:
        df[c] = _to_num_series(df[c])

    return df


# ============================================================
# MONTHLY STRUCTURE
# ============================================================

def load_monthly_mapping(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Monthly_Mapping"))

    required = [
        "Internal Tax Head",
        "Department",
        "Monthly Label",
        "Weight",
        "Sign",
        "Use for actual",
        "Use for share",
    ]
    _require_columns(df, required, "Monthly_Mapping")

    df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)
    df["Department"] = df["Department"].map(_clean)
    df["Monthly Label"] = df["Monthly Label"].map(_clean)

    df["Weight"] = _to_num_series(df["Weight"]).fillna(1.0)
    df["Sign"] = _to_num_series(df["Sign"]).fillna(1.0)

    df["Use_for_actual"] = df["Use for actual"].map(_bool_like)
    df["Use_for_share"] = df["Use for share"].map(_bool_like)

    return df


def load_monthly_collections_normalized(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Monthly_Collections_Normalized"))

    base_required = ["Department", "Monthly Label", "Fiscal Year", "Month Index", "Month Name"]
    _require_columns(df, base_required, "Monthly_Collections_Normalized")

    value_candidates = [
        "Collection",
        "Monthly Collection",
        "Mapped Monthly Collection",
        "Value",
        "Amount",
        "Collections",
    ]
    value_col = _find_first_existing_column(
        df, value_candidates, "Monthly_Collections_Normalized"
    )

    text_cols = ["Department", "Monthly Label", "Fiscal Year", "Month Name"]
    for c in text_cols:
        df[c] = df[c].map(_clean)

    df["Month Index"] = _to_num_series(df["Month Index"])
    df["Collection"] = _to_num_series(df[value_col]).fillna(0.0)

    return df[
        ["Department", "Monthly Label", "Fiscal Year", "Month Index", "Month Name", "Collection"]
    ].copy()


def load_monthly_taxhead_actuals(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Monthly_TaxHead_Actuals"))

    required = [
        "Internal Tax Head",
        "Fiscal Year",
        "Month Index",
        "Month Name",
        "Mapped Monthly Collection",
        "Load Flag",
    ]
    _require_columns(df, required, "Monthly_TaxHead_Actuals")

    df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)
    df["Fiscal Year"] = df["Fiscal Year"].map(_clean)
    df["Month Name"] = df["Month Name"].map(_clean)

    df["Month Index"] = _to_num_series(df["Month Index"])
    df["Mapped Monthly Collection"] = _to_num_series(df["Mapped Monthly Collection"]).fillna(0.0)
    df["Load Flag"] = _to_num_series(df["Load Flag"]).fillna(0.0).astype(int)

    return df


def load_monthly_shares(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Monthly_Shares_2024_25"))

    required = [
        "Internal Tax Head",
        "Reference Fiscal Year",
        "Month Index",
        "Month Name",
        "Reference Monthly Value",
        "Reference Annual Total",
        "Monthly Share",
    ]
    _require_columns(df, required, "Monthly_Shares_2024_25")

    df["Internal Tax Head"] = df["Internal Tax Head"].map(_clean)
    df["Reference Fiscal Year"] = df["Reference Fiscal Year"].map(_clean)
    df["Month Name"] = df["Month Name"].map(_clean)

    for c in ["Month Index", "Reference Monthly Value", "Reference Annual Total", "Monthly Share"]:
        df[c] = _to_num_series(df[c])

    return df


# ============================================================
# FORECAST ACCURACY
# ============================================================

def load_forecast_accuracy(file_path: str | Path) -> pd.DataFrame:
    df = _standardize_columns(_read_sheet(file_path, "Forecast_Accuracy_2025_26"))

    required = [
        "Internal Tax Head",
        "Year Status",
        "Forecast Final 2025/26",
        "Frozen Forecast for Accuracy",
        "Actual Final 2025/26",
        "Forecast Error",
        "APE",
        "PE",
        "Opening Base Applied 2026/27",
        "Base Source 2026/27",
        "Accuracy Ready?",
    ]
    _require_columns(df, required, "Forecast_Accuracy_2025_26")

    text_cols = ["Internal Tax Head", "Year Status", "Base Source 2026/27", "Accuracy Ready?"]
    for c in text_cols:
        df[c] = df[c].map(_clean)

    numeric_cols = [c for c in df.columns if c not in text_cols]
    for c in numeric_cols:
        df[c] = _to_num_series(df[c])

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_loaded_data(data: Dict[str, Any]) -> None:
    rolling = data["rolling_control"]

    selected_year = rolling["selected_year"]
    if selected_year not in FORECAST_YEARS:
        raise RollingLoaderError(
            f"Selected forecast year '{selected_year}' is invalid. Allowed: {FORECAST_YEARS}"
        )

    months_loaded = rolling["actual_months_loaded"]
    if not (0 <= months_loaded <= 12):
        raise RollingLoaderError("Actual Months Loaded must lie between 0 and 12.")

    year_status = rolling["year_status_2025_26"]
    if year_status not in {"OPEN", "CLOSED"}:
        raise RollingLoaderError("2025/26 Year Status must be either 'OPEN' or 'CLOSED'.")

    tax_heads = data["tax_heads_input"]
    if tax_heads["Internal Tax Head"].duplicated().any():
        dups = tax_heads.loc[tax_heads["Internal Tax Head"].duplicated(), "Internal Tax Head"].unique().tolist()
        raise RollingLoaderError(f"Tax_Heads_Input contains duplicate Internal Tax Head values: {dups}")

    mapping = data["monthly_mapping"]
    active_mapping = mapping.loc[mapping["Use_for_actual"] | mapping["Use_for_share"]].copy()
    dup_labels = active_mapping["Monthly Label"].duplicated()
    if dup_labels.any():
        dups = active_mapping.loc[dup_labels, "Monthly Label"].unique().tolist()
        raise RollingLoaderError(f"Monthly_Mapping has duplicated active Monthly Label values: {dups}")

    shares = data["monthly_shares"]
    if not shares.empty:
        share_sums = (
            shares.groupby(["Internal Tax Head", "Reference Fiscal Year"], as_index=False)["Monthly Share"]
            .sum()
        )
        bad = share_sums.loc[(share_sums["Monthly Share"] - 1.0).abs() > 1e-4]
        if not bad.empty:
            pairs = bad[["Internal Tax Head", "Reference Fiscal Year", "Monthly Share"]].to_dict(orient="records")
            raise RollingLoaderError(f"Monthly share sums do not equal 1 for some tax heads: {pairs}")

    rolling_table = data["tax_heads_rolling"]
    if not rolling_table.empty:
        if "Opening Base 2026/27" in rolling_table.columns and "Final 2025/26" in rolling_table.columns:
            check_1 = (rolling_table["Opening Base 2026/27"] - rolling_table["Final 2025/26"]).abs()
            bad_1 = rolling_table.loc[check_1 > 1e-4]
            if not bad_1.empty:
                raise RollingLoaderError("Tax_Heads_Rolling fails 2026/27 roll identity for some tax heads.")

        if "Opening Base 2027/28" in rolling_table.columns and "Final 2026/27" in rolling_table.columns:
            check_2 = (rolling_table["Opening Base 2027/28"] - rolling_table["Final 2026/27"]).abs()
            bad_2 = rolling_table.loc[check_2 > 1e-4]
            if not bad_2.empty:
                raise RollingLoaderError("Tax_Heads_Rolling fails 2027/28 roll identity for some tax heads.")


# ============================================================
# MASTER ENTRY POINT
# ============================================================

def load_all_inputs(file_path: str | Path, validate: bool = True) -> Dict[str, Any]:
    file_path = Path(file_path).resolve()

    data: Dict[str, Any] = {
        "file_path": str(file_path),
        "control": load_control(file_path),
        "rolling_control": load_rolling_control(file_path),
        "macro": load_macro(file_path),
        "macro_dict": load_macro_dict(file_path),
        "elasticities": load_elasticities(file_path),
        "elasticities_dict": load_elasticities_dict(file_path),
        "tax_heads_input": load_tax_heads_input(file_path),
        "policy_measures": load_policy_measures(file_path),
        "targets": load_targets(file_path),
        "scenarios": load_scenarios(file_path),
        "monthly_collections_normalized": load_monthly_collections_normalized(file_path),
        "monthly_mapping": load_monthly_mapping(file_path),
        "monthly_taxhead_actuals": load_monthly_taxhead_actuals(file_path),
        "monthly_shares": load_monthly_shares(file_path),
        "tax_heads_rolling": load_tax_heads_rolling(file_path),
        "forecast_accuracy_2025_26": load_forecast_accuracy(file_path),
    }

    if validate:
        validate_loaded_data(data)

    return data


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).resolve().parent
    TEST_FILE = CURRENT_DIR / "kra_forecast_input_template_final_for_python.xlsx"

    print("=" * 90)
    print("ROLLING LOADER V2 TEST")
    print("=" * 90)
    print(f"Workbook: {TEST_FILE}")
    print(f"Exists: {TEST_FILE.exists()}")

    try:
        loaded = load_all_inputs(TEST_FILE, validate=True)

        print("\n[1] CONTROL")
        print("Selected Year:", loaded["rolling_control"]["selected_year"])
        print("Current Fiscal Year:", loaded["rolling_control"]["current_fiscal_year"])
        print("Reference Share Year:", loaded["rolling_control"]["reference_share_year"])
        print("Actual Months Loaded:", loaded["rolling_control"]["actual_months_loaded"])
        print("2025/26 Year Status:", loaded["rolling_control"]["year_status_2025_26"])

        print("\n[2] TABLE SHAPES")
        for key in [
            "macro",
            "elasticities",
            "tax_heads_input",
            "policy_measures",
            "targets",
            "scenarios",
            "monthly_collections_normalized",
            "monthly_mapping",
            "monthly_taxhead_actuals",
            "monthly_shares",
            "tax_heads_rolling",
            "forecast_accuracy_2025_26",
        ]:
            obj = loaded[key]
            if isinstance(obj, pd.DataFrame):
                print(f"{key:30s} -> {obj.shape}")

        print("\n[3] QUICK CHECKS")
        print("Tax heads loaded:", loaded["tax_heads_input"]["Internal Tax Head"].nunique())
        print("Monthly mapping rows:", len(loaded["monthly_mapping"]))
        print("Monthly share rows:", len(loaded["monthly_shares"]))

        share_check = (
            loaded["monthly_shares"]
            .groupby(["Internal Tax Head", "Reference Fiscal Year"])["Monthly Share"]
            .sum()
            .reset_index()
        )
        print("Unique share sums:", sorted(share_check["Monthly Share"].round(6).unique().tolist())[:10])

        print("\nROLLING LOADER V2 TEST PASSED.")

    except Exception:
        print("\nROLLING LOADER V2 TEST FAILED.")
        raise