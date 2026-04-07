# -*- coding: utf-8 -*-
"""
SCENARIO BUILDER V5
Stable scenario construction for KRA model.

Responsibilities
----------------
- Read CGE_Scenarios from loaded inputs
- Select workbook scenario
- Optionally apply UI overrides
- Build baseline macro table
- Apply additive macro shocks for the selected year only
- Return clean scenario metadata for downstream engines
- Respect duration override passed from app/Streamlit via rolling_control
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from macro_identities import build_macro_driver_table


class ScenarioBuilderError(Exception):
    """Raised when scenario shock construction fails."""
    pass


# ============================================================
# HELPERS
# ============================================================

def _clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _clean_text(x: Any) -> str:
    """
    Keep metadata fields as text.
    Prevent junk values like '0.0', 'nan', 'None' from leaking through.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    if s in {"0.0", "0"}:
        return ""
    return s


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if pd.notna(v) else default
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _to_bool_yes_no(x: Any, default: bool = False) -> bool:
    s = _clean_text(x).lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return default


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ScenarioBuilderError(f"{label} is missing or empty.")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ScenarioBuilderError(f"{label} missing required columns: {missing}")


def _get_selected_year(data: Dict[str, Any]) -> str:
    year = _clean(data.get("rolling_control", {}).get("selected_year", ""))
    if year not in {"2025/26", "2026/27", "2027/28"}:
        raise ScenarioBuilderError(
            f"Invalid selected year '{year}'. Expected one of: 2025/26, 2026/27, 2027/28."
        )
    return year


def _get_selected_scenario(data: Dict[str, Any]) -> str:
    return _clean(data.get("control", {}).get("scenario", "Baseline")) or "Baseline"


def _get_duration_override_from_controls(data: Dict[str, Any]) -> Optional[int]:
    rolling_control = data.get("rolling_control", {})
    raw = rolling_control.get("scenario_duration_months", None)
    if raw is None:
        return None
    try:
        val = int(float(raw))
        return val if val >= 0 else None
    except Exception:
        return None


# ============================================================
# SCENARIO ROW
# ============================================================

def _get_scenario_row(
    scenarios_df: pd.DataFrame,
    selected_scenario: str,
    ui_override_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    work = scenarios_df.copy()
    work.columns = [_clean(c) for c in work.columns]

    _require_columns(work, ["Scenario"], "CGE_Scenarios")

    work["Scenario"] = work["Scenario"].astype(str).str.strip()
    row = work.loc[work["Scenario"] == selected_scenario]

    if row.empty:
        available = sorted(work["Scenario"].dropna().astype(str).str.strip().unique().tolist())
        raise ScenarioBuilderError(
            f"Selected Scenario '{selected_scenario}' not found in CGE_Scenarios. Available: {available}"
        )

    out = row.iloc[0].copy()

    if ui_override_df is not None:
        if not isinstance(ui_override_df, pd.DataFrame) or ui_override_df.empty:
            raise ScenarioBuilderError("ui_override_df was supplied but is empty or invalid.")

        override = ui_override_df.iloc[0].to_dict()
        for k, v in override.items():
            k_clean = _clean(k)
            if k_clean != "" and pd.notna(v):
                out[k_clean] = v

    return out


# ============================================================
# SCENARIO METADATA
# ============================================================

def _extract_scenario_metadata(
    scenario_row: pd.Series,
    selected_scenario: str = "",
    duration_override: Optional[int] = None,
) -> Dict[str, Any]:
    def text_or_blank(*keys: str) -> str:
        for k in keys:
            if k in scenario_row.index:
                v = _clean_text(scenario_row.get(k, ""))
                if v != "":
                    return v
        return ""

    start_month = _to_int(scenario_row.get("Start Month", 10), 10)
    duration_months = _to_int(scenario_row.get("Duration Months", 0), 0)

    # IMPORTANT: UI/app override wins over sheet value
    if duration_override is not None:
        duration_months = int(duration_override)

    carryover = _to_bool_yes_no(scenario_row.get("Carryover To Next FY", "No"), False)

    recovery_profile = text_or_blank("Recovery Profile")
    scenario_type = text_or_blank("Scenario Type")
    severity = text_or_blank("Severity")
    description = text_or_blank("Description")
    notes = text_or_blank("Notes")

    # Safe fallbacks if Excel metadata is blank or dirty
    scen_name = _clean(selected_scenario).lower()

    if recovery_profile == "":
        if "baseline" in scen_name:
            recovery_profile = "Immediate"
        elif "gt3m" in scen_name:
            recovery_profile = "Gradual"
        elif "3m" in scen_name or "1m" in scen_name:
            recovery_profile = "Fast"
        elif "oil" in scen_name:
            recovery_profile = "Fast"
        elif "shipping" in scen_name:
            recovery_profile = "Gradual"
        elif "composite" in scen_name:
            recovery_profile = "Persistent"
        else:
            recovery_profile = "Fast"

    if scenario_type == "":
        if "baseline" in scen_name:
            scenario_type = "Baseline"
        elif "corrective" in scen_name:
            scenario_type = "Corrective"
        elif "oil" in scen_name:
            scenario_type = "Oil"
        elif "shipping" in scen_name or "trade" in scen_name:
            scenario_type = "Trade"
        elif "composite" in scen_name:
            scenario_type = "Composite"
        else:
            scenario_type = "Stress"

    if severity == "":
        if "baseline" in scen_name:
            severity = "None"
        elif "severe" in scen_name:
            severity = "Severe"
        elif "moderate" in scen_name:
            severity = "Moderate"
        elif "mild" in scen_name:
            severity = "Mild"
        else:
            severity = "Severe"

    if description == "" and selected_scenario:
        description = _clean(selected_scenario)

    return {
        "scenario_start_month": start_month,
        "scenario_duration_months": duration_months,
        "carryover_to_next_fy": carryover,
        "recovery_profile": recovery_profile,
        "scenario_type": scenario_type,
        "severity": severity,
        "description": description,
        "notes": notes,
    }


# ============================================================
# SHOCK APPLICATION
# ============================================================

def _apply_additive_shocks(
    baseline_macro_df: pd.DataFrame,
    scenario_row: pd.Series,
    selected_year: str,
) -> pd.DataFrame:
    shocked = baseline_macro_df.copy()

    if "year" not in shocked.columns:
        raise ScenarioBuilderError("Baseline macro dataframe missing 'year' column.")

    shocked["year"] = shocked["year"].astype(str).str.strip()
    mask = shocked["year"] == selected_year

    if mask.sum() == 0:
        raise ScenarioBuilderError(f"Selected year '{selected_year}' not found in baseline macro.")

    def add_shock(macro_col: str, scenario_col: str) -> None:
        if macro_col in shocked.columns and scenario_col in scenario_row.index:
            base_val = _to_float(shocked.loc[mask, macro_col].iloc[0], 0.0)
            shock_val = _to_float(scenario_row[scenario_col], 0.0)
            shocked.loc[mask, macro_col] = base_val + shock_val

    add_shock("real_gdp_growth", "Real GDP growth shock")
    add_shock("gdp_deflator", "GDP deflator shock")
    add_shock("cpi", "CPI shock")
    add_shock("wage_growth", "Wage growth shock")
    add_shock("import_growth", "Import Value Growth shock")
    add_shock("export_growth", "Export Value Growth shock")
    add_shock("non_oil_import_growth", "Non oil import value growth shock")
    add_shock("oil_volume_change", "Oil volume shock")
    add_shock("oil_price_change", "Oil world price shock")
    add_shock("implied_profitability_growth", "Profitability growth shock")
    add_shock("exchange_rate_change", "Exchange rate shock")

    # Rebuild identities after shocks
    if {"real_gdp_growth", "gdp_deflator", "nominal_gdp_growth"}.issubset(shocked.columns):
        shocked.loc[mask, "nominal_gdp_growth"] = (
            pd.to_numeric(shocked.loc[mask, "real_gdp_growth"], errors="coerce").fillna(0.0)
            + pd.to_numeric(shocked.loc[mask, "gdp_deflator"], errors="coerce").fillna(0.0)
        )

    if {"oil_price_change", "oil_volume_change", "oil_value_growth"}.issubset(shocked.columns):
        shocked.loc[mask, "oil_value_growth"] = (
            pd.to_numeric(shocked.loc[mask, "oil_price_change"], errors="coerce").fillna(0.0)
            + pd.to_numeric(shocked.loc[mask, "oil_volume_change"], errors="coerce").fillna(0.0)
        )

    return shocked


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def build_scenario_package(
    data: Dict[str, Any],
    ui_override_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if "macro" not in data:
        raise ScenarioBuilderError("Loaded data missing 'macro'.")
    if "scenarios" not in data:
        raise ScenarioBuilderError("Loaded data missing 'scenarios'.")
    if "rolling_control" not in data:
        raise ScenarioBuilderError("Loaded data missing 'rolling_control'.")
    if "control" not in data:
        raise ScenarioBuilderError("Loaded data missing 'control'.")

    selected_year = _get_selected_year(data)
    selected_scenario = _get_selected_scenario(data)
    duration_override = _get_duration_override_from_controls(data)

    baseline_macro_df = build_macro_driver_table(
        macro_input_df=data["macro"],
        overwrite_existing=True,
    )

    if selected_scenario.lower() == "baseline":
        duration_final = duration_override if duration_override is not None else 0
        return {
            "baseline_macro_df": baseline_macro_df,
            "shocked_macro_df": baseline_macro_df.copy(),
            "selected_scenario": selected_scenario,
            "selected_year": selected_year,
            "scenario_start_month": 10,
            "scenario_duration_months": duration_final,
            "carryover_to_next_fy": False,
            "recovery_profile": "Immediate",
            "scenario_type": "Baseline",
            "severity": "None",
            "description": "No shock",
            "notes": "",
            "ui_override_applied": ui_override_df is not None,
            "scenario_row": pd.DataFrame(),
        }

    scenario_row = _get_scenario_row(
        scenarios_df=data["scenarios"],
        selected_scenario=selected_scenario,
        ui_override_df=ui_override_df,
    )

    scenario_meta = _extract_scenario_metadata(
        scenario_row=scenario_row,
        selected_scenario=selected_scenario,
        duration_override=duration_override,
    )

    shocked_macro_df = _apply_additive_shocks(
        baseline_macro_df=baseline_macro_df,
        scenario_row=scenario_row,
        selected_year=selected_year,
    )

    return {
        "baseline_macro_df": baseline_macro_df,
        "shocked_macro_df": shocked_macro_df,
        "selected_scenario": selected_scenario,
        "selected_year": selected_year,
        "scenario_start_month": scenario_meta["scenario_start_month"],
        "scenario_duration_months": scenario_meta["scenario_duration_months"],
        "carryover_to_next_fy": scenario_meta["carryover_to_next_fy"],
        "recovery_profile": scenario_meta["recovery_profile"],
        "scenario_type": scenario_meta["scenario_type"],
        "severity": scenario_meta["severity"],
        "description": scenario_meta["description"],
        "notes": scenario_meta["notes"],
        "ui_override_applied": ui_override_df is not None,
        "scenario_row": pd.DataFrame([scenario_row.to_dict()]),
    }