from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from rolling_loader_v4 import load_all_inputs
from macro_identities import build_macro_driver_table
from scenario_builder_v1 import build_scenario_package
from scenario_runner_v1 import run_baseline_only, run_baseline_and_scenario
from dashboard_builder_v2 import build_dashboard_pack
from decomposition_engine_v1 import build_decomposition_pack
from validation_engine_v1 import (
    validate_loaded_inputs,
    validate_rolling_outputs,
    validate_scenario_runner_package,
    validate_dashboard_pack,
    validate_decomposition_pack,
)
from export_engine_v1 import export_baseline_outputs, export_full_package


class AppRunnerError(Exception):
    pass


def _clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _resolve_path(p: str | Path) -> Path:
    return Path(p).resolve()


def _apply_in_memory_controls(
    data: Dict[str, Any],
    selected_year: Optional[str] = None,
    selected_scenario: Optional[str] = None,
    scenario_duration_months: Optional[int] = None,
) -> Dict[str, Any]:
    out = dict(data)

    rolling_control = dict(out.get("rolling_control", {}))
    control = dict(out.get("control", {}))

    if selected_year is not None and _clean(selected_year) != "":
        # selected_year is the reporting / forecast year requested by the user.
        # Do NOT overwrite current_fiscal_year here.
        # current_fiscal_year must remain the workbook-defined live/open FY
        # so that future years are not treated as fresh shock years.
        rolling_control["selected_year"] = _clean(selected_year)

    if selected_scenario is not None and _clean(selected_scenario) != "":
        control["scenario"] = _clean(selected_scenario)

    if scenario_duration_months is not None:
        rolling_control["scenario_duration_months"] = int(scenario_duration_months)

    out["rolling_control"] = rolling_control
    out["control"] = control
    return out


def prepare_baseline_macro(data: Dict[str, Any]) -> pd.DataFrame:
    if "macro" not in data:
        raise AppRunnerError("Loaded data missing 'macro'.")

    macro_df = build_macro_driver_table(
        macro_input_df=data["macro"],
        overwrite_existing=True,
    )

    if "year" not in macro_df.columns:
        raise AppRunnerError("Prepared baseline macro dataframe missing 'year' column.")

    return macro_df


def run_app_baseline(
    workbook_path: str | Path,
    export: bool = False,
    export_path: Optional[str | Path] = None,
    selected_year: Optional[str] = None,
) -> Dict[str, Any]:
    workbook_path = _resolve_path(workbook_path)

    data = load_all_inputs(workbook_path, validate=True)
    data = _apply_in_memory_controls(data=data, selected_year=selected_year)
    validate_loaded_inputs(data)

    baseline_macro_df = prepare_baseline_macro(data)

    baseline_package = run_baseline_only(
        data=data,
        baseline_macro_df=baseline_macro_df,
    )

    if "baseline" not in baseline_package:
        raise AppRunnerError("Baseline package missing 'baseline'.")

    validate_rolling_outputs(baseline_package["baseline"])

    exported_files: Dict[str, Path] = {}
    if export:
        if export_path is None:
            export_path = workbook_path.parent / "baseline_outputs.xlsx"

        exported_files["baseline"] = export_baseline_outputs(
            filepath=export_path,
            baseline_outputs=baseline_package["baseline"],
        )

    return {
        "metadata": {
            "mode": "baseline",
            "workbook_path": str(workbook_path),
            "selected_year": _clean(data["rolling_control"].get("selected_year", "")),
            "selected_scenario": _clean(data["control"].get("scenario", "")),
        },
        "inputs": data,
        "macro": {
            "baseline_macro_df": baseline_macro_df,
        },
        "baseline": baseline_package["baseline"],
        "exports": exported_files,
    }


def run_app_scenario(
    workbook_path: str | Path,
    ui_override_df: Optional[pd.DataFrame] = None,
    export: bool = False,
    export_path: Optional[str | Path] = None,
    selected_year: Optional[str] = None,
    selected_scenario: Optional[str] = None,
    scenario_duration_months: Optional[int] = None,
) -> Dict[str, Any]:
    workbook_path = _resolve_path(workbook_path)

    data = load_all_inputs(workbook_path, validate=True)
    data = _apply_in_memory_controls(
        data=data,
        selected_year=selected_year,
        selected_scenario=selected_scenario,
        scenario_duration_months=scenario_duration_months,
    )
    validate_loaded_inputs(data)

    baseline_macro_df = prepare_baseline_macro(data)

    scenario_build = build_scenario_package(
        data=data,
        ui_override_df=ui_override_df,
    )
    shocked_macro_df = scenario_build["shocked_macro_df"]

    scenario_package = run_baseline_and_scenario(
        data=data,
        baseline_macro_df=baseline_macro_df,
        shocked_macro_df=shocked_macro_df,
        scenario_name=scenario_build["selected_scenario"],
        scenario_start_month=scenario_build["scenario_start_month"],
        scenario_duration_months=scenario_build["scenario_duration_months"],
        carryover_to_next_fy=scenario_build["carryover_to_next_fy"],
        recovery_profile=scenario_build["recovery_profile"],
        scenario_type=scenario_build["scenario_type"],
        severity=scenario_build["severity"],
    )

    validate_scenario_runner_package(scenario_package)

    dashboard_pack = build_dashboard_pack(scenario_package)
    validate_dashboard_pack(dashboard_pack)

    decomposition_pack = build_decomposition_pack(dashboard_pack)
    validate_decomposition_pack(decomposition_pack)

    exported_files: Dict[str, Path] = {}
    if export:
        if export_path is None:
            export_path = workbook_path.parent / "full_scenario_package.xlsx"

        exported_files["full_package"] = export_full_package(
            filepath=export_path,
            baseline_outputs=scenario_package["baseline"],
            scenario_outputs=scenario_package["scenario"],
            comparisons=scenario_package["comparisons"],
            dashboard_pack=dashboard_pack,
            decomposition_pack=decomposition_pack,
            metadata={
                "selected_year": scenario_package["metadata"]["selected_year"],
                "scenario_name": scenario_package["metadata"]["scenario_name"],
                "scenario_start_month": scenario_package["metadata"]["scenario_start_month"],
                "scenario_duration_months": scenario_package["metadata"]["scenario_duration_months"],
                "carryover_to_next_fy": scenario_package["metadata"]["carryover_to_next_fy"],
                "recovery_profile": scenario_package["metadata"]["recovery_profile"],
                "scenario_type": scenario_package["metadata"]["scenario_type"],
                "severity": scenario_package["metadata"]["severity"],
                "workbook_path": str(workbook_path),
                "ui_override_applied": scenario_build["ui_override_applied"],
            },
        )

    return {
        "metadata": {
            "mode": "scenario",
            "workbook_path": str(workbook_path),
            "selected_year": scenario_package["metadata"]["selected_year"],
            "scenario_name": scenario_package["metadata"]["scenario_name"],
            "scenario_start_month": scenario_package["metadata"]["scenario_start_month"],
            "scenario_duration_months": scenario_package["metadata"]["scenario_duration_months"],
            "carryover_to_next_fy": scenario_package["metadata"]["carryover_to_next_fy"],
            "recovery_profile": scenario_package["metadata"]["recovery_profile"],
            "scenario_type": scenario_package["metadata"]["scenario_type"],
            "severity": scenario_package["metadata"]["severity"],
            "ui_override_applied": scenario_build["ui_override_applied"],
        },
        "inputs": data,
        "macro": {
            "baseline_macro_df": baseline_macro_df,
            "shocked_macro_df": shocked_macro_df,
        },
        "baseline": scenario_package["baseline"],
        "scenario": scenario_package["scenario"],
        "comparisons": scenario_package["comparisons"],
        "dashboard_pack": dashboard_pack,
        "decomposition_pack": decomposition_pack,
        "exports": exported_files,
    }


def run_app(
    workbook_path: str | Path,
    mode: str = "baseline",
    ui_override_df: Optional[pd.DataFrame] = None,
    export: bool = False,
    export_path: Optional[str | Path] = None,
    selected_year: Optional[str] = None,
    selected_scenario: Optional[str] = None,
    scenario_duration_months: Optional[int] = None,
) -> Dict[str, Any]:
    mode_clean = _clean(mode).lower()

    if mode_clean == "baseline":
        return run_app_baseline(
            workbook_path=workbook_path,
            export=export,
            export_path=export_path,
            selected_year=selected_year,
        )

    if mode_clean == "scenario":
        return run_app_scenario(
            workbook_path=workbook_path,
            ui_override_df=ui_override_df,
            export=export,
            export_path=export_path,
            selected_year=selected_year,
            selected_scenario=selected_scenario,
            scenario_duration_months=scenario_duration_months,
        )

    raise AppRunnerError(f"Unsupported mode '{mode}'. Use 'baseline' or 'scenario'.")