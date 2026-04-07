# -*- coding: utf-8 -*-
"""
APP RUNNER V3
Workbook-driven orchestrator for KRA forecasting model

Purpose
-------
- load workbook inputs through rolling_loader_v4
- prepare baseline macro identities
- build shocked macro through scenario_builder_v1
- run baseline or baseline+scenario
- validate outputs
- build dashboard pack
- build decomposition pack
- optionally export exactly what the UI shows
"""

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
from export_engine_v1 import (
    export_baseline_outputs,
    export_full_package,
)


class AppRunnerError(Exception):
    """Raised when end-to-end app orchestration fails."""
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _resolve_path(p: str | Path) -> Path:
    return Path(p).resolve()


# ============================================================
# MACRO PREPARATION
# ============================================================

def prepare_baseline_macro(data: Dict[str, Any]) -> pd.DataFrame:
    if "macro" not in data:
        raise AppRunnerError("Loaded data missing 'macro'.")

    macro_input = data["macro"]
    macro_df = build_macro_driver_table(
        macro_input_df=macro_input,
        overwrite_existing=True,
    )

    if "year" not in macro_df.columns:
        raise AppRunnerError("Prepared baseline macro dataframe missing 'year' column.")

    return macro_df


# ============================================================
# BASELINE APP RUN
# ============================================================

def run_app_baseline(
    workbook_path: str | Path,
    export: bool = False,
    export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    workbook_path = _resolve_path(workbook_path)

    # 1. Load workbook inputs
    data = load_all_inputs(workbook_path, validate=True)
    validate_loaded_inputs(data)

    # 2. Build baseline macro
    baseline_macro_df = prepare_baseline_macro(data)

    # 3. Run baseline
    baseline_package = run_baseline_only(
        data=data,
        baseline_macro_df=baseline_macro_df,
    )

    if "baseline" not in baseline_package:
        raise AppRunnerError("Baseline package missing 'baseline'.")

    validate_rolling_outputs(baseline_package["baseline"])

    # 4. Optional export
    exported_files: Dict[str, Path] = {}
    if export:
        if export_path is None:
            export_path = workbook_path.parent / "baseline_outputs.xlsx"

        exported_files["baseline"] = export_baseline_outputs(
            filepath=export_path,
            baseline_outputs=baseline_package["baseline"],
        )

    # 5. Return
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


# ============================================================
# SCENARIO APP RUN
# ============================================================

def run_app_scenario(
    workbook_path: str | Path,
    ui_override_df: Optional[pd.DataFrame] = None,
    export: bool = False,
    export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    workbook_path = _resolve_path(workbook_path)

    # 1. Load workbook inputs
    data = load_all_inputs(workbook_path, validate=True)
    validate_loaded_inputs(data)

    # 2. Build baseline macro
    baseline_macro_df = prepare_baseline_macro(data)

    # 3. Build shocked macro from workbook-selected scenario
    scenario_build = build_scenario_package(
        data=data,
        ui_override_df=ui_override_df,
    )
    shocked_macro_df = scenario_build["shocked_macro_df"]

    # 4. Run baseline + scenario
    scenario_package = run_baseline_and_scenario(
        data=data,
        baseline_macro_df=baseline_macro_df,
        shocked_macro_df=shocked_macro_df,
        scenario_name=scenario_build["selected_scenario"],
        scenario_duration_months=scenario_build["scenario_duration_months"],
    )

    validate_scenario_runner_package(scenario_package)

    # 5. Dashboard
    dashboard_pack = build_dashboard_pack(scenario_package)
    validate_dashboard_pack(dashboard_pack)

    # 6. Decomposition
    decomposition_pack = build_decomposition_pack(dashboard_pack)
    validate_decomposition_pack(decomposition_pack)

    # 7. Optional export
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
                "scenario_duration_months": scenario_package["metadata"]["scenario_duration_months"],
                "workbook_path": str(workbook_path),
                "ui_override_applied": scenario_build["ui_override_applied"],
            },
        )

    # 8. Return
    return {
        "metadata": {
            "mode": "scenario",
            "workbook_path": str(workbook_path),
            "selected_year": scenario_package["metadata"]["selected_year"],
            "scenario_name": scenario_package["metadata"]["scenario_name"],
            "scenario_duration_months": scenario_package["metadata"]["scenario_duration_months"],
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


# ============================================================
# UNIFIED RUNNER
# ============================================================

def run_app(
    workbook_path: str | Path,
    mode: str = "baseline",
    ui_override_df: Optional[pd.DataFrame] = None,
    export: bool = False,
    export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    mode_clean = _clean(mode).lower()

    if mode_clean == "baseline":
        return run_app_baseline(
            workbook_path=workbook_path,
            export=export,
            export_path=export_path,
        )

    if mode_clean == "scenario":
        return run_app_scenario(
            workbook_path=workbook_path,
            ui_override_df=ui_override_df,
            export=export,
            export_path=export_path,
        )

    raise AppRunnerError(f"Unsupported mode '{mode}'. Use 'baseline' or 'scenario'.")


if __name__ == "__main__":
    print("=" * 90)
    print("APP RUNNER V3 TEST")
    print("=" * 90)
    workbook = Path("kra_forecast_input_template_final.xlsx").resolve()

    print("\n[1] Baseline smoke test")
    baseline = run_app(
        workbook_path=workbook,
        mode="baseline",
        export=False,
    )
    print("Baseline total:")
    print(baseline["baseline"]["total_summary"])

    print("\n[2] Scenario smoke test using workbook-selected scenario")
    scenario = run_app(
        workbook_path=workbook,
        mode="scenario",
        export=False,
    )
    print("Scenario total:")
    print(scenario["scenario"]["total_summary"])
    print("Total comparison:")
    print(scenario["comparisons"]["total_comparison"])