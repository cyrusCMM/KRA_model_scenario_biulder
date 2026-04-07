from __future__ import annotations

# ============================================================
# APP RUNNER V2
# Flat-import version for single-folder workflow
# Scope:
# - load workbook inputs
# - prepare macro tables
# - run baseline or baseline+scenario
# - validate outputs
# - build dashboard pack
# - build decomposition pack
# - optionally export outputs
# ============================================================

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from rolling_loader_v3 import load_all_inputs
from macro_identities import build_macro_driver_table
from scenario_runner_v1 import run_baseline_only, run_baseline_and_scenario
from dashboard_builder_v1 import build_dashboard_pack
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
    """
    Takes workbook macro sheet and rebuilds all macro identities into
    year-shaped macro drivers.
    """
    if "macro" not in data:
        raise AppRunnerError("data missing 'macro'.")

    macro_input = data["macro"]
    macro_df = build_macro_driver_table(
        macro_input_df=macro_input,
        overwrite_existing=True,
    )

    if "year" not in macro_df.columns:
        raise AppRunnerError("Prepared macro dataframe missing 'year' column.")

    return macro_df


def prepare_shocked_macro(shocked_macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a shocked macro dataframe already prepared upstream and ensures
    identities are rebuilt.
    """
    if not isinstance(shocked_macro_df, pd.DataFrame):
        raise AppRunnerError("shocked_macro_df must be a pandas DataFrame.")

    macro_df = build_macro_driver_table(
        macro_input_df=shocked_macro_df,
        overwrite_existing=True,
    )

    if "year" not in macro_df.columns:
        raise AppRunnerError("Prepared shocked macro dataframe missing 'year' column.")

    return macro_df


# ============================================================
# BASELINE APP RUN
# ============================================================

def run_app_baseline(
    workbook_path: str | Path,
    export: bool = False,
    export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end baseline-only app pipeline.
    """
    workbook_path = _resolve_path(workbook_path)

    # --------------------------------------------------------
    # 1. Load inputs
    # --------------------------------------------------------
    data = load_all_inputs(workbook_path, validate=True)
    validate_loaded_inputs(data)

    # --------------------------------------------------------
    # 2. Prepare macro
    # --------------------------------------------------------
    baseline_macro_df = prepare_baseline_macro(data)

    # --------------------------------------------------------
    # 3. Run baseline
    # --------------------------------------------------------
    baseline_package = run_baseline_only(
        data=data,
        baseline_macro_df=baseline_macro_df,
    )

    if "baseline" not in baseline_package:
        raise AppRunnerError("Baseline package missing 'baseline'.")

    validate_rolling_outputs(baseline_package["baseline"])

    # --------------------------------------------------------
    # 4. Optional export
    # --------------------------------------------------------
    exported_files: Dict[str, Path] = {}

    if export:
        if export_path is None:
            export_path = workbook_path.parent / "baseline_outputs.xlsx"

        exported_files["baseline"] = export_baseline_outputs(
            filepath=export_path,
            baseline_outputs=baseline_package["baseline"],
        )

    # --------------------------------------------------------
    # 5. Final return object
    # --------------------------------------------------------
    return {
        "metadata": {
            "mode": "baseline",
            "workbook_path": str(workbook_path),
            "selected_year": _clean(data["rolling_control"].get("selected_year", "")),
        },
        "inputs": data,
        "macro": {
            "baseline_macro_df": baseline_macro_df,
        },
        "baseline": baseline_package["baseline"],
        "exports": exported_files,
    }


# ============================================================
# FULL APP RUN
# ============================================================

def run_app_scenario(
    workbook_path: str | Path,
    shocked_macro_df: pd.DataFrame,
    scenario_name: str = "Custom Scenario",
    scenario_duration_months: Optional[int] = None,
    export: bool = False,
    export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end baseline + scenario app pipeline.
    """
    workbook_path = _resolve_path(workbook_path)

    # --------------------------------------------------------
    # 1. Load inputs
    # --------------------------------------------------------
    data = load_all_inputs(workbook_path, validate=True)
    validate_loaded_inputs(data)

    # --------------------------------------------------------
    # 2. Prepare macro tables
    # --------------------------------------------------------
    baseline_macro_df = prepare_baseline_macro(data)
    shocked_macro_prepared = prepare_shocked_macro(shocked_macro_df)

    # --------------------------------------------------------
    # 3. Run scenario package
    # --------------------------------------------------------
    scenario_package = run_baseline_and_scenario(
        data=data,
        baseline_macro_df=baseline_macro_df,
        shocked_macro_df=shocked_macro_prepared,
        scenario_name=scenario_name,
        scenario_duration_months=scenario_duration_months,
    )

    validate_scenario_runner_package(scenario_package)

    # --------------------------------------------------------
    # 4. Build dashboard pack
    # --------------------------------------------------------
    dashboard_pack = build_dashboard_pack(scenario_package)
    validate_dashboard_pack(dashboard_pack)

    # --------------------------------------------------------
    # 5. Build decomposition pack
    # --------------------------------------------------------
    decomposition_pack = build_decomposition_pack(dashboard_pack)
    validate_decomposition_pack(decomposition_pack)

    # --------------------------------------------------------
    # 6. Optional export
    # --------------------------------------------------------
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
            },
        )

    # --------------------------------------------------------
    # 7. Final return object
    # --------------------------------------------------------
    return {
        "metadata": {
            "mode": "scenario",
            "workbook_path": str(workbook_path),
            "selected_year": scenario_package["metadata"]["selected_year"],
            "scenario_name": scenario_package["metadata"]["scenario_name"],
            "scenario_duration_months": scenario_package["metadata"]["scenario_duration_months"],
        },
        "inputs": data,
        "macro": {
            "baseline_macro_df": baseline_macro_df,
            "shocked_macro_df": shocked_macro_prepared,
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
    shocked_macro_df: Optional[pd.DataFrame] = None,
    scenario_name: str = "Custom Scenario",
    scenario_duration_months: Optional[int] = None,
    export: bool = False,
    export_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Unified app-facing runner.

    mode:
    - "baseline"
    - "scenario"
    """
    mode_clean = _clean(mode).lower()

    if mode_clean == "baseline":
        return run_app_baseline(
            workbook_path=workbook_path,
            export=export,
            export_path=export_path,
        )

    if mode_clean == "scenario":
        if shocked_macro_df is None:
            raise AppRunnerError("shocked_macro_df must be provided when mode='scenario'.")

        return run_app_scenario(
            workbook_path=workbook_path,
            shocked_macro_df=shocked_macro_df,
            scenario_name=scenario_name,
            scenario_duration_months=scenario_duration_months,
            export=export,
            export_path=export_path,
        )

    raise AppRunnerError(f"Unsupported mode '{mode}'. Use 'baseline' or 'scenario'.")


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    print("=" * 90)
    print("APP RUNNER V2 TEST")
    print("=" * 90)
    print("This file is orchestration only.")
    print("To test fully, point it to a real workbook and a real shocked macro table.")
    print("=" * 90)

    workbook_path = Path("kra_forecast_input_template_final_for_python.xlsx").resolve()

    shocked_macro_df = pd.DataFrame({
        "Variable": [
            "Real GDP growth",
            "GDP deflator",
            "CPI",
            "Import Value Growth",
            "Export Value Growth",
            "Non oil import value growth (Dry)",
            "Oil World price change (in US$)",
            "Oil (% volume change)",
        ],
        "2025/26": [0.02, 0.05, 0.07, 0.04, 0.03, 0.05, 0.08, 0.00],
        "2026/27": [0.04, 0.05, 0.06, 0.06, 0.04, 0.06, 0.05, 0.01],
        "2027/28": [0.05, 0.05, 0.06, 0.07, 0.05, 0.07, 0.04, 0.01],
    })

    print("\nExample usage:")
    print("1. Baseline only")
    print("   result = run_app(workbook_path, mode='baseline', export=False)")
    print("\n2. Scenario run")
    print("   result = run_app(workbook_path, mode='scenario', shocked_macro_df=shocked_macro_df, export=False)")
    print("\nAPP RUNNER V2 READY.")