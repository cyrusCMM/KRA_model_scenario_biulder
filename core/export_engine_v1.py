# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:42:21 2026

@author: hp
"""

from __future__ import annotations

# ============================================================
# EXPORT ENGINE V1
# Fresh export layer
# Scope:
# - export baseline outputs
# - export scenario outputs
# - export dashboard tables
# - export decomposition tables
# - export full package workbook
# ============================================================

from pathlib import Path
from typing import Dict, Any
import importlib.util

import pandas as pd


class ExportEngineError(Exception):
    """Raised when export packaging fails."""
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def _require_dict(label: str, obj: Any) -> None:
    if not isinstance(obj, dict):
        raise ExportEngineError(f"{label} must be a dictionary.")


def _require_dataframe_dict(label: str, data: Dict[str, Any]) -> None:
    _require_dict(label, data)
    for k, v in data.items():
        if not isinstance(v, pd.DataFrame):
            raise ExportEngineError(f"{label}['{k}'] is not a pandas DataFrame.")


def _safe_sheet_name(name: str) -> str:
    """
    Excel limits sheet names to 31 chars and disallows []:*?/\
    """
    s = str(name)
    for ch in ['[', ']', ':', '*', '?', '/', '\\']:
        s = s.replace(ch, "_")
    return s[:31]


def _choose_excel_engine() -> str:
    """
    Prefer xlsxwriter if available, otherwise openpyxl.
    """
    if importlib.util.find_spec("xlsxwriter") is not None:
        return "xlsxwriter"
    if importlib.util.find_spec("openpyxl") is not None:
        return "openpyxl"
    raise ExportEngineError("Neither 'xlsxwriter' nor 'openpyxl' is available.")


def _ensure_parent_dir(filepath: str | Path) -> Path:
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_dict_of_dfs_to_writer(
    writer: pd.ExcelWriter,
    data: Dict[str, pd.DataFrame],
    prefix: str | None = None,
) -> None:
    _require_dataframe_dict("data", data)

    for name, df in data.items():
        sheet_name = f"{prefix} - {name}" if prefix else name
        df.to_excel(
            writer,
            sheet_name=_safe_sheet_name(sheet_name),
            index=False,
        )


def export_to_excel(
    filepath: str | Path,
    data: Dict[str, pd.DataFrame],
) -> Path:
    """
    Generic export for any dict of DataFrames.
    """
    _require_dataframe_dict("data", data)
    path = _ensure_parent_dir(filepath)
    engine = _choose_excel_engine()

    with pd.ExcelWriter(path, engine=engine) as writer:
        _write_dict_of_dfs_to_writer(writer, data)

    return path


# ============================================================
# BASELINE EXPORT
# ============================================================

def export_baseline_outputs(
    filepath: str | Path,
    baseline_outputs: Dict[str, Any],
) -> Path:
    """
    Exports the main baseline package from rolling_engine_v1.
    """
    _require_dict("baseline_outputs", baseline_outputs)

    export_map: Dict[str, pd.DataFrame] = {}

    for key in [
        "detail",
        "annex_summary",
        "department_summary",
        "total_summary",
        "base_switch_table",
        "forecast_accuracy_by_head",
        "forecast_accuracy_summary",
    ]:
        if key in baseline_outputs and isinstance(baseline_outputs[key], pd.DataFrame):
            export_map[key] = baseline_outputs[key]

    if "monthly_outputs" in baseline_outputs and isinstance(baseline_outputs["monthly_outputs"], dict):
        for k, v in baseline_outputs["monthly_outputs"].items():
            if isinstance(v, pd.DataFrame):
                export_map[f"monthly_{k}"] = v

    if not export_map:
        raise ExportEngineError("No exportable baseline DataFrames were found.")

    return export_to_excel(filepath, export_map)


# ============================================================
# SCENARIO EXPORT
# ============================================================

def export_scenario_outputs(
    filepath: str | Path,
    scenario_outputs: Dict[str, Any],
) -> Path:
    """
    Exports the main scenario package from simulation_engine_v1.
    """
    _require_dict("scenario_outputs", scenario_outputs)

    export_map: Dict[str, pd.DataFrame] = {}

    for key in [
        "annual_delta",
        "monthly_delta",
        "scenario_monthly_path",
        "scenario_annual_rebuild",
        "detail",
        "annex_summary",
        "department_summary",
        "total_summary",
    ]:
        if key in scenario_outputs and isinstance(scenario_outputs[key], pd.DataFrame):
            export_map[key] = scenario_outputs[key]

    if not export_map:
        raise ExportEngineError("No exportable scenario DataFrames were found.")

    return export_to_excel(filepath, export_map)


# ============================================================
# DASHBOARD EXPORT
# ============================================================

def export_dashboard_pack(
    filepath: str | Path,
    dashboard_pack: Dict[str, pd.DataFrame],
) -> Path:
    _require_dataframe_dict("dashboard_pack", dashboard_pack)
    return export_to_excel(filepath, dashboard_pack)


# ============================================================
# DECOMPOSITION EXPORT
# ============================================================

def export_decomposition_pack(
    filepath: str | Path,
    decomposition_pack: Dict[str, pd.DataFrame],
) -> Path:
    _require_dataframe_dict("decomposition_pack", decomposition_pack)
    return export_to_excel(filepath, decomposition_pack)


# ============================================================
# FULL PACKAGE EXPORT
# ============================================================

def export_full_package(
    filepath: str | Path,
    baseline_outputs: Dict[str, Any],
    scenario_outputs: Dict[str, Any],
    comparisons: Dict[str, pd.DataFrame],
    dashboard_pack: Dict[str, pd.DataFrame],
    decomposition_pack: Dict[str, pd.DataFrame] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """
    Exports one combined workbook with:
    - metadata
    - baseline
    - scenario
    - comparisons
    - dashboard tables
    - decomposition tables
    """
    _require_dict("baseline_outputs", baseline_outputs)
    _require_dict("scenario_outputs", scenario_outputs)
    _require_dataframe_dict("comparisons", comparisons)
    _require_dataframe_dict("dashboard_pack", dashboard_pack)

    if decomposition_pack is not None:
        _require_dataframe_dict("decomposition_pack", decomposition_pack)

    path = _ensure_parent_dir(filepath)
    engine = _choose_excel_engine()

    with pd.ExcelWriter(path, engine=engine) as writer:
        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------
        if metadata is not None:
            meta_rows = []
            for k, v in metadata.items():
                meta_rows.append({"Key": k, "Value": v})
            meta_df = pd.DataFrame(meta_rows)
            meta_df.to_excel(writer, sheet_name=_safe_sheet_name("Metadata"), index=False)

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------
        baseline_export_map: Dict[str, pd.DataFrame] = {}
        for key, val in baseline_outputs.items():
            if isinstance(val, pd.DataFrame):
                baseline_export_map[key] = val

        if "monthly_outputs" in baseline_outputs and isinstance(baseline_outputs["monthly_outputs"], dict):
            for k, v in baseline_outputs["monthly_outputs"].items():
                if isinstance(v, pd.DataFrame):
                    baseline_export_map[f"monthly_{k}"] = v

        _write_dict_of_dfs_to_writer(writer, baseline_export_map, prefix="Baseline")

        # ----------------------------------------------------
        # SCENARIO
        # ----------------------------------------------------
        scenario_export_map: Dict[str, pd.DataFrame] = {}
        for key, val in scenario_outputs.items():
            if isinstance(val, pd.DataFrame):
                scenario_export_map[key] = val

        _write_dict_of_dfs_to_writer(writer, scenario_export_map, prefix="Scenario")

        # ----------------------------------------------------
        # COMPARISONS
        # ----------------------------------------------------
        _write_dict_of_dfs_to_writer(writer, comparisons, prefix="Compare")

        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------
        _write_dict_of_dfs_to_writer(writer, dashboard_pack, prefix="Dashboard")

        # ----------------------------------------------------
        # DECOMPOSITION
        # ----------------------------------------------------
        if decomposition_pack is not None:
            _write_dict_of_dfs_to_writer(writer, decomposition_pack, prefix="Decomp")

    return path


# ============================================================
# QUICK MULTI-FILE EXPORT
# ============================================================

def export_all_separate_files(
    output_dir: str | Path,
    baseline_outputs: Dict[str, Any],
    scenario_outputs: Dict[str, Any],
    comparisons: Dict[str, pd.DataFrame],
    dashboard_pack: Dict[str, pd.DataFrame],
    decomposition_pack: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Path]:
    """
    Writes separate workbooks for each package.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: Dict[str, Path] = {}

    exported["baseline"] = export_baseline_outputs(
        output_dir / "baseline_outputs.xlsx",
        baseline_outputs,
    )
    exported["scenario"] = export_scenario_outputs(
        output_dir / "scenario_outputs.xlsx",
        scenario_outputs,
    )
    exported["comparisons"] = export_to_excel(
        output_dir / "comparison_outputs.xlsx",
        comparisons,
    )
    exported["dashboard"] = export_dashboard_pack(
        output_dir / "dashboard_outputs.xlsx",
        dashboard_pack,
    )

    if decomposition_pack is not None:
        exported["decomposition"] = export_decomposition_pack(
            output_dir / "decomposition_outputs.xlsx",
            decomposition_pack,
        )

    return exported


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    print("=" * 90)
    print("EXPORT ENGINE V1 TEST")
    print("=" * 90)
    print("This test writes synthetic files to ./test_exports")
    print("=" * 90)

    baseline_outputs = {
        "detail": pd.DataFrame({
            "Internal Tax Head": ["PAYE"],
            "Final Forecast": [1100.0],
        }),
        "annex_summary": pd.DataFrame({
            "Tax head": ["PAYE"],
            "Projected Collection 2025/26": [1100.0],
        }),
        "department_summary": pd.DataFrame({
            "Department": ["Domestic"],
            "Final Forecast": [1100.0],
        }),
        "total_summary": pd.DataFrame({
            "Final Forecast": [1100.0],
        }),
        "monthly_outputs": {
            "baseline_monthly_path": pd.DataFrame({
                "Month Index": [1, 2],
                "Baseline Monthly Forecast": [90.0, 91.0],
            })
        }
    }

    scenario_outputs = {
        "annual_delta": pd.DataFrame({
            "Internal Tax Head": ["PAYE"],
            "Scenario Impact": [-20.0],
        }),
        "detail": pd.DataFrame({
            "Internal Tax Head": ["PAYE"],
            "Final Forecast": [1080.0],
        }),
        "annex_summary": pd.DataFrame({
            "Tax head": ["PAYE"],
            "Projected Collection 2025/26": [1080.0],
        }),
        "department_summary": pd.DataFrame({
            "Department": ["Domestic"],
            "Final Forecast": [1080.0],
        }),
        "total_summary": pd.DataFrame({
            "Final Forecast": [1080.0],
        }),
    }

    comparisons = {
        "total_comparison": pd.DataFrame({
            "Baseline Final": [1100.0],
            "Scenario Final": [1080.0],
            "Scenario Impact": [-20.0],
        }),
        "tax_head_comparison": pd.DataFrame({
            "Internal Tax Head": ["PAYE"],
            "Scenario Impact": [-20.0],
        }),
    }

    dashboard_pack = {
        "executive_summary": pd.DataFrame({
            "Baseline Final": [1100.0],
            "Scenario Final": [1080.0],
            "Scenario Impact": [-20.0],
        }),
        "minister_brief": pd.DataFrame({
            "Scenario Impact": [-20.0],
            "Top Positive Movers": [""],
            "Top Negative Movers": ["PAYE"],
        }),
    }

    decomposition_pack = {
        "scenario_decomposition": pd.DataFrame({
            "Internal Tax Head": ["PAYE"],
            "Scenario Impact": [-20.0],
        }),
        "scenario_decomposition_summary": pd.DataFrame({
            "Scenario Impact": [-20.0],
        }),
    }

    out_dir = Path("test_exports")

    combined = export_full_package(
        filepath=out_dir / "full_package.xlsx",
        baseline_outputs=baseline_outputs,
        scenario_outputs=scenario_outputs,
        comparisons=comparisons,
        dashboard_pack=dashboard_pack,
        decomposition_pack=decomposition_pack,
        metadata={
            "selected_year": "2025/26",
            "scenario_name": "Synthetic Test",
        },
    )

    separate = export_all_separate_files(
        output_dir=out_dir,
        baseline_outputs=baseline_outputs,
        scenario_outputs=scenario_outputs,
        comparisons=comparisons,
        dashboard_pack=dashboard_pack,
        decomposition_pack=decomposition_pack,
    )

    print(f"\nCombined package: {combined}")
    for k, v in separate.items():
        print(f"{k}: {v}")

    print("\nEXPORT ENGINE V1 TEST PASSED.")