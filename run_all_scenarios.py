# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:15:45 2026

@author: hp
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

import app_runner_v3
from export_engine_v1 import export_full_package


WORKBOOK_PATH = r"C:\Users\hp\Documents\Q\KRA model\kra_forecast_model\inputs\kra_forecast_input_template_final.xlsx"
OUT_DIR_NAME = "out"
INCLUDE_BASELINE = True   # set False if you only want non-baseline scenarios


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_sheet_name(name: str, used: set[str]) -> str:
    name = re.sub(r"[:\\/?*\[\]]", "_", str(name))
    name = name[:31]
    base = name
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = (base[: 31 - len(suffix)] + suffix)
        i += 1
    used.add(name)
    return name


def read_scenarios(workbook_path: str) -> pd.DataFrame:
    df = pd.read_excel(workbook_path, sheet_name="CGE_Scenarios")
    df.columns = [clean(c) for c in df.columns]
    if "Scenario" not in df.columns:
        raise ValueError("CGE_Scenarios sheet must contain a 'Scenario' column.")
    df["Scenario"] = df["Scenario"].astype(str).str.strip()
    df = df.loc[df["Scenario"] != ""].copy()
    return df


def build_run_summary(result: dict) -> pd.DataFrame:
    meta = result.get("metadata", {})
    total = result["comparisons"]["total_comparison"].copy()

    row = total.iloc[0].to_dict()
    row.update({
        "Scenario": meta.get("scenario_name", ""),
        "Selected Year": meta.get("selected_year", ""),
        "Start Month": meta.get("scenario_start_month", ""),
        "Duration Months": meta.get("scenario_duration_months", ""),
        "Carryover To Next FY": meta.get("carryover_to_next_fy", ""),
        "Recovery Profile": meta.get("recovery_profile", ""),
        "Scenario Type": meta.get("scenario_type", ""),
        "Severity": meta.get("severity", ""),
    })
    return pd.DataFrame([row])


def build_department_summary(result: dict) -> pd.DataFrame:
    df = result["scenario"]["department_summary"].copy()
    meta = result.get("metadata", {})
    df.insert(0, "Scenario", meta.get("scenario_name", ""))
    df.insert(1, "Selected Year", meta.get("selected_year", ""))
    return df


def build_monthly_summary(result: dict) -> pd.DataFrame:
    df = result["scenario"]["monthly_delta"].copy()
    meta = result.get("metadata", {})
    df.insert(0, "Scenario", meta.get("scenario_name", ""))
    df.insert(1, "Selected Year", meta.get("selected_year", ""))
    return df


def build_taxhead_summary(result: dict) -> pd.DataFrame:
    df = result["comparisons"]["tax_head_comparison"].copy()
    meta = result.get("metadata", {})
    df.insert(0, "Scenario", meta.get("scenario_name", ""))
    df.insert(1, "Selected Year", meta.get("selected_year", ""))
    return df


def run_all_scenarios(
    workbook_path: str,
    include_baseline: bool = True,
) -> None:
    workbook = Path(workbook_path).resolve()
    out_dir = workbook.parent / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios_df = read_scenarios(str(workbook))
    scenario_names = scenarios_df["Scenario"].tolist()

    if not include_baseline:
        scenario_names = [s for s in scenario_names if clean(s).lower() != "baseline"]

    master_summary = []
    master_departments = []
    master_monthly = []
    master_taxheads = []
    run_log = []

    for scen in scenario_names:
        print(f"Running scenario: {scen}")

        try:
            result = app_runner_v3.run_app(
                workbook_path=str(workbook),
                mode="scenario",
                selected_scenario=scen,
                export=False,
            )

            meta = result.get("metadata", {})

            master_summary.append(build_run_summary(result))
            master_departments.append(build_department_summary(result))
            master_monthly.append(build_monthly_summary(result))
            master_taxheads.append(build_taxhead_summary(result))

            # save one detailed package per scenario
            scenario_file = out_dir / f"{scen}_package.xlsx"
            export_full_package(
                filepath=scenario_file,
                baseline_outputs=result["baseline"],
                scenario_outputs=result["scenario"],
                comparisons=result["comparisons"],
                dashboard_pack=result["dashboard_pack"],
                decomposition_pack=result["decomposition_pack"],
                metadata=result["metadata"],
            )

            run_log.append({
                "Scenario": scen,
                "Status": "Success",
                "Selected Year": meta.get("selected_year", ""),
                "Start Month": meta.get("scenario_start_month", ""),
                "Duration Months": meta.get("scenario_duration_months", ""),
                "Carryover To Next FY": meta.get("carryover_to_next_fy", ""),
                "Output File": str(scenario_file),
            })

        except Exception as e:
            run_log.append({
                "Scenario": scen,
                "Status": "Failed",
                "Selected Year": "",
                "Start Month": "",
                "Duration Months": "",
                "Carryover To Next FY": "",
                "Output File": "",
                "Error": str(e),
            })
            print(f"Failed: {scen} -> {e}")

    # combine master tables
    summary_df = pd.concat(master_summary, ignore_index=True) if master_summary else pd.DataFrame()
    dept_df = pd.concat(master_departments, ignore_index=True) if master_departments else pd.DataFrame()
    monthly_df = pd.concat(master_monthly, ignore_index=True) if master_monthly else pd.DataFrame()
    taxhead_df = pd.concat(master_taxheads, ignore_index=True) if master_taxheads else pd.DataFrame()
    log_df = pd.DataFrame(run_log)

    # save master workbook
    master_file = out_dir / "all_scenarios_master.xlsx"
    used_sheets = set()

    with pd.ExcelWriter(master_file, engine="openpyxl") as writer:
        log_df.to_excel(writer, sheet_name=safe_sheet_name("Run_Log", used_sheets), index=False)
        summary_df.to_excel(writer, sheet_name=safe_sheet_name("Scenario_Summary", used_sheets), index=False)
        dept_df.to_excel(writer, sheet_name=safe_sheet_name("Department_Impact", used_sheets), index=False)
        monthly_df.to_excel(writer, sheet_name=safe_sheet_name("Monthly_Delta", used_sheets), index=False)
        taxhead_df.to_excel(writer, sheet_name=safe_sheet_name("Taxhead_Impact", used_sheets), index=False)

    print("\nDone.")
    print(f"Master workbook: {master_file}")
    print(f"Detailed scenario files saved in: {out_dir}")


if __name__ == "__main__":
    run_all_scenarios(
        workbook_path=WORKBOOK_PATH,
        include_baseline=INCLUDE_BASELINE,
    )