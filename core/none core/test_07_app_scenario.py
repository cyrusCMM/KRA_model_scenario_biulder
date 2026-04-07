# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:24:05 2026

@author: hp
"""

from pathlib import Path
import sys
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from app_runner_v2 import run_app

WORKBOOK = CURRENT_DIR / "kra_forecast_input_template_final_for_python.xlsx"

print("=" * 90)
print("TEST 07 - APP RUNNER SCENARIO")
print("=" * 90)
print("Workbook:", WORKBOOK)
print("Workbook exists:", WORKBOOK.exists())

# ------------------------------------------------------------
# Controlled shock input
# Keep it simple and transparent
# ------------------------------------------------------------
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

result = run_app(
    workbook_path=WORKBOOK,
    mode="scenario",
    shocked_macro_df=shocked_macro_df,
    scenario_name="Controlled test shock",
    scenario_duration_months=None,
    export=False,
)

print("\nMetadata:")
print(result["metadata"])

print("\nBaseline total summary:")
print(result["baseline"]["total_summary"])

print("\nScenario total summary:")
print(result["scenario"]["total_summary"])

print("\nTotal comparison:")
print(result["comparisons"]["total_comparison"])

print("\nDashboard keys:")
for k in result["dashboard_pack"].keys():
    print("-", k)

print("\nDecomposition keys:")
for k in result["decomposition_pack"].keys():
    print("-", k)

print("\nTEST 07 PASSED")