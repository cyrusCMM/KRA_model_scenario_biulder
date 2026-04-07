# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:30:19 2026

@author: hp
"""

from pathlib import Path
import sys
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from app_runner_v2 import run_app
from export_engine_v1 import (
    export_full_package,
    export_all_separate_files,
)

WORKBOOK = CURRENT_DIR / "kra_forecast_input_template_final_for_python.xlsx"
OUTPUT_DIR = CURRENT_DIR / "test_exports"

print("=" * 90)
print("TEST 08 - EXPORT ENGINE")
print("=" * 90)
print("Workbook:", WORKBOOK)
print("Workbook exists:", WORKBOOK.exists())
print("Output dir:", OUTPUT_DIR)

# ------------------------------------------------------------
# Controlled shock input
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

# ------------------------------------------------------------
# Run full scenario package
# ------------------------------------------------------------
result = run_app(
    workbook_path=WORKBOOK,
    mode="scenario",
    shocked_macro_df=shocked_macro_df,
    scenario_name="Controlled test shock",
    scenario_duration_months=None,
    export=False,
)

print("\nScenario package built successfully.")

# ------------------------------------------------------------
# Export combined workbook
# ------------------------------------------------------------
combined_file = export_full_package(
    filepath=OUTPUT_DIR / "full_package_test.xlsx",
    baseline_outputs=result["baseline"],
    scenario_outputs=result["scenario"],
    comparisons=result["comparisons"],
    dashboard_pack=result["dashboard_pack"],
    decomposition_pack=result["decomposition_pack"],
    metadata=result["metadata"],
)

print("\nCombined export created:")
print(combined_file)
print("Exists:", combined_file.exists())

# ------------------------------------------------------------
# Export separate workbooks
# ------------------------------------------------------------
separate_files = export_all_separate_files(
    output_dir=OUTPUT_DIR / "separate",
    baseline_outputs=result["baseline"],
    scenario_outputs=result["scenario"],
    comparisons=result["comparisons"],
    dashboard_pack=result["dashboard_pack"],
    decomposition_pack=result["decomposition_pack"],
)

print("\nSeparate exports created:")
for k, v in separate_files.items():
    print(f"{k}: {v} | exists={v.exists()}")

print("\nTEST 08 PASSED")