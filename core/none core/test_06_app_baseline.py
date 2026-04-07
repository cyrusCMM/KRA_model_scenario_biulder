# -*- coding: utf-8 -*-

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from app_runner_v2 import run_app

WORKBOOK = CURRENT_DIR / "kra_forecast_input_template_final_for_python.xlsx"

print("=" * 90)
print("TEST 06 - APP RUNNER BASELINE")
print("=" * 90)
print("Current dir:", CURRENT_DIR)
print("Workbook:", WORKBOOK)
print("Workbook exists:", WORKBOOK.exists())

result = run_app(
    workbook_path=WORKBOOK,
    mode="baseline",
    export=False,
)

print("\nMetadata:")
print(result["metadata"])

print("\nBaseline total summary:")
print(result["baseline"]["total_summary"])

print("\nAvailable baseline keys:")
for k in result["baseline"].keys():
    print("-", k)

print("\nTEST 06 PASSED")