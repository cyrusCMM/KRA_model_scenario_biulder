# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:18:36 2026

@author: hp
"""

from pathlib import Path
from rolling_loader_v3 import load_all_inputs
from macro_identities import build_macro_driver_table
from rolling_engine_v1 import run_rolling_engine

WORKBOOK = Path("kra_forecast_input_template_final_for_python.xlsx").resolve()

print("=" * 90)
print("TEST 04 - ROLLING ENGINE")
print("=" * 90)

data = load_all_inputs(WORKBOOK, validate=True)
macro_df = build_macro_driver_table(data["macro"], overwrite_existing=True)

outputs = run_rolling_engine(data=data, macro_df=macro_df)

print("\nFinal total summary:")
print(outputs["total_summary"])

if "monthly_outputs" in outputs:
    print("\nMonthly rebuild check:")
    print(outputs["monthly_outputs"]["rebuild_check"].head(20))

if "base_switch_table" in outputs:
    print("\nBase switch table:")
    print(outputs["base_switch_table"].head(20))

print("\nTEST 04 COMPLETED")