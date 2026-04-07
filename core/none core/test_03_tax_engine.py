# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:09:27 2026

@author: hp
"""

from pathlib import Path
from rolling_loader_v3 import load_all_inputs
from macro_identities import build_macro_driver_table
from tax_engine_v1 import run_tax_engine

WORKBOOK = Path("kra_forecast_input_template_final_for_python.xlsx").resolve()

print("=" * 90)
print("TEST 03 - TAX ENGINE")
print("=" * 90)

data = load_all_inputs(WORKBOOK, validate=True)
macro_df = build_macro_driver_table(data["macro"], overwrite_existing=True)

outputs = run_tax_engine(data=data, macro_df=macro_df)

print("\nDetail head:")
print(outputs["detail"].head(20))

print("\nAnnex summary:")
print(outputs["annex_summary"].head(20))

print("\nTotal summary:")
print(outputs["total_summary"])

print("\nTEST 03 COMPLETED")