# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:07:28 2026

@author: hp
"""

from pathlib import Path
from rolling_loader_v3 import load_all_inputs

WORKBOOK = Path("kra_forecast_input_template_final_for_python.xlsx").resolve()

print("=" * 90)
print("TEST 01 - LOADER")
print("=" * 90)
print("Workbook:", WORKBOOK)
print("Exists:", WORKBOOK.exists())

data = load_all_inputs(WORKBOOK, validate=True)

print("\nLoaded keys:")
for k in data.keys():
    print("-", k)

print("\nRolling control:")
for k, v in data["rolling_control"].items():
    if k != "raw":
        print(f"{k}: {v}")

print("\nShapes:")
for k, v in data.items():
    if hasattr(v, "shape"):
        print(f"{k:30s} -> {v.shape}")

print("\nTEST 01 PASSED")