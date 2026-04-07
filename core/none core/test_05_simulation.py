# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:19:36 2026

@author: hp
"""

from pathlib import Path
from rolling_loader_v3 import load_all_inputs
from macro_identities import build_macro_driver_table
from rolling_engine_v1 import run_rolling_engine
from simulation_engine_v1 import run_simulation_engine

WORKBOOK = Path("kra_forecast_input_template_final_for_python.xlsx").resolve()

print("=" * 90)
print("TEST 05 - SIMULATION ENGINE")
print("=" * 90)

data = load_all_inputs(WORKBOOK, validate=True)

baseline_macro = build_macro_driver_table(data["macro"], overwrite_existing=True)
baseline_outputs = run_rolling_engine(data=data, macro_df=baseline_macro)

shocked_macro = baseline_macro.copy()
mask = shocked_macro["year"] == data["rolling_control"]["selected_year"]
shocked_macro.loc[mask, "real_gdp_growth"] -= 0.02
shocked_macro.loc[mask, "nominal_gdp_growth"] -= 0.02
shocked_macro.loc[mask, "import_growth"] -= 0.03
shocked_macro.loc[mask, "non_oil_import_growth"] -= 0.03

outputs = run_simulation_engine(
    data=data,
    baseline_outputs=baseline_outputs,
    shocked_macro_df=shocked_macro,
    scenario_duration_months=None,
)

print("\nAnnual delta:")
print(outputs["annual_delta"].head(20))

print("\nMonthly delta totals:")
print(outputs["monthly_delta"].groupby("Internal Tax Head", as_index=False)["Monthly Delta"].sum().head(20))

print("\nScenario total summary:")
print(outputs["total_summary"])

print("\nTEST 05 COMPLETED")