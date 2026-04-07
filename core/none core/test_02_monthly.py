# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:08:36 2026

@author: hp
"""

from pathlib import Path
from rolling_loader_v3 import load_all_inputs
from monthly_engine_v1 import run_monthly_baseline_pipeline

WORKBOOK = Path("kra_forecast_input_template_final_for_python.xlsx").resolve()

print("=" * 90)
print("TEST 02 - MONTHLY ENGINE")
print("=" * 90)

data = load_all_inputs(WORKBOOK, validate=True)

# Dummy annual frame from workbook rolling table for test only
annual_detail = data["tax_heads_rolling"][["Internal Tax Head", "Final 2025/26"]].copy()
annual_detail = annual_detail.rename(columns={"Final 2025/26": "Final Forecast"})

outputs = run_monthly_baseline_pipeline(
    annual_detail_df=annual_detail,
    monthly_collections_df=data["monthly_collections_normalized"],
    monthly_mapping_df=data["monthly_mapping"],
    reference_fiscal_year=data["rolling_control"]["reference_share_year"],
    current_fiscal_year=data["rolling_control"]["current_fiscal_year"],
    months_loaded=data["rolling_control"]["actual_months_loaded"],
    annual_value_col="Final Forecast",
)

print("\nActual YTD:")
print(outputs["actual_ytd"].head())

print("\nRebuild check:")
print(outputs["rebuild_check"].head(20))

bad = outputs["rebuild_check"].loc[~outputs["rebuild_check"]["Pass"].astype(bool)]
print("\nFailed rebuild rows:", len(bad))

print("\nTEST 02 COMPLETED")