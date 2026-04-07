# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:16:10 2026

@author: hp
"""

from pathlib import Path
from model.rolling_loader import load_all_inputs

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_ROOT / "inputs" / "kra_forecast_input_template_final_for_python.xlsx"

print("Running model...")

data = load_all_inputs(INPUT_FILE, validate=True)

print("Loaded successfully.")
print("Selected year:", data["rolling_control"]["selected_year"])