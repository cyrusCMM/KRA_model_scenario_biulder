# scenario_builder_v1.py

import pandas as pd


def build_scenario_macro(
    baseline_macro_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    selected_scenario: str,
    selected_year: str,
) -> pd.DataFrame:
    """
    Build shocked macro by applying scenario shocks to baseline macro.

    Returns FULL macro dataframe (not dict).
    """

    if baseline_macro_df is None or baseline_macro_df.empty:
        raise ValueError("Baseline macro is empty")

    if scenarios_df is None or scenarios_df.empty:
        raise ValueError("Scenario table is empty")

    # Clean inputs
    baseline = baseline_macro_df.copy()
    baseline["year"] = baseline["year"].astype(str).str.strip()

    scen = scenarios_df.copy()
    scen["Scenario"] = scen["Scenario"].astype(str).str.strip()
    scen["year"] = scen["year"].astype(str).str.strip()

    # Filter scenario
    scen_row = scen[
        (scen["Scenario"] == selected_scenario)
        & (scen["year"] == selected_year)
    ]

    if scen_row.empty:
        # fallback → no shock
        return baseline.copy()

    scen_row = scen_row.iloc[0].to_dict()

    # Apply shocks
    shocked = baseline.copy()

    shock_map = {
        "real_gdp_growth": "real_gdp_growth",
        "inflation": "inflation",
        "import_value_growth": "import_value_growth",
        "export_value_growth": "export_value_growth",
        "oil_value_growth": "oil_value_growth",
        "implied_profitability_growth": "implied_profitability_growth",
    }

    for scen_col, macro_col in shock_map.items():
        if scen_col in scen_row and macro_col in shocked.columns:
            val = scen_row[scen_col]

            if pd.notna(val):
                shocked.loc[
                    shocked["year"] == selected_year,
                    macro_col
                ] = val

    return shocked


def build_scenario_package(
    baseline_macro_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    selected_scenario: str,
    selected_year: str,
) -> dict:
    """
    Wrapper used by simulation engine.
    """

    shocked_macro = build_scenario_macro(
        baseline_macro_df,
        scenarios_df,
        selected_scenario,
        selected_year,
    )

    return {
        "baseline_macro": baseline_macro_df,
        "shocked_macro": shocked_macro,
        "scenario_name": selected_scenario,
        "selected_year": selected_year,
    }