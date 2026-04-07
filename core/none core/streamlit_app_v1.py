import streamlit as st
import pandas as pd
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from app_runner_v2 import run_app
from dashboard_builder_v2 import build_monthly_chart_table, build_monthly_impact_table
from export_engine_v1 import export_full_package, export_baseline_outputs

st.set_page_config(page_title="KRA Revenue Forecasting Model", layout="wide")
st.title("KRA Revenue Forecasting Model")


def get_default_shock() -> pd.DataFrame:
    return pd.DataFrame({
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


def show_df(title: str, df: pd.DataFrame):
    st.subheader(title)
    st.dataframe(df, use_container_width=True)


st.sidebar.header("Inputs")
workbook_path = st.sidebar.text_input(
    "Workbook path",
    value=str(CURRENT_DIR / "kra_forecast_input_template_final_for_python.xlsx")
)
mode = st.sidebar.selectbox("Mode", ["baseline", "scenario"])
scenario_name = st.sidebar.text_input("Scenario name", value="Controlled test shock")
run_button = st.sidebar.button("Run Model")

shocked_macro_df = None
if mode == "scenario":
    st.sidebar.subheader("Scenario Shock")
    shocked_macro_df = get_default_shock()
    st.sidebar.dataframe(shocked_macro_df, use_container_width=True)

if run_button:
    try:
        workbook = Path(workbook_path).resolve()
        if not workbook.exists():
            st.error(f"Workbook not found: {workbook}")
            st.stop()

        st.info("Running model...")

        if mode == "baseline":
            result = run_app(
                workbook_path=workbook,
                mode="baseline",
                export=False,
            )
        else:
            result = run_app(
                workbook_path=workbook,
                mode="scenario",
                shocked_macro_df=shocked_macro_df,
                scenario_name=scenario_name,
                scenario_duration_months=None,
                export=False,
            )

        st.success("Model run complete.")

        st.header("Baseline Results")
        show_df("Baseline Total Summary", result["baseline"]["total_summary"])
        show_df("Baseline Tax Head Detail", result["baseline"]["detail"])

        if "monthly_outputs" in result["baseline"]:
            show_df("Baseline Monthly Path", result["baseline"]["monthly_outputs"]["baseline_monthly_path"])

        if mode == "scenario":
            st.header("Scenario Results")
            show_df("Scenario Total Summary", result["scenario"]["total_summary"])
            show_df("Total Comparison", result["comparisons"]["total_comparison"])
            show_df("Tax Head Comparison", result["comparisons"]["tax_head_comparison"])
            show_df("Monthly Total Comparison", result["dashboard_pack"]["monthly_total_comparison"])

            st.subheader("Monthly Baseline vs Scenario")
            chart_df = build_monthly_chart_table(result["dashboard_pack"]["monthly_total_comparison"])
            st.line_chart(chart_df)

            st.subheader("Monthly Impact")
            impact_df = build_monthly_impact_table(result["dashboard_pack"]["monthly_total_comparison"])
            st.bar_chart(impact_df)

            col1, col2 = st.columns(2)
            with col1:
                show_df("Top Gainers", result["dashboard_pack"]["top_gainers"])
            with col2:
                show_df("Top Losers", result["dashboard_pack"]["top_losers"])

            show_df("Decomposition Summary", result["decomposition_pack"]["scenario_decomposition_summary"])

        st.header("Download Output")

        if st.button("Generate Excel Export"):
            output_file = CURRENT_DIR / "streamlit_export.xlsx"

            if mode == "baseline":
                export_baseline_outputs(
                    filepath=output_file,
                    baseline_outputs=result["baseline"],
                )
            else:
                export_full_package(
                    filepath=output_file,
                    baseline_outputs=result["baseline"],
                    scenario_outputs=result["scenario"],
                    comparisons=result["comparisons"],
                    dashboard_pack=result["dashboard_pack"],
                    decomposition_pack=result["decomposition_pack"],
                    metadata=result["metadata"],
                )

            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download Excel",
                    data=f,
                    file_name="kra_forecast_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error: {e}")