# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import traceback

import altair as alt
import pandas as pd
import streamlit as st

from app_runner_v3 import run_app
from export_engine_v1 import export_full_package


FISCAL_ORDER = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
MONTH_MAP = {
    1: "Jul",
    2: "Aug",
    3: "Sep",
    4: "Oct",
    5: "Nov",
    6: "Dec",
    7: "Jan",
    8: "Feb",
    9: "Mar",
    10: "Apr",
    11: "May",
    12: "Jun",
}


st.set_page_config(
    page_title="KRA Scenario Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def _clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    if s in {"0", "0.0"}:
        return ""
    return s


def _to_float(x, default=0.0):
    try:
        v = float(x)
        return v if pd.notna(v) else default
    except Exception:
        return default


def _to_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _fmt0(x):
    return f"{_to_float(x):,.0f}"


def _fmt_pct(x):
    return f"{_to_float(x) * 100:,.2f}%"


def _safe_df(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy()


def _month_name_from_index(idx: int) -> str:
    return MONTH_MAP.get(int(idx), f"M{idx}")


def _normalize_override_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _safe_df(df)
    if out.empty:
        return pd.DataFrame()

    drop_cols = [c for c in out.columns if str(c).startswith("_")]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")

    out = out.iloc[[0]].copy().reset_index(drop=True)
    out.columns = [_clean(c) for c in out.columns]
    return out


def _extract_row_meta(base_row: pd.DataFrame) -> dict:
    if base_row.empty:
        return {
            "description": "",
            "scenario_type": "",
            "severity": "",
            "start_month": 10,
            "duration_months": 0,
            "carryover": "No",
            "recovery_profile": "",
            "notes": "",
        }

    row = base_row.iloc[0]

    return {
        "description": _clean_text(row.get("Description", "")),
        "scenario_type": _clean_text(row.get("Scenario Type", "")),
        "severity": _clean_text(row.get("Severity", "")),
        "start_month": _to_int(row.get("Start Month", 10), 10),
        "duration_months": _to_int(row.get("Duration Months", 0), 0),
        "carryover": _clean_text(row.get("Carryover To Next FY", "No")),
        "recovery_profile": _clean_text(row.get("Recovery Profile", "")),
        "notes": _clean_text(row.get("Notes", "")),
    }


@st.cache_data(show_spinner=False)
def load_workbook_objects(workbook_path: str):
    control = pd.read_excel(workbook_path, sheet_name="Control", header=None)
    scenarios = pd.read_excel(workbook_path, sheet_name="CGE_Scenarios")

    selected_year = None
    selected_scenario = None

    for i in range(len(control)):
        key = _clean(control.iloc[i, 0])
        val = control.iloc[i, 1] if control.shape[1] > 1 else None
        if key == "Selected Year":
            selected_year = _clean(val)
        elif key == "Selected Scenario":
            selected_scenario = _clean(val)

    scenarios.columns = [_clean(c) for c in scenarios.columns]
    if "Scenario" in scenarios.columns:
        scenarios["Scenario"] = scenarios["Scenario"].astype(str).str.strip()

    scenario_names = (
        scenarios["Scenario"].dropna().astype(str).str.strip().tolist()
        if "Scenario" in scenarios.columns else []
    )

    return {
        "selected_year": selected_year,
        "selected_scenario": selected_scenario,
        "scenarios_df": scenarios,
        "scenario_names": scenario_names,
    }


def build_editable_scenario_row(base_row: pd.DataFrame) -> pd.DataFrame:
    if base_row.empty:
        return pd.DataFrame()

    preferred_cols = [
        "Scenario",
        "Description",
        "Scenario Type",
        "Severity",
        "Start Month",
        "Duration Months",
        "Carryover To Next FY",
        "Recovery Profile",
        "Real GDP growth shock",
        "GDP deflator shock",
        "CPI shock",
        "Wage growth shock",
        "Import Value Growth shock",
        "Export Value Growth shock",
        "Non oil import value growth shock",
        "Oil volume shock",
        "Oil world price shock",
        "Profitability growth shock",
        "Exchange rate shock",
        "Notes",
    ]

    cols = [c for c in preferred_cols if c in base_row.columns]
    if not cols:
        return base_row.reset_index(drop=True)

    return base_row[cols].reset_index(drop=True)


def build_department_comparison(current_result: dict) -> pd.DataFrame:
    base = _safe_df(current_result["baseline"].get("department_summary"))
    scen = _safe_df(current_result["scenario"].get("department_summary"))

    if base.empty or scen.empty:
        return pd.DataFrame(columns=["Department", "Baseline Final", "Scenario Final", "Impact", "Impact %"])

    left = base[["Department", "Final Forecast"]].rename(columns={"Final Forecast": "Baseline Final"})
    right = scen[["Department", "Final Forecast"]].rename(columns={"Final Forecast": "Scenario Final"})

    out = left.merge(right, on="Department", how="outer")
    out["Baseline Final"] = pd.to_numeric(out["Baseline Final"], errors="coerce").fillna(0.0)
    out["Scenario Final"] = pd.to_numeric(out["Scenario Final"], errors="coerce").fillna(0.0)
    out["Impact"] = out["Scenario Final"] - out["Baseline Final"]
    out["Impact %"] = out.apply(
        lambda r: r["Impact"] / r["Baseline Final"] if _to_float(r["Baseline Final"]) != 0 else 0.0,
        axis=1,
    )
    return out.sort_values("Impact", ascending=False).reset_index(drop=True)


def build_management_monthly_view(current_result: dict) -> pd.DataFrame:
    df = _safe_df(current_result.get("dashboard_pack", {}).get("monthly_total_comparison"))

    if df.empty:
        scen_path = _safe_df(current_result.get("scenario", {}).get("scenario_monthly_path"))
        if scen_path.empty:
            return pd.DataFrame()

        group_cols = ["Fiscal Year", "Month Index", "Month Name"]
        df = (
            scen_path.groupby(group_cols, as_index=False)[
                ["Baseline Monthly Forecast", "Scenario Monthly Forecast", "Monthly Delta"]
            ]
            .sum()
            .rename(columns={"Monthly Delta": "Monthly Impact"})
        )

    for c in ["Baseline Monthly Forecast", "Scenario Monthly Forecast", "Monthly Impact"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["Month Index"] = pd.to_numeric(df["Month Index"], errors="coerce").fillna(0).astype(int)
    df["Month Name"] = df["Month Name"].astype(str).str.strip()
    df = df.sort_values(["Fiscal Year", "Month Index"]).reset_index(drop=True)

    meta = current_result.get("metadata", {})
    start_month = _to_int(meta.get("scenario_start_month", 10), 10)
    duration = _to_int(meta.get("scenario_duration_months", 0), 0)

    if duration > 0:
        end_month = min(12, start_month + duration - 1)
        df["Is Scenario Period"] = (df["Month Index"] >= start_month) & (df["Month Index"] <= end_month)
    else:
        df["Is Scenario Period"] = df["Month Index"] >= start_month

    df["Management Impact"] = df.apply(
        lambda r: r["Monthly Impact"] if bool(r["Is Scenario Period"]) else 0.0,
        axis=1,
    )
    df["Display Month"] = df["Fiscal Year"].astype(str) + "-" + df["Month Name"].astype(str)

    return df


def build_impact_only_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Display Month", "Management Impact"])

    out = df.loc[df["Is Scenario Period"]].copy()
    return out[[
        "Fiscal Year",
        "Month Index",
        "Month Name",
        "Display Month",
        "Baseline Monthly Forecast",
        "Scenario Monthly Forecast",
        "Management Impact",
    ]].reset_index(drop=True)


def build_management_tax_head_view(current_result: dict, top_n: int = 12) -> pd.DataFrame:
    df = _safe_df(current_result["comparisons"].get("tax_head_comparison"))
    if df.empty:
        return pd.DataFrame()

    keep = ["Internal Tax Head", "Baseline Final", "Scenario Final", "Scenario Impact", "Scenario Impact %"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["Scenario Impact"] = pd.to_numeric(out["Scenario Impact"], errors="coerce").fillna(0.0)
    out = out.loc[out["Scenario Impact"].abs() > 0].copy()
    out["Abs Impact"] = out["Scenario Impact"].abs()
    return out.sort_values("Abs Impact", ascending=False).head(top_n).drop(columns=["Abs Impact"]).reset_index(drop=True)


def build_totals_chart(total_comparison: pd.DataFrame) -> alt.Chart:
    row = total_comparison.iloc[0]
    chart_df = pd.DataFrame({
        "Series": ["Baseline", "Scenario"],
        "Revenue": [_to_float(row["Baseline Final"]), _to_float(row["Scenario Final"])],
    })

    return (
        alt.Chart(chart_df)
        .mark_bar(size=90)
        .encode(
            x=alt.X("Series:N", title=""),
            y=alt.Y("Revenue:Q", title="Revenue"),
            tooltip=["Series", alt.Tooltip("Revenue:Q", format=",.0f")],
        )
        .properties(height=320)
    )


def build_monthly_path_chart(monthly_df: pd.DataFrame) -> alt.Chart:
    if monthly_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    long_df = monthly_df.melt(
        id_vars=["Display Month", "Month Index"],
        value_vars=["Baseline Monthly Forecast", "Scenario Monthly Forecast"],
        var_name="Series",
        value_name="Revenue",
    )

    sort_order = monthly_df.sort_values("Month Index")["Display Month"].tolist()

    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Display Month:N", sort=sort_order, title="Month"),
            y=alt.Y("Revenue:Q", title="Revenue"),
            color=alt.Color("Series:N", title="Series"),
            tooltip=["Display Month", "Series", alt.Tooltip("Revenue:Q", format=",.0f")],
        )
        .properties(height=340)
    )


def build_monthly_impact_chart(impact_df: pd.DataFrame) -> alt.Chart:
    if impact_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    sort_order = impact_df.sort_values("Month Index")["Display Month"].tolist()

    return (
        alt.Chart(impact_df)
        .mark_bar()
        .encode(
            x=alt.X("Display Month:N", sort=sort_order, title="Scenario Month"),
            y=alt.Y("Management Impact:Q", title="Scenario Impact"),
            tooltip=[
                "Display Month",
                alt.Tooltip("Baseline Monthly Forecast:Q", format=",.0f"),
                alt.Tooltip("Scenario Monthly Forecast:Q", format=",.0f"),
                alt.Tooltip("Management Impact:Q", format=",.0f"),
            ],
        )
        .properties(height=300)
    )


def build_department_impact_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    sort_order = df.sort_values("Impact")["Department"].tolist()

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Impact:Q", title="Scenario Impact"),
            y=alt.Y("Department:N", sort=sort_order, title="Component"),
            tooltip=[
                "Department",
                alt.Tooltip("Baseline Final:Q", format=",.0f"),
                alt.Tooltip("Scenario Final:Q", format=",.0f"),
                alt.Tooltip("Impact:Q", format=",.0f"),
                alt.Tooltip("Impact %:Q", format=".2%"),
            ],
        )
        .properties(height=240)
    )


def build_tax_head_impact_chart(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    sort_order = df.sort_values("Scenario Impact")["Internal Tax Head"].tolist()

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Scenario Impact:Q", title="Scenario Impact"),
            y=alt.Y("Internal Tax Head:N", sort=sort_order, title="Tax Head"),
            tooltip=[
                "Internal Tax Head",
                alt.Tooltip("Baseline Final:Q", format=",.0f"),
                alt.Tooltip("Scenario Final:Q", format=",.0f"),
                alt.Tooltip("Scenario Impact:Q", format=",.0f"),
                alt.Tooltip("Scenario Impact %:Q", format=".2%"),
            ],
        )
        .properties(height=420)
    )


def render_department_cards(dept_df: pd.DataFrame):
    if dept_df.empty:
        return

    wanted = ["Domestic", "Customs", "Traffic"]
    display = []
    for name in wanted:
        hit = dept_df.loc[dept_df["Department"].astype(str).str.strip().str.lower() == name.lower()]
        if not hit.empty:
            display.append(hit.iloc[0])

    if not display:
        display = [r for _, r in dept_df.head(3).iterrows()]

    cols = st.columns(max(1, len(display)))
    for i, row in enumerate(display):
        cols[i].metric(
            f"{row['Department']} Impact",
            _fmt0(row["Impact"]),
            _fmt_pct(row["Impact %"]),
        )


def render_scenario_configuration(meta: dict):
    st.markdown("### Scenario Configuration")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start Month", _month_name_from_index(_to_int(meta.get("scenario_start_month", 10), 10)))
    c2.metric("Duration (months)", _to_int(meta.get("scenario_duration_months", 0), 0))
    c3.metric("Severity", _clean_text(meta.get("severity", "")))
    c4.metric("Type", _clean_text(meta.get("scenario_type", "")))

    c5, c6 = st.columns(2)
    c5.metric("Recovery", _clean_text(meta.get("recovery_profile", "")))
    c6.metric("Carryover", str(meta.get("carryover_to_next_fy", False)))


def main():
    st.title("KRA Scenario Simulator")
    st.caption("Workbook-driven revenue simulation with timing-aware scenario shocks.")

    st.sidebar.header("Baseline Data")

    default_workbook = str(Path("kra_forecast_input_template_final.xlsx").resolve())
    workbook_path = st.sidebar.text_input("Workbook path", value=default_workbook)

    meta = None
    if Path(workbook_path).exists():
        try:
            meta = load_workbook_objects(workbook_path)
        except Exception as e:
            st.sidebar.error(f"Workbook read error: {e}")

    selected_year = st.sidebar.selectbox(
        "Fiscal Year",
        options=["2025/26", "2026/27", "2027/28"],
        index=0 if meta is None or meta["selected_year"] not in ["2025/26", "2026/27", "2027/28"]
        else ["2025/26", "2026/27", "2027/28"].index(meta["selected_year"]),
    )

    scenario_options = meta["scenario_names"] if meta is not None and meta["scenario_names"] else ["Baseline"]
    selected_scenario = st.sidebar.selectbox(
        "Scenario",
        options=scenario_options,
        index=0 if meta is None or meta["selected_scenario"] not in scenario_options
        else scenario_options.index(meta["selected_scenario"]),
    )

    scenario_mode = st.sidebar.radio("Scenario mode", ["Use Saved Scenario", "Adjust Scenario"])

    base_row = pd.DataFrame()
    editable_row = pd.DataFrame()
    row_meta = {
        "description": "",
        "scenario_type": "",
        "severity": "",
        "start_month": 10,
        "duration_months": 0,
        "carryover": "No",
        "recovery_profile": "",
        "notes": "",
    }

    if meta is not None and not meta["scenarios_df"].empty and selected_scenario in meta["scenario_names"]:
        base_row = meta["scenarios_df"].loc[
            meta["scenarios_df"]["Scenario"].astype(str).str.strip() == selected_scenario
        ].copy()

        if not base_row.empty:
            editable_row = build_editable_scenario_row(base_row)
            row_meta = _extract_row_meta(base_row)

    st.subheader("Scenario Builder")

    c1, c2, c3 = st.columns([1.2, 1.8, 1.2])
    with c1:
        st.markdown("**Scenario Name**")
        st.write(selected_scenario)

    with c2:
        st.markdown("**Description**")
        st.write(row_meta["description"] if row_meta["description"] else "No description provided.")

    with c3:
        st.markdown("**Duration (months)**")
        scenario_duration_months = st.number_input(
            "Duration",
            min_value=0,
            max_value=24,
            value=int(row_meta["duration_months"]),
            step=1,
            label_visibility="collapsed",
            key="duration_input",
        )

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        st.markdown("**Type**")
        st.write(row_meta["scenario_type"] or "-")
    with cc2:
        st.markdown("**Severity**")
        st.write(row_meta["severity"] or "-")
    with cc3:
        st.markdown("**Start Month**")
        st.write(_month_name_from_index(row_meta["start_month"]))
    with cc4:
        st.markdown("**Carryover**")
        st.write(row_meta["carryover"] or "No")

    if row_meta["recovery_profile"] or row_meta["notes"]:
        st.caption(
            f"Recovery: {row_meta['recovery_profile'] or '-'} | Notes: {row_meta['notes'] or '-'}"
        )

    st.caption("Edits apply to this run only. The source workbook is not modified.")

    override_df = None
    if scenario_mode == "Adjust Scenario" and not editable_row.empty:
        edited = st.data_editor(
            editable_row,
            use_container_width=True,
            num_rows="fixed",
            key="scenario_editor",
        )
        override_df = _normalize_override_df(edited)

        if not override_df.empty:
            override_df.loc[0, "Duration Months"] = int(scenario_duration_months)

    elif not editable_row.empty:
        st.dataframe(editable_row, use_container_width=True)

    run_button = st.button("Run Scenario", type="primary")

    if not run_button:
        st.info("Choose a scenario and click **Run Scenario**.")
        return

    if not Path(workbook_path).exists():
        st.error("Workbook path not found.")
        return

    with st.spinner("Running scenario..."):
        current_result = run_app(
            workbook_path=workbook_path,
            mode="scenario",
            ui_override_df=override_df if scenario_mode == "Adjust Scenario" else None,
            selected_year=selected_year,
            selected_scenario=selected_scenario,
            scenario_duration_months=int(scenario_duration_months) if int(scenario_duration_months) > 0 else None,
            export=False,
        )

    total_comparison = current_result["comparisons"]["total_comparison"].copy()
    dept_comparison = build_department_comparison(current_result)
    mgmt_monthly = build_management_monthly_view(current_result)
    impact_monthly = build_impact_only_monthly(mgmt_monthly)
    tax_head_view = build_management_tax_head_view(current_result, top_n=12)

    st.success("Scenario run complete.")

    run_meta = current_result.get("metadata", {})
    st.markdown(
        f"**FY:** {selected_year} | "
        f"**Scenario:** {selected_scenario} | "
        f"**Mode Used by Engine:** {'Adjusted' if scenario_mode == 'Adjust Scenario' else 'Saved'}"
    )

    render_scenario_configuration(run_meta)

    row = total_comparison.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Revenue", _fmt0(row["Baseline Final"]))
    m2.metric("Scenario Revenue", _fmt0(row["Scenario Final"]))
    m3.metric("Scenario Impact", _fmt0(row["Scenario Impact"]))
    m4.metric("Impact %", _fmt_pct(row["Scenario Impact %"]))

    render_department_cards(dept_comparison)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive View", "Monthly View", "Department View", "Detailed Tables"]
    )

    with tab1:
        st.subheader("Revenue Comparison")
        st.altair_chart(build_totals_chart(total_comparison), use_container_width=True)

        st.subheader("Top Tax Head Impacts")
        st.altair_chart(build_tax_head_impact_chart(tax_head_view), use_container_width=True)

    with tab2:
        st.subheader("Baseline vs Scenario — Monthly Revenue Path")
        if mgmt_monthly.empty:
            st.info("Monthly comparison is not available.")
        else:
            st.altair_chart(build_monthly_path_chart(mgmt_monthly), use_container_width=True)

        st.subheader("Scenario Impact by Month")
        if impact_monthly.empty:
            st.info("No scenario-period monthly impact available for the selected year.")
        else:
            first_month = impact_monthly["Month Name"].iloc[0]
            st.caption(f"Impact starts from scenario start month: {first_month}.")
            st.altair_chart(build_monthly_impact_chart(impact_monthly), use_container_width=True)

    with tab3:
        st.subheader("Domestic / Customs / Traffic Components")
        if dept_comparison.empty:
            st.info("Department comparison not available.")
        else:
            st.altair_chart(build_department_impact_chart(dept_comparison), use_container_width=True)
            st.dataframe(dept_comparison, use_container_width=True)

    with tab4:
        st.subheader("Scenario Definition Used")
        used_df = override_df if (scenario_mode == "Adjust Scenario" and override_df is not None and not override_df.empty) else editable_row
        st.dataframe(used_df, use_container_width=True)

        st.subheader("Run Metadata Returned by Engine")
        st.json(run_meta)

        st.subheader("Total Comparison")
        st.dataframe(total_comparison, use_container_width=True)

        st.subheader("Tax Head Comparison")
        st.dataframe(current_result["comparisons"]["tax_head_comparison"], use_container_width=True)

        if "annex_comparison" in current_result["comparisons"]:
            st.subheader("Annex Comparison")
            st.dataframe(current_result["comparisons"]["annex_comparison"], use_container_width=True)

        st.subheader("Monthly Comparison")
        st.dataframe(mgmt_monthly, use_container_width=True)

    st.header("Download Output")
    if st.button("Generate Excel Export"):
        output_file = Path("streamlit_export.xlsx").resolve()

        export_full_package(
            filepath=output_file,
            baseline_outputs=current_result["baseline"],
            scenario_outputs=current_result["scenario"],
            comparisons=current_result["comparisons"],
            dashboard_pack=current_result["dashboard_pack"],
            decomposition_pack=current_result["decomposition_pack"],
            metadata=current_result["metadata"],
        )

        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Excel",
                data=f,
                file_name="kra_forecast_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.code(traceback.format_exc())