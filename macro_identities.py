# -*- coding: utf-8 -*-
"""
MACRO IDENTITIES
Single source of macro truth for KRA forecasting model

Purpose
-------
- Convert workbook macro inputs into consistent annual drivers
- Enforce identity relationships
- Produce clean year-level macro table
- Add derived variables needed by tax_engine_v2
"""

from typing import Dict, Any
import pandas as pd


class MacroIdentitiesError(Exception):
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _norm(x: Any) -> str:
    s = _clean(x).lower()
    s = s.replace("&", " and ")
    for ch in ["/", "-", "_", ",", ".", "(", ")", "%"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if pd.notna(v) else default
    except Exception:
        return default


# ============================================================
# CORE EXTRACTION
# ============================================================

def _macro_to_dict(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    required = ["Variable", "2025/26", "2026/27", "2027/28"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MacroIdentitiesError(f"Macro sheet missing columns: {missing}")

    out: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        var = _clean(row["Variable"])
        if not var:
            continue

        out[var] = {
            "2025/26": _to_float(row["2025/26"]),
            "2026/27": _to_float(row["2026/27"]),
            "2027/28": _to_float(row["2027/28"]),
        }

    return out


def _get_macro_value(
    macro_dict: Dict[str, Dict[str, float]],
    candidates: list[str],
    year: str,
    default: float = 0.0,
) -> float:
    for key in candidates:
        if key in macro_dict:
            return _to_float(macro_dict[key].get(year, default), default)

    target_candidates = {_norm(c) for c in candidates}
    for k, v in macro_dict.items():
        if _norm(k) in target_candidates:
            return _to_float(v.get(year, default), default)

    return default


# ============================================================
# IDENTITY BUILDER
# ============================================================

def build_macro_driver_table(
    macro_input_df: pd.DataFrame,
    overwrite_existing: bool = True,
) -> pd.DataFrame:
    """
    Output columns:
    - year
    - nominal_gdp_growth
    - real_gdp_growth
    - gdp_deflator
    - cpi
    - wage_growth
    - import_growth
    - export_growth
    - non_oil_import_growth
    - oil_price_change
    - oil_volume_change
    - oil_value_growth
    - implied_profitability_growth
    - exchange_rate_change
    """

    macro_dict = _macro_to_dict(macro_input_df)
    years = ["2025/26", "2026/27", "2027/28"]

    rows = []

    for y in years:
        real_gdp = _get_macro_value(
            macro_dict,
            ["Real GDP growth"],
            y,
            0.0,
        )

        deflator = _get_macro_value(
            macro_dict,
            ["GDP deflator"],
            y,
            0.0,
        )

        cpi = _get_macro_value(
            macro_dict,
            ["CPI"],
            y,
            0.0,
        )

        wage_growth = _get_macro_value(
            macro_dict,
            ["Wage rate (% change)", "Wage growth", "Wage rate change"],
            y,
            0.0,
        )

        import_growth = _get_macro_value(
            macro_dict,
            ["Import Value Growth"],
            y,
            0.0,
        )

        export_growth = _get_macro_value(
            macro_dict,
            ["Export Value Growth"],
            y,
            0.0,
        )

        non_oil_import_growth = _get_macro_value(
            macro_dict,
            ["Non oil import value growth (Dry)", "Non oil import value growth", "Non oil import growth"],
            y,
            0.0,
        )

        oil_price_change = _get_macro_value(
            macro_dict,
            ["Oil World price change (in US$)", "Oil world price shock", "Oil world price growth", "Oil price growth"],
            y,
            0.0,
        )

        oil_volume_change = _get_macro_value(
            macro_dict,
            ["Oil (% volume change)", "Oil volume shock", "Oil volume growth"],
            y,
            0.0,
        )

        exchange_rate_change = _get_macro_value(
            macro_dict,
            ["Exchange rate shock", "Exchange rate growth", "Exchange rate change"],
            y,
            0.0,
        )

        implied_profitability_growth = _get_macro_value(
            macro_dict,
            ["Implied profitability growth", "Profitability growth", "Profitability growth shock"],
            y,
            0.0,
        )

        nominal_gdp_growth = real_gdp + deflator
        oil_value_growth = oil_price_change + oil_volume_change

        rows.append({
            "year": y,
            "nominal_gdp_growth": nominal_gdp_growth,
            "real_gdp_growth": real_gdp,
            "gdp_deflator": deflator,
            "cpi": cpi,
            "wage_growth": wage_growth,
            "import_growth": import_growth,
            "export_growth": export_growth,
            "non_oil_import_growth": non_oil_import_growth,
            "oil_price_change": oil_price_change,
            "oil_volume_change": oil_volume_change,
            "oil_value_growth": oil_value_growth,
            "implied_profitability_growth": implied_profitability_growth,
            "exchange_rate_change": exchange_rate_change,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise MacroIdentitiesError("Macro driver table is empty")

    if df["year"].duplicated().any():
        raise MacroIdentitiesError("Duplicate years in macro table")

    return df


# ============================================================
# SCENARIO SHOCK APPLICATION
# ============================================================

def apply_macro_shock(
    baseline_macro_df: pd.DataFrame,
    shocked_macro_df: pd.DataFrame,
) -> pd.DataFrame:
    if "year" not in baseline_macro_df.columns:
        raise MacroIdentitiesError("baseline macro missing 'year'")
    if "year" not in shocked_macro_df.columns:
        raise MacroIdentitiesError("shocked macro missing 'year'")

    required_cols = [
        "nominal_gdp_growth",
        "real_gdp_growth",
        "gdp_deflator",
        "cpi",
        "wage_growth",
        "import_growth",
        "export_growth",
        "non_oil_import_growth",
        "oil_price_change",
        "oil_volume_change",
        "oil_value_growth",
        "implied_profitability_growth",
        "exchange_rate_change",
    ]

    missing = [c for c in required_cols if c not in shocked_macro_df.columns]
    if missing:
        raise MacroIdentitiesError(f"Shocked macro missing columns: {missing}")

    if set(baseline_macro_df["year"]) != set(shocked_macro_df["year"]):
        raise MacroIdentitiesError("Baseline and shocked macro years mismatch")

    return shocked_macro_df.copy()