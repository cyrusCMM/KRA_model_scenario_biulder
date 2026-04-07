# -*- coding: utf-8 -*-
"""
TAX ENGINE V2
Schema-faithful annual forecast engine for KRA model.

Key revisions
-------------
- stronger driver alias resolution
- stronger elasticity alias resolution
- better customs/import fallback logic
- clearer bridge row handling
- richer audit traces for resolved schema pairs
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class TaxEngineError(Exception):
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
    for ch in ["/", "-", "_", ",", ".", "(", ")", "%", "$", ":"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if pd.notna(v) else default
    except Exception:
        return default


def _clip_growth(g: float, floor: float = -0.95, cap: float = 5.0) -> float:
    return max(floor, min(cap, _to_float(g, 0.0)))


def _safe_growth_from_level(new_value: float, old_value: float) -> float:
    new_value = _to_float(new_value, 0.0)
    old_value = _to_float(old_value, 0.0)
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def _require_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise TaxEngineError(f"{label} missing required columns: {missing}")


# ============================================================
# YEAR / POLICY HELPERS
# ============================================================

def _get_selected_year(data: Dict[str, Any]) -> str:
    selected_year = _clean(data.get("rolling_control", {}).get("selected_year", ""))
    if selected_year not in {"2025/26", "2026/27", "2027/28"}:
        raise TaxEngineError(
            f"Invalid selected year '{selected_year}'. Expected one of: 2025/26, 2026/27, 2027/28."
        )
    return selected_year


def _get_target_column(selected_year: str) -> str:
    return {
        "2025/26": "Target 2025/26",
        "2026/27": "Target 2026/27",
        "2027/28": "Target 2027/28",
    }[selected_year]


def _get_opening_base_column(selected_year: str) -> str:
    return {
        "2025/26": "Opening Base 2025/26",
        "2026/27": "Opening Base 2026/27",
        "2027/28": "Opening Base 2027/28",
    }[selected_year]


def _get_policy_total_column(selected_year: str) -> str:
    return {
        "2025/26": "Policy Total 2025/26",
        "2026/27": "Policy Total 2026/27",
        "2027/28": "Policy Total 2027/28",
    }[selected_year]


# ============================================================
# LOOKUPS
# ============================================================

def _lookup_macro_col(macro_row: pd.Series, col_name: str, default: float = 0.0) -> float:
    if col_name in macro_row.index:
        return _to_float(macro_row[col_name], default)
    target = _norm(col_name)
    for k in macro_row.index:
        if _norm(k) == target:
            return _to_float(macro_row[k], default)
    return default


def _lookup_param(name: str, params: Dict[str, float], default: float = 0.0) -> float:
    target = _norm(name)
    for k, v in params.items():
        if _norm(k) == target:
            return _to_float(v, default)
    return default


def _build_target_map(targets_df: pd.DataFrame, selected_year: str) -> Dict[str, float]:
    target_col = _get_target_column(selected_year)
    out: Dict[str, float] = {}
    if targets_df is None or targets_df.empty:
        return out
    if "Annex Tax Head" not in targets_df.columns or target_col not in targets_df.columns:
        return out

    work = targets_df[["Annex Tax Head", target_col]].copy()
    work["Annex Tax Head"] = work["Annex Tax Head"].map(_clean)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").fillna(0.0)
    for _, r in work.iterrows():
        out[_clean(r["Annex Tax Head"])] = _to_float(r[target_col], 0.0)
    return out


def _build_policy_map(policy_df: pd.DataFrame, selected_year: str) -> Dict[str, float]:
    col = _get_policy_total_column(selected_year)
    out: Dict[str, float] = {}
    if policy_df is None or policy_df.empty:
        return out
    if "Internal Tax Head" not in policy_df.columns or col not in policy_df.columns:
        return out

    work = policy_df[["Internal Tax Head", col]].copy()
    work["Internal Tax Head"] = work["Internal Tax Head"].map(_clean)
    work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    grouped = work.groupby("Internal Tax Head", as_index=False)[col].sum()
    for _, r in grouped.iterrows():
        out[_clean(r["Internal Tax Head"])] = _to_float(r[col], 0.0)
    return out


def _get_macro_row(macro_df: pd.DataFrame, selected_year: str) -> pd.Series:
    if "year" not in macro_df.columns:
        raise TaxEngineError("macro_df must contain a 'year' column.")
    mask = macro_df["year"].astype(str).str.strip() == selected_year
    if mask.sum() == 0:
        raise TaxEngineError(f"Selected year '{selected_year}' not found in macro data.")
    return macro_df.loc[mask].iloc[0]


# ============================================================
# OPENING BASE PREPARATION
# ============================================================

def build_tax_base_frame(data: Dict[str, Any]) -> pd.DataFrame:
    selected_year = _get_selected_year(data)
    tax_input = data.get("tax_heads_input", pd.DataFrame()).copy()
    tax_roll = data.get("tax_heads_rolling", pd.DataFrame()).copy()

    _require_columns(
        tax_input,
        ["Department", "Revenue Type", "Internal Tax Head", "Annex Tax Head", "Actual 2024/25"],
        "tax_heads_input",
    )
    _require_columns(
        tax_roll,
        ["Internal Tax Head", "Annex Tax Head", _get_opening_base_column(selected_year)],
        "tax_heads_rolling",
    )

    base_col = _get_opening_base_column(selected_year)
    left = tax_input.copy()
    right = tax_roll[["Internal Tax Head", "Annex Tax Head", base_col]].copy()

    left["Internal Tax Head"] = left["Internal Tax Head"].map(_clean)
    left["Annex Tax Head"] = left["Annex Tax Head"].map(_clean)
    right["Internal Tax Head"] = right["Internal Tax Head"].map(_clean)
    right["Annex Tax Head"] = right["Annex Tax Head"].map(_clean)

    merged = left.merge(right, on=["Internal Tax Head", "Annex Tax Head"], how="left")
    merged["Actual 2024/25"] = pd.to_numeric(merged["Actual 2024/25"], errors="coerce").fillna(0.0)
    merged[base_col] = pd.to_numeric(merged[base_col], errors="coerce").fillna(0.0)
    merged = merged.rename(columns={base_col: "Opening Base"})
    return merged


# ============================================================
# SCHEMA RESOLUTION
# ============================================================

_DRIVER_ALIASES = {
    # GDP / prices
    "real gdp growth": "real_gdp_growth",
    "nominal gdp growth": "nominal_gdp_growth",
    "gdp deflator": "gdp_deflator",
    "cpi": "cpi",

    # wages
    "wage rate % change": "wage_growth",
    "wage rate change": "wage_growth",
    "wage growth": "wage_growth",
    "bracket creep": "cpi",

    # external / trade
    "import value growth": "import_growth",
    "imports growth": "import_growth",
    "export value growth": "export_growth",
    "exports growth": "export_growth",
    "non oil import value growth dry": "non_oil_import_growth",
    "non oil import value growth": "non_oil_import_growth",
    "non oil import growth": "non_oil_import_growth",
    "oil value growth": "oil_value_growth",
    "oil import growth": "oil_value_growth",

    # oil
    "oil world price change in us": "oil_price_change",
    "oil world price growth": "oil_price_change",
    "oil world price shock": "oil_price_change",
    "oil price growth": "oil_price_change",
    "oil % volume change": "oil_volume_change",
    "oil volume growth": "oil_volume_change",
    "oil volume shock": "oil_volume_change",

    # other
    "implied profitability growth": "implied_profitability_growth",
    "profitability growth": "implied_profitability_growth",
    "profitability growth shock": "implied_profitability_growth",
    "exchange rate change": "exchange_rate_change",
    "exchange rate growth": "exchange_rate_change",
    "exchange rate shock": "exchange_rate_change",

    # parameter-like driver
    "aal default growth": "__PARAM__aal_default_growth",
}

_ELASTICITY_ALIASES = {
    "tariff elasticity for non oil imports": "tariff_non_oil",
    "tariff elasticity for oil": "tariff_oil",

    "oil excise elasticity": "excise_oil_gdp",
    "excise oil crude price": "excise_oil_crude_price",
    "non oil import excise elasticity": "excise_non_oil_imports",

    "vat elasticity on imports": "vat_import_non_oil",
    "vat elasticity on ordinary imports": "vat_import_non_oil",
    "vat elasticity on non oil imports": "vat_import_non_oil",

    "vat elasticity on oil": "vat_import_oil_gdp",
    "vat elasticity on imports oil": "vat_import_oil_gdp",
    "vat elasticity on oil imports": "vat_import_oil_gdp",

    "domestic vat elasticity w r t gdp": "vat_domestic_gdp",
    "domestic vat elasticity wrt gdp": "vat_domestic_gdp",
    "domestic vat elasticity with respect to gdp": "vat_domestic_gdp",

    "other income tax elasticity": "other_income",
    "non oil domestic excise elasticity": "excise_non_oil_domestic",

    "idf elasticity": "idf_elasticity",
    "rdl elasticity": "rdl_elasticity",
    "export levy elasticity": "export_levy_elasticity",

    "traffic proxy": "traffic_proxy",
    "aal default growth": "aal_default_growth",

    "paye wage": "paye_wage",
    "paye bracket": "paye_bracket",
    "proxy nominal": "proxy_nominal",
}


def _resolve_driver_value(driver_name: str, macro_row: pd.Series, params: Dict[str, float]) -> float:
    d = _norm(driver_name)
    if d == "":
        return 0.0

    alias = _DRIVER_ALIASES.get(d)
    if alias and alias.startswith("__PARAM__"):
        return _lookup_param(alias.replace("__PARAM__", ""), params, 0.0)

    if alias:
        return _lookup_macro_col(macro_row, alias, 0.0)

    return _lookup_macro_col(macro_row, driver_name, 0.0)


def _resolve_elasticity_value(elasticity_name: str, params: Dict[str, float], default: float = 1.0) -> float:
    e = _clean(elasticity_name)
    if e == "":
        return default

    try:
        return float(e)
    except Exception:
        pass

    lookup_key = _ELASTICITY_ALIASES.get(_norm(e), e)
    return _lookup_param(lookup_key, params, default)


def _schema_pairs(row: pd.Series) -> List[Tuple[str, str]]:
    pair_order = [
        ("Improved Driver 1", "Improved Elasticity 1"),
        ("Improved Driver 2", "Improved Elasticity 2"),
        ("Driver 1", "Elasticity 1"),
        ("Driver 2", "Elasticity 2"),
    ]

    pairs: List[Tuple[str, str]] = []
    seen = set()

    for dcol, ecol in pair_order:
        if dcol not in row.index or ecol not in row.index:
            continue

        d = _clean(row.get(dcol, ""))
        e = _clean(row.get(ecol, ""))

        if not d or not e:
            continue

        key = (_norm(d), _norm(e))
        if key not in seen:
            pairs.append((d, e))
            seen.add(key)

    return pairs


def _has_schema_fields(row: pd.Series) -> bool:
    candidates = [
        "Formula Type",
        "Driver 1", "Elasticity 1",
        "Driver 2", "Elasticity 2",
        "Improved Driver 1", "Improved Elasticity 1",
        "Improved Driver 2", "Improved Elasticity 2",
    ]
    existing = [c for c in candidates if c in row.index]
    return any(_clean(row.get(c, "")) != "" for c in existing)


def _compute_schema_growth(
    row: pd.Series,
    macro_row: pd.Series,
    params: Dict[str, float],
) -> Tuple[Optional[float], List[Tuple[str, float, str, float]]]:
    if not _has_schema_fields(row):
        return None, []

    pair_defs = _schema_pairs(row)
    if not pair_defs:
        return None, []

    resolved = []
    components = []

    for dname, ename in pair_defs:
        dval = _resolve_driver_value(dname, macro_row, params)
        eval_ = _resolve_elasticity_value(ename, params, default=1.0)
        resolved.append((dname, dval, ename, eval_))
        components.append(dval * eval_)

    return _clip_growth(sum(components)), resolved


# ============================================================
# FALLBACK LOGIC
# ============================================================

def _compute_fallback_growth(
    internal_raw: str,
    annex_raw: str,
    macro_row: pd.Series,
    params: Dict[str, float],
) -> float:
    internal = _norm(internal_raw)
    annex = _norm(annex_raw)

    real_gdp = _lookup_macro_col(macro_row, "real_gdp_growth", 0.0)
    nom_gdp = _lookup_macro_col(macro_row, "nominal_gdp_growth", 0.0)
    cpi = _lookup_macro_col(macro_row, "cpi", 0.0)
    wage_growth = _lookup_macro_col(macro_row, "wage_growth", 0.0)
    import_growth = _lookup_macro_col(macro_row, "import_growth", 0.0)
    export_growth = _lookup_macro_col(macro_row, "export_growth", 0.0)
    non_oil_import_growth = _lookup_macro_col(macro_row, "non_oil_import_growth", 0.0)
    oil_value_growth = _lookup_macro_col(macro_row, "oil_value_growth", 0.0)
    profitability_growth = _lookup_macro_col(macro_row, "implied_profitability_growth", 0.0)

    # Customs / import block
    if internal in {"import duty ordinary (net)", "import duty ordinary"} or annex == "import duty":
        return _clip_growth(
            non_oil_import_growth * _lookup_param("tariff_non_oil", params, 0.0)
            + oil_value_growth * _lookup_param("tariff_oil", params, 0.0)
        )

    if internal == "excise duty oil":
        coeff = _lookup_param("excise_oil_crude_price", params, _lookup_param("excise_oil_gdp", params, 0.0))
        return _clip_growth(oil_value_growth * coeff)

    if internal == "excise duty ordinary":
        return _clip_growth(non_oil_import_growth * _lookup_param("excise_non_oil_imports", params, 0.0))

    if internal == "less provision for refunds":
        return 0.0

    if internal == "vat imports ordinary":
        return _clip_growth(non_oil_import_growth * _lookup_param("vat_import_non_oil", params, 0.0))

    if internal == "vat imports oil":
        return _clip_growth(oil_value_growth * _lookup_param("vat_import_oil_gdp", params, 0.0))

    if internal == "idf fees" or annex == "idf fees":
        return _clip_growth(import_growth * _lookup_param("idf_elasticity", params, 1.0))

    if internal == "railway development levy" or annex == "railway development levy":
        return _clip_growth(import_growth * _lookup_param("rdl_elasticity", params, 1.0))

    if internal in {"export and investment promotion levy", "export levy"} or annex in {"export and investment promotion levy", "export levy"}:
        return _clip_growth(export_growth * _lookup_param("export_levy_elasticity", params, 1.0))

    if internal == "anti adulteration levy" or annex == "anti adulteration levy":
        return _clip_growth(_lookup_param("aal_default_growth", params, 0.025))

    # Domestic / income block
    if internal == "paye" or annex == "paye":
        return _clip_growth(
            wage_growth * _lookup_param("paye_wage", params, 1.0)
            + cpi * _lookup_param("paye_bracket", params, 0.0)
        )

    if internal in {"other income taxes", "capital gains tax"} or annex in {"other income taxes", "capital gains tax"}:
        return _clip_growth(profitability_growth * _lookup_param("other_income", params, 0.0))

    if internal == "domestic vat" or annex == "domestic vat":
        return _clip_growth(nom_gdp * _lookup_param("vat_domestic_gdp", params, 0.0))

    if internal in {
        "excise domestic",
        "excise financial transactions",
        "excise on financial transactions",
        "excise financial transaction",
        "excise - financial transactions",
        "excise on airtime",
        "excise betting services",
        "betting tax",
        "digital service tax",
        "significant economic presence tax",
        "sep tax",
    } or annex in {
        "excise domestic",
        "excise financial transactions",
        "excise on financial transactions",
        "excise financial transaction",
        "excise - financial transactions",
        "excise on airtime",
        "excise betting services",
        "betting tax",
        "digital service tax",
        "significant economic presence tax",
        "sep tax",
    }:
        return _clip_growth(nom_gdp * _lookup_param("excise_non_oil_domestic", params, 0.0))

    if internal in {"rent of land", "stamp duty", "rental income", "turnover tax", "surplus funds"} or annex in {
        "rent of land", "stamp duty", "rental income", "turnover tax", "surplus funds"
    }:
        return _clip_growth(nom_gdp * _lookup_param("proxy_nominal", params, 1.0))

    if internal == "traffic exchequer revenue" or annex == "traffic exchequer revenue":
        return _clip_growth(real_gdp * _lookup_param("traffic_proxy", params, 1.0))

    return 0.0


def _prefer_fallback_over_schema(
    internal_raw: str,
    annex_raw: str,
    schema_growth: float,
    fallback_growth: float,
    resolved_pairs: List[Tuple[str, float, str, float]],
) -> bool:
    internal = _norm(internal_raw)
    annex = _norm(annex_raw)

    critical_heads = {
        "import duty ordinary (net)",
        "import duty ordinary",
        "excise duty oil",
        "excise duty ordinary",
        "vat imports ordinary",
        "vat imports oil",
        "idf fees",
        "railway development levy",
        "export and investment promotion levy",
        "export levy",
    }

    if internal in critical_heads or annex in {
        "import duty",
        "idf fees",
        "railway development levy",
        "export levy",
        "export and investment promotion levy",
    }:
        if abs(schema_growth) < 1e-12 and abs(fallback_growth) > 1e-12:
            return True

        if resolved_pairs:
            all_zero = all(
                abs(_to_float(dval, 0.0) * _to_float(eval_, 0.0)) < 1e-12
                for _, dval, _, eval_ in resolved_pairs
            )
            if all_zero and abs(fallback_growth) > 1e-12:
                return True

    return False


# ============================================================
# CORE FORECAST ENGINE
# ============================================================

def build_tax_forecast(data: Dict[str, Any], macro_df: pd.DataFrame) -> pd.DataFrame:
    selected_year = _get_selected_year(data)
    base_df = build_tax_base_frame(data)
    params = dict(data.get("elasticities_dict", {}))
    targets_df = data.get("targets", pd.DataFrame()).copy()
    policy_df = data.get("policy_measures", pd.DataFrame()).copy()

    macro_row = _get_macro_row(macro_df, selected_year)
    target_map = _build_target_map(targets_df, selected_year)
    policy_map = _build_policy_map(policy_df, selected_year)

    rows = []

    for _, r in base_df.iterrows():
        dept = _clean(r.get("Department", ""))
        rev_type = _clean(r.get("Revenue Type", ""))
        annex_raw = _clean(r.get("Annex Tax Head", ""))
        internal_raw = _clean(r.get("Internal Tax Head", ""))
        opening_base = _to_float(r.get("Opening Base", 0.0), 0.0)
        formula_type = _clean(r.get("Formula Type", ""))

        schema_growth, audit_pairs = _compute_schema_growth(r, macro_row, params)
        fallback_growth = _compute_fallback_growth(internal_raw, annex_raw, macro_row, params)

        if schema_growth is None:
            growth = fallback_growth
            logic_source = "fallback"
        else:
            if _prefer_fallback_over_schema(internal_raw, annex_raw, schema_growth, fallback_growth, audit_pairs):
                growth = fallback_growth
                logic_source = "fallback_override"
            else:
                growth = schema_growth
                logic_source = "schema"

        baseline_forecast = opening_base * (1.0 + growth)
        policy_adjustment = _to_float(policy_map.get(internal_raw, 0.0), 0.0)
        final_forecast = baseline_forecast + policy_adjustment

        rows.append({
            "Department": dept,
            "Revenue Type": rev_type,
            "Annex Tax Head": annex_raw,
            "Internal Tax Head": internal_raw,
            "Selected Year": selected_year,
            "Actual 2024/25": _to_float(r.get("Actual 2024/25", 0.0), 0.0),
            "Opening Base": opening_base,
            "Baseline Forecast": baseline_forecast,
            "Policy Adjustment": policy_adjustment,
            "Final Forecast": final_forecast,
            "Target": _to_float(target_map.get(annex_raw, 0.0), 0.0),
            "Applied Growth": growth,
            "Formula Type": formula_type,
            "Logic Source": logic_source,
            "Resolved Pair Count": len(audit_pairs),
            "Resolved Pairs": " | ".join(
                [f"{d}={dval:.6f} * {e}={eval_:.6f}" for d, dval, e, eval_ in audit_pairs]
            ),
        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # BRIDGE LINES
    # --------------------------------------------------------
    excise = df[df["Internal Tax Head"].astype(str).map(_norm).isin(["excise duty oil", "excise duty ordinary"])].copy()
    df = df[df["Annex Tax Head"].astype(str).map(_norm) != "import excise duty"].copy()

    if not excise.empty:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{
                        "Department": "Customs",
                        "Revenue Type": "Bridge",
                        "Annex Tax Head": "Import Excise Duty",
                        "Internal Tax Head": "Import Excise Duty",
                        "Selected Year": selected_year,
                        "Actual 2024/25": excise["Actual 2024/25"].sum(),
                        "Opening Base": excise["Opening Base"].sum(),
                        "Baseline Forecast": excise["Baseline Forecast"].sum(),
                        "Policy Adjustment": excise["Policy Adjustment"].sum(),
                        "Final Forecast": excise["Final Forecast"].sum(),
                        "Target": _to_float(target_map.get("Import Excise Duty", 0.0), 0.0),
                        "Applied Growth": _safe_growth_from_level(excise["Final Forecast"].sum(), excise["Opening Base"].sum()),
                        "Formula Type": "bridge",
                        "Logic Source": "bridge",
                        "Resolved Pair Count": 0,
                        "Resolved Pairs": "",
                    }]
                )
            ],
            ignore_index=True,
        )

    vat = df[df["Internal Tax Head"].astype(str).map(_norm).isin(["vat imports ordinary", "vat imports oil"])].copy()
    df = df[df["Annex Tax Head"].astype(str).map(_norm) != "vat imports"].copy()
    df = df[df["Annex Tax Head"].astype(str).map(_norm) != "vat imports ordinary"].copy()
    df = df[df["Annex Tax Head"].astype(str).map(_norm) != "vat imports oil"].copy()

    if not vat.empty:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [{
                        "Department": "Customs",
                        "Revenue Type": "Bridge",
                        "Annex Tax Head": "VAT, imports",
                        "Internal Tax Head": "VAT, imports",
                        "Selected Year": selected_year,
                        "Actual 2024/25": vat["Actual 2024/25"].sum(),
                        "Opening Base": vat["Opening Base"].sum(),
                        "Baseline Forecast": vat["Baseline Forecast"].sum(),
                        "Policy Adjustment": vat["Policy Adjustment"].sum(),
                        "Final Forecast": vat["Final Forecast"].sum(),
                        "Target": _to_float(target_map.get("VAT, imports", 0.0), 0.0),
                        "Applied Growth": _safe_growth_from_level(vat["Final Forecast"].sum(), vat["Opening Base"].sum()),
                        "Formula Type": "bridge",
                        "Logic Source": "bridge",
                        "Resolved Pair Count": 0,
                        "Resolved Pairs": "",
                    }]
                )
            ],
            ignore_index=True,
        )

    return df.reset_index(drop=True)


# ============================================================
# SUMMARIES
# ============================================================

def build_annex_output(detail_df: pd.DataFrame, selected_year: str) -> pd.DataFrame:
    _require_columns(
        detail_df,
        ["Annex Tax Head", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"],
        "detail_df",
    )
    return (
        detail_df.groupby("Annex Tax Head", as_index=False)[
            ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"]
        ]
        .sum()
        .rename(columns={"Annex Tax Head": "Tax head", "Final Forecast": f"Projected Collection {selected_year}"})
    )


def build_department_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        detail_df,
        ["Department", "Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"],
        "detail_df",
    )
    return detail_df.groupby("Department", as_index=False)[
        ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast", "Target"]
    ].sum()


def build_total_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(detail_df, ["Opening Base", "Baseline Forecast", "Policy Adjustment", "Final Forecast"], "detail_df")
    opening = pd.to_numeric(detail_df["Opening Base"], errors="coerce").fillna(0.0).sum()
    baseline = pd.to_numeric(detail_df["Baseline Forecast"], errors="coerce").fillna(0.0).sum()
    policy = pd.to_numeric(detail_df["Policy Adjustment"], errors="coerce").fillna(0.0).sum()
    final = pd.to_numeric(detail_df["Final Forecast"], errors="coerce").fillna(0.0).sum()
    return pd.DataFrame([{
        "Opening Base": opening,
        "Baseline Forecast": baseline,
        "Policy Adjustment": policy,
        "Final Forecast": final,
        "Forecast Growth": ((final - opening) / opening) if opening != 0 else 0.0,
    }])


def run_tax_engine(data: Dict[str, Any], macro_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    selected_year = _get_selected_year(data)
    detail = build_tax_forecast(data, macro_df)
    annex_summary = build_annex_output(detail, selected_year)
    department_summary = build_department_summary(detail)
    total_summary = build_total_summary(detail)
    return {
        "detail": detail,
        "annex_summary": annex_summary,
        "department_summary": department_summary,
        "total_summary": total_summary,
    }


# ============================================================
# AUDIT
# ============================================================

def audit_schema_against_workbook(data: Dict[str, Any], macro_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base_df = build_tax_base_frame(data)
    params = dict(data.get("elasticities_dict", {}))
    macro_row = _get_macro_row(macro_df, _get_selected_year(data))

    rows = []
    for _, r in base_df.iterrows():
        growth, audit_pairs = _compute_schema_growth(r, macro_row, params)
        fallback_growth = _compute_fallback_growth(
            _clean(r.get("Internal Tax Head", "")),
            _clean(r.get("Annex Tax Head", "")),
            macro_row,
            params,
        )
        formula_type = _clean(r.get("Formula Type", ""))

        rows.append({
            "Internal Tax Head": _clean(r.get("Internal Tax Head", "")),
            "Formula Type": formula_type,
            "Schema Present": _has_schema_fields(r),
            "Resolved Pair Count": len(audit_pairs),
            "Schema Growth Computed": growth is not None,
            "Schema Growth": growth,
            "Fallback Growth": fallback_growth,
            "Resolved Pairs": " | ".join(
                [f"{d}={dval:.6f} * {e}={eval_:.6f}" for d, dval, e, eval_ in audit_pairs]
            ),
        })

    audit_df = pd.DataFrame(rows)
    issues = audit_df.loc[
        (audit_df["Schema Present"]) & (~audit_df["Schema Growth Computed"])
    ].copy()

    return {"audit": audit_df, "issues": issues}


if __name__ == "__main__":
    print("=" * 90)
    print("TAX ENGINE V2 REVISED LOADED")
    print("=" * 90)