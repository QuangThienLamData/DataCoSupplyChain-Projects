"""Combine the three layers' hits into a single alert per (product, month),
assign a severity, and attribute a suspected driver.

Severity rules
--------------
- **critical** — three layers fire OR (two layers AND STL z-score > 5)
- **warn**     — two layers fire OR forecast-dev severe single breach OR
                  STL z-score > 4.5
- **info**     — exactly one layer fires

Suspected driver
----------------
We look at which series triggered STL, which risk feature is highest at the
month (relative to product baseline), and whether the calendar fired.
Returns one short label per alert (e.g., "demand-down", "price-up",
"fraud-spike", "disaster-known", "disaster-anomaly").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def _combine_hits(stl: pd.DataFrame,
                   iforest: pd.DataFrame,
                   fdev: pd.DataFrame) -> pd.DataFrame:
    keys = ["product_card_id", "year_month"]
    layers = pd.concat([
        stl[keys].assign(stl=True),
        iforest[keys].assign(iforest=True),
        fdev[keys].assign(forecast_dev=True),
    ], ignore_index=True)
    layers = layers.groupby(keys, as_index=False).agg({
        "stl": "any", "iforest": "any", "forecast_dev": "any",
    }).fillna(False)
    layers["n_layers"] = layers[["stl", "iforest", "forecast_dev"]].sum(axis=1)
    return layers


def _max_stl_z(stl: pd.DataFrame) -> pd.DataFrame:
    if stl.empty:
        return pd.DataFrame(columns=["product_card_id", "year_month",
                                      "max_stl_z", "stl_top_series"])
    g = stl.assign(abs_z=stl["z_score"].abs()) \
           .sort_values("abs_z", ascending=False) \
           .drop_duplicates(["product_card_id", "year_month"])
    return g[["product_card_id", "year_month", "abs_z", "series"]] \
             .rename(columns={"abs_z": "max_stl_z", "series": "stl_top_series"})


def _attribute_driver(row: pd.Series, panel_lookup: dict,
                       risk_lookup: dict, known_lookup: dict) -> str:
    """Heuristic driver attribution. Order of checks reflects severity of cause."""
    key = (row["product_card_id"], row["year_month"])
    known = known_lookup.get(key, 0.0)
    if known > 0:
        return "disaster-known"

    risk = risk_lookup.get(key, {})
    disaster_idx = risk.get("disaster_index", 0.0)
    if disaster_idx > 0.5:
        return "disaster-anomaly"
    if risk.get("p_fraud", 0.0) > 0.05:
        return "fraud-spike"
    if risk.get("p_cancel", 0.0) > 0.05:
        return "cancel-spike"

    stl_series = row.get("stl_top_series", None)
    if isinstance(stl_series, str):
        # Map series to direction by checking the panel value vs rolling median
        panel_row = panel_lookup.get(key)
        if panel_row is not None and stl_series in panel_row:
            val = panel_row[stl_series]
            med = panel_row.get(f"{stl_series}_median", val)
            direction = "down" if val < med else "up"
            base = {"qty": "demand", "p_eff": "price",
                    "revenue_realized": "revenue", "elasticity": "elasticity"}
            return f"{base.get(stl_series, stl_series)}-{direction}"
        return f"{stl_series}-shock"
    return "unspecified"


def assign_severity(hits: pd.DataFrame,
                     stl_summary: pd.DataFrame,
                     panel: pd.DataFrame,
                     risk_drag: pd.DataFrame,
                     known_severity: pd.DataFrame) -> pd.DataFrame:
    df = hits.merge(stl_summary, on=["product_card_id", "year_month"], how="left")
    df["max_stl_z"] = df["max_stl_z"].fillna(0.0)

    # Severity rules
    n = df["n_layers"]; z = df["max_stl_z"]
    df["severity"] = np.select(
        [
            (n == 3) | ((n == 2) & (z > 5)),
            (n == 2) | (z > 4.5),
            n == 1,
        ],
        ["critical", "warn", "info"],
        default="info",
    )

    # Build lookup tables for driver attribution
    pkeys = ["product_card_id", "year_month"]
    pl = panel.set_index(pkeys)[["qty", "p_eff", "revenue_realized", "fraud_rate", "cancel_rate", "late_rate"]]
    # Add rolling medians for direction inference
    pmed = (panel.sort_values(pkeys)
                  .groupby("product_card_id")[["qty", "p_eff", "revenue_realized"]]
                  .transform(lambda s: s.rolling(12, min_periods=3).median()))
    pmed.columns = [c + "_median" for c in pmed.columns]
    pl = pl.join(pmed.set_index(panel.set_index(pkeys).index))
    panel_lookup = pl.to_dict(orient="index")

    risk_lookup = (risk_drag.set_index(pkeys)
                            [["p_fraud", "p_cancel", "p_late", "disaster_index"]]
                            .to_dict(orient="index"))
    known_lookup = (known_severity.set_index(pkeys)["known_severity"].to_dict()
                    if not known_severity.empty else {})

    df["suspected_driver"] = df.apply(
        lambda r: _attribute_driver(r, panel_lookup, risk_lookup, known_lookup),
        axis=1,
    )
    return df
