"""Layer 3 — Forecast-deviation detector.

For every (product, month) in val + test that M1 forecasted, flag when the
actual realised value falls outside the P5–P95 band. We compute the band
from M1's (q10, q50, q90) by extending to the 5/95 quantiles assuming a
lognormal shape (consistent with M4's MC).

Disaster-adjustment
-------------------
Per the M1↔M4 architecture decision (see memory), M1 doesn't see disasters.
For known-disaster months (the hurricane calendar fired), we widen the
expected band by the disaster damping: the lower bound becomes
`q5 × (1 − DISASTER_DAMPING · disaster_index)`. This prevents the layer
from re-flagging every known-disaster month as a fresh anomaly.

Threshold
---------
We require **2 consecutive months** of breach, OR a single breach beyond
P1/P99, to trigger an alert.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sales.forecast import DISASTER_DAMPING

ROOT = Path(__file__).resolve().parents[3]
M1_FORECAST_PATH = ROOT / "forecasts" / "m1_demand.parquet"
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
RISK_DRAG_PATH = ROOT / "forecasts" / "m3_risk_drag.parquet"

# Approx normal quantile (Z) for 5% / 95% from q10 / q90
# q10 = exp(μ − Z90 σ), q90 = exp(μ + Z90 σ);
# q5  = exp(μ − Z95 σ),  q95 = exp(μ + Z95 σ).
Z90 = 1.2816
Z95 = 1.6449
WIDEN = Z95 / Z90  # ≈ 1.283


def widen_to_5_95(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extend (q10, q90) to (q5, q95) assuming lognormal-symmetric in log space."""
    log_lo = np.log(np.maximum(q10, 1e-6))
    log_mid = np.log(np.maximum(q50, 1e-6))
    log_hi = np.log(np.maximum(q90, 1e-6))
    half_log_band = (log_hi - log_lo) / 2
    q5 = np.exp(log_mid - WIDEN * half_log_band)
    q95 = np.exp(log_mid + WIDEN * half_log_band)
    return q5, q95


def detect_forecast_deviation(
    m1: pd.DataFrame,
    panel: pd.DataFrame,
    risk_drag: pd.DataFrame,
    model_name: str = "timesfm",
) -> pd.DataFrame:
    """Per (product, month) in M1 val+test, flag breaches."""
    fc = m1[m1["model"] == model_name].copy()
    fc = fc.sort_values(["product_card_id", "year_month"]).reset_index(drop=True)

    q5, q95 = widen_to_5_95(fc["q10"].to_numpy(),
                             fc["q50"].to_numpy(),
                             fc["q90"].to_numpy())
    fc["q5"] = q5; fc["q95"] = q95

    # Disaster widening — pull disaster_index from risk_drag, scale band
    fc = fc.merge(
        risk_drag[["product_card_id", "year_month", "disaster_index"]],
        on=["product_card_id", "year_month"], how="left",
    )
    fc["disaster_index"] = fc["disaster_index"].fillna(0.0)
    fc["q5_adj"] = fc["q5"] * (1 - DISASTER_DAMPING * fc["disaster_index"])

    # Actuals
    actuals = panel[(panel["data_quality"] == "ok")][
        ["product_card_id", "year_month", "gross_qty"]]
    fc = fc.merge(actuals, on=["product_card_id", "year_month"], how="left")

    fc["above_95"] = fc["gross_qty"] > fc["q95"]
    fc["below_5"]  = fc["gross_qty"] < fc["q5_adj"]
    fc["breach"]   = fc["above_95"] | fc["below_5"]

    # Single severe breach beyond P1 / P99 (use 2× widening as proxy)
    fc["q1"]   = np.maximum(fc["q5_adj"] - 2 * (fc["q5_adj"] - fc["q5"]), 0)
    fc["q99"]  = fc["q95"] + 2 * (fc["q95"] - fc["q90"])
    fc["severe_single"] = (fc["gross_qty"] > fc["q99"]) | \
                          (fc["gross_qty"] < np.maximum(fc["q1"], 0))

    # 2-consecutive-month rule per product
    fc = fc.sort_values(["product_card_id", "year_month"])
    fc["breach_prev"] = fc.groupby("product_card_id")["breach"].shift(1).fillna(False)
    fc["two_consecutive"] = fc["breach"] & fc["breach_prev"]

    hits = fc[fc["severe_single"] | fc["two_consecutive"]].copy()
    if hits.empty:
        return pd.DataFrame(columns=[
            "product_card_id", "year_month", "score", "layer",
            "direction", "disaster_widened"
        ])
    hits["direction"] = np.where(hits["above_95"], "up",
                                  np.where(hits["below_5"], "down", "n/a"))
    hits["disaster_widened"] = hits["disaster_index"] > 0
    # Score = relative excess vs band
    band = fc["q95"] - fc["q5_adj"]
    hits["score"] = np.maximum(
        (hits["gross_qty"] - hits["q95"]) / band.loc[hits.index].replace(0, np.nan),
        (hits["q5_adj"] - hits["gross_qty"]) / band.loc[hits.index].replace(0, np.nan),
    ).fillna(1.0).clip(0, None)
    hits["layer"] = "forecast_dev"
    return hits[[
        "product_card_id", "year_month", "score", "layer",
        "direction", "disaster_widened"
    ]]
