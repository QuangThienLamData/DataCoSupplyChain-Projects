"""Layer 1 — STL residual anomaly detector.

For each product, decompose each of the watched series (`qty`, `p_eff`,
`revenue_realized`, `elasticity`) into trend + seasonal + residual via STL.
Flag any month whose residual is beyond ±k·σ of the in-sample residual
standard deviation. Default k = 3.

Notes
-----
- STL on monthly data needs at least 2 full seasonal cycles (24 obs). For
  products with <24 obs in the clean window we fall back to a simpler
  detrended-rolling-std approach.
- Elasticity series come from M2's monthly file. They're noisier than
  quantity / price; bump the threshold to 3.5σ for that series.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
ELASTICITY_MONTHLY = ROOT / "forecasts" / "m2_elasticity_monthly.parquet"

WATCHED_SERIES = ["qty", "p_eff", "revenue_realized"]
ELASTICITY_SERIES = "elasticity"

DEFAULT_K_SIGMA = 3.0
ELASTICITY_K_SIGMA = 3.5
SEASONAL_PERIOD = 12
MIN_OBS_FOR_STL = 24


def _resid_stl(series: pd.Series) -> pd.Series:
    """Return the STL residual for a series (NaN-padded to original length)."""
    from statsmodels.tsa.seasonal import STL
    y = series.astype(float).copy()
    y = y.interpolate(limit_direction="both")
    if y.notna().sum() < MIN_OBS_FOR_STL:
        # Fallback: detrend (12-month rolling mean) → residual
        trend = y.rolling(SEASONAL_PERIOD, min_periods=3, center=True).mean()
        return (y - trend).reindex(series.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = STL(y.values, period=SEASONAL_PERIOD, robust=True).fit()
            return pd.Series(res.resid, index=series.index)
        except Exception:
            trend = y.rolling(SEASONAL_PERIOD, min_periods=3, center=True).mean()
            return (y - trend).reindex(series.index)


def detect_stl(panel: pd.DataFrame,
               elasticity_monthly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run STL detector across the full panel. Returns long-form alerts
    with one row per (product, month, series) that fires."""
    clean = panel[panel["data_quality"] == "ok"].copy()
    out: list[pd.DataFrame] = []

    for pid, g in clean.sort_values(["product_card_id", "year_month"]).groupby("product_card_id"):
        for col in WATCHED_SERIES:
            if col not in g.columns or g[col].notna().sum() < 6:
                continue
            resid = _resid_stl(g.set_index("year_month")[col])
            sigma = float(np.nanstd(resid)) if np.isfinite(resid).any() else 0.0
            if sigma == 0:
                continue
            z = resid / sigma
            mask = z.abs() > DEFAULT_K_SIGMA
            if mask.any():
                hits = pd.DataFrame({
                    "product_card_id": pid,
                    "year_month": z.index[mask],
                    "series": col,
                    "z_score": z[mask].values,
                    "resid_value": resid[mask].values,
                    "layer": "stl",
                })
                out.append(hits)

    # Elasticity series — separate source
    if elasticity_monthly is not None and not elasticity_monthly.empty:
        for pid, g in elasticity_monthly.sort_values(
            ["product_card_id", "year_month"]).groupby("product_card_id"):
            resid = _resid_stl(g.set_index("year_month")[ELASTICITY_SERIES])
            sigma = float(np.nanstd(resid)) if np.isfinite(resid).any() else 0.0
            if sigma == 0:
                continue
            z = resid / sigma
            mask = z.abs() > ELASTICITY_K_SIGMA
            if mask.any():
                hits = pd.DataFrame({
                    "product_card_id": pid,
                    "year_month": z.index[mask],
                    "series": "elasticity",
                    "z_score": z[mask].values,
                    "resid_value": resid[mask].values,
                    "layer": "stl",
                })
                out.append(hits)

    if not out:
        return pd.DataFrame(columns=[
            "product_card_id", "year_month", "series", "z_score",
            "resid_value", "layer"
        ])
    return pd.concat(out, ignore_index=True)
