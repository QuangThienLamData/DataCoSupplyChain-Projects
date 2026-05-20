"""Time-varying elasticity series ε_{p,month}.

Per the plan we need a *monthly* elasticity. With only 24–33 obs/product and
narrow price variation, per-product rolling regressions are too noisy. We use
a two-level decomposition instead:

    ε_{p, t}  =  β_pool(t)      +  Δβ_p
                 ──────────       ──────
                 pooled rolling   product-specific
                 elasticity at t  deviation (full-panel)

- `β_pool(t)`: pooled OLS w/ product FE, fit on a trailing 12-month window
  ending at month t. Gives a robust population-level elasticity that varies
  over time.
- `Δβ_p`: each product's deviation from the pooled mean, estimated once on
  the full panel via fixed effects on log_p_eff × product_id. If
  non-identifiable, Δβ_p = 0.

This sidesteps the unidentifiability problem while still emitting a per-
product, per-month number that can be consumed by the M4 sales model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

WINDOW_MONTHS = 12
MIN_OBS_IN_WINDOW = 100  # pooled across products; 12 months × ~10 products


def _fit_pool_window(window_df: pd.DataFrame) -> tuple[float, float] | None:
    """Pooled OLS on a single window; returns (β, se) or None if unfittable."""
    sub = window_df.dropna(subset=["log_qty_lag1"]).copy()
    if len(sub) < MIN_OBS_IN_WINDOW:
        return None
    sub["product_id_str"] = sub["product_card_id"].astype(str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.ols(
                "log_qty ~ log_p_eff + log_p_eff_cat + log_qty_lag1 + "
                "C(month) + C(product_id_str)",
                data=sub,
            ).fit(cov_type="HC3")
        return float(m.params["log_p_eff"]), float(m.bse["log_p_eff"])
    except Exception:
        return None


def pooled_rolling(frame: pd.DataFrame, products: list,
                   window: int = WINDOW_MONTHS) -> pd.DataFrame:
    """Run pooled-FE rolling-window regression; return one row per month."""
    sub = frame[frame["product_card_id"].isin(products)].copy()
    sub = sub.sort_values("year_month")
    months = pd.to_datetime(sorted(sub["year_month"].unique()))

    rows: list[dict] = []
    for end_idx in range(window - 1, len(months)):
        end = months[end_idx]
        start = months[end_idx - window + 1]
        window_df = sub[(sub["year_month"] >= start) &
                        (sub["year_month"] <= end)]
        fit = _fit_pool_window(window_df)
        if fit is None:
            beta, se = float("nan"), float("nan")
            n_obs = len(window_df)
        else:
            beta, se = fit
            n_obs = len(window_df)
        rows.append({
            "year_month": end,
            "window_start": start,
            "window_end": end,
            "n_obs": n_obs,
            "elasticity_pool": beta,
            "elasticity_pool_se": se,
        })
    return pd.DataFrame(rows)


def product_deviations(frame: pd.DataFrame, products: list,
                       per_product: pd.DataFrame) -> pd.DataFrame:
    """For identifiable products, Δβ_p = β_p − mean(β_p).
    For non-identifiable, Δβ_p = 0."""
    pp = per_product[per_product["product_card_id"].isin(products)].copy()
    mean_beta = float(pp.loc[pp["identifiable"], "elasticity"].mean())
    pp["delta_beta"] = np.where(
        pp["identifiable"],
        pp["elasticity"] - mean_beta,
        0.0,
    )
    return pp[["product_card_id", "delta_beta", "identifiable"]]


def build_monthly_elasticity(
    frame: pd.DataFrame,
    products: list,
    per_product: pd.DataFrame,
    window: int = WINDOW_MONTHS,
) -> pd.DataFrame:
    """Final long-form output: one row per (product, month) with elasticity."""
    pool = pooled_rolling(frame, products, window)
    devs = product_deviations(frame, products, per_product)

    rows: list[pd.DataFrame] = []
    months = pool["year_month"]
    for _, prow in devs.iterrows():
        df = pool[["year_month", "elasticity_pool", "elasticity_pool_se"]].copy()
        df["product_card_id"] = prow["product_card_id"]
        df["delta_beta"] = prow["delta_beta"]
        df["identifiable"] = prow["identifiable"]
        df["elasticity"] = df["elasticity_pool"] + df["delta_beta"]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out[
        ["product_card_id", "year_month", "elasticity",
         "elasticity_pool", "elasticity_pool_se", "delta_beta", "identifiable"]
    ]
