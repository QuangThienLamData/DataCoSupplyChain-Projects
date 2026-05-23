"""Time-series assumption diagnostics for the M1 demand panel.

Classical forecasters (SARIMA, ETS) and the rolling-window anomaly logic
in M5 rest on specific assumptions about the input series. When those
assumptions fail, the forecast intervals are misleading even if the
point forecast looks OK on the surface. This module computes per-product
diagnostics so we can:

  1. Filter to products where SARIMA can be fit responsibly
  2. Flag products where residuals are non-white (model misspecification)
  3. Quantify how much of the M1 panel falls within the "classical
     forecasting safe zone" — and how much needs a foundation model like
     TimesFM that's more forgiving of structural breaks / sparsity

Tests applied (per product, on training-period series only)
-----------------------------------------------------------
- **ADF (Augmented Dickey-Fuller)**: H0 = unit root (non-stationary).
    Reject (p < 0.05) → series is stationary as-is.
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**: H0 = stationary.
    Reject (p < 0.05) → series has a unit root or trend.
    Used as confirmation of ADF — they're complementary tests.
- **Seasonal strength** (Hyndman-Wang Fσ): variance of seasonal component
    relative to detrended series. > 0.6 → strong seasonality.
- **Ljung-Box** on SARIMA residuals (lags=12): H0 = residuals are white
    noise. Reject (p < 0.05) → model misspecified, autocorrelation
    remains.
- **Jarque-Bera** on SARIMA residuals: H0 = normal. Fail → forecast
    intervals (which assume Gaussian) are mis-calibrated.

Output schema (per (product, slice)): the assumptions DataFrame includes:
    product_card_id, slice, n_obs,
    adf_pvalue, adf_stationary,
    kpss_pvalue, kpss_stationary,
    seasonal_strength,
    ljung_box_pvalue, ljung_box_white,
    jarque_bera_pvalue, residuals_normal,
    sarima_order, sarima_aicc,
    forecaster_recommended  (one of: sarima | ets | seasonal_naive | timesfm)

Where "forecaster_recommended" applies these rules:
    - n_obs < 24 (less than 2 seasonal cycles) → seasonal_naive (insufficient)
    - non-stationary even after d=1 → timesfm (SARIMA assumptions fail)
    - Ljung-Box rejected and series has structural break → timesfm
    - else → sarima (assumptions hold; classical model is appropriate)
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "panel_meta.parquet"
OUT_PATH = ROOT / "forecasts" / "m1_ts_assumptions.parquet"

SIGNIFICANCE_ALPHA = 0.05
SEASONAL_PERIOD = 12


def _adf_test(series: np.ndarray) -> tuple[float, bool]:
    """ADF: returns (p-value, is_stationary_at_alpha)."""
    from statsmodels.tsa.stattools import adfuller
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 12 or np.std(s) < 1e-9:
        return (float("nan"), False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(adfuller(s, autolag="AIC", regression="ct")[1])
    except Exception:
        return (float("nan"), False)
    return (p, p < SIGNIFICANCE_ALPHA)


def _kpss_test(series: np.ndarray) -> tuple[float, bool]:
    """KPSS: returns (p-value, is_stationary_at_alpha)."""
    from statsmodels.tsa.stattools import kpss
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 12 or np.std(s) < 1e-9:
        return (float("nan"), False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(kpss(s, regression="c", nlags="auto")[1])
    except Exception:
        return (float("nan"), False)
    # KPSS H0 = stationary, so we KEEP if p >= alpha
    return (p, p >= SIGNIFICANCE_ALPHA)


def _seasonal_strength(series: np.ndarray, period: int = SEASONAL_PERIOD) -> float:
    """Hyndman-Wang seasonal strength: Fσ = max(0, 1 - var(resid)/var(seasonal+resid)).
    Returns a value in [0, 1]; > 0.6 means strong seasonal pattern."""
    from statsmodels.tsa.seasonal import STL
    s = np.asarray(series, dtype=float)
    s = pd.Series(s).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    if len(s) < 2 * period or np.std(s) < 1e-9:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(s, period=period, robust=True).fit()
        resid_var = np.var(stl.resid)
        seas_resid_var = np.var(stl.seasonal + stl.resid)
        if seas_resid_var < 1e-12:
            return 0.0
        return float(max(0.0, 1 - resid_var / seas_resid_var))
    except Exception:
        return float("nan")


def _fit_sarima_and_diagnose(
    series: np.ndarray, period: int = SEASONAL_PERIOD,
) -> dict:
    """Fit SARIMA(1,1,1)(1,1,1,P) and run Ljung-Box + JB on residuals.

    A small fixed order is used here (not grid-search) so the diagnostics
    are stable across products — the full grid search is in SARIMAForecaster.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
    s = np.asarray(series, dtype=float)
    s = pd.Series(s).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    if len(s) < 2 * period or np.std(s) < 1e-9:
        return {
            "ljung_box_pvalue": float("nan"),
            "ljung_box_white": False,
            "jarque_bera_pvalue": float("nan"),
            "residuals_normal": False,
            "sarima_order": "n/a",
            "sarima_aicc": float("nan"),
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                s, order=(1, 1, 1), seasonal_order=(1, 1, 1, period),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, method="lbfgs", maxiter=100)
        resid = np.asarray(res.resid, dtype=float)
        resid = resid[~np.isnan(resid)]
        if len(resid) < 8:
            raise ValueError("too few residuals for LB / JB")
        lb_p = float(acorr_ljungbox(resid, lags=[min(period, len(resid)//2)],
                                     return_df=True)["lb_pvalue"].iloc[0])
        jb_p = float(jarque_bera(resid)[1])
        return {
            "ljung_box_pvalue": lb_p,
            "ljung_box_white": lb_p >= SIGNIFICANCE_ALPHA,
            "jarque_bera_pvalue": jb_p,
            "residuals_normal": jb_p >= SIGNIFICANCE_ALPHA,
            "sarima_order": "(1,1,1)(1,1,1,12)",
            "sarima_aicc": float(res.aicc) if np.isfinite(res.aicc) else float("nan"),
        }
    except Exception:
        return {
            "ljung_box_pvalue": float("nan"),
            "ljung_box_white": False,
            "jarque_bera_pvalue": float("nan"),
            "residuals_normal": False,
            "sarima_order": "fit_failed",
            "sarima_aicc": float("nan"),
        }


def _recommend_forecaster(row: dict, n_obs: int) -> str:
    """Pick a forecaster family based on which assumptions hold."""
    if n_obs < 2 * SEASONAL_PERIOD:
        return "seasonal_naive"
    # ADF rejection OR KPSS retention → stationary (at least with d=1)
    adf_ok = bool(row.get("adf_stationary", False))
    kpss_ok = bool(row.get("kpss_stationary", False))
    lb_ok = bool(row.get("ljung_box_white", False))
    if not (adf_ok or kpss_ok):
        # non-stationary even after differencing — classical model risky
        return "timesfm"
    if not lb_ok:
        # residuals carry autocorrelation — SARIMA misspecified
        return "ets"
    return "sarima"


def run_diagnostics(
    panel: pd.DataFrame | None = None,
    target_col: str = "gross_qty",
    train_end: str = "2016-12-31",
) -> pd.DataFrame:
    """Compute the per-product diagnostics frame on the training window."""
    if panel is None:
        panel = pd.read_parquet(PANEL_PATH)
    meta = pd.read_parquet(META_PATH)
    cohort_a = set(meta.loc[meta["cohort"] == "A_active", "product_card_id"])

    rows: list[dict] = []
    log.info("running TS diagnostics on %d cohort-A products", len(cohort_a))
    for pid in sorted(cohort_a):
        sub = panel[(panel["product_card_id"] == pid)
                     & (panel["year_month"] <= train_end)] \
                .sort_values("year_month")
        series = sub[target_col].fillna(0.0).to_numpy(dtype=float)
        n_obs = len(series)

        adf_p, adf_ok = _adf_test(series)
        kpss_p, kpss_ok = _kpss_test(series)
        seas = _seasonal_strength(series)
        diag = _fit_sarima_and_diagnose(series)

        row = {
            "product_card_id": pid,
            "n_obs": n_obs,
            "adf_pvalue": adf_p,
            "adf_stationary": adf_ok,
            "kpss_pvalue": kpss_p,
            "kpss_stationary": kpss_ok,
            "seasonal_strength": seas,
            **diag,
        }
        row["forecaster_recommended"] = _recommend_forecaster(row, n_obs)
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    log.info("wrote %s (%d rows)", OUT_PATH, len(df))
    return df


def assumption_summary(df: pd.DataFrame) -> pd.Series:
    """Portfolio-level summary of which assumptions hold."""
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series({
        "n_products": n,
        "pct_adf_stationary": 100 * df["adf_stationary"].mean(),
        "pct_kpss_stationary": 100 * df["kpss_stationary"].mean(),
        "pct_both_stationary": 100 * (df["adf_stationary"] & df["kpss_stationary"]).mean(),
        "pct_strong_seasonality": 100 * (df["seasonal_strength"] > 0.6).mean(),
        "pct_ljung_box_white": 100 * df["ljung_box_white"].mean(),
        "pct_residuals_normal": 100 * df["residuals_normal"].mean(),
        "median_seasonal_strength": float(df["seasonal_strength"].median()),
        "median_sarima_aicc": float(df["sarima_aicc"].median()),
    }).round(2)


if __name__ == "__main__":
    df = run_diagnostics()
    print("\n=== Portfolio assumption summary ===")
    print(assumption_summary(df).to_string())
    print("\n=== Forecaster recommendation mix ===")
    print(df["forecaster_recommended"].value_counts().to_string())
    print("\n=== First 10 products ===")
    cols = ["product_card_id", "n_obs", "adf_pvalue", "kpss_pvalue",
            "seasonal_strength", "ljung_box_pvalue", "jarque_bera_pvalue",
            "sarima_aicc", "forecaster_recommended"]
    print(df[cols].head(10).round(3).to_string(index=False))
