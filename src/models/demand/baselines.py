"""Classical baselines for demand forecasting.

Three families, all with a common interface so they're swappable with the
TimesFM model in the eval harness:

- SeasonalNaiveForecaster — y_hat_{t+h} = y_{t+h-12}; quantiles via residual std.
- ETSForecaster — Holt-Winters additive (level + trend + 12-month seasonality).
- ProphetForecaster — Facebook Prophet with yearly seasonality, monthly grain.

Interface
---------
Each class exposes:

    def forecast(self, history: np.ndarray, horizon: int) -> dict
        # returns {"q10": np.ndarray[h], "q50": np.ndarray[h], "q90": np.ndarray[h]}

`history` is the in-sample quantity series (most recent value last). NaN-safe:
each forecaster fills internal NaNs with 0 (zero-demand months).
"""
from __future__ import annotations

# Force torch-first import order to avoid Windows c10.dll conflicts.
from . import _preamble  # noqa: F401

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# statsmodels & prophet are imported lazily so the module remains importable
# even if one of them is missing.

SEASONAL_LAG = 12


def _z_from_p(p: float) -> float:
    """Approximate normal quantile (avoids importing scipy just for this)."""
    return float(np.sqrt(2) * _erfinv(2 * p - 1))


def _erfinv(x: float) -> float:
    # Rational approx good to ~1e-6 in [-0.95, 0.95]; we only call for p=0.1/0.9
    a = 0.147
    ln = np.log(1 - x * x)
    term = 2 / (np.pi * a) + ln / 2
    return float(np.sign(x) * np.sqrt(np.sqrt(term * term - ln / a) - term))


def _empty_quantiles(horizon: int) -> dict[str, np.ndarray]:
    return {"q10": np.zeros(horizon), "q50": np.zeros(horizon), "q90": np.zeros(horizon)}


def _post(q10, q50, q90):
    """Enforce non-negative + monotone quantiles."""
    q50 = np.maximum(q50, 0.0)
    q10 = np.minimum(np.maximum(q10, 0.0), q50)
    q90 = np.maximum(q90, q50)
    return {"q10": q10, "q50": q50, "q90": q90}


# ---------------------------------------------------------------------------
# Seasonal-Naive
# ---------------------------------------------------------------------------
@dataclass
class SeasonalNaiveForecaster:
    seasonality: int = SEASONAL_LAG
    name: str = "seasonal_naive"

    def forecast(self, history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        h = np.nan_to_num(np.asarray(history, dtype=float), nan=0.0)
        if len(h) < self.seasonality:
            mean = float(np.nanmean(h)) if len(h) else 0.0
            std = float(np.nanstd(h)) if len(h) else 0.0
            q50 = np.full(horizon, mean)
        else:
            # Take last `seasonality` values, cycle to length `horizon`
            cycle = h[-self.seasonality:]
            idx = np.arange(horizon) % self.seasonality
            q50 = cycle[idx]
            # Residual std from in-sample seasonal-naive errors
            resid = h[self.seasonality:] - h[:-self.seasonality]
            std = float(np.nanstd(resid))
        # Quantile bands grow with sqrt(horizon)
        spread = std * np.sqrt(np.arange(1, horizon + 1))
        z90 = _z_from_p(0.9)  # ~1.2816
        return _post(q50 - z90 * spread, q50, q50 + z90 * spread)


# ---------------------------------------------------------------------------
# ETS (Exponential Smoothing — Holt-Winters)
# ---------------------------------------------------------------------------
@dataclass
class ETSForecaster:
    seasonality: int = SEASONAL_LAG
    trend: str | None = "add"  # 'add' or None
    seasonal: str | None = "add"  # 'add' or None
    name: str = "ets"

    def forecast(self, history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        h = np.nan_to_num(np.asarray(history, dtype=float), nan=0.0)
        # Need at least 2 full seasonal cycles for seasonal ETS
        if len(h) < 2 * self.seasonality:
            return SeasonalNaiveForecaster(self.seasonality).forecast(history, horizon)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ExponentialSmoothing(
                    h,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonality,
                    initialization_method="estimated",
                ).fit(disp=False)
            except Exception:
                return SeasonalNaiveForecaster(self.seasonality).forecast(history, horizon)

            q50 = np.asarray(model.forecast(horizon), dtype=float)
            resid = h - model.fittedvalues
            std = float(np.nanstd(resid))

        spread = std * np.sqrt(np.arange(1, horizon + 1))
        z90 = _z_from_p(0.9)
        return _post(q50 - z90 * spread, q50, q50 + z90 * spread)


# ---------------------------------------------------------------------------
# SARIMA — classical Box-Jenkins seasonal ARIMA
# ---------------------------------------------------------------------------
#
# Used as the rigorous statistical-baseline against TimesFM. SARIMA has
# specific assumptions (stationarity after differencing, white-noise
# residuals, identifiable AR/MA orders) which we validate per product in
# `src/models/demand/assumptions.py`. When the assumptions fail (sparse
# series, structural break, non-converging residuals), the forecaster
# falls back to seasonal-naive.
#
# Order search: small grid over (p,d,q)(P,D,Q,12), pick AICc-minimum;
# falls back to (1,1,1)(1,1,1,12) if no candidate fits.
@dataclass
class SARIMAForecaster:
    seasonality: int = SEASONAL_LAG
    name: str = "sarima"
    # Explicit candidate orders, chosen to be identifiable on short
    # monthly series (24-36 observations). Combinations of seasonal
    # differencing (D=1) with seasonal AR/MA terms (P>0 or Q>0) tend to
    # produce non-finite AICc on short series, so we exclude them.
    # Sorted by complexity (number of estimated parameters); the search
    # stops after the first finite-AICc fit is found beyond a minimum.
    candidate_orders: tuple[tuple, ...] = (
        # (p, d, q, P, D, Q)  — sorted by parameter count
        (0, 1, 1, 0, 1, 0),   # (0,1,1)(0,1,0)_12   simple seasonal-IMA
        (1, 1, 0, 0, 1, 0),   # (1,1,0)(0,1,0)_12   seasonal-ARI
        (1, 1, 1, 0, 1, 0),   # (1,1,1)(0,1,0)_12   seasonal-ARIMA
        (0, 1, 1, 1, 0, 0),   # (0,1,1)(1,0,0)_12   AR-seasonal
        (0, 1, 1, 0, 0, 1),   # (0,1,1)(0,0,1)_12   MA-seasonal
        (1, 1, 0, 1, 0, 0),
        (1, 1, 1, 1, 0, 0),
        (2, 1, 1, 0, 1, 0),
        (1, 1, 2, 0, 1, 0),
        (0, 1, 0, 0, 1, 0),   # very parsimonious — seasonal random walk + drift
    )

    def _fit_best(self, h: np.ndarray):
        """Try a small explicit list of identifiable orders, return lowest-AICc fit."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        best = None
        for (p, d, q, P, D, Q) in self.candidate_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(
                        h, order=(p, d, q),
                        seasonal_order=(P, D, Q, self.seasonality),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, method="lbfgs", maxiter=100)
                aicc = float(res.aicc)
                if not np.isfinite(aicc):
                    continue
                if best is None or aicc < best[0]:
                    best = (aicc, res, (p, d, q, P, D, Q))
            except Exception:
                continue
        return best   # None if every candidate failed

    def forecast(self, history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        h = np.nan_to_num(np.asarray(history, dtype=float), nan=0.0)
        # SARIMA needs at least 2 full seasonal cycles; below that the
        # seasonal terms are unidentifiable.
        if len(h) < 2 * self.seasonality:
            return SeasonalNaiveForecaster(self.seasonality).forecast(history, horizon)

        best = self._fit_best(h)
        if best is None:
            return SeasonalNaiveForecaster(self.seasonality).forecast(history, horizon)
        _aicc, res, _order = best

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = res.get_forecast(steps=horizon)
                mean = np.asarray(fc.predicted_mean, dtype=float)
                ci = fc.conf_int(alpha=0.2)
                # statsmodels returns ndarray here (not DataFrame); handle both.
                ci_arr = ci.to_numpy() if hasattr(ci, "to_numpy") else np.asarray(ci)
                ci10 = ci_arr[:, 0].astype(float)
                ci90 = ci_arr[:, 1].astype(float)
        except Exception:
            return SeasonalNaiveForecaster(self.seasonality).forecast(history, horizon)

        return _post(ci10, mean, ci90)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------
@dataclass
class ProphetForecaster:
    name: str = "prophet"
    yearly_seasonality: bool = True
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    interval_width: float = 0.8

    def forecast(self, history_df: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
        """history_df: columns ['ds' datetime, 'y' float] sorted by ds."""
        from prophet import Prophet

        df = history_df.copy()
        df["y"] = df["y"].fillna(0.0)
        if len(df) < 6:
            # Prophet needs at least a handful of points
            return SeasonalNaiveForecaster().forecast(df["y"].to_numpy(), horizon)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    interval_width=self.interval_width,
                )
                model.fit(df)
            future = model.make_future_dataframe(periods=horizon, freq="MS")
            fc = model.predict(future).iloc[-horizon:]
            return _post(
                fc["yhat_lower"].to_numpy(dtype=float),
                fc["yhat"].to_numpy(dtype=float),
                fc["yhat_upper"].to_numpy(dtype=float),
            )
        except Exception:
            return SeasonalNaiveForecaster().forecast(df["y"].to_numpy(), horizon)


# ---------------------------------------------------------------------------
# Convenience: run baselines over the full panel and produce a forecast frame
# ---------------------------------------------------------------------------
def forecast_panel_baseline(
    panel: pd.DataFrame,
    forecaster,
    origin: pd.Timestamp,
    horizon: int,
    product_ids: list | None = None,
) -> pd.DataFrame:
    """Generate forecasts for every product (or `product_ids` subset) given
    history through `origin` (inclusive). Returns long-format frame:
        product_card_id, year_month, horizon (1..H), q10, q50, q90, model
    """
    clean = panel[panel["data_quality"] == "ok"].copy()
    months = pd.to_datetime(sorted(clean["year_month"].unique()))
    origin = pd.Timestamp(origin)
    if origin not in months:
        raise ValueError(f"origin {origin} not in panel months")
    origin_idx = list(months).index(origin)
    horizon = min(horizon, len(months) - origin_idx - 1)
    if horizon <= 0:
        return pd.DataFrame(columns=[
            "product_card_id", "year_month", "horizon", "q10", "q50", "q90", "model"
        ])
    forecast_months = months[origin_idx + 1: origin_idx + 1 + horizon]

    pids = product_ids if product_ids is not None else sorted(clean["product_card_id"].unique())
    out_rows: list[pd.DataFrame] = []
    for pid in pids:
        hist = clean[(clean["product_card_id"] == pid) & (clean["year_month"] <= origin)]
        hist = hist.sort_values("year_month")
        if isinstance(forecaster, ProphetForecaster):
            q = forecaster.forecast(hist[["year_month", "qty"]].rename(
                columns={"year_month": "ds", "qty": "y"}), horizon)
        else:
            q = forecaster.forecast(hist["qty"].to_numpy(), horizon)
        out_rows.append(pd.DataFrame({
            "product_card_id": pid,
            "year_month": forecast_months,
            "horizon": np.arange(1, horizon + 1),
            "q10": q["q10"], "q50": q["q50"], "q90": q["q90"],
            "model": forecaster.name,
        }))
    return pd.concat(out_rows, ignore_index=True)
