"""Evaluation harness for the M1 Demand forecaster (and any other
quantile-emitting forecaster in this project).

Metrics
-------
- sMAPE — symmetric MAPE, scale-free, robust to zero-actuals (returns 0 when
  both forecast and actual are 0).
- WAPE  — weighted absolute % error; portfolio-friendly because it can be
  aggregated by summing numerator/denominator independently.
- MASE  — mean absolute scaled error; baseline = seasonal-naive (lag-12) on the
  train slice. Comparable across products of different scales.
- Quantile coverage — fraction of actuals inside [P10, P90].

Conventions
-----------
- `forecast_df` has columns: product_card_id, year_month, horizon (int),
  q10, q50, q90 (floats). q50 = point forecast.
- `actual_df`   has columns: product_card_id, year_month, qty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# core metrics (vectorised; nan-safe)
# ---------------------------------------------------------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return float(np.nanmean(out))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    num = np.nansum(np.abs(y_pred - y_true))
    den = np.nansum(np.abs(y_true))
    return float(num / den) if den > 0 else float("nan")


def mase(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray,
         seasonality: int = 12) -> float:
    """Mean Absolute Scaled Error. Scale = mean abs seasonal-naive error on train.

    If the training slice is too short for the seasonality, falls back to lag-1.
    Returns NaN if scale is 0 (constant training series).
    """
    train_y = np.asarray(train_y, float)
    lag = seasonality if len(train_y) > seasonality else 1
    scale = np.nanmean(np.abs(train_y[lag:] - train_y[:-lag])) if len(train_y) > lag else np.nan
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    err = np.nanmean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)))
    return float(err / scale)


def coverage(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    inside = (y_true >= np.asarray(q_lo, float)) & (y_true <= np.asarray(q_hi, float))
    return float(np.nanmean(inside.astype(float)))


# ---------------------------------------------------------------------------
# scoring frames
# ---------------------------------------------------------------------------
@dataclass
class EvalSlice:
    """One evaluation cut: a forecast frame + the actuals to score against,
    plus the training history for MASE scaling."""
    forecast: pd.DataFrame   # product_card_id, year_month, horizon, q10, q50, q90
    actual: pd.DataFrame     # product_card_id, year_month, qty
    train_history: pd.DataFrame  # product_card_id, year_month, qty

    def merged(self) -> pd.DataFrame:
        df = self.forecast.merge(
            self.actual[["product_card_id", "year_month", "qty"]],
            on=["product_card_id", "year_month"],
            how="inner",
            validate="one_to_one",
        )
        return df


def score_per_product(slice_: EvalSlice) -> pd.DataFrame:
    """Per-product metrics over the slice."""
    m = slice_.merged()
    train = slice_.train_history.sort_values(["product_card_id", "year_month"])

    rows: list[dict] = []
    for pid, g in m.groupby("product_card_id", sort=True):
        train_y = train.loc[train["product_card_id"] == pid, "qty"].to_numpy()
        rows.append({
            "product_card_id": pid,
            "n_obs": len(g),
            "smape": smape(g["qty"], g["q50"]),
            "wape":  wape(g["qty"], g["q50"]),
            "mase":  mase(g["qty"], g["q50"], train_y),
            "coverage_80": coverage(g["qty"], g["q10"], g["q90"]),
        })
    return pd.DataFrame(rows)


def score_portfolio(slice_: EvalSlice) -> dict:
    """Portfolio-level metrics. WAPE is weighted by total demand."""
    m = slice_.merged()
    return {
        "n_obs": len(m),
        "smape_mean": float(score_per_product(slice_)["smape"].mean()),
        "wape": wape(m["qty"], m["q50"]),
        "coverage_80": coverage(m["qty"], m["q10"], m["q90"]),
    }


# ---------------------------------------------------------------------------
# rolling-origin backtest helper
# ---------------------------------------------------------------------------
def rolling_origins(
    panel: pd.DataFrame,
    first_origin: str,
    last_origin: str,
    horizon: int = 1,
) -> Iterable[pd.Timestamp]:
    """Yield each forecast origin t (the last observed month) so that t+1..t+h
    sit inside [first_origin, last_origin] in the panel."""
    months = pd.to_datetime(sorted(panel["year_month"].unique()))
    first = pd.Timestamp(first_origin)
    last = pd.Timestamp(last_origin)
    for i, m in enumerate(months[:-horizon]):
        # m is the origin; predictions are months[i+1 .. i+horizon]
        first_pred = months[i + 1]
        last_pred = months[i + horizon]
        if first_pred >= first and last_pred <= last:
            yield m


def compare_models(per_product_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-product tables from multiple models into one comparison frame."""
    frames = [df.assign(model=name) for name, df in per_product_tables.items()]
    return pd.concat(frames, ignore_index=True)
